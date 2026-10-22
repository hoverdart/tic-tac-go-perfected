"""Read date-keyed boards from Google's published game data, without executing JS.

The bundle format is undocumented. Reject incomplete or ambiguous catalogs rather
than guessing at cells or silently accepting a partial extraction. Only successful
snapshots are cached; a short failure backoff avoids repeated title-lookup retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, UTC
from hashlib import sha256
from html import unescape
import json
import logging
import re
from threading import Lock
import time
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


OFFICIAL_GAME_URL = "https://dailygames.discover.google.com/games/tic-tac-go"
_SCRIPT_RE = re.compile(r'<script[^>]+src=["\']([^"\']*GoogleGamesPlatformUi[^"\']*)', re.I)
_ENTRY_START_RE = re.compile(r'\["(\d{8})",\s*\{')
_ENTRY_RE = re.compile(
    r'\["(?P<id>\d{8})",\s*\{name:"(?P<name>(?:\\.|[^"\\])*)",\s*'
    r'I:new [\w$]+\((?P<year>\d+),\s*(?P<month>\d+),\s*(?P<day>\d+)\),\s*'
    r'width:(?P<width>\d+),\s*height:(?P<height>\d+),\s*'
    r'level:"(?P<level>[^"\\]*)"\}\]'
)
_CELL_MAP = {"W": "B", "P": "U", "-": "", "X": "X", "O": "O"}
logger = logging.getLogger("tic_tac_go.catalog")


class CatalogError(RuntimeError):
    """Catalog unavailable, unsupported, invalid, or missing the requested date."""


@dataclass(frozen=True)
class CatalogEntry:
    puzzle_date: date
    title: str
    width: int
    height: int
    level: str

    @property
    def board_sha256(self) -> str:
        return sha256(f"{self.width}x{self.height}:{self.level}".encode()).hexdigest()

    def board(self) -> list[list[str]]:
        # Preserve the published coordinates, including its outer wall border.
        board = [[_CELL_MAP[c] for c in self.level[r*self.width:(r+1)*self.width]]
                 for r in range(self.height)]
        return [row + ["B"] * (8-self.width) for row in board] + [
            ["B"] * 8 for _ in range(8-self.height)
        ]


@dataclass(frozen=True)
class CatalogSnapshot:
    entries: Mapping[date, CatalogEntry]
    source_url: str
    source_sha256: str
    fetched_at: str


def parse_catalog(source: str, *, source_url: str = "") -> CatalogSnapshot:
    entries: dict[date, CatalogEntry] = {}
    starts = _ENTRY_START_RE.findall(source)
    matches = list(_ENTRY_RE.finditer(source))
    if not starts or len(matches) != len(starts):
        raise CatalogError("Unsupported or incomplete official board catalog.")
    for match in matches:
        raw = match.groupdict()
        try:
            puzzle_date = date(int(raw["year"]), int(raw["month"]), int(raw["day"]))
            if puzzle_date.strftime("%Y%m%d") != raw["id"] or puzzle_date in entries:
                raise ValueError("duplicate or mismatched date")
            width, height = int(raw["width"]), int(raw["height"])
            level = raw["level"]
            title = json.loads('"' + raw["name"] + '"').strip()
            if not (3 <= width <= 8 and 3 <= height <= 8):
                raise ValueError("unsupported dimensions")
            if len(level) != width * height or set(level) - _CELL_MAP.keys():
                raise ValueError("invalid cell data")
            if level.count("P") != 1 or level.count("O") < 2:
                raise ValueError("invalid player or O count")
            if not title or len(title) > 80:
                raise ValueError("invalid title")
        except (ValueError, TypeError) as exc:
            raise CatalogError(f"Invalid official board {raw['id']}: {exc}") from exc
        entries[puzzle_date] = CatalogEntry(puzzle_date, title, width, height, level)
    return CatalogSnapshot(
        MappingProxyType(entries), source_url, sha256(source.encode()).hexdigest(),
        datetime.now(UTC).isoformat(),
    )


def _fetch_text(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000:
        raise CatalogError("Official catalog response is too large.")
    return data.decode("utf-8")


def fetch_catalog() -> CatalogSnapshot:
    try:
        page = _fetch_text(OFFICIAL_GAME_URL, 12)
        match = _SCRIPT_RE.search(page)
        if not match:
            raise CatalogError("Official game bundle was not found.")
        script_url = urljoin(OFFICIAL_GAME_URL, unescape(match.group(1)))
        module_url = re.sub(r"m=_b(?=$|[;&])", "m=Aeygpd", script_url)
        parsed = urlparse(module_url)
        if module_url == script_url or parsed.scheme != "https" or parsed.hostname != "www.gstatic.com":
            raise CatalogError("Unsupported official game bundle URL.")
        return parse_catalog(_fetch_text(module_url, 18), source_url=module_url)
    except CatalogError:
        raise
    except Exception as exc:
        raise CatalogError(f"Could not load official board catalog: {exc}") from exc


_lock = Lock()
_snapshot: CatalogSnapshot | None = None
_expires_at = 0.0
_retry_after = 0.0
_last_error = ""


def load_catalog(*, force_refresh: bool = False) -> CatalogSnapshot:
    global _snapshot, _expires_at, _retry_after, _last_error
    with _lock:
        now = time.monotonic()
        if not force_refresh and _snapshot is not None and now < _expires_at:
            return _snapshot
        if not force_refresh and now < _retry_after:
            raise CatalogError(_last_error)
        try:
            snapshot = fetch_catalog()
        except CatalogError as exc:
            _retry_after = time.monotonic() + 30
            _last_error = str(exc)
            raise
        _snapshot = snapshot
        _expires_at = time.monotonic() + 3600
        _retry_after = 0
        logger.info("catalog.loaded entries=%s source_sha256=%s source_url=%s",
                    len(snapshot.entries), snapshot.source_sha256, snapshot.source_url)
        return snapshot


def entry_for_date(puzzle_date: date) -> tuple[CatalogEntry, CatalogSnapshot]:
    snapshot = load_catalog()
    entry = snapshot.entries.get(puzzle_date)
    if entry is None:
        raise CatalogError(f"Official catalog has no board for {puzzle_date}.")
    return entry, snapshot
