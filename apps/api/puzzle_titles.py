"""
Puzzle title lookup for Tic Tac Go boards.

This stays separate from image parsing on purpose: the parser only turns pixels
into a board, while this module finds a human-readable puzzle name from either
the live Google DOM or the historical board manifest.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
import logging
import re
from typing import Any


logger = logging.getLogger("tic_tac_go.daily_solve")

_TITLE_CSS_SELECTORS = [
    "[data-attrid='title']",
    "[data-attrid*='title']",
    "g-card [role='heading']",
    "[role='heading']",
    "h1",
    "h2",
    "h3",
    ".lnXdpd",
    ".Bc2kGd",
    ".yTXDjf",
    ".UWnNse",
    "canvas.board ~ h3",
    "canvas.board ~ * h3",
]

_SKIP_EXACT = {
    "back",
    "close",
    "google",
    "how to play",
    "play",
    "search",
    "skip",
    "tic tac go",
}

_SKIP_PHRASES = (
    "google search",
    "search results",
    "sign in",
)

_TITLE_PREFIX_RE = re.compile(r"^tic\s*tac\s*go(?:\s*[-:|]\s*|\s+)", re.IGNORECASE)
_TITLE_SUFFIX_RE = re.compile(r"\s*(?:[-|]\s*)?(?:google\s*search|tic\s*tac\s*go)\s*$", re.IGNORECASE)

_GENERIC_TITLES = {
    "a google game",
    "tic tac go",
    "tic tac go daily solver",
    "tic tac go solution",
}

_SLASH_DATE_RE = re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*(?:\d{2}|\d{4})$")


def _normalize_title_text(text: str) -> str:
    """Normalize case and punctuation before comparing generic title text."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _date_key(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")

    stripped = value.strip()
    if re.fullmatch(r"\d{8}", stripped):
        return stripped
    try:
        return date.fromisoformat(stripped).strftime("%Y%m%d")
    except ValueError:
        return None


def is_generic_title(text: Any) -> bool:
    """Return whether text is a generic product heading, ignoring punctuation."""
    if not isinstance(text, str):
        return False
    return _normalize_title_text(text) in _GENERIC_TITLES


def clean_puzzle_title(text: Any) -> str | None:
    """Return a usable puzzle title or None for generic/non-title text."""
    if not isinstance(text, str):
        return None

    cleaned = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'")
    if is_generic_title(cleaned):
        return None
    if _SLASH_DATE_RE.fullmatch(cleaned):
        return None
    cleaned = _TITLE_PREFIX_RE.sub("", cleaned).strip(" \t\r\n\"'-:|")
    cleaned = _TITLE_SUFFIX_RE.sub("", cleaned).strip(" \t\r\n\"'-:|")
    if not 3 <= len(cleaned) <= 60:
        return None

    lowered = cleaned.lower()
    if lowered in _SKIP_EXACT:
        return None
    if any(phrase in lowered for phrase in _SKIP_PHRASES):
        return None

    return cleaned


# Retained for compatibility with existing callers and tests.
_clean_title = clean_puzzle_title


@lru_cache(maxsize=1)
def _historical_titles() -> dict[str, str]:
    try:
        from backfill_solutions import ALL_PAST_DAYS
    except Exception as exc:
        logger.debug("title_fetcher.backfill_import_failed error=%s", exc)
        return {}

    titles: dict[str, str] = {}
    for entry in ALL_PAST_DAYS:
        if not isinstance(entry, dict):
            continue
        key = _date_key(str(entry.get("id", "")))
        title = _clean_title(entry.get("name"))
        if key and title:
            titles[key] = title
    return titles


def title_from_past_days(puzzle_date: date | datetime | str | None) -> str | None:
    """Return the manifest title for a date, if the historical list has one."""
    key = _date_key(puzzle_date)
    if not key:
        return None
    title = _historical_titles().get(key)
    if title:
        logger.info("title_fetcher.backfill_found date=%s title=%r", key, title)
    return title


def _title_from_css(page) -> str | None:
    for selector in _TITLE_CSS_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
        except Exception as exc:
            logger.debug("title_fetcher.css_count_failed selector=%r error=%s", selector, exc)
            continue

        for index in range(min(count, 6)):
            try:
                candidate = locator.nth(index)
                if not candidate.is_visible(timeout=300):
                    continue
                title = _clean_title(candidate.inner_text(timeout=1_000))
                if title:
                    logger.info(
                        "title_fetcher.css_found selector=%r index=%s title=%r",
                        selector,
                        index,
                        title,
                    )
                    return title
            except Exception as exc:
                logger.debug(
                    "title_fetcher.css_candidate_failed selector=%r index=%s error=%s",
                    selector,
                    index,
                    exc,
                )
    return None


def _title_from_dom_walk(page) -> str | None:
    try:
        candidates = page.evaluate(
            """
            () => {
                const skipExact = new Set([
                    'a google game',
                    'back',
                    'close',
                    'google',
                    'how to play',
                    'play',
                    'search',
                    'skip',
                    'tic tac go',
                ]);
                const skipPhrases = ['google search', 'search results', 'sign in'];

                function clean(text) {
                    if (!text) return null;
                    const value = String(text).replace(/\\s+/g, ' ').trim().replace(/^["']|["']$/g, '');
                    const lower = value.toLowerCase();
                    const normalized = lower.replace(/[^a-z0-9]+/g, ' ').trim();
                    if (value.length < 3 || value.length > 60) return null;
                    if (skipExact.has(normalized)) return null;
                    if (/^\d{1,2}\s*\/\s*\d{1,2}\s*\/\s*(?:\d{2}|\d{4})$/.test(value)) return null;
                    if (lower.startsWith('tic tac go ')) return null;
                    if (skipPhrases.some((phrase) => lower.includes(phrase))) return null;
                    return value;
                }

                function visible(el) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.visibility !== 'hidden'
                        && style.display !== 'none'
                        && rect.width > 0
                        && rect.height > 0;
                }

                const selectors = [
                    '[data-attrid="title"]',
                    '[data-attrid*="title"]',
                    'h1,h2,h3,h4,h5,[role="heading"]',
                    '[aria-label]'
                ];
                const roots = [];
                const canvas = document.querySelector('canvas.board');
                let node = canvas ? canvas.parentElement : null;
                for (let depth = 0; depth < 14 && node; depth++) {
                    roots.push(node);
                    node = node.parentElement;
                }
                roots.push(document);

                const seen = new Set();
                const results = [];
                for (const root of roots) {
                    for (const selector of selectors) {
                        for (const el of Array.from(root.querySelectorAll(selector))) {
                            if (seen.has(el) || !visible(el)) continue;
                            seen.add(el);
                            const title = clean(el.innerText || el.textContent || el.getAttribute('aria-label'));
                            if (title) results.push(title);
                        }
                    }
                    if (results.length) break;
                }
                return results;
            }
            """
        )
    except Exception as exc:
        logger.debug("title_fetcher.dom_walk_failed error=%s", exc)
        return None

    if not isinstance(candidates, list):
        return None

    for candidate in candidates:
        title = _clean_title(candidate)
        if title:
            logger.info("title_fetcher.dom_walk_found title=%r", title)
            return title
    return None


def _title_from_page_metadata(page) -> str | None:
    """Use document-level title/meta text as a fallback when page markup moves."""
    candidates: list[Any] = []

    try:
        candidates.append(page.title())
    except Exception as exc:
        logger.debug("title_fetcher.page_title_failed error=%s", exc)

    try:
        metadata = page.evaluate(
            """
            () => [
                document.querySelector('meta[property="og:title"]')?.content,
                document.querySelector('meta[name="twitter:title"]')?.content,
                document.querySelector('meta[name="title"]')?.content,
                document.querySelector('[aria-label*="Tic Tac Go" i]')?.getAttribute('aria-label'),
            ].filter(Boolean)
            """
        )
        if isinstance(metadata, list):
            candidates.extend(metadata)
    except Exception as exc:
        logger.debug("title_fetcher.metadata_failed error=%s", exc)

    for candidate in candidates:
        title = _clean_title(candidate)
        if title:
            logger.info("title_fetcher.metadata_found title=%r", title)
            return title
    return None


def extract_puzzle_title(page, puzzle_date: date | datetime | str | None = None) -> str | None:
    """
    Read the puzzle title from the page, falling back to the historical manifest.

    The DOM is tried first because it reflects the live Google page. The
    historical manifest is a strong fallback for known dates and also keeps
    pending/failed records from losing a title when Google changes markup.
    """
    title = _title_from_css(page) or _title_from_dom_walk(page) or _title_from_page_metadata(page)
    if title:
        return title

    fallback = title_from_past_days(puzzle_date)
    if fallback:
        return fallback

    logger.info("title_fetcher.not_found date=%s", _date_key(puzzle_date))
    return None
