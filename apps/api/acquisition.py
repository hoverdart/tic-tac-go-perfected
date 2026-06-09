"""
Board acquisition: visit Google Search, locate the daily Tic Tac Go puzzle,
and return a screenshot of the game canvas together with the puzzle title.

The module supports three browser backends, chosen automatically at runtime:

  1. Local Chromium  — default when running on a developer machine.
  2. Browserless     — a hosted Chrome-as-a-service reachable via WebSocket/CDP.
                       Configured via BROWSERLESS_TOKEN (or BROWSERLESS_WS_URL).
  3. Browserbase     — another hosted browser service that requires creating a
                       session through its REST API before connecting over CDP.
                       Configured via BROWSERBASE_API_KEY (and optionally
                       BROWSERBASE_PROJECT_ID).

Vercel and other serverless runtimes cannot bundle a full Chromium binary, so
the code refuses to start without one of the remote backends when it detects
it is running in that environment.

Public surface
--------------
capture_google_board_screenshot()  — the main entry point called by the solver
remote_browser_diagnostics()       — health-check helper exposed by the API
BoardCaptureError                  — raised on any unrecoverable failure
"""

from __future__ import annotations

import os
import tempfile
import logging
import json
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path


DEFAULT_GOOGLE_TIC_TAC_GO_URL = "https://www.google.com/search?q=tic+tac+go&hl=en&gl=US"
BROWSERBASE_SESSIONS_URL = "https://api.browserbase.com/v1/sessions"
BROWSERLESS_DEFAULT_REGION = "production-sfo"
logger = logging.getLogger("tic_tac_go.daily_solve")

# Google's minified class names change between deployments, so we maintain a
# prioritised list of selectors to try when hunting for the puzzle title.
# The JavaScript DOM-walk in _extract_puzzle_title is the more durable path;
# these CSS selectors are a fast first pass that avoids the JS round-trip when
# a known class happens to be present.
_TITLE_CSS_SELECTORS = [
    ".lnXdpd",              # Google game info container (seen on some Knowledge Panel games)
    ".Bc2kGd",              # Google game header
    ".yTXDjf",              # Google Knowledge Panel heading variant
    ".UWnNse",              # Another Knowledge Panel title class
    "[role='heading']",     # Any ARIA heading near the board
    "canvas.board ~ h3",    # h3 immediately after the board canvas
    "canvas.board ~ * h3",  # h3 inside a sibling of the canvas
]


class BoardCaptureError(RuntimeError):
    """Raised when the Google board cannot be captured."""


@dataclass(frozen=True)
class CaptureResult:
    """
    Everything the caller needs from one acquisition run.

    Both fields travel together because the screenshot and the puzzle title
    are extracted in the same browser session. Bundling them prevents a second
    browser launch just to recover a title that was already on the page.
    puzzle_title is None when title extraction fails; the screenshot is still
    usable for the solver in that case.
    """
    screenshot_path: Path
    puzzle_title: str | None


@dataclass(frozen=True)
class RemoteBrowserTarget:
    """A resolved remote-browser connection target, ready to pass to Playwright."""
    provider: str   # "browserless", "browserbase", or "direct"
    source: str     # which env var(s) supplied the configuration
    url: str        # WebSocket/CDP URL to connect to


def _dismiss_tutorial_overlay(page) -> bool:
    """
    Dismiss the first-visit tutorial overlay that Google occasionally shows
    before the Tic Tac Go board is interactive.

    Google's DOM structure for this overlay is not stable across deployments,
    so we try several selector strategies in priority order rather than
    committing to a single one. If a visible Skip button is found, it is
    clicked and we return True. If nothing visible is found we send Escape
    as a last-ditch attempt and return False — the caller can log the outcome
    but should not treat a False return as fatal.
    """
    candidates = [
        ("button[name=Skip]", page.get_by_role("button", name="Skip")),
        ("text=Skip", page.get_by_text("Skip", exact=True)),
        ("role/button text=Skip", page.locator("button:has-text('Skip'), [role='button']:has-text('Skip')")),
    ]

    hidden_candidates = 0
    for selector_name, locator in candidates:
        try:
            count = locator.count()
            if count == 0:
                continue
        except Exception as exc:
            logger.debug("capture.skip_count_failed selector=%s error=%s", selector_name, exc)
            continue

        for index in range(min(count, 5)):
            candidate = locator.nth(index)
            try:
                if not candidate.is_visible(timeout=500):
                    hidden_candidates += 1
                    continue

                candidate.click(timeout=2_000)
                logger.info("capture.tutorial_dismissed selector=%s index=%s", selector_name, index)
                page.wait_for_timeout(1_000)
                return True
            except Exception as exc:
                logger.debug(
                    "capture.skip_click_candidate_failed selector=%s index=%s error=%s",
                    selector_name,
                    index,
                    exc,
                )

    # Last resort: Escape sometimes closes modal overlays that have no Skip button
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
    except Exception as exc:
        logger.debug("capture.escape_dismiss_failed error=%s", exc)

    if hidden_candidates:
        logger.info("capture.tutorial_not_visible hidden_skip_candidates=%s", hidden_candidates)
    else:
        logger.info("capture.tutorial_not_found")

    return False


def _response_status(response) -> int | None:
    """Return the HTTP status code from a Playwright response, or None if unavailable."""
    if response is None:
        return None

    try:
        return response.status
    except Exception:
        return None


def _wait_for_network_idle(page) -> None:
    """
    Wait until the page has no in-flight network requests.

    We swallow the timeout exception intentionally — network-idle is a
    best-effort signal, not a hard requirement. Some Google Search pages keep
    long-polling connections open indefinitely, so treating a timeout here as
    fatal would break the common case.
    """
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception as exc:
        logger.info("capture.networkidle_timeout_or_unavailable error=%s", exc)


def _log_page_state(page, phase: str) -> None:
    """Log the current page title and URL as a breadcrumb for debugging."""
    try:
        title = page.title()
    except Exception:
        title = "<unavailable>"

    logger.info("capture.page_state phase=%s title=%r current_url=%s", phase, title, page.url)


def _reveal_board_area(page) -> None:
    """
    Scroll down to bring the game board into the viewport.

    Google Search results place the Knowledge Panel game widget below the fold
    on some screen sizes. A small wheel scroll is usually enough to reveal it
    without accidentally scrolling past it.
    """
    try:
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(1_000)
        logger.info("capture.board_reveal_wheel_done")
    except Exception as exc:
        logger.info("capture.board_reveal_wheel_failed error=%s", exc)


_GAME_CONTAINER_SELECTORS = [
    "canvas.board",
    "[data-hveid] .iKdzV",   # Google knowledge-panel game widget wrapper
    "div[id*='ttt']",
    "div[class*='game']",
    "g-scrolling-carousel",
]


def _extract_puzzle_title(page) -> str | None:
    """
    Attempt to read the daily puzzle name (e.g. "Gear Shift") from the loaded page.

    Strategy
    --------
    1. CSS selector pass: try each entry in _TITLE_CSS_SELECTORS in order.
       Google's minified class names are volatile, so this may hit or miss
       depending on the current deployment. When a match is found we validate
       it with a length/content filter to avoid returning UI chrome ("Skip",
       "Tic Tac Go", etc.) as a puzzle title.

    2. JavaScript DOM walk: if no CSS selector matched, inject a script that
       climbs up from `canvas.board` through up to 12 ancestor nodes looking
       for any heading element. This approach is independent of class names and
       is therefore more resilient to Google's minification churn.

    Returns None on any failure so callers are never blocked by a missing title.
    """
    # CSS selector pass
    for selector in _TITLE_CSS_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
            if not count:
                continue
            for i in range(min(count, 3)):
                el = locator.nth(i)
                if not el.is_visible(timeout=300):
                    continue
                text = el.inner_text(timeout=1_000).strip()
                # Filter out UI strings that look like headings but are not puzzle titles
                if 3 <= len(text) <= 60 and text.lower() not in {"tic tac go", "skip", "close"}:
                    logger.info("capture.title_css_found selector=%r title=%r", selector, text)
                    return text
        except Exception as exc:
            logger.debug("capture.title_css_failed selector=%r error=%s", selector, exc)

    # JavaScript fallback: walk up from the board canvas looking for headings.
    # We anchor on canvas.board because it is a stable landmark — its class is
    # set by Google's game code, not by their CSS build pipeline.
    try:
        title = page.evaluate("""
            () => {
                const canvas = document.querySelector('canvas.board');
                if (!canvas) return null;
                const skip = new Set(['tic tac go', 'google', 'skip', 'close', 'back', 'how to play']);
                let node = canvas.parentElement;
                for (let depth = 0; depth < 12 && node; depth++) {
                    const els = Array.from(
                        node.querySelectorAll('h1,h2,h3,h4,h5,[role="heading"]')
                    );
                    for (const el of els) {
                        const text = (el.innerText || el.textContent || '').trim();
                        if (text.length >= 3 && text.length <= 60 && !skip.has(text.toLowerCase())) {
                            return text;
                        }
                    }
                    node = node.parentElement;
                }
                return null;
            }
        """)
        if title:
            logger.info("capture.title_js_found title=%r", title)
            return title
    except Exception as exc:
        logger.debug("capture.title_js_failed error=%s", exc)

    logger.info("capture.title_not_found")
    return None


def _capture_board_image(page, screenshot_path: Path) -> None:
    """
    Save a screenshot of the game board canvas to screenshot_path.

    We wait 8 seconds before attempting the screenshot to give Google's canvas
    rendering time to settle after any animations. If the canvas locator is
    available and visible we screenshot just that element; otherwise we fall
    back to a full-page screenshot so we always produce some output for
    debugging even if the board did not load cleanly.
    """
    page.wait_for_timeout(8_000)
    try:
        board = page.locator("canvas.board").first
        board.wait_for(state="visible", timeout=5_000)
        board.screenshot(path=str(screenshot_path))
        logger.info("capture.board_canvas_screenshot_done")
        return
    except Exception as exc:
        logger.info("capture.board_canvas_screenshot_failed error=%s", exc)

    # Fallback: capture the whole page so we have something to inspect
    page.screenshot(path=str(screenshot_path), full_page=True)
    logger.info("capture.full_page_screenshot_fallback")


def google_tic_tac_go_url() -> str:
    """Return the Google Search URL for Tic Tac Go, overrideable via env var."""
    return os.getenv("GOOGLE_TIC_TAC_GO_URL", DEFAULT_GOOGLE_TIC_TAC_GO_URL)


def _env_value(*names: str) -> str | None:
    """
    Look up the first non-empty value among the given environment variable names.

    Also handles the edge case where a value was accidentally set as
    `VAR_NAME=actual_value` (i.e. the shell assignment leaked into the value
    string), stripping the prefix and surrounding quotes before returning.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            stripped = value.strip().strip('"').strip("'")
            for env_name in names:
                assignment_prefix = f"{env_name}="
                if stripped.startswith(assignment_prefix):
                    stripped = stripped[len(assignment_prefix):].strip().strip('"').strip("'")
                    break
            else:
                return stripped
            return stripped
    return None


def _redacted_url(url: str) -> str:
    """Return the URL with token/key query parameters replaced by '<redacted>' for safe logging."""
    parsed = urlparse(url)
    if not parsed.query:
        return url

    redacted_query = []
    for part in parsed.query.split("&"):
        key = part.split("=", 1)[0]
        if key.lower() in {"token", "api_key", "apikey", "key"}:
            redacted_query.append(f"{key}=<redacted>")
        else:
            redacted_query.append(part)

    return parsed._replace(query="&".join(redacted_query)).geturl()


def _diagnostic_host(url: str) -> str | None:
    """Extract the host:port portion of a URL for log messages."""
    return urlparse(url).netloc or None


def _browserbase_connect_url(create_session: bool = True) -> RemoteBrowserTarget | None:
    """
    Resolve a Browserbase CDP connection URL, optionally creating a new session.

    Browserbase is a managed browser service. Unlike Browserless, it requires
    an explicit REST API call to spin up a session before Playwright can
    connect. That call returns a one-time `connectUrl` (a WebSocket address
    scoped to the new session) which we hand off to Playwright's
    connect_over_cdp().

    When create_session=False (used by the diagnostics endpoint) we return a
    placeholder target without making a network call, so callers can check
    whether Browserbase is configured without incurring a session creation cost.

    Returns None if BROWSERBASE_API_KEY is not configured.
    Raises BoardCaptureError on session creation failures.
    """
    api_key = _env_value("BROWSERBASE_API_KEY", "browserbase_api_key")
    if not api_key:
        return None

    if not create_session:
        return RemoteBrowserTarget(
            provider="browserbase",
            source="BROWSERBASE_API_KEY",
            url=BROWSERBASE_SESSIONS_URL,
        )

    project_id = _env_value("BROWSERBASE_PROJECT_ID", "browserbase_project_id")
    payload: dict[str, object] = {
        "browserSettings": {
            "viewport": {"width": 1280, "height": 1400},
        },
        "timeout": 120,
    }
    if project_id:
        payload["projectId"] = project_id

    request = Request(
        BROWSERBASE_SESSIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-BB-API-Key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BoardCaptureError(
            f"Browserbase session creation failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise BoardCaptureError(f"Could not reach Browserbase: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BoardCaptureError("Browserbase returned invalid JSON.") from exc

    connect_url = body.get("connectUrl")
    if not isinstance(connect_url, str) or not connect_url:
        raise BoardCaptureError("Browserbase did not return a connectUrl.")

    logger.info("capture.browserbase_session_created session_id=%s", body.get("id"))
    return RemoteBrowserTarget(
        provider="browserbase",
        source="BROWSERBASE_API_KEY",
        url=connect_url,
    )


def _browserless_token_from_value(value: str) -> str:
    """
    Extract a bare token string from a value that might be a full WebSocket URL.

    Operators sometimes paste a full Browserless connection URL into the token
    env var. We handle that gracefully by pulling the token= query parameter
    out rather than failing with a confusing auth error.
    """
    if "token=" not in value:
        return value

    parsed = urlparse(value)
    tokens = parse_qs(parsed.query).get("token")
    return tokens[0].strip() if tokens else value.split("token=", 1)[1].split("&", 1)[0].strip()


def _browserless_connect_url() -> RemoteBrowserTarget | None:
    """Build a Browserless WebSocket URL from BROWSERLESS_TOKEN (and optionally BROWSERLESS_REGION).

    Returns None if BROWSERLESS_TOKEN is not set.
    """
    token = _env_value("BROWSERLESS_TOKEN", "browserless_token")
    if not token:
        return None

    token = _browserless_token_from_value(token)
    region = _env_value("BROWSERLESS_REGION", "browserless_region") or BROWSERLESS_DEFAULT_REGION
    return RemoteBrowserTarget(
        provider="browserless",
        source="BROWSERLESS_TOKEN",
        url=f"wss://{region}.browserless.io?token={quote(token, safe='')}",
    )


def _direct_remote_target(source: str, url: str) -> RemoteBrowserTarget:
    """
    Validate and wrap an explicitly provided WebSocket or CDP URL.

    Raises BoardCaptureError with a descriptive message if the URL looks
    obviously wrong (e.g. an HTTP REST endpoint instead of a WebSocket address,
    or a Browserless URL that is missing its token parameter).
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss", "http", "https"}:
        raise BoardCaptureError(
            f"{source} must be a WebSocket/CDP URL. Got unsupported scheme {parsed.scheme!r}."
        )
    if "browserless.io" in parsed.netloc and "/pdf" in parsed.path:
        raise BoardCaptureError(
            f"{source} is a Browserless REST PDF endpoint. Use "
            "`BROWSERLESS_TOKEN` or `wss://production-sfo.browserless.io?token=...` instead."
        )
    if "browserless.io" in parsed.netloc and "token" not in parse_qs(parsed.query):
        raise BoardCaptureError(
            f"{source} points at Browserless but has no token query parameter."
        )

    provider = "browserless" if "browserless.io" in parsed.netloc else "direct"
    return RemoteBrowserTarget(provider=provider, source=source, url=url)


def _remote_browser_target(create_browserbase_session: bool = True) -> RemoteBrowserTarget | None:
    """
    Resolve which remote browser to use, returning a ready-to-connect target.

    Resolution order
    ----------------
    1. REMOTE_BROWSER_PROVIDER — if set, the named provider is used exclusively
       and an error is raised if its required credentials are missing. This
       prevents silent fallthrough to a different provider than the operator
       intended.

    2. BROWSERLESS_TOKEN       — construct a Browserless WSS URL from the token.

    3. BROWSERLESS_WS_URL      — use the raw WebSocket URL as-is.

    4. PLAYWRIGHT_CDP_URL      — use a raw CDP URL (e.g. a self-hosted Chrome).

    5. BROWSERBASE_API_KEY     — create a Browserbase session and connect to it.

    Returns None only when none of the above env vars are configured, which
    means the caller should fall back to launching a local Chromium instance.
    """
    provider = (_env_value("REMOTE_BROWSER_PROVIDER", "remote_browser_provider") or "").lower()

    if provider:
        if provider == "browserless":
            target = _browserless_connect_url()
            if not target:
                raise BoardCaptureError(
                    "REMOTE_BROWSER_PROVIDER=browserless requires BROWSERLESS_TOKEN."
                )
            return target
        if provider == "browserbase":
            target = _browserbase_connect_url(create_session=create_browserbase_session)
            if not target:
                raise BoardCaptureError(
                    "REMOTE_BROWSER_PROVIDER=browserbase requires BROWSERBASE_API_KEY."
                )
            return target
        if provider in {"direct", "url"}:
            direct_url = _env_value("BROWSERLESS_WS_URL") or _env_value("PLAYWRIGHT_CDP_URL")
            if not direct_url:
                raise BoardCaptureError(
                    "REMOTE_BROWSER_PROVIDER=direct requires BROWSERLESS_WS_URL or PLAYWRIGHT_CDP_URL."
                )
            return _direct_remote_target("BROWSERLESS_WS_URL or PLAYWRIGHT_CDP_URL", direct_url)
        if provider not in {"direct", "url"}:
            raise BoardCaptureError(
                "REMOTE_BROWSER_PROVIDER must be browserless, browserbase, direct, or unset."
            )

    # No explicit provider — auto-detect from whichever credentials are present
    browserless_target = _browserless_connect_url()
    if browserless_target:
        return browserless_target

    browserless_url = _env_value("BROWSERLESS_WS_URL")
    if browserless_url:
        return _direct_remote_target("BROWSERLESS_WS_URL", browserless_url)

    cdp_url = _env_value("PLAYWRIGHT_CDP_URL")
    if cdp_url:
        return _direct_remote_target("PLAYWRIGHT_CDP_URL", cdp_url)

    return _browserbase_connect_url(create_session=create_browserbase_session)


def remote_browser_diagnostics() -> dict[str, object]:
    """
    Return a summary of the current remote-browser configuration for the API
    health-check endpoint.

    Avoids creating an actual Browserbase session (create_browserbase_session=False)
    so this can be called cheaply on every health-check request.
    """
    configured = {
        name: _env_value(name) is not None
        for name in (
            "REMOTE_BROWSER_PROVIDER",
            "BROWSERLESS_TOKEN",
            "BROWSERLESS_WS_URL",
            "PLAYWRIGHT_CDP_URL",
            "BROWSERBASE_API_KEY",
            "BROWSERBASE_PROJECT_ID",
        )
    }

    try:
        target = _remote_browser_target(create_browserbase_session=False)
    except BoardCaptureError as exc:
        return {
            "configured": configured,
            "selected_provider": None,
            "selected_source": None,
            "selected_host": None,
            "error": str(exc),
        }

    return {
        "configured": configured,
        "selected_provider": target.provider if target else None,
        "selected_source": target.source if target else None,
        "selected_host": _diagnostic_host(target.url) if target else None,
        "selected_url": _redacted_url(target.url) if target else None,
        "error": None,
    }


def _remote_browser_url() -> str | None:
    """Resolve and log the remote browser URL, returning None if no remote browser is configured."""
    target = _remote_browser_target()
    if target:
        logger.info(
            "capture.remote_browser.selected provider=%s source=%s host=%s",
            target.provider,
            target.source,
            _diagnostic_host(target.url),
        )
        return target.url

    return None


def _missing_remote_browser_message() -> str:
    """Return the error message shown when Vercel is detected but no remote browser is configured."""
    return (
        "Vercel cannot bundle Chromium for Playwright. Recommended setup: set "
        "REMOTE_BROWSER_PROVIDER=browserless and BROWSERLESS_TOKEN to your "
        "Browserless token. Alternatively set BROWSERLESS_WS_URL, "
        "PLAYWRIGHT_CDP_URL, or BROWSERBASE_API_KEY."
    )


def _running_on_vercel() -> bool:
    """
    Detect whether the process is running inside a Vercel (or AWS Lambda) serverless environment.

    Vercel's build pipeline strips native binaries, so the Playwright Chromium
    bundle is unavailable at runtime. We check both Vercel's own markers and
    the underlying Lambda markers that Vercel's infrastructure sets, since
    future Vercel runtime versions might drop the VERCEL_* variables while
    keeping the Lambda ones.
    """
    serverless_markers = (
        "VERCEL",
        "VERCEL_ENV",
        "VERCEL_URL",
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_LAMBDA_RUNTIME_API",
        "AWS_EXECUTION_ENV",
        "LAMBDA_TASK_ROOT",
    )
    return any(os.getenv(marker) for marker in serverless_markers)


def capture_google_board_screenshot(source_url: str | None = None) -> CaptureResult:
    """
    Navigate to the Google Tic Tac Go page, capture the game canvas, and
    extract the daily puzzle title.

    Flow
    ----
    1. Choose a browser backend (remote CDP or local Chromium).
    2. Navigate to the Google Search URL for Tic Tac Go.
    3. Wait for the page to settle (network idle + fixed delay).
    4. Dismiss the tutorial overlay if one appears.
    5. Wait again for any post-dismiss animations.
    6. Screenshot the canvas element (falling back to a full-page screenshot).
    7. Extract the puzzle title from the surrounding DOM.
    8. Return a CaptureResult with both artifacts.

    puzzle_title is initialised to None before the browser block so that if
    title extraction fails partway through — or raises — we can still return a
    valid CaptureResult with the screenshot we already saved rather than losing
    both pieces of data to an exception.

    Raises BoardCaptureError on any unrecoverable failure (timeout, missing
    screenshot file, misconfigured remote browser, etc.).
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BoardCaptureError(
            "Missing dependency: install Playwright with "
            "`python3 -m pip install playwright` and run `playwright install chromium`."
        ) from exc

    url = source_url or google_tic_tac_go_url()
    output_dir = Path(tempfile.mkdtemp(prefix="tic-tac-go-"))
    screenshot_path = output_dir / "google-tic-tac-go.png"
    # Initialised here so it remains accessible after the browser block even if
    # title extraction raises inside the try/except below.
    puzzle_title: str | None = None

    logger.info("capture.start url=%s screenshot_path=%s", url, screenshot_path)

    try:
        with sync_playwright() as playwright:
            remote_browser_url = _remote_browser_url()
            if remote_browser_url:
                browser = playwright.chromium.connect_over_cdp(remote_browser_url)
                logger.info("capture.browser_connected chromium=remote")
            elif _running_on_vercel():
                # No remote browser configured and we can't launch Chromium locally
                raise BoardCaptureError(_missing_remote_browser_message())
            else:
                browser = playwright.chromium.launch(headless=True)
                logger.info("capture.browser_launched chromium=headless")

            page = browser.new_page(
                viewport={"width": 1280, "height": 1400},
                device_scale_factor=1,
            )
            response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            logger.info("capture.goto_done status=%s target_url=%s", _response_status(response), url)
            # Expand the viewport after initial load so the Knowledge Panel game
            # widget has more horizontal room to render fully
            page.set_viewport_size({"width": 1920, "height": 1080})
            logger.info("capture.viewport_set width=1920 height=1080")
            _wait_for_network_idle(page)
            _log_page_state(page, "after_goto")
            page.wait_for_timeout(2_000)
            dismissed = _dismiss_tutorial_overlay(page)
            logger.info("capture.tutorial_dismissed=%s", dismissed)
            page.wait_for_timeout(2_500)
            _log_page_state(page, "before_screenshot")
            _capture_board_image(page, screenshot_path)
            puzzle_title = _extract_puzzle_title(page)
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise BoardCaptureError(f"Timed out while loading Google Tic Tac Go: {url}") from exc
    except Exception as exc:
        raise BoardCaptureError(f"Could not capture Google Tic Tac Go: {exc}") from exc

    if not screenshot_path.is_file():
        raise BoardCaptureError("Playwright finished without producing a screenshot.")

    logger.info(
        "capture.done screenshot_path=%s bytes=%s puzzle_title=%r",
        screenshot_path,
        screenshot_path.stat().st_size,
        puzzle_title,
    )
    return CaptureResult(screenshot_path=screenshot_path, puzzle_title=puzzle_title)
