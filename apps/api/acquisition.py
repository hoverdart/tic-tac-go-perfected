"""
    This is Playwright, a module that allows us to open 
    a local browser, navigate to the Tic Tac Go page on 
    Google, and capture a screenshot of the game board.
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


class BoardCaptureError(RuntimeError):
    """Raised when the Google board cannot be captured."""


@dataclass(frozen=True)
class RemoteBrowserTarget:
    provider: str
    source: str
    url: str


def _dismiss_tutorial_overlay(page) -> bool:
    """Click the Google Tic Tac Go tutorial Skip button when it appears."""
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
    if response is None:
        return None

    try:
        return response.status
    except Exception:
        return None


def _wait_for_network_idle(page) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception as exc:
        logger.info("capture.networkidle_timeout_or_unavailable error=%s", exc)


def _log_page_state(page, phase: str) -> None:
    try:
        title = page.title()
    except Exception:
        title = "<unavailable>"

    logger.info("capture.page_state phase=%s title=%r current_url=%s", phase, title, page.url)


def google_tic_tac_go_url() -> str:
    return os.getenv("GOOGLE_TIC_TAC_GO_URL", DEFAULT_GOOGLE_TIC_TAC_GO_URL)


def _env_value(*names: str) -> str | None:
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
    return urlparse(url).netloc or None


def _browserbase_connect_url(create_session: bool = True) -> RemoteBrowserTarget | None:
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
    if "token=" not in value:
        return value

    parsed = urlparse(value)
    tokens = parse_qs(parsed.query).get("token")
    return tokens[0].strip() if tokens else value.split("token=", 1)[1].split("&", 1)[0].strip()


def _browserless_connect_url() -> RemoteBrowserTarget | None:
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
    return (
        "Vercel cannot bundle Chromium for Playwright. Recommended setup: set "
        "REMOTE_BROWSER_PROVIDER=browserless and BROWSERLESS_TOKEN to your "
        "Browserless token. Alternatively set BROWSERLESS_WS_URL, "
        "PLAYWRIGHT_CDP_URL, or BROWSERBASE_API_KEY."
    )


def _running_on_vercel() -> bool:
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


def capture_google_board_screenshot(source_url: str | None = None) -> Path:
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

    logger.info("capture.start url=%s screenshot_path=%s", url, screenshot_path)

    try:
        with sync_playwright() as playwright:
            remote_browser_url = _remote_browser_url()
            if remote_browser_url:
                browser = playwright.chromium.connect_over_cdp(remote_browser_url)
                logger.info("capture.browser_connected chromium=remote")
            elif _running_on_vercel():
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
            _wait_for_network_idle(page)
            _log_page_state(page, "after_goto")
            page.wait_for_timeout(2_000)
            dismissed = _dismiss_tutorial_overlay(page)
            logger.info("capture.tutorial_dismissed=%s", dismissed)
            _log_page_state(page, "before_screenshot")
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise BoardCaptureError(f"Timed out while loading Google Tic Tac Go: {url}") from exc
    except Exception as exc:
        raise BoardCaptureError(f"Could not capture Google Tic Tac Go: {exc}") from exc

    if not screenshot_path.is_file():
        raise BoardCaptureError("Playwright finished without producing a screenshot.")

    logger.info(
        "capture.done screenshot_path=%s bytes=%s",
        screenshot_path,
        screenshot_path.stat().st_size,
    )
    return screenshot_path
