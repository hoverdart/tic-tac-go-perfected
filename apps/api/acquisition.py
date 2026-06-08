"""
    This is Playwright, a module that allows us to open 
    a local browser, navigate to the Tic Tac Go page on 
    Google, and capture a screenshot of the game board.
"""


import os
import tempfile
import logging
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path


DEFAULT_GOOGLE_TIC_TAC_GO_URL = "https://www.google.com/search?q=tic+tac+go&hl=en&gl=US"
BROWSERBASE_SESSIONS_URL = "https://api.browserbase.com/v1/sessions"
BROWSERLESS_DEFAULT_REGION = "production-sfo"
logger = logging.getLogger("tic_tac_go.daily_solve")


class BoardCaptureError(RuntimeError):
    """Raised when the Google board cannot be captured."""


def _dismiss_tutorial_overlay(page) -> bool:
    """Click the Google Tic Tac Go tutorial Skip button when it appears."""
    candidates = [
        page.get_by_text("Skip", exact=True),
        page.get_by_role("button", name="Skip"),
        page.locator("text=Skip"),
    ]

    for locator in candidates:
        try:
            if locator.count() == 0:
                continue
            first = locator.first
            first.wait_for(state="visible", timeout=2_000)
            first.click(timeout=2_000)
            logger.info("capture.tutorial_dismissed selector=%s", locator)
            page.wait_for_timeout(1_000)
            return True
        except Exception as exc:
            logger.info("capture.skip_click_candidate_failed error=%s", exc)

    logger.info("capture.tutorial_not_found")
    return False


def google_tic_tac_go_url() -> str:
    return os.getenv("GOOGLE_TIC_TAC_GO_URL", DEFAULT_GOOGLE_TIC_TAC_GO_URL)


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            stripped = value.strip().strip('"').strip("'")
            assignment_prefix = f"{name}="
            if stripped.startswith(assignment_prefix):
                stripped = stripped[len(assignment_prefix):].strip().strip('"').strip("'")
            return stripped
    return None


def _browserbase_connect_url() -> str | None:
    api_key = _env_value("BROWSERBASE_API_KEY", "browserbase_api_key")
    if not api_key:
        return None

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
    return connect_url


def _browserless_connect_url() -> str | None:
    token = _env_value("BROWSERLESS_TOKEN", "browserless_token")
    if not token:
        return None

    region = _env_value("BROWSERLESS_REGION", "browserless_region") or BROWSERLESS_DEFAULT_REGION
    return f"wss://{region}.browserless.io?token={token}"


def _remote_browser_url() -> str | None:
    return (
        _env_value("PLAYWRIGHT_CDP_URL")
        or _env_value("BROWSERLESS_WS_URL")
        or _browserless_connect_url()
        or _browserbase_connect_url()
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
                raise BoardCaptureError(
                    "Vercel cannot bundle Chromium for Playwright. Set "
                    "PLAYWRIGHT_CDP_URL, BROWSERLESS_WS_URL, "
                    "BROWSERLESS_TOKEN, or BROWSERBASE_API_KEY to use a "
                    "remote Chromium endpoint."
                )
            else:
                browser = playwright.chromium.launch(headless=True)
                logger.info("capture.browser_launched chromium=headless")

            page = browser.new_page(
                viewport={"width": 1280, "height": 1400},
                device_scale_factor=1,
            )
            page.goto(url, wait_until="networkidle", timeout=45_000)
            logger.info("capture.page_loaded title=%r current_url=%s", page.title(), page.url)
            page.wait_for_timeout(2_000)
            dismissed = _dismiss_tutorial_overlay(page)
            logger.info("capture.tutorial_dismissed=%s", dismissed)
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
