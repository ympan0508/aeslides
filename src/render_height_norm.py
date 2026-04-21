"""
Adaptive-height Playwright renderer.

This module renders HTML with a minimal initial viewport, measures the full
document height, and captures a viewport-only JPEG at the adaptive height.
"""

from __future__ import annotations

from typing import Any, Dict

from playwright.sync_api import sync_playwright


_FULL_HEIGHT_JS = """() => {
    const body = document.body;
    const html = document.documentElement;
    return Math.max(
        body ? body.scrollHeight : 0,
        body ? body.offsetHeight : 0,
        html ? html.clientHeight : 0,
        html ? html.scrollHeight : 0,
        html ? html.offsetHeight : 0,
    ) || 1;
}"""


def render_adaptive_height(
    html_content: str,
    *,
    viewport_width: int = 1280,
    init_viewport_height: int = 10,
    max_viewport_height: int = 7680,
    wait_for_load_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Render HTML with adaptive viewport height and capture a viewport-only JPEG.

    Parameters
    ----------
    html_content:
        HTML content to render.
    viewport_width:
        Target viewport width in pixels.
    init_viewport_height:
        Initial tiny viewport height used before measuring the full document.
    max_viewport_height:
        Upper bound for the final adaptive viewport height.
    wait_for_load_ms:
        Additional wait time in milliseconds after `networkidle`.

    Returns
    -------
    dict
        {
            "image_bytes": bytes,
            "view_width": int,
            "view_height": int,
            "full_height": int,
        }
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": viewport_width, "height": init_viewport_height}
        )

        try:
            page.set_content(html_content)
            page.wait_for_load_state("networkidle")

            if wait_for_load_ms > 0:
                page.wait_for_timeout(wait_for_load_ms)

            full_height = int(page.evaluate(_FULL_HEIGHT_JS))
            view_height = max(1, min(full_height, max_viewport_height))

            page.set_viewport_size({
                "width": viewport_width,
                "height": view_height,
            })

            image_bytes = page.screenshot(full_page=False, type="jpeg")

            return {
                "image_bytes": image_bytes,
                "view_width": viewport_width,
                "view_height": view_height,
                "full_height": full_height,
            }
        finally:
            page.close()
            browser.close()
