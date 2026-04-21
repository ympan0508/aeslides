"""
Playwright renderer for screenshots and rich bbox trees.

This module renders HTML in Playwright, captures a screenshot, and extracts a
rich bbox tree enriched with lightweight DOM semantics.

Due to proprietary constraints, this script only shows a subset of the
heuristic rules. Please extend it based on your actual use case and add more
heuristic rules as needed to ensure detection accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from playwright.sync_api import sync_playwright


STRICT_FIX720_WIDTH = 1280
STRICT_FIX720_HEIGHT = 720


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


_RICH_BBOX_TREE_JS = """() => {
    const root = document.documentElement;
    if (!root) {
        return {
            surface: { width: 0, height: 0 },
            tree: null,
        };
    }

    const surfaceWidth = Math.max(
        window.innerWidth || 0,
        document.documentElement ? document.documentElement.clientWidth : 0,
        0,
    );
    const surfaceHeight = Math.max(
        window.innerHeight || 0,
        document.documentElement ? document.documentElement.clientHeight : 0,
        0,
    );

    const clamp = (value, minimum, maximum) => {
        if (!Number.isFinite(value)) {
            return minimum;
        }
        return Math.min(Math.max(value, minimum), maximum);
    };

    const clipRect = (rect) => {
        const left = clamp(rect.left, 0, surfaceWidth);
        const top = clamp(rect.top, 0, surfaceHeight);
        const right = clamp(rect.right, 0, surfaceWidth);
        const bottom = clamp(rect.bottom, 0, surfaceHeight);
        return {
            x: left,
            y: top,
            width: Math.max(0, right - left),
            height: Math.max(0, bottom - top),
        };
    };

    const parseNumeric = (raw) => {
        const value = Number.parseFloat(raw || "");
        return Number.isFinite(value) ? value : null;
    };

    const parseColor = (raw) => {
        if (!raw || raw === "transparent") {
            return [0, 0, 0, 0];
        }
        const rgbaMatch = raw.match(/rgba?\\(([^)]+)\\)/i);
        if (!rgbaMatch) {
            return null;
        }
        const parts = rgbaMatch[1].split(",").map((part) => part.trim());
        if (parts.length < 3) {
            return null;
        }
        const r = clamp(Number.parseFloat(parts[0]), 0, 255);
        const g = clamp(Number.parseFloat(parts[1]), 0, 255);
        const b = clamp(Number.parseFloat(parts[2]), 0, 255);
        const a = parts.length >= 4 ? clamp(Number.parseFloat(parts[3]), 0, 1) : 1;
        return [r, g, b, a];
    };

    const normalizeText = (raw, limit) => {
        const cleaned = String(raw || "").replace(/\\s+/g, " ").trim();
        if (!cleaned) {
            return "";
        }
        return cleaned.length <= limit ? cleaned : cleaned.slice(0, limit);
    };

    const directTextContent = (el) => {
        let result = "";
        for (const node of el.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                result += ` ${node.textContent || ""}`;
            }
        }
        return normalizeText(result, 160);
    };

    const collectDirectTextRects = (el) => {
        const rects = [];
        for (const node of el.childNodes) {
            if (node.nodeType !== Node.TEXT_NODE) {
                continue;
            }
            const text = normalizeText(node.textContent || "", 400);
            if (!text) {
                continue;
            }
            const range = document.createRange();
            range.selectNodeContents(node);
            for (const rawRect of Array.from(range.getClientRects())) {
                const clipped = clipRect(rawRect);
                if (clipped.width <= 0 || clipped.height <= 0) {
                    continue;
                }
                rects.push(clipped);
                if (rects.length >= 12) {
                    return rects;
                }
            }
        }
        return rects;
    };

    const isRenderable = (el, inheritedVisible) => {
        if (!inheritedVisible) {
            return false;
        }
        const style = window.getComputedStyle(el);
        if (!style) {
            return true;
        }
        if (style.display === "none" || style.visibility === "hidden") {
            return false;
        }
        return Number.parseFloat(style.opacity || "1") > 0;
    };

    const buildMeta = (el) => {
        const style = window.getComputedStyle(el);
        const classTokens = Array.from(el.classList || []);
        const innerText = normalizeText(el.innerText || el.textContent || "", 240);
        const directText = directTextContent(el);
        const textRects = collectDirectTextRects(el);
        const fontFamily = normalizeText(style.fontFamily || "", 120).toLowerCase();
        const tagName = el.tagName.toLowerCase();

        const isIconLike =
            classTokens.some((token) => token.toLowerCase().includes("icon"))
            || fontFamily.includes("material icons")
            || (tagName === "i" && innerText.length <= 16);

        const isImageLike = ["img", "picture", "canvas", "video", "svg"].includes(tagName);

        const isTextLike = !!innerText && (
            ["p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "small", "strong", "label", "li", "td", "th", "a", "button"].includes(tagName)
            || textRects.length > 0
            || innerText.length >= 24
        );

        return {
            id: el.id || null,
            class_tokens: classTokens,
            text: innerText,
            text_len: innerText.length,
            direct_text: directText,
            direct_text_len: directText.length,
            text_rects: textRects,
            style: {
                position: style.position || null,
                z_index: parseNumeric(style.zIndex),
                opacity: parseNumeric(style.opacity),
                overflow_x: style.overflowX || null,
                overflow_y: style.overflowY || null,
                pointer_events: style.pointerEvents || null,
                font_size_px: parseNumeric(style.fontSize),
                line_height_px: parseNumeric(style.lineHeight),
                font_weight: style.fontWeight || null,
                text_align: style.textAlign || null,
                color_rgba: parseColor(style.color),
                background_rgba: parseColor(style.backgroundColor),
                border_radius_px: parseNumeric(style.borderRadius),
                background_image: style.backgroundImage && style.backgroundImage !== "none"
                    ? normalizeText(style.backgroundImage, 120)
                    : null,
            },
            flags: {
                is_text_like: isTextLike,
                is_image_like: isImageLike,
                is_icon_like: isIconLike,
                has_background_image: !!(style.backgroundImage && style.backgroundImage !== "none"),
            },
        };
    };

    const buildNode = (el, inheritedVisible) => {
        const selfVisible = isRenderable(el, inheritedVisible);
        const children = [];

        for (const child of el.children) {
            const childNode = buildNode(child, selfVisible);
            if (childNode) {
                children.push(childNode);
            }
        }

        const bbox = clipRect(el.getBoundingClientRect());
        const hasArea = bbox.width > 0 && bbox.height > 0;

        if ((!selfVisible || !hasArea) && children.length === 0) {
            return null;
        }

        return {
            node_type: el.tagName.toLowerCase(),
            bbox,
            meta: buildMeta(el),
            children,
        };
    };

    return {
        surface: {
            width: surfaceWidth,
            height: surfaceHeight,
        },
        tree: buildNode(root, true),
    };
}"""


def render_rich_bbox_tree_fullpage(
    html_content: str,
    *,
    viewport_width: int = 1280,
    initial_viewport_height: int = 720,
    max_viewport_height: int = 7680,
    wait_for_load_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Render HTML, expand viewport to full page height, capture screenshot,
    and extract a rich bbox tree.

    Returns
    -------
    dict
        {
            "image_bytes": bytes,
            "view_width": int,
            "view_height": int,
            "bbox_tree": dict,
            "surface_width": int,
            "surface_height": int,
            "full_height": int,
        }
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": viewport_width, "height": initial_viewport_height}
        )
        try:
            page.set_content(html_content)
            page.wait_for_load_state("networkidle")
            if wait_for_load_ms > 0:
                page.wait_for_timeout(wait_for_load_ms)

            full_height = int(page.evaluate(_FULL_HEIGHT_JS))
            view_height = max(1, min(full_height, max_viewport_height))
            page.set_viewport_size({"width": viewport_width, "height": view_height})

            bbox_tree, surface_width, surface_height = _collect_rich_bbox_tree(page)

            image_bytes = page.screenshot(full_page=True, type="jpeg")

            return {
                "image_bytes": image_bytes,
                "view_width": viewport_width,
                "view_height": view_height,
                "bbox_tree": bbox_tree,
                "surface_width": surface_width,
                "surface_height": surface_height,
                "full_height": full_height,
            }
        finally:
            page.close()
            browser.close()


def render_rich_bbox_tree_fixed_720p(
    html_content: str,
    *,
    wait_for_load_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Render HTML in a strict 1280x720 viewport, capture screenshot,
    and extract a rich bbox tree clipped to that viewport.

    Returns
    -------
    dict
        {
            "image_bytes": bytes,
            "view_width": int,
            "view_height": int,
            "bbox_tree": dict,
            "surface_width": int,
            "surface_height": int,
            "full_height": int,
            "overflow_height": int,
        }
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": STRICT_FIX720_WIDTH, "height": STRICT_FIX720_HEIGHT}
        )
        try:
            page.set_content(html_content)
            page.wait_for_load_state("networkidle")
            if wait_for_load_ms > 0:
                page.wait_for_timeout(wait_for_load_ms)

            full_height = int(page.evaluate(_FULL_HEIGHT_JS))
            page.set_viewport_size(
                {"width": STRICT_FIX720_WIDTH, "height": STRICT_FIX720_HEIGHT}
            )

            bbox_tree, surface_width, surface_height = _collect_rich_bbox_tree(page)

            image_bytes = page.screenshot(full_page=False, type="jpeg")

            return {
                "image_bytes": image_bytes,
                "view_width": STRICT_FIX720_WIDTH,
                "view_height": STRICT_FIX720_HEIGHT,
                "bbox_tree": bbox_tree,
                "surface_width": surface_width,
                "surface_height": surface_height,
                "full_height": full_height,
                "overflow_height": max(0, full_height - STRICT_FIX720_HEIGHT),
            }
        finally:
            page.close()
            browser.close()


def _collect_rich_bbox_tree(page: Any) -> tuple[Dict[str, Any], int, int]:
    payload = page.evaluate(_RICH_BBOX_TREE_JS)
    if not isinstance(payload, Mapping):
        raise ValueError("rich bbox tree JS must return a mapping payload")

    tree = payload.get("tree")
    if not isinstance(tree, Mapping):
        raise ValueError("rich bbox tree JS did not return a valid root node")

    surface = payload.get("surface", {})
    if isinstance(surface, Mapping):
        try:
            surface_width = int(surface.get("width", 0))
            surface_height = int(surface.get("height", 0))
        except (TypeError, ValueError):
            surface_width = 0
            surface_height = 0
    else:
        surface_width = 0
        surface_height = 0

    return dict(tree), surface_width, surface_height
