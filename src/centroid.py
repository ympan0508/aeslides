"""
Rich semantic visual-centroid metric.

This module computes a weighted visual centroid from a rich bbox tree and
measures layout imbalance relative to the viewport center.

Due to proprietary constraints, this script only shows a subset of the
heuristic rules. Please extend it based on your actual use case and add more
heuristic rules as needed to ensure detection accuracy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class WeightedItem:
    center_x: float
    center_y: float
    weight: float
    source: str


def compute_visual_centroid_v2(
    *,
    bbox_tree: Mapping[str, Any],
    view_width: float,
    view_height: float,
    x_tolerance_norm: float = 0.0075,
    y_tolerance_norm: float = 0.145,
    min_area: float = 4.0,
    min_rect_area: float = 2.0,
    min_opacity: float = 0.05,
    text_rect_switch_text_len: int = 40,
    icon_weight: float = 0.5,
    bg_alpha_min: float = 0.05,
    min_background_area_ratio: float = 0.002,
    max_background_area_ratio: float = 0.75,
    min_background_residual_ratio: float = 0.08,
    max_background_children: int = 12,
    exclude_node_types: Sequence[str] = ("html", "body"),
    atomic_node_types: Sequence[str] = (
        "svg",
        "img",
        "picture",
        "canvas",
        "video",
        "iframe",
        "object",
        "embed",
        "button",
        "input",
        "textarea",
        "select",
        "option",
        "progress",
        "meter",
    ),
) -> Dict[str, Any]:
    """
    Compute the visual centroid and imbalance score from a rich bbox tree.

    Parameters
    ----------
    bbox_tree:
        Root node of a rich bbox tree.
    view_width, view_height:
        Viewport dimensions in pixels.

    Returns
    -------
    dict
        {
            "score": float,
            "is_visually_imbalanced": bool,
            "center_x_px": float,
            "center_y_px": float,
            "center_x_norm": float,
            "center_y_norm": float,
            "offset_x_px": float,
            "offset_y_px": float,
            "offset_x_norm": float,
            "offset_y_norm": float,
            "x_tolerance_px": float,
            "y_tolerance_px": float,
            "nodes_used": int,
            "total_area": float,
            "source_breakdown": dict[str, int],
        }
    """
    excluded_types = _normalize_type_set(exclude_node_types)
    atomic_types = _normalize_type_set(atomic_node_types)
    viewport_area = view_width * view_height

    items = _collect_rich_items(
        bbox_tree,
        excluded_types=excluded_types,
        atomic_types=atomic_types,
        min_area=min_area,
        min_rect_area=min_rect_area,
        min_opacity=min_opacity,
        text_rect_switch_text_len=text_rect_switch_text_len,
        icon_weight=icon_weight,
        bg_alpha_min=bg_alpha_min,
        min_background_area_ratio=min_background_area_ratio,
        max_background_area_ratio=max_background_area_ratio,
        min_background_residual_ratio=min_background_residual_ratio,
        max_background_children=max_background_children,
        view_area=viewport_area,
    )

    return _score_items(
        items,
        view_width=view_width,
        view_height=view_height,
        x_tolerance_norm=x_tolerance_norm,
        y_tolerance_norm=y_tolerance_norm,
    )


def _to_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _to_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _normalize_node_type(raw: Any) -> str:
    return str(raw).strip().lower()


def _normalize_type_set(raw: Sequence[str]) -> set[str]:
    return {
        _normalize_node_type(item)
        for item in raw
        if isinstance(item, str) and item.strip()
    }


def _bbox_values(node: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = node.get("bbox")
    if not isinstance(bbox, Mapping):
        return None
    return (
        _to_float(bbox.get("x")),
        _to_float(bbox.get("y")),
        max(0.0, _to_float(bbox.get("width"))),
        max(0.0, _to_float(bbox.get("height"))),
    )


def _bbox_area(node: Mapping[str, Any]) -> float:
    values = _bbox_values(node)
    if values is None:
        return 0.0
    return values[2] * values[3]


def _bbox_center(node: Mapping[str, Any]) -> tuple[float, float]:
    values = _bbox_values(node)
    if values is None:
        raise ValueError("node is missing bbox")
    x, y, width, height = values
    return x + width / 2.0, y + height / 2.0


def _rect_center(rect: Mapping[str, Any]) -> tuple[float, float]:
    x = _to_float(rect.get("x"))
    y = _to_float(rect.get("y"))
    width = max(0.0, _to_float(rect.get("width")))
    height = max(0.0, _to_float(rect.get("height")))
    return x + width / 2.0, y + height / 2.0


def _rect_area(rect: Mapping[str, Any]) -> float:
    return max(0.0, _to_float(rect.get("width"))) * max(0.0, _to_float(rect.get("height")))


def _iter_children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    children = node.get("children")
    if not isinstance(children, list):
        return []
    return [child for child in children if isinstance(child, Mapping)]


def _meta(node: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = node.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _style(node: Mapping[str, Any]) -> Mapping[str, Any]:
    style = _meta(node).get("style")
    return style if isinstance(style, Mapping) else {}


def _flags(node: Mapping[str, Any]) -> Mapping[str, Any]:
    flags = _meta(node).get("flags")
    return flags if isinstance(flags, Mapping) else {}


def _rects(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rects = _meta(node).get("text_rects")
    if not isinstance(rects, list):
        return []
    return [rect for rect in rects if isinstance(rect, Mapping)]


def _text_len(node: Mapping[str, Any]) -> int:
    return max(0, _to_int(_meta(node).get("text_len")))


def _opacity(node: Mapping[str, Any]) -> float:
    value = _to_float(_style(node).get("opacity"), 1.0)
    return max(0.0, min(1.0, value if value > 0 else 0.0))


def _background_alpha(node: Mapping[str, Any]) -> float:
    rgba = _style(node).get("background_rgba")
    if isinstance(rgba, list) and len(rgba) >= 4:
        return max(0.0, min(1.0, _to_float(rgba[3])))
    return 0.0


def _has_background_image(node: Mapping[str, Any]) -> bool:
    if bool(_flags(node).get("has_background_image")):
        return True
    background_image = _style(node).get("background_image")
    return isinstance(background_image, str) and background_image.strip().lower() != "none"


def _is_text_like(node: Mapping[str, Any]) -> bool:
    return bool(_flags(node).get("is_text_like")) or _text_len(node) > 0


def _is_image_like(node: Mapping[str, Any]) -> bool:
    return bool(_flags(node).get("is_image_like"))


def _is_icon_like(node: Mapping[str, Any]) -> bool:
    return bool(_flags(node).get("is_icon_like"))


def _intersection_area(parent: Mapping[str, Any], child: Mapping[str, Any]) -> float:
    parent_values = _bbox_values(parent)
    child_values = _bbox_values(child)
    if parent_values is None or child_values is None:
        return 0.0

    px, py, pw, ph = parent_values
    cx, cy, cw, ch = child_values

    left = max(px, cx)
    top = max(py, cy)
    right = min(px + pw, cx + cw)
    bottom = min(py + ph, cy + ch)

    return max(0.0, right - left) * max(0.0, bottom - top)


def _intersection_area_with_rects(
    node: Mapping[str, Any],
    rects: Sequence[Mapping[str, Any]],
) -> float:
    parent_values = _bbox_values(node)
    if parent_values is None:
        return 0.0

    px, py, pw, ph = parent_values
    total = 0.0

    for rect in rects:
        rx = _to_float(rect.get("x"))
        ry = _to_float(rect.get("y"))
        rw = max(0.0, _to_float(rect.get("width")))
        rh = max(0.0, _to_float(rect.get("height")))

        left = max(px, rx)
        top = max(py, ry)
        right = min(px + pw, rx + rw)
        bottom = min(py + ph, ry + rh)

        total += max(0.0, right - left) * max(0.0, bottom - top)

    return total


def _collect_rich_items(
    node: Mapping[str, Any],
    *,
    excluded_types: set[str],
    atomic_types: set[str],
    min_area: float,
    min_rect_area: float,
    min_opacity: float,
    text_rect_switch_text_len: int,
    icon_weight: float,
    bg_alpha_min: float,
    min_background_area_ratio: float,
    max_background_area_ratio: float,
    min_background_residual_ratio: float,
    max_background_children: int,
    view_area: float,
) -> list[WeightedItem]:
    children = _iter_children(node)
    items: list[WeightedItem] = []

    for child in children:
        items.extend(
            _collect_rich_items(
                child,
                excluded_types=excluded_types,
                atomic_types=atomic_types,
                min_area=min_area,
                min_rect_area=min_rect_area,
                min_opacity=min_opacity,
                text_rect_switch_text_len=text_rect_switch_text_len,
                icon_weight=icon_weight,
                bg_alpha_min=bg_alpha_min,
                min_background_area_ratio=min_background_area_ratio,
                max_background_area_ratio=max_background_area_ratio,
                min_background_residual_ratio=min_background_residual_ratio,
                max_background_children=max_background_children,
                view_area=view_area,
            )
        )

    area = _bbox_area(node)
    if area < min_area or _opacity(node) < min_opacity:
        return items

    node_type = _normalize_node_type(node.get("node_type", ""))
    if node_type in excluded_types:
        return items

    text_like = _is_text_like(node)
    image_like = _is_image_like(node)
    icon_like = _is_icon_like(node)
    text_rects = _rects(node)
    text_len = _text_len(node)
    has_media = image_like or _has_background_image(node)

    if text_like:
        if text_len > text_rect_switch_text_len and text_rects:
            for rect in text_rects:
                rect_area = _rect_area(rect)
                if rect_area < min_rect_area:
                    continue
                center_x, center_y = _rect_center(rect)
                items.append(
                    WeightedItem(
                        center_x=center_x,
                        center_y=center_y,
                        weight=rect_area,
                        source="text_rect",
                    )
                )
        else:
            center_x, center_y = _bbox_center(node)
            items.append(
                WeightedItem(
                    center_x=center_x,
                    center_y=center_y,
                    weight=area,
                    source="text_bbox",
                )
            )

    if image_like:
        center_x, center_y = _bbox_center(node)
        items.append(
            WeightedItem(
                center_x=center_x,
                center_y=center_y,
                weight=area,
                source="image",
            )
        )
    elif icon_like:
        center_x, center_y = _bbox_center(node)
        items.append(
            WeightedItem(
                center_x=center_x,
                center_y=center_y,
                weight=area * icon_weight,
                source="icon",
            )
        )
    elif not text_like and not children and node_type in atomic_types:
        center_x, center_y = _bbox_center(node)
        items.append(
            WeightedItem(
                center_x=center_x,
                center_y=center_y,
                weight=area,
                source="atomic_fallback",
            )
        )

    if has_media:
        area_ratio = (area / view_area) if view_area > 0 else 0.0
        allow_background = (
            area_ratio >= min_background_area_ratio
            and area_ratio <= max_background_area_ratio
            and len(children) <= max_background_children
            and not image_like
            and (_background_alpha(node) >= bg_alpha_min or _has_background_image(node))
        )

        if allow_background:
            occupied_area = sum(_intersection_area(node, child) for child in children)
            occupied_area += _intersection_area_with_rects(node, text_rects)

            residual_area = max(0.0, area - min(area, occupied_area))
            residual_ratio = (residual_area / area) if area > 0 else 0.0

            if residual_area >= min_area and residual_ratio >= min_background_residual_ratio:
                center_x, center_y = _bbox_center(node)
                items.append(
                    WeightedItem(
                        center_x=center_x,
                        center_y=center_y,
                        weight=residual_area,
                        source="background_residual",
                    )
                )

    return items


def _score_items(
    items: Sequence[WeightedItem],
    *,
    view_width: float,
    view_height: float,
    x_tolerance_norm: float,
    y_tolerance_norm: float,
) -> Dict[str, Any]:
    total_weight = sum(item.weight for item in items)
    if total_weight <= 0:
        raise ValueError("no positive-area visual items were collected")

    center_x = sum(item.center_x * item.weight for item in items) / total_weight
    center_y = sum(item.center_y * item.weight for item in items) / total_weight

    center_x_norm = center_x / view_width
    center_y_norm = center_y / view_height

    offset_x_px = center_x - (view_width / 2.0)
    offset_y_px = center_y - (view_height / 2.0)
    offset_x_norm = offset_x_px / view_width
    offset_y_norm = offset_y_px / view_height

    score = (
        ((abs(offset_x_norm) / x_tolerance_norm) ** 2)
        + ((abs(offset_y_norm) / y_tolerance_norm) ** 2)
    ) ** 0.5

    return {
        "score": score,
        "is_visually_imbalanced": score > 1.0,
        "center_x_px": center_x,
        "center_y_px": center_y,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "offset_x_px": offset_x_px,
        "offset_y_px": offset_y_px,
        "offset_x_norm": offset_x_norm,
        "offset_y_norm": offset_y_norm,
        "x_tolerance_px": x_tolerance_norm * view_width,
        "y_tolerance_px": y_tolerance_norm * view_height,
        "nodes_used": len(items),
        "total_area": total_weight,
        "source_breakdown": dict(Counter(item.source for item in items)),
    }
