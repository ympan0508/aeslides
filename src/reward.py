import math


def calculate_asymmetric_quadratic_reward(
    w,
    h,
    target=1280 / 720,
    alpha=16.0,
    beta=64.0,
    margin=0.04,
    flat_low_ar=1280 / 720,
    flat_high_ar=1280 / 720,
):
    if target <= 0:
        raise ValueError(f"target must be > 0, got {target!r}")
    if w <= 0 or h <= 0:
        return 0.0
    if flat_low_ar <= 0 or flat_high_ar <= 0:
        raise ValueError(
            f"flat_low_ar and flat_high_ar must be > 0, "
            f"got {flat_low_ar!r}, {flat_high_ar!r}"
        )
    if flat_low_ar > flat_high_ar:
        raise ValueError(
            f"flat_low_ar must be <= flat_high_ar, "
            f"got {flat_low_ar!r} > {flat_high_ar!r}"
        )

    aspect_ratio = w / h

    if flat_low_ar <= aspect_ratio <= flat_high_ar:
        return 1.0

    if aspect_ratio > flat_high_ar:
        e = math.log(aspect_ratio / flat_high_ar)
    else:
        e = math.log(aspect_ratio / flat_low_ar)

    base = alpha * (e**2)
    tall_excess = max(-e - margin, 0.0)
    asym = beta * (tall_excess**2)
    return math.exp(-base - asym)


def calculate_smoothstep_reward(
    x: float,
    scale: float = 1.0,
    lower: float = 0.8,
    upper: float = 0.995,
    gamma: float = 1.0,
) -> float:
    x = x * scale

    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0

    t = (x - lower) / (upper - lower)

    a = t ** gamma
    b = (1.0 - t) ** gamma
    u = a / (a + b)

    # smoothstep
    return 3.0 * u * u - 2.0 * u * u * u
