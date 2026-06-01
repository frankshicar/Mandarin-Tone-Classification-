from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/imagegen/mandarin_hearing_ai_architecture_gpt.png"
OUT = ROOT / "output/imagegen/mandarin_hearing_ai_architecture_gpt_fixed_v2.png"

SCALE = 4

BLUE = (12, 67, 178, 255)


def scaled(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x * SCALE, y * SCALE) for x, y in points]


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(scaled(points), fill=color, width=width * SCALE, joint="curve")


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
    dash: int = 10,
    gap: int = 8,
) -> None:
    dash *= SCALE
    gap *= SCALE
    pts = scaled(points)
    for start, end in zip(pts, pts[1:]):
        seg_len = distance(start, end)
        if seg_len == 0:
            continue
        dx = (end[0] - start[0]) / seg_len
        dy = (end[1] - start[1]) / seg_len
        drawn = 0.0
        while drawn < seg_len:
            dash_end = min(seg_len, drawn + dash)
            p1 = (start[0] + dx * drawn, start[1] + dy * drawn)
            p2 = (start[0] + dx * dash_end, start[1] + dy * dash_end)
            draw.line([p1, p2], fill=color, width=width * SCALE)
            drawn += dash + gap


def draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    tail: tuple[float, float],
    head: tuple[float, float],
    color: tuple[int, int, int, int],
    size: int = 15,
) -> None:
    tail_x, tail_y = tail[0] * SCALE, tail[1] * SCALE
    head_x, head_y = head[0] * SCALE, head[1] * SCALE
    angle = math.atan2(head_y - tail_y, head_x - tail_x)
    size *= SCALE
    spread = math.radians(29)
    left = (
        head_x - size * math.cos(angle - spread),
        head_y - size * math.sin(angle - spread),
    )
    right = (
        head_x - size * math.cos(angle + spread),
        head_y - size * math.sin(angle + spread),
    )
    draw.polygon([(head_x, head_y), left, right], fill=color)


def paste_patch(
    image: Image.Image,
    src_box: tuple[int, int, int, int],
    dest_xy: tuple[int, int],
) -> None:
    patch = image.crop(src_box)
    image.paste(patch, dest_xy)


def main() -> None:
    base = Image.open(SRC).convert("RGBA")
    w, h = base.size

    # Restore clean background patches copied from neighboring regions of the base image.
    paste_patch(base, (600, 548, 714, 594), (600, 652))
    paste_patch(base, (700, 728, 764, 756), (700, 824))

    overlay = Image.new("RGBA", (w * SCALE, h * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Restore the full dashed return path into the Qwen3-TTS box.
    draw_dashed_polyline(draw, [(158, 367), (200, 367)], BLUE, width=4)
    draw_arrowhead(draw, (175, 367), (200, 367), BLUE, size=18)

    # Re-route approved stimuli into participant listening/repetition.
    draw_polyline(draw, [(613, 840), (710, 840), (710, 327), (793, 327)], BLUE, width=4)
    draw_arrowhead(draw, (768, 327), (793, 327), BLUE, size=17)

    overlay = overlay.resize((w, h), Image.Resampling.LANCZOS)
    base.alpha_composite(overlay)
    base.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
