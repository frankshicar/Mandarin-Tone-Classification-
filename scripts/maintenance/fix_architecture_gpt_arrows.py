from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/imagegen/mandarin_hearing_ai_architecture_gpt.png"
OUT = ROOT / "output/imagegen/mandarin_hearing_ai_architecture_gpt_fixed.png"

SCALE = 4

BLUE = (42, 92, 182, 255)
PURPLE = (66, 27, 156, 255)
GREEN_BORDER = (28, 111, 24, 255)
WHITE = (255, 255, 255, 255)
PANEL_BLUE = (241, 247, 253, 255)
PANEL_GREEN = (238, 251, 241, 255)
BOX_GREEN = (247, 253, 248, 255)


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
    dash: int = 16,
    gap: int = 10,
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
    size: int = 16,
) -> None:
    tail_x, tail_y = tail[0] * SCALE, tail[1] * SCALE
    head_x, head_y = head[0] * SCALE, head[1] * SCALE
    angle = math.atan2(head_y - tail_y, head_x - tail_x)
    size *= SCALE
    spread = math.radians(28)
    left = (
        head_x - size * math.cos(angle - spread),
        head_y - size * math.sin(angle - spread),
    )
    right = (
        head_x - size * math.cos(angle + spread),
        head_y - size * math.sin(angle + spread),
    )
    draw.polygon([(head_x, head_y), left, right], fill=color)


def erase_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE), fill=color)


def erase_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(scaled(points), fill=color, width=width * SCALE, joint="curve")


def main() -> None:
    base = Image.open(SRC).convert("RGBA")
    w, h = base.size

    # Apply small cleanup patches directly on the base image.
    cleanup = Image.new("RGBA", (w * SCALE, h * SCALE), (0, 0, 0, 0))
    cleanup_draw = ImageDraw.Draw(cleanup)

    # Remove the old dashed return line that stopped below the Qwen3-TTS box.
    erase_line(cleanup_draw, [(101, 367), (145, 367)], WHITE, width=10)
    erase_line(cleanup_draw, [(145, 367), (194, 367)], PANEL_BLUE, width=10)
    draw_arrowhead(cleanup_draw, (170, 367), (194, 367), PANEL_BLUE, size=18)

    # Remove the incorrect arrow from approved stimuli to confusion analysis.
    erase_rect(cleanup_draw, (612, 826, 758, 852), WHITE)
    erase_rect(cleanup_draw, (758, 826, 801, 852), PANEL_GREEN)
    erase_rect(cleanup_draw, (792, 820, 806, 860), BOX_GREEN)

    # Restore the left border segment of the confusion-analysis box.
    draw_polyline(cleanup_draw, [(793, 820), (793, 860)], GREEN_BORDER, width=4)

    cleanup = cleanup.resize((w, h), Image.Resampling.LANCZOS)
    base.alpha_composite(cleanup)

    # Draw the corrected arrow layer with smoother antialiasing.
    overlay = Image.new("RGBA", (w * SCALE, h * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dashed review loop now returns fully to Qwen3-TTS.
    dashed_path = [(194, 669), (101, 669), (101, 328), (194, 328)]
    draw_dashed_polyline(draw, dashed_path, BLUE, width=4, dash=14, gap=8)
    draw_arrowhead(draw, (170, 328), (194, 328), BLUE, size=16)

    # Approved stimuli now feed participant listening/repetition in Stage 2.
    stage2_path = [(613, 840), (742, 840), (742, 327), (793, 327)]
    draw_polyline(draw, stage2_path, BLUE, width=4)
    draw_arrowhead(draw, (766, 327), (793, 327), BLUE, size=18)

    # Crisp overlay for the answer-key routing so the purple line is aligned.
    draw_polyline(draw, [(532, 165), (532, 194), (1170, 194)], PURPLE, width=4)
    draw_polyline(draw, [(399, 194), (399, 229)], PURPLE, width=4)
    draw_arrowhead(draw, (399, 208), (399, 229), PURPLE, size=16)
    draw_polyline(draw, [(1170, 194), (1170, 210), (1288, 210), (1288, 669), (1207, 669)], PURPLE, width=4)
    draw_arrowhead(draw, (1235, 669), (1207, 669), PURPLE, size=18)

    overlay = overlay.resize((w, h), Image.Resampling.LANCZOS)
    base.alpha_composite(overlay)
    base.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
