# -*- coding: utf-8 -*-
"""網站 logo 的唯一來源：圓環（放大鏡／分析）＋環內三根上升 K 線（股市）＋環外斜線。

頁首、載入動畫（stock_dashboard.py）與 assets/logo/ 圖檔都用這裡的座標（54×54）。

  python site_logo.py     # 改完座標後重新產生 assets/logo/ 全部圖檔
"""
from __future__ import annotations

import math
from pathlib import Path

COLOR = "#8e2b2f"
STROKE = 4
RING_ROTATE = (-22, 27, 27)
RING_POINTS = "26.27,10.00 27.73,10.00 27.73,10.00 28.37,10.01 29.01,10.05 29.64,10.11 30.28,10.20 30.90,10.31 31.53,10.45 32.15,10.61 32.76,10.80 33.36,11.01 33.96,11.24 34.54,11.49 35.12,11.77 35.68,12.07 36.23,12.40 36.77,12.74 37.29,13.11 37.80,13.49 38.30,13.90 38.77,14.32 39.23,14.77 39.68,15.23 40.10,15.70 40.51,16.20 40.89,16.71 41.26,17.23 41.60,17.77 41.93,18.32 42.23,18.88 42.51,19.46 42.76,20.04 42.99,20.64 43.20,21.24 43.39,21.85 43.55,22.47 43.69,23.10 43.80,23.72 43.89,24.36 43.95,24.99 43.99,25.63 44.00,26.27 44.00,29.19 44.00,29.19 43.99,29.77 43.95,30.35 43.90,30.93 43.82,31.51 43.72,32.08 43.59,32.65 43.44,33.21 43.28,33.77 43.08,34.32 42.87,34.86 42.64,35.39 42.39,35.91 42.11,36.43 41.82,36.93 41.50,37.42 41.17,37.90 40.82,38.36 40.45,38.81 40.07,39.24 39.66,39.66 39.24,40.07 38.81,40.45 38.36,40.82 37.90,41.17 37.42,41.50 36.93,41.82 36.43,42.11 35.91,42.39 35.39,42.64 34.86,42.87 34.32,43.08 33.77,43.28 33.21,43.44 32.65,43.59 32.08,43.72 31.51,43.82 30.93,43.90 30.35,43.95 29.77,43.99 29.19,44.00 27.73,44.00 27.73,44.00 27.03,43.99 26.34,43.95 25.65,43.88 24.96,43.78 24.27,43.66 23.59,43.51 22.92,43.33 22.25,43.13 21.59,42.90 20.95,42.65 20.31,42.37 19.68,42.07 19.07,41.74 18.47,41.39 17.88,41.01 17.31,40.61 16.75,40.19 16.22,39.75 15.70,39.29 15.19,38.81 14.71,38.30 14.25,37.78 13.81,37.25 13.39,36.69 12.99,36.12 12.61,35.53 12.26,34.93 11.93,34.32 11.63,33.69 11.35,33.05 11.10,32.41 10.87,31.75 10.67,31.08 10.49,30.41 10.34,29.73 10.22,29.04 10.12,28.35 10.05,27.66 10.01,26.97 10.00,26.27 10.00,26.27 10.00,26.27 10.01,25.63 10.05,24.99 10.11,24.36 10.20,23.72 10.31,23.10 10.45,22.47 10.61,21.85 10.80,21.24 11.01,20.64 11.24,20.04 11.49,19.46 11.77,18.88 12.07,18.32 12.40,17.77 12.74,17.23 13.11,16.71 13.49,16.20 13.90,15.70 14.32,15.23 14.77,14.77 15.23,14.32 15.70,13.90 16.20,13.49 16.71,13.11 17.23,12.74 17.77,12.40 18.32,12.07 18.88,11.77 19.46,11.49 20.04,11.24 20.64,11.01 21.24,10.80 21.85,10.61 22.47,10.45 23.10,10.31 23.72,10.20 24.36,10.11 24.99,10.05 25.63,10.01 26.27,10.00"
# (x, 實體上緣 y, 實體下緣 y, 上影線 y, 下影線 y)：由左下往右上走，都在圓環內
CANDLES = ((20, 30.5, 36, 28, 38.2), (26.8, 24, 31.5, 21.5, 34), (33.6, 18.5, 25.5, 16, 28))
CANDLE_WIDTH, WICK_WIDTH = 4.4, 1.7
TAIL = (39.4, 39.4, 47, 47)


def candle_svg(i: int, attrs: str = 'fill="%s" stroke="%s"' % (COLOR, COLOR)) -> str:
    x, top, bottom, wick_top, wick_bottom = CANDLES[i]
    return (f'<g {attrs}><line x1="{x}" x2="{x}" y1="{wick_top}" y2="{wick_bottom}" stroke-width="{WICK_WIDTH}" stroke-linecap="round"/>'
            f'<rect x="{x - CANDLE_WIDTH / 2}" y="{top}" width="{CANDLE_WIDTH}" height="{bottom - top}" rx=".6" stroke="none"/></g>')


def ring_svg(attrs: str = "") -> str:
    a, cx, cy = RING_ROTATE
    return (f'<polygon {attrs} points="{RING_POINTS}" transform="rotate({a} {cx} {cy})" fill="none" '
            f'stroke="{COLOR}" stroke-width="{STROKE}" stroke-linejoin="round"/>')


def tail_svg(attrs: str = "") -> str:
    x1, y1, x2, y2 = TAIL
    return f'<line {attrs} x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLOR}" stroke-width="{STROKE}" stroke-linecap="round"/>'


def svg_markup(size: int | str = 54, css_class: str = "") -> str:
    """一行的 inline SVG（放進 st.markdown 不能有換行）。"""
    cls = f' class="{css_class}"' if css_class else ""
    body = ring_svg() + "".join(candle_svg(i) for i in range(len(CANDLES))) + tail_svg()
    return f'<svg{cls} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 54 54" width="{size}" height="{size}" role="img" aria-label="台股即時互動式分析系統">{body}</svg>'


def render_png(size: int, background=None):
    """用 PIL 畫（4 倍超取樣）；只有產生圖檔時才需要。"""
    from PIL import Image, ImageDraw

    k = size * 4 / 54
    img = Image.new("RGBA", (size * 4, size * 4), background or (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rgb = tuple(int(COLOR[i:i + 2], 16) for i in (1, 3, 5)) + (255,)

    def line(x1, y1, x2, y2, w):
        d.line([(x1 * k, y1 * k), (x2 * k, y2 * k)], fill=rgb, width=max(1, round(w * k)))
        for x, y in ((x1, y1), (x2, y2)):
            r = w * k / 2
            d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=rgb)

    a, cx, cy = RING_ROTATE
    t = math.radians(a)
    pts = []
    for pair in RING_POINTS.split():
        x, y = map(float, pair.split(","))
        pts.append((cx + (x - cx) * math.cos(t) - (y - cy) * math.sin(t), cy + (x - cx) * math.sin(t) + (y - cy) * math.cos(t)))
    for p, q in zip(pts, pts[1:] + pts[:1]):
        line(*p, *q, STROKE)
    for x, top, bottom, wick_top, wick_bottom in CANDLES:
        line(x, wick_top, x, wick_bottom, WICK_WIDTH)
        d.rounded_rectangle([(x - CANDLE_WIDTH / 2) * k, top * k, (x + CANDLE_WIDTH / 2) * k, bottom * k], radius=.6 * k, fill=rgb)
    line(*TAIL, STROKE)
    return img.resize((size, size), Image.LANCZOS)


def write_assets(folder: Path | None = None) -> None:
    folder = folder or Path(__file__).resolve().parent / "assets" / "logo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "logo.svg").write_text(svg_markup(512), encoding="utf-8")
    render_png(512).save(folder / "logo.png")
    render_png(512, (255, 255, 255, 255)).save(folder / "logo_white_bg.png")
    render_png(192).save(folder / "logo_192.png")
    render_png(64).save(folder / "favicon.png")
    render_png(64).save(folder / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])


if __name__ == "__main__":
    write_assets()
