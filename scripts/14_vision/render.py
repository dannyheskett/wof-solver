"""Synthetic screencaps of the puzzle board, each with its exact labels.

render(rng, puzzle) draws one frame and returns (image, label):

    label = {"present": bool,         False for frames with no board
             "grid": [4 x 14 chars],  board.CLASSES per square, ' ' off-grid
             "category": str,         as written on screen, "" if not shown
             "corners": [[x, y] x 4], the 14x4 grid's TL TR BR BL in pixels
             "era": "new" | "old", "source": "broadcast" | "offair" | "phone"}

Two boards are drawn. "new" is the LED board from Season 40 (2022) on: green
unused squares with a pale logo, a dark frame with a cyan edge, the
category in a pill under the board. "old" is the 1997-2022 board: teal
unused squares, a thick light-blue frame, the category in a bar across the
foot of the screen. Both have white squares, near-black letters and blue lit
squares.

Each frame then passes through one source:
    broadcast  a clean capture: camera framing, mild compression
    offair     a home recording: lower resolution, 4:3 crops, blur,
               station logos, picture-in-picture, heavy compression
    phone      a phone pointed at a TV: bezel, room, perspective, moire,
               glare, scanlines, colour cast, noise

A square counts as covered ('~') when more than half of it is hidden behind
the host, an inset, a caption or the frame edge. Everything is drawn with
Pillow and numpy from a seeded numpy Generator, so a seed always gives the
same frame.

    python render.py --preview 24 /tmp/render.jpg   # a contact sheet
"""
import argparse
import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import board as B
import wof_common as wc

W, H = 960, 540
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
]
# pod/pack.sh copies these into data/vision/fonts/, and that copy wins when
# it exists, so frames render with the same fonts here and on a pod.
FONT_DIR = wc.DATA / "vision" / "fonts"
MONO_NAME = "DejaVuSansMono.ttf"
if FONT_DIR.is_dir():
    FONTS = sorted(str(p) for p in FONT_DIR.iterdir() if p.name != MONO_NAME)
    MONO = str(FONT_DIR / MONO_NAME)
else:
    FONTS = [p for p in FONT_PATHS if Path(p).exists()]
    MONO = "/usr/share/fonts/truetype/dejavu/" + MONO_NAME
_font_cache = {}


def font(path, size):
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, max(6, int(size)))
    return _font_cache[key]


def jitter(rng, rgb, amount):
    return tuple(int(np.clip(c + rng.normal(0, amount), 0, 255)) for c in rgb)


def rand_rgb(rng, lo=0, hi=255):
    return tuple(int(x) for x in rng.integers(lo, hi, 3))


# --- puzzle state -----------------------------------------------------------

FREQ = "ETAOINSRHLDCUMFPGWYBVKXJQZ"


def pick_state(rng, answer):
    """Called letters, a lit letter (or None) and whether it is a bonus round."""
    letters = sorted(set(ch for ch in answer if ch in wc.IDX))
    bonus = rng.random() < 0.15
    if bonus:
        called = set("RSTLNE")
        if rng.random() < 0.7:
            cons = [c for c in wc.CONSONANTS if c not in called]
            called |= set(rng.choice(cons, 3, replace=False))
            called.add(str(rng.choice(["A", "I", "O", "U"])))
        if rng.random() < 0.15:
            called = set(letters)
    elif rng.random() < 0.12:
        called = set(letters)
    else:
        n = int(rng.integers(0, 15))
        w = np.array([1.0 / (1 + i) ** 0.6 for i in range(26)])
        called = set(rng.choice(list(FREQ), n, replace=False, p=w / w.sum()))
    lit = None
    hits = [c for c in called if c in letters]
    if hits and rng.random() < 0.25:
        lit = str(rng.choice(hits))
    return called, lit, bonus


def pick_layout(rng, answer):
    lays = B.layouts(answer)
    if not lays:
        return None
    scores = np.array([B.layout_score(l) for l in lays[:6]])
    p = np.exp(-(scores - scores.min()) * 1.2)
    return lays[int(rng.choice(len(p), p=p / p.sum()))]


# --- the board ----------------------------------------------------------------

def board_style(rng, era):
    if era == "new":
        return dict(
            off=jitter(rng, (38, 168, 72), 12), logo=jitter(rng, (120, 225, 140), 15),
            lattice=jitter(rng, (90, 200, 120), 20), back=jitter(rng, (12, 22, 30), 6),
            frame=jitter(rng, (16, 26, 40), 8), edge=jitter(rng, (40, 210, 240), 20),
            edge_w=0.05, frame_w=0.32, lit=jitter(rng, (35, 140, 235), 15),
            white=jitter(rng, (246, 246, 244), 5), ink=jitter(rng, (32, 32, 36), 8),
            radius=0.06)
    return dict(
        off=jitter(rng, (38, 150, 128), 14), logo=jitter(rng, (170, 220, 210), 15),
        lattice=jitter(rng, (20, 60, 70), 10), back=jitter(rng, (18, 50, 60), 8),
        frame=jitter(rng, (205, 225, 245), 15), edge=jitter(rng, (110, 190, 245), 20),
        edge_w=0.12, frame_w=0.45, lit=jitter(rng, (90, 170, 245), 20),
        white=jitter(rng, (242, 244, 246), 6), ink=jitter(rng, (25, 25, 30), 8),
        radius=0.02)


def frame_polygon(x0, y0, pw, ph, m):
    """The stepped outline of the board: rows 1 and 4 are a column narrower
    on each side. m is the margin around the squares in pixels."""
    L, R = x0 - m, x0 + 14 * pw + m
    l1, r1 = x0 + pw - m, x0 + 13 * pw + m
    T, Bo = y0 - m, y0 + 4 * ph + m
    return [(l1, T), (r1, T), (r1, y0 + ph - m * 0.3), (R, y0 + ph - m * 0.3),
            (R, y0 + 3 * ph + m * 0.3), (r1, y0 + 3 * ph + m * 0.3), (r1, Bo),
            (l1, Bo), (l1, y0 + 3 * ph + m * 0.3), (L, y0 + 3 * ph + m * 0.3),
            (L, y0 + ph - m * 0.3), (l1, y0 + ph - m * 0.3)]


def draw_board(img, rng, grid, x0, y0, pw, ph, st, era):
    """Draws the board with its grid's top-left at (x0, y0)."""
    d = ImageDraw.Draw(img)
    m = st["frame_w"] * pw
    if era == "new":
        d.polygon(frame_polygon(x0, y0, pw, ph, m * 1.15), fill=st["frame"])
    else:
        d.polygon(frame_polygon(x0, y0, pw, ph, m * 1.25), fill=st["edge"])
        d.polygon(frame_polygon(x0, y0, pw, ph, m), fill=st["frame"])
    d.polygon(frame_polygon(x0, y0, pw, ph, m * 0.3), fill=st["back"])
    if era == "new":
        d.line(frame_polygon(x0, y0, pw, ph, m * 0.85) + [frame_polygon(x0, y0, pw, ph, m * 0.85)[0]],
               fill=st["edge"], width=max(2, int(st["edge_w"] * pw)))
    # lattice lines between squares
    lw = max(1, int(pw * 0.04))
    for r in range(5):
        y = y0 + r * ph
        d.line([(x0, y), (x0 + 14 * pw, y)], fill=st["lattice"], width=lw)
    for c in range(15):
        x = x0 + c * pw
        d.line([(x, y0), (x, y0 + 4 * ph)], fill=st["lattice"], width=lw)
    fpath = FONTS[int(rng.integers(len(FONTS)))]
    f = font(fpath, ph * rng.uniform(0.6, 0.75))
    inset = pw * rng.uniform(0.07, 0.11)
    rad = int(st["radius"] * pw)
    for r, c in B.SQUARES:
        ch = grid[r][c]
        bx0, by0 = x0 + c * pw + inset, y0 + r * ph + inset
        bx1, by1 = x0 + (c + 1) * pw - inset, y0 + (r + 1) * ph - inset
        if ch == B.OFF:
            d.rounded_rectangle([bx0, by0, bx1, by1], rad, fill=st["off"])
            cx, cy, rr = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) * 0.24
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=st["logo"],
                      width=max(1, int(rr * 0.35)))
            if era == "new":
                d.ellipse([cx - rr * 0.35, cy - rr * 0.35, cx + rr * 0.35, cy + rr * 0.35],
                          fill=st["logo"])
        elif ch == B.LIT:
            d.rounded_rectangle([bx0, by0, bx1, by1], rad, fill=st["lit"])
        else:
            d.rounded_rectangle([bx0, by0, bx1, by1], rad, fill=st["white"])
            if ch != B.HIDDEN:
                d.text(((bx0 + bx1) / 2 + rng.normal(0, pw * 0.02),
                        (by0 + by1) / 2 + rng.normal(0, ph * 0.02)),
                       ch, font=f, fill=st["ink"], anchor="mm")


# --- the scene ----------------------------------------------------------------

def background(rng, era):
    a, b = rand_rgb(rng, 20, 230), rand_rgb(rng, 0, 200)
    t = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(a) * (1 - t) + np.array(b) * t) * np.ones((1, W, 1))
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    if rng.random() < 0.6:   # wheel spokes behind the board
        cx, cy = W * rng.uniform(0.2, 0.8), H * rng.uniform(-0.2, 0.4)
        col = rand_rgb(rng, 150, 255)
        for k in range(int(rng.integers(12, 40))):
            ang = rng.uniform(0, 2 * math.pi)
            d.line([(cx, cy), (cx + 2000 * math.cos(ang), cy + 2000 * math.sin(ang))],
                   fill=col, width=int(rng.integers(2, 8)))
    for _ in range(int(rng.integers(0, 14))):   # trees, lights, props
        x, y, s = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(20, 200)
        shape = d.ellipse if rng.random() < 0.6 else d.rectangle
        shape([x, y, x + s * rng.uniform(0.5, 2), y + s], fill=rand_rgb(rng))
    if era == "new" and rng.random() < 0.8:   # the gold arch around the board
        col = jitter(rng, (210, 175, 90), 20)
        cx, cy = W / 2 + rng.normal(0, 40), H * 0.4
        rx, ry = W * rng.uniform(0.42, 0.55), H * rng.uniform(0.38, 0.5)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
        for k in range(48):
            ang = 2 * math.pi * k / 48
            d.line([(cx, cy), (cx + rx * math.cos(ang), cy + ry * math.sin(ang))],
                   fill=jitter(rng, col, 30), width=int(rng.integers(3, 9)))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 3)))


def draw_person(rng, mask_draw, d, x, foot, height):
    """A host-like figure: head, body and dress. Drawn on the frame and on
    the covering mask."""
    skin = jitter(rng, (205, 160, 130), 30)
    dress = rand_rgb(rng)
    w = height * rng.uniform(0.17, 0.24)
    head = height * 0.11
    top = foot - height
    for dr, fill in ((d, None), (mask_draw, 255)):
        dr.ellipse([x - head * 0.8, top, x + head * 0.8, top + head * 2],
                   fill=fill if fill else skin)
        dr.rounded_rectangle([x - w / 2, top + head * 1.9, x + w / 2, foot],
                             int(w * 0.3), fill=fill if fill else dress)


CARD_WORDS = ["WHEEL OF FORTUNE", "#WHEELOFFORTUNE", "BONUS ROUND", "PRIZE PUZZLE", "EXPRESS",
              "TOSS-UP", "MYSTERY ROUND", "JACKPOT", "SPIN", "COMING UP", "$1,000,000", "AMERICA'S GAME"]


def title_card(rng, img, d):
    """A graphic over the whole frame: big words on colour, as the show's
    title cards and end screens. Letters in rows, but no board."""
    for _ in range(int(rng.integers(1, 4))):
        txt = str(rng.choice(CARD_WORDS))
        f = font(FONTS[int(rng.integers(len(FONTS)))], H * rng.uniform(0.06, 0.2))
        d.text((W * rng.uniform(0.2, 0.8), H * rng.uniform(0.15, 0.85)), txt, font=f,
               fill=rand_rgb(rng, 150, 255), anchor="mm")
    if rng.random() < 0.4:   # a row of icons or small boxes along the foot
        y = H * rng.uniform(0.8, 0.95)
        for k in range(int(rng.integers(3, 8))):
            x = W * (0.3 + 0.06 * k)
            d.ellipse([x, y, x + H * 0.04, y + H * 0.04], fill=rand_rgb(rng, 100, 255))


def wheel_frame(rng):
    """A frame with no board: a close-up of the wheel, people, or a title card."""
    img = background(rng, "old")
    d = ImageDraw.Draw(img)
    kind = rng.random()
    if kind < 0.3:
        title_card(rng, img, d)
        if rng.random() < 0.5:
            mask = Image.new("L", (W, H))
            draw_person(rng, ImageDraw.Draw(mask), d, rng.uniform(100, W - 100), H * 1.05, H * rng.uniform(0.8, 1.4))
    elif kind < 0.7:
        cx, cy, r = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.6, 1.4), W * rng.uniform(0.4, 0.9)
        n = 24
        f = font(FONTS[0], r * 0.06)
        for k in range(n):
            a0, a1 = 360 * k / n, 360 * (k + 1) / n
            d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=rand_rgb(rng, 40, 255))
            am = math.radians((a0 + a1) / 2)
            d.text((cx + r * 0.8 * math.cos(am), cy + r * 0.8 * math.sin(am)),
                   str(int(rng.integers(5, 10)) * 100), font=f, fill=(20, 20, 20), anchor="mm")
    else:
        mask = Image.new("L", (W, H))
        md = ImageDraw.Draw(mask)
        for _ in range(int(rng.integers(1, 4))):
            draw_person(rng, md, d, rng.uniform(100, W - 100), H * 1.05, H * rng.uniform(0.8, 1.6))
    return img


def category_text(rng, cat, bonus):
    t = B.show_category(cat, rng)
    return t


def draw_category(img, rng, text, era, x0, y0, pw, ph):
    d = ImageDraw.Draw(img)
    gw = 14 * pw
    if era == "new":
        cy = y0 + 4 * ph * rng.uniform(1.18, 1.32)
        hh = 4 * ph * 0.08
        x_l, x_r = x0 + gw * 0.17, x0 + gw * 0.83
        d.rounded_rectangle([x_l, cy - hh, x_r, cy + hh], int(hh), fill=jitter(rng, (25, 25, 35), 8),
                            outline=jitter(rng, (200, 165, 80), 20), width=max(1, int(hh * 0.08)))
        f = font(FONTS[int(rng.integers(len(FONTS)))], hh * 1.0)
        d.text(((x_l + x_r) / 2, cy), text, font=f, fill=(245, 245, 245), anchor="mm")
    else:
        cy = H * rng.uniform(0.86, 0.93)
        hh = H * rng.uniform(0.025, 0.04)
        d.rectangle([0, cy - hh, W, cy + hh], fill=jitter(rng, (20, 40, 110), 15))
        f = font(FONTS[int(rng.integers(len(FONTS)))], hh * 1.3)
        d.text((W / 2 + rng.normal(0, 30), cy), text, font=f, fill=(245, 245, 245), anchor="mm")


def overlays(img, rng, mask, bonus, called):
    """Score bar, RSTLNE strip, picture-in-picture, station logo, captions.
    Anything that can sit over the board is drawn on the mask as well."""
    d, md = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    if rng.random() < 0.2:   # contestant scores across the top
        y = H * rng.uniform(0.0, 0.2)
        for k in range(3):
            x = W * (0.25 + 0.2 * k)
            d.rectangle([x, y, x + W * 0.15, y + H * 0.06], fill=rand_rgb(rng, 100, 255))
            d.text((x + W * 0.075, y + H * 0.03), f"${int(rng.integers(0, 30)) * 250:,}",
                   font=font(FONTS[0], H * 0.035), fill=(255, 255, 255), anchor="mm")
    if bonus and rng.random() < 0.6:   # the called letters, on screen in bonus rounds
        y = H * rng.uniform(0.92, 0.97)
        d.text((W * rng.uniform(0.15, 0.35), y), " ".join(sorted(called)),
               font=font(FONTS[0], H * 0.03), fill=(240, 240, 240), anchor="lm")
    if rng.random() < 0.15:   # picture-in-picture of a contestant
        s = rng.uniform(0.15, 0.3)
        x, y = W * rng.uniform(0.0, 1 - s), H * rng.uniform(0.2, 0.9 - s)
        box = [x, y, x + W * s, y + H * s]
        if rng.random() < 0.5:
            d.ellipse(box, fill=rand_rgb(rng)); md.ellipse(box, fill=255)
        else:
            d.rectangle(box, fill=rand_rgb(rng)); md.rectangle(box, fill=255)
    if rng.random() < 0.3:   # station logo bug
        f = font(FONTS[int(rng.integers(len(FONTS)))], H * rng.uniform(0.03, 0.06))
        txt = rng.choice(["7", "GSN", "WABC", "13", "NBC", "abc", "CBS 4", "KHON2"])
        d.text((W * rng.uniform(0.82, 0.95), H * rng.uniform(0.8, 0.95)), str(txt), font=f,
               fill=(255, 255, 255, 140), anchor="mm")
    if rng.random() < 0.12:   # closed captions
        f = font(MONO, H * 0.04)
        words = ["TOGETHER", "THE", "PUZZLE", "SOLVE", "I'D LIKE", "TO", "BUY", "A", "VOWEL", "PAT"]
        line = " ".join(rng.choice(words, int(rng.integers(3, 7))))
        x, y = W * rng.uniform(0.05, 0.4), H * rng.uniform(0.02, 0.8)
        bb = d.textbbox((x, y), line, font=f)
        d.rectangle(bb, fill=(0, 0, 0)); md.rectangle(bb, fill=255)
        d.text((x, y), line, font=f, fill=(240, 240, 240))


# --- geometry -----------------------------------------------------------------

def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, dtype=np.float64))
    return (vt[-1] / vt[-1, -1]).reshape(3, 3)


def apply_h(Hm, pts):
    p = np.c_[np.asarray(pts, dtype=np.float64), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:]


def warp(img, Hm, size, fill=0):
    """img warped by Hm (img pixels -> output pixels) onto an output of size."""
    inv = np.linalg.inv(Hm)
    inv = inv / inv[2, 2]
    if img.mode == "RGB" and not isinstance(fill, tuple):
        fill = (fill,) * 3
    return img.transform(size, Image.Transform.PERSPECTIVE, tuple(inv.flatten()[:8]),
                         Image.Resampling.BILINEAR, fillcolor=fill)


# --- sources ------------------------------------------------------------------

def jpeg(img, q):
    buf = BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    return Image.open(BytesIO(buf.getvalue())).convert("RGB")


def camera(rng, img, mask, corners):
    """Broadcast framing: a zoom and slight shift of the stage."""
    gw = corners[:, 0].max() - corners[:, 0].min()
    fit = W * 0.97 / max(gw, 1)               # the zoom that just fits the board
    z = 1.0
    if rng.random() < 0.5:
        z = rng.uniform(1.0, max(1.0, min(1.5, fit)))
    if rng.random() < 0.1:                    # now and then, a shot that cuts the board
        z = rng.uniform(1.0, 1.6)
    cw, ch = W / z, H / z
    gx, gy = corners[:, 0].mean(), corners[:, 1].mean()
    x = np.clip(gx - cw / 2 + rng.normal(0, cw * 0.08), 0, W - cw)
    y = np.clip(gy - ch / 2 + rng.normal(0, ch * 0.08), 0, H - ch)
    Hm = np.array([[z, 0, -x * z], [0, z, -y * z], [0, 0, 1]])
    return warp(img, Hm, (W, H)), warp(mask, Hm, (W, H), 255), apply_h(Hm, corners)


def offair(rng, img, mask, corners):
    if rng.random() < 0.35:   # a 4:3 crop of the 16:9 frame
        cw = H * 4 / 3
        x = rng.uniform(0, W - cw)
        img, mask = img.crop((x, 0, x + cw, H)), mask.crop((x, 0, x + cw, H))
        corners = corners - [x, 0]
    w, h = img.size
    s = rng.uniform(0.25, 0.6)
    small = img.resize((int(w * s), int(h * s)), Image.Resampling.BILINEAR)
    img = small.resize((w, h), Image.Resampling.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.92, 1.06, 3) + rng.normal(0, rng.uniform(1, 7), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return jpeg(img, rng.uniform(15, 60)), mask, corners


def phone(rng, img, mask, corners):
    """The frame on a TV in a room, photographed or filmed by a phone."""
    out_w, out_h = (W, H) if rng.random() < 0.8 else (H, W)
    room = background(rng, "old").resize((out_w, out_h))
    # where the screen lands: a random quad, mostly the full frame, at an angle
    s = rng.uniform(0.6, 1.15) * min(out_w / W, out_h / H)
    cx, cy = out_w / 2 + rng.normal(0, out_w * 0.06), out_h / 2 + rng.normal(0, out_h * 0.06)
    hw, hh = W * s / 2, H * s / 2
    quad = np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
    quad += rng.normal(0, 0.05 * hw, (4, 2))
    rot = rng.normal(0, 0.06)
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    quad = (quad - [cx, cy]) @ R.T + [cx, cy]
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    Hm = homography(src, quad)
    # the bezel
    bezel = quad + (quad - quad.mean(0)) * rng.uniform(0.02, 0.06)
    d = ImageDraw.Draw(room)
    d.polygon([tuple(p) for p in bezel], fill=jitter(rng, (15, 15, 18), 6))
    # the picture: moire and scanlines first, in screen space
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.7:
        yy, xx = np.mgrid[0:H, 0:W]
        ang, fr = rng.uniform(0, math.pi), rng.uniform(0.05, 0.6)
        grating = np.sin((xx * math.cos(ang) + yy * math.sin(ang)) * fr)
        arr *= (1 - rng.uniform(0.03, 0.15) * (grating[..., None] * 0.5 + 0.5))
    if rng.random() < 0.4:
        arr[::int(rng.integers(2, 5))] *= rng.uniform(0.7, 0.92)
    screen = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    room.paste(warp(screen, Hm, (out_w, out_h)), mask=warp(Image.new("L", (W, H), 255), Hm, (out_w, out_h)))
    mask = warp(mask, Hm, (out_w, out_h), 255)
    corners = apply_h(Hm, corners)
    # glare, colour cast, blur, noise
    arr = np.asarray(room).astype(np.float32)
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0:out_h, 0:out_w]
        gx, gy = rng.uniform(0, out_w), rng.uniform(0, out_h)
        r = rng.uniform(0.05, 0.3) * out_w
        arr += rng.uniform(40, 160) * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r))[..., None]
    arr = arr * rng.uniform(0.85, 1.1, 3) * rng.uniform(0.75, 1.15) + rng.uniform(-20, 20)
    arr += rng.normal(0, rng.uniform(2, 10), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:   # hand shake
        k = int(rng.integers(3, 9))
        img = img.filter(ImageFilter.BoxBlur(k // 3))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
    return jpeg(img, rng.uniform(30, 85)), mask, corners


# --- labels -------------------------------------------------------------------

def square_quad(corners, r, c):
    Hm = homography([(0, 0), (B.COLS, 0), (B.COLS, B.ROWS), (0, B.ROWS)], corners)
    return apply_h(Hm, [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)])


def covered(mask, corners, r, c):
    """True when more than half the square is under the mask or off the image."""
    q = square_quad(corners, r, c)
    w, h = mask.size
    pts = []
    for a in np.linspace(0.15, 0.85, 4):
        for b in np.linspace(0.15, 0.85, 4):
            top = q[0] + (q[1] - q[0]) * a
            bot = q[3] + (q[2] - q[3]) * a
            pts.append(top + (bot - top) * b)
    m = np.asarray(mask)
    hid = 0
    for x, y in pts:
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h) or m[yi, xi] > 127:
            hid += 1
    return hid > len(pts) / 2


# --- one frame ----------------------------------------------------------------

def render(rng, puzzle):
    answer, cat = puzzle
    era = "new" if rng.random() < 0.5 else "old"
    source = rng.choice(["broadcast", "offair", "phone"], p=[0.4, 0.3, 0.3])
    if rng.random() < 0.10:
        img, mask = wheel_frame(rng), Image.new("L", (W, H), 0)
        corners = np.zeros((4, 2))
        label = dict(present=False, grid=B.grid_text(B.blank_grid()), category="",
                     era=era, source=str(source))
    else:
        layout = pick_layout(rng, answer)
        called, lit, bonus = pick_state(rng, answer)
        grid = B.place(layout, called)
        if lit:
            for r, c in B.SQUARES:
                if grid[r][c] == lit and rng.random() < 0.85:
                    grid[r][c] = B.LIT
        st = board_style(rng, era)
        img = background(rng, era)
        pw = W * rng.uniform(0.6, 0.82) / 14
        ph = pw * rng.uniform(1.1, 1.24)
        x0 = W / 2 - 7 * pw + rng.normal(0, W * 0.02)
        y0 = H * rng.uniform(0.12, 0.3)
        # floor reflection, drawn first so the board sits over it
        if rng.random() < 0.6:
            refl = Image.new("RGB", (W, H))
            draw_board(refl, rng, grid, x0, y0, pw, ph, st, era)
            gy1 = int(y0 + 4 * ph + ph * 0.6)
            flipped = refl.crop((0, int(y0 - ph), W, gy1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            flipped = flipped.filter(ImageFilter.GaussianBlur(rng.uniform(1, 4)))
            ry = int(gy1 + ph * rng.uniform(1.0, 1.8))
            img.paste(Image.blend(img.crop((0, ry, W, ry + flipped.height)), flipped,
                                  rng.uniform(0.3, 0.6)), (0, ry))
        draw_board(img, rng, grid, x0, y0, pw, ph, st, era)
        corners = np.array([(x0, y0), (x0 + 14 * pw, y0), (x0 + 14 * pw, y0 + 4 * ph), (x0, y0 + 4 * ph)])
        mask = Image.new("L", (W, H), 0)
        d, md = ImageDraw.Draw(img), ImageDraw.Draw(mask)
        if rng.random() < 0.7:
            side = rng.random()
            if side < 0.4:
                hx = x0 + pw * rng.uniform(-1.2, 1.5)
            elif side < 0.8:
                hx = x0 + 14 * pw - pw * rng.uniform(-1.2, 1.5)
            else:
                hx = x0 + 14 * pw * rng.uniform(0.2, 0.8)
            draw_person(rng, md, d, hx, y0 + 4 * ph + ph * rng.uniform(0.2, 0.8),
                        4 * ph * rng.uniform(0.95, 1.25))
        shown_cat = category_text(rng, cat, bonus) if rng.random() < 0.9 else ""
        if shown_cat:
            draw_category(img, rng, shown_cat, era, x0, y0, pw, ph)
        overlays(img, rng, mask, bonus, called)
        label = dict(present=True, grid=None, category=shown_cat, era=era, source=str(source))
    img, mask, corners = camera(rng, img, mask, corners)
    if source == "offair":
        img, mask, corners = offair(rng, img, mask, corners)
    elif source == "phone":
        img, mask, corners = phone(rng, img, mask, corners)
    elif rng.random() < 0.7:
        img = jpeg(img, rng.uniform(60, 95))
    if label["present"]:
        g = [row[:] for row in grid]
        for r, c in B.SQUARES:
            if covered(mask, corners, r, c):
                g[r][c] = B.COVERED
        label["grid"] = B.grid_text(g)
        # a category cut off by the frame cannot be read
    label["corners"] = corners.round(1).tolist()
    return img, label


def puzzles(name):
    return [(a, c) for a, c in wc.read_puzzles(name) if B.layouts(a)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", action="store_true", help="draw the grid labels on each tile")
    ap.add_argument("out")
    a = ap.parse_args()
    ps = wc.read_puzzles("train.csv")
    tw, th = 480, 270
    cols = 3
    sheet = Image.new("RGB", (tw * cols, th * ((a.preview + cols - 1) // cols)))
    for k in range(a.preview):
        rng = np.random.default_rng([a.seed, k])
        while True:
            p = ps[int(rng.integers(len(ps)))]
            if B.layouts(p[0]):
                break
        img, lab = render(rng, p)
        sx, sy = tw / img.width, th / img.height
        s = min(sx, sy)
        im = img.resize((int(img.width * s), int(img.height * s)))
        d = ImageDraw.Draw(im)
        if a.labels and lab["present"]:
            d.polygon([(x * s, y * s) for x, y in lab["corners"]], outline=(255, 0, 0))
        d.text((3, 3), f"{lab['source']} {lab['era']} {lab['category']}", fill=(255, 0, 0),
               font=font(FONTS[0], 12))
        if a.labels and lab["present"]:
            d.text((3, th - 60), "\n".join(lab["grid"]), fill=(255, 255, 0), font=font(MONO, 11))
        sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
    sheet.save(a.out, quality=85)


if __name__ == "__main__":
    main()
