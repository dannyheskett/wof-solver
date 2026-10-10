#!/usr/bin/env python3
"""Check the C board decoder (server/src/decoder.c) against PyTorch on the
same images: the letterboxed input byte for byte against Pillow, and the
square probabilities, category and present probability against the model.

    make -C server build/decoder_dump
    python3 server/tests/decoder_parity.py runs/vision/dec1/best.pt runs/vision/dec1/decoder.bin \\
        IMAGE [IMAGE ...]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "14_vision"))
import decoder_net as D  # noqa: E402
from make_frames import letterbox  # noqa: E402


def main():
    ckpt, weights, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    model = D.BoardDecoder()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    tmp = Path(tempfile.mkdtemp())
    lines, lbs = [], []
    for k, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        raw = tmp / f"in{k}.raw"
        raw.write_bytes(np.asarray(img).tobytes())
        lines.append(f"{img.width} {img.height} {raw}")
        lbs.append(np.asarray(letterbox(img)[0]))
    out = subprocess.run([str(ROOT / "server/build/decoder_dump"), weights, str(tmp)],
                         input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    sys.stderr.write(out.stderr)
    worst_px = worst_p = 0.0
    same_cls = total = cat_same = 0
    for k, line in enumerate(out.stdout.splitlines()):
        c = json.loads(line)
        c_in = np.frombuffer((tmp / f"{k}.rgb").read_bytes(), np.uint8).reshape(D.IN_H, D.IN_W, 3)
        worst_px = max(worst_px, float(np.abs(c_in.astype(int) - lbs[k].astype(int)).max()))
        # the model on C's own letterboxed input, so the forward pass is
        # compared on identical numbers
        x = torch.from_numpy(c_in.astype(np.float32) / 255).permute(2, 0, 1)[None]
        with torch.no_grad():
            sq, cat, pr = model(x)
        p_py = sq.softmax(-1)[0].numpy()
        p_c = np.array(c["probs"])
        worst_p = max(worst_p, float(np.abs(p_py - p_c).max()))
        same_cls += int((p_py.argmax(-1) == p_c.argmax(-1)).sum())
        total += D.N_SQ
        cat_py = D.decode_category(cat.argmax(-1)[0].tolist())
        cat_same += cat_py == c["category"]
        print(f"{images[k]}: present py {torch.sigmoid(pr).item():.4f} c {c['present']:.4f}, "
              f"category py {cat_py!r} c {c['category']!r}")
    print(f"letterbox: largest byte difference from Pillow {worst_px:.0f}")
    print(f"squares: largest probability difference {worst_p:.2e}, same class {same_cls}/{total}, "
          f"same category {cat_same}/{len(images)}")


if __name__ == "__main__":
    main()
