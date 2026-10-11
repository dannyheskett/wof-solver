"""Runs a board decoder checkpoint on images.

    from infer import Decoder
    dec = Decoder("runs/vision/dec1/best.pt")
    out = dec.read([PIL images or paths])   # one dict per image; paths open a batch at a time

Each dict: present (probability a board is there), grid (4 strings of 14,
board.CLASSES, ' ' off-grid), conf (52 probabilities of the chosen class, in
board.SQUARES order), category (the strip as read), board and guessed
(board.to_board on the grid). With probs=True, also probs: 52 lists of
board.CLASSES probabilities.
"""
import numpy as np
import torch
from PIL import Image

import board as B
import decoder_net as D
from make_frames import letterbox


class Decoder:
    def __init__(self, ckpt, device=None, threads=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if threads:
            torch.set_num_threads(threads)
        self.model = D.BoardDecoder().to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device)["model"])

    @torch.no_grad()
    def read(self, images, batch=32, probs=False):
        out = []
        for i in range(0, len(images), batch):
            xs = []
            for img in images[i:i + batch]:
                if not isinstance(img, Image.Image):
                    with Image.open(img) as f:
                        img = f.convert("RGB")
                lb, *_ = letterbox(img.convert("RGB"))
                xs.append(torch.from_numpy(np.asarray(lb, dtype=np.float32) / 255).permute(2, 0, 1))
            x = torch.stack(xs).to(self.device)
            sq, cat, pr = self.model(x)
            sq_p = sq.float().softmax(-1).cpu()
            cat_ids = cat.argmax(-1).cpu()
            pr = pr.float().sigmoid().cpu()
            for k in range(len(xs)):
                conf, cls = sq_p[k].max(-1)
                g = B.blank_grid()
                for j, (r, c) in enumerate(B.SQUARES):
                    g[r][c] = B.CLASSES[int(cls[j])]
                board, guessed = B.to_board(g)
                d = dict(present=float(pr[k]), grid=B.grid_text(g), conf=conf.tolist(),
                         category=D.decode_category(cat_ids[k].tolist()),
                         board=board, guessed=guessed)
                if probs:
                    d["probs"] = sq_p[k].numpy()
                out.append(d)
        return out
