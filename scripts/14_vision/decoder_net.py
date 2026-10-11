"""The board decoder: a screencap in, the 52 squares, the category and
whether a board is there at all out.

    image 3 x 360 x 640
      -> CNN backbone (ReLU convs, batch norm), features at 1/8 and 1/16 scale
      -> 1/16 features upsampled and added to the 1/8 ones: 45 x 80 tokens
      -> + 2D sine position code
      -> transformer decoder: 77 learned queries, each layer self-attention
         among the queries, cross-attention to the tokens, an MLP
      -> heads: 52 square queries -> board.CLASSES
                24 category queries -> CAT_CHARS (index 0 is padding)
                1 board query -> present logit

Every op is a convolution, batch norm, ReLU, linear layer, layer norm or
softmax attention, so the forward pass can be written in C the way
server/src/model.c runs the transformer. Batch norm folds into the
convolutions at export.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import board as B

IN_W, IN_H = 640, 360
CAT_CHARS = "\0 ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&?!'.-/"
CAT_IDX = {c: i for i, c in enumerate(CAT_CHARS)}
CAT_LEN = 24
N_SQ = len(B.SQUARES)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


def encode_category(text):
    ids = [CAT_IDX.get(c, CAT_IDX["?"]) for c in text.upper()[:CAT_LEN]]
    return ids + [0] * (CAT_LEN - len(ids))


def decode_category(ids):
    return "".join(CAT_CHARS[i] for i in ids if i).strip()


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Block(nn.Module):
    """Two 3x3 convs with a residual, the first one strided when downsampling."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = conv_bn(cin, cout, stride)
        self.b = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        s = x if self.skip is None else self.skip(x)
        return F.relu(self.b(self.a(x)) + s)


def sine_pos(h, w, d):
    """2D sine position code, d/2 channels for y and d/2 for x."""
    y = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
    x = torch.arange(w, dtype=torch.float32)[None, :].expand(h, w)
    q = d // 4
    freq = torch.exp(-math.log(1000.0) * torch.arange(q, dtype=torch.float32) / q)
    parts = []
    for v in (y, x):
        a = v[..., None] * freq
        parts += [a.sin(), a.cos()]
    return torch.cat(parts, -1).view(h * w, d)


class DecoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, mem, qpos, mpos):
        h = self.n1(q)
        q = q + self.self_attn(h + qpos, h + qpos, h, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.cross(h + qpos, mem + mpos, mem, need_weights=False)[0]
        return q + self.mlp(self.n3(q))


class BoardDecoder(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, widths=(32, 48, 64, 96, 128)):
        super().__init__()
        w = widths
        self.stem = conv_bn(3, w[0], 2)                             # 1/2
        self.s1 = nn.Sequential(Block(w[0], w[1], 2), Block(w[1], w[1], 1))   # 1/4
        self.s2 = nn.Sequential(Block(w[1], w[2], 2), Block(w[2], w[2], 1))   # 1/8
        self.s3 = nn.Sequential(Block(w[2], w[3], 2), Block(w[3], w[3], 1))   # 1/16
        self.s4 = nn.Sequential(Block(w[3], w[4], 2), Block(w[4], w[4], 1))   # 1/32
        self.lat8 = nn.Conv2d(w[2], d, 1)
        self.lat16 = nn.Conv2d(w[3], d, 1)
        self.lat32 = nn.Conv2d(w[4], d, 1)
        self.d = d
        self.register_buffer("pos", sine_pos(math.ceil(IN_H / 8), IN_W // 8, d), persistent=False)
        n_q = N_SQ + CAT_LEN + 1
        self.query = nn.Parameter(torch.randn(n_q, d) * 0.02)
        self.qpos = nn.Parameter(torch.randn(n_q, d) * 0.02)
        self.layers = nn.ModuleList(DecoderLayer(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.sq_head = nn.Linear(d, len(B.CLASSES))
        self.cat_head = nn.Linear(d, len(CAT_CHARS))
        self.present_head = nn.Linear(d, 1)

    def forward(self, x):
        """x: N x 3 x 360 x 640 in [0, 1]. Returns square logits N x 52 x C,
        category logits N x 24 x K and present logits N."""
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        f8 = self.s2(self.s1(self.stem(x)))
        f16 = self.s3(f8)
        f32 = self.s4(f16)
        p16 = self.lat16(f16) + F.interpolate(self.lat32(f32), size=f16.shape[-2:], mode="nearest")
        p8 = self.lat8(f8) + F.interpolate(p16, size=f8.shape[-2:], mode="nearest")
        n = x.shape[0]
        mem = p8.flatten(2).transpose(1, 2)                         # N x (45*80) x d
        mpos = self.pos[None, :mem.shape[1]]
        q = self.query[None].expand(n, -1, -1)
        qpos = self.qpos[None]
        for layer in self.layers:
            q = layer(q, mem, qpos, mpos)
        q = self.norm(q)
        return (self.sq_head(q[:, :N_SQ]),
                self.cat_head(q[:, N_SQ:N_SQ + CAT_LEN]),
                self.present_head(q[:, -1]).squeeze(-1))


if __name__ == "__main__":
    m = BoardDecoder()
    n = sum(p.numel() for p in m.parameters())
    out = m(torch.rand(2, 3, IN_H, IN_W))
    print(f"{n / 1e6:.2f}M parameters", [tuple(o.shape) for o in out])
