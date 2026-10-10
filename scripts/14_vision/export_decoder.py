"""Writes a board decoder checkpoint as the flat weights file the C server
loads (server/src/decoder.c).

    python export_decoder.py runs/vision/dec1/best.pt runs/vision/dec1/decoder.bin

Layout, all little-endian:
    magic   "WOFD" and a uint32 version (1)
    uint32  number of tensors
    per tensor:
        uint32  name length, then the name (ASCII)
        uint32  number of dimensions, then each dimension
        float32 values, row-major
Batch norm is folded into the convolution before it, so every conv has a
weight [out][in][k][k] and a bias [out]. The decoder's position code is
written as "pos" [tokens][d], so C does not recompute it. Linear weights are
[out][in], as in PyTorch. About 8 MB.
"""
import struct
import sys

import torch

import decoder_net as D


def fold(conv, bn):
    """conv followed by batch norm, as one conv with a bias."""
    w = conv.weight.detach().double()
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    b = bn.bias.detach().double() - bn.running_mean.double() * scale
    if conv.bias is not None:
        b = b + conv.bias.detach().double() * scale
    return (w * scale[:, None, None, None]).float(), b.float()


def tensors(model):
    out = []

    def conv_bn(name, seq):
        w, b = fold(seq[0], seq[1])
        out.extend([(name + ".w", w), (name + ".b", b)])

    def block(name, blk):
        conv_bn(name + ".a", blk.a)
        conv_bn(name + ".b", blk.b)
        if blk.skip is not None:
            conv_bn(name + ".skip", blk.skip)

    conv_bn("stem", model.stem)
    for s in ("s1", "s2", "s3", "s4"):
        for k, blk in enumerate(getattr(model, s)):
            block(f"{s}.{k}", blk)
    for lat in ("lat8", "lat16", "lat32"):
        c = getattr(model, lat)
        out.extend([(lat + ".w", c.weight.detach()), (lat + ".b", c.bias.detach())])
    out.append(("pos", model.pos))
    out.extend([("query", model.query.detach()), ("qpos", model.qpos.detach())])
    for k, L in enumerate(model.layers):
        p = f"layer{k}."
        for n in ("n1", "n2", "n3"):
            ln = getattr(L, n)
            out.extend([(p + n + ".w", ln.weight.detach()), (p + n + ".b", ln.bias.detach())])
        for n in ("self_attn", "cross"):
            a = getattr(L, n)
            out.extend([(p + n + ".in_w", a.in_proj_weight.detach()), (p + n + ".in_b", a.in_proj_bias.detach()),
                        (p + n + ".out_w", a.out_proj.weight.detach()), (p + n + ".out_b", a.out_proj.bias.detach())])
        out.extend([(p + "fc.w", L.mlp[0].weight.detach()), (p + "fc.b", L.mlp[0].bias.detach()),
                    (p + "proj.w", L.mlp[2].weight.detach()), (p + "proj.b", L.mlp[2].bias.detach())])
    out.extend([("norm.w", model.norm.weight.detach()), ("norm.b", model.norm.bias.detach())])
    for h in ("sq_head", "cat_head", "present_head"):
        lin = getattr(model, h)
        out.extend([(h + ".w", lin.weight.detach()), (h + ".b", lin.bias.detach())])
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = D.BoardDecoder()
    model.load_state_dict(torch.load(src, map_location="cpu")["model"])
    model.eval()
    ts = tensors(model)
    n = 0
    with open(dst, "wb") as f:
        f.write(b"WOFD" + struct.pack("<2I", 1, len(ts)))
        for name, t in ts:
            t = t.float().contiguous()
            f.write(struct.pack("<I", len(name)) + name.encode("ascii"))
            f.write(struct.pack("<I", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape))
            f.write(t.numpy().astype("<f4").tobytes())
            n += t.numel()
    print(f"{dst}: {len(ts)} tensors, {n:,} values as float32")


if __name__ == "__main__":
    main()
