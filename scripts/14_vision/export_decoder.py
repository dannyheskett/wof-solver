"""Writes a board decoder checkpoint as the flat weights file the C server
loads (server/src/decoder.c).

    python export_decoder.py runs/vision/dec1/best.pt runs/vision/dec1/decoder.bin

Layout, all little-endian:
    magic   "WOFD" and a uint32 version (2)
    uint32  number of tensors
    per tensor:
        uint32  name length, then the name (ASCII)
        uint32  number of dimensions, then each dimension
        zero bytes up to the next multiple of 16 in the file
        float32 values, row-major
The server maps the file rather than copying it, so the weights are page
cache the kernel can drop, and the 16-byte alignment lets it use them in
place. Batch norm is folded into the convolution before it, so every conv
has a weight [out][in][k][k] and a bias [out]. Every linear layer is stored
transposed, as <name>.wt [in][out], with its bias <name>.b [out], so the
server's inner loops run along the outputs; attention's packed input
projection is split into q, k and v. The decoder's position code is written
as "pos" [tokens][d], so C does not recompute it. About 10 MB.
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

    def linear(name, w, b):
        out.extend([(name + ".wt", w.t()), (name + ".b", b)])

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
            w, b = a.in_proj_weight.detach(), a.in_proj_bias.detach()
            d = w.shape[1]
            for k, part in enumerate("qkv"):
                linear(p + n + "." + part, w[k * d:(k + 1) * d], b[k * d:(k + 1) * d])
            linear(p + n + ".o", a.out_proj.weight.detach(), a.out_proj.bias.detach())
        linear(p + "fc", L.mlp[0].weight.detach(), L.mlp[0].bias.detach())
        linear(p + "proj", L.mlp[2].weight.detach(), L.mlp[2].bias.detach())
    out.extend([("norm.w", model.norm.weight.detach()), ("norm.b", model.norm.bias.detach())])
    for h in ("sq_head", "cat_head", "present_head"):
        lin = getattr(model, h)
        linear(h, lin.weight.detach(), lin.bias.detach())
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = D.BoardDecoder()
    model.load_state_dict(torch.load(src, map_location="cpu")["model"])
    model.eval()
    ts = tensors(model)
    n = 0
    with open(dst, "wb") as f:
        f.write(b"WOFD" + struct.pack("<2I", 2, len(ts)))
        for name, t in ts:
            t = t.float().contiguous()
            f.write(struct.pack("<I", len(name)) + name.encode("ascii"))
            f.write(struct.pack("<I", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape))
            f.write(b"\0" * (-f.tell() % 16))
            f.write(t.numpy().astype("<f4").tobytes())
            n += t.numel()
    print(f"{dst}: {len(ts)} tensors, {n:,} values as float32")


if __name__ == "__main__":
    main()
