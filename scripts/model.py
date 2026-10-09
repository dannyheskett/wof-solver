"""A small GPT-style character model, after nanoGPT: token and position
embeddings, n_layer pre-norm blocks of causal self-attention and a GELU
MLP, a final LayerNorm, and an output layer tied to the token embedding.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 48
    block: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.n_head, self.dropout = c.n_head, c.dropout

    def forward(self, x, cache=None):
        """cache, when given, is a dict holding this layer's keys and values
        so far. It is extended in place, so generation can feed one new
        token at a time."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        causal = True
        if cache is not None:
            if "k" in cache:
                assert T == 1, "after the prompt, feed one token at a time"
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
                causal = False   # the one new token may see everything before it
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd, bias=False),
                                 nn.Dropout(c.dropout))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab, c.n_embd)
        self.pos = nn.Embedding(c.block, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()

    def forward(self, idx, targets=None, caches=None, start=0):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(start, start + T, device=idx.device))
        for k, b in enumerate(self.blocks):
            x = b(x, caches[k] if caches is not None else None)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    @torch.no_grad()
    def complete(self, prompts, stop, max_new, ban=None):
        """Greedy completion of each token-id list in prompts, stopping at
        the `stop` token, with a key/value cache so each new token costs
        one position. Returns the new ids per prompt, without `stop`.
        ban(i, new_ids) may return token ids prompt i must not write next.
        self.banned counts, per prompt, the steps where a ban changed the
        choice. self.logp holds, per prompt, the log probability of each
        new token among the tokens allowed at that step."""
        out = [[] for _ in prompts]
        self.banned = [0] * len(prompts)
        self.logp = [[] for _ in prompts]
        device = self.head.weight.device
        for i, p in enumerate(prompts):
            assert len(p) + max_new <= self.config.block, "prompt too long for the block"
            caches = [{} for _ in self.blocks]
            logits = self(torch.tensor([p], device=device), caches=caches)[0, -1]
            pos = len(p)
            for _ in range(max_new):
                row = logits.float()
                first = int(row.argmax())
                banned = ban(i, out[i]) if ban else ()
                if banned:
                    row = row.clone()
                    row[list(banned)] = float("-inf")
                best = int(row.argmax())
                if best != first:
                    self.banned[i] += 1
                self.logp[i].append(float(torch.log_softmax(row, dim=0)[best]))
                if best == stop:
                    break
                out[i].append(best)
                logits = self(torch.tensor([[best]], device=device), caches=caches, start=pos)[0, -1]
                pos += 1
        return out
