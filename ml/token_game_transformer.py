"""
A transformer-encoder policy/value network for Step A of the diagonal token game.

This is a drop-in alternative to the dilated CNN of `token_game_policy.py`: same task, same
data pipeline, same metrics, same (policy, value) output contract -- only the trunk changes,
from stacked dilated convolutions to a stack of self-attention encoder layers. The point is
an architecture ablation: attention has a global receptive field by construction (every
diagonal attends to every other), so if it *also* saturates on move accuracy while
greedy-solve stays near 0, that corroborates the paper's claim that the wall is the factoring
step, not the network.

Input/output match the CNN exactly, so the shared helpers are reused verbatim:
  * input  : featurized profile of shape (channels, 2N-1)   [channels set by --cond]
  * output : (policy logits (B, 2N), value logit (B, 1))     [2N-1 moves + STOP; value]

Run:  .venv/bin/python token_game_transformer.py --demo
      .venv/bin/python token_game_transformer.py --N 16 --cond pop --d-model 160 --layers 6
"""

from __future__ import annotations
import argparse
import random

# Shared game logic, data generation, metrics -- identical to the CNN's pipeline.
from token_game_policy import (
    generate_dataset, rollout_solves, cond_channels, pick_device,
)


def _build_model_parts():
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class TokenGameTransformer(nn.Module):
        """features (B, in_channels, 2N-1) -> (policy_logits (B, 2N), value_logit (B, 1)).

        Each of the 2N-1 diagonals is a token; a learned positional embedding fixes their
        order; a pre-LN self-attention encoder mixes them globally. A per-token linear gives
        the 2N-1 move logits; a mean-pooled summary drives the STOP logit and the scalar value.
        """

        def __init__(self, N: int, in_channels: int = 1, d_model: int = 128, nhead: int = 8,
                     layers: int = 4, dim_ff: int = 256, dropout: float = 0.0):
            super().__init__()
            self.N = N
            self.L = 2 * N - 1
            self.in_channels = in_channels
            self.d_model = d_model
            self.in_proj = nn.Linear(in_channels, d_model)
            self.pos = nn.Parameter(torch.zeros(1, self.L, d_model))   # learned positions (L fixed)
            nn.init.normal_(self.pos, std=0.02)
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                             activation="gelu", batch_first=True, norm_first=True)
            # enable_nested_tensor is a padded-batch speedup incompatible with norm_first (and
            # irrelevant here: every sequence is length 2N-1). Disable it to silence the warning.
            self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.move_head = nn.Linear(d_model, 1)                     # per-token "duplicate here"
            self.stop_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
            self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        def forward(self, x):                              # x: (B, in_channels, L)
            h = x.transpose(1, 2)                          # (B, L, in_channels)
            h = self.in_proj(h) + self.pos                 # (B, L, d_model)
            h = self.norm(self.encoder(h))                 # (B, L, d_model)
            move_logits = self.move_head(h).squeeze(-1)    # (B, L)
            pooled = h.mean(dim=1)                          # (B, d_model) global summary
            stop_logit = self.stop_head(pooled)            # (B, 1)
            policy = torch.cat([move_logits, stop_logit], dim=1)   # (B, L+1)
            value = self.value_head(pooled)                # (B, 1) raw logit
            return policy, value

    return torch, nn, TensorDataset, DataLoader, TokenGameTransformer


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------

def save_model(model, path, N, cond, d_model, nhead, layers, dim_ff, dropout):
    import os
    import torch
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"arch": "transformer", "state_dict": model.state_dict(), "N": N, "cond": cond,
                "d_model": d_model, "nhead": nhead, "layers": layers,
                "dim_ff": dim_ff, "dropout": dropout}, path)


def load_model(path, device=None):
    """Rebuild a transformer policy from a checkpoint. Returns (model, N, cond)."""
    import torch
    dev = pick_device(device)
    ck = torch.load(path, map_location=dev, weights_only=False)
    _, _, _, _, TokenGameTransformer = _build_model_parts()
    model = TokenGameTransformer(ck["N"], in_channels=cond_channels(ck["cond"]),
                                 d_model=ck["d_model"], nhead=ck["nhead"], layers=ck["layers"],
                                 dim_ff=ck["dim_ff"], dropout=ck["dropout"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["N"], ck["cond"]


# --------------------------------------------------------------------------------------
# Training (mirrors token_game_policy.train; masked policy CE + value BCE)
# --------------------------------------------------------------------------------------

def train(N: int = 16, rounds: int = 200, instances_per_round: int = 128, batch_size: int = 512,
          lr: float = 3e-4, d_model: int = 128, nhead: int = 8, layers: int = 4, dim_ff: int = 256,
          dropout: float = 0.0, stop_weight: float = 20.0, eval_instances: int = 64,
          device: str | None = None, cond: str = "none", save: str | None = None,
          resume: str | None = None):
    torch, nn, TensorDataset, DataLoader, TokenGameTransformer = _build_model_parts()
    dev = pick_device(device)
    L = 2 * N - 1
    ch = cond_channels(cond)

    model = TokenGameTransformer(N, in_channels=ch, d_model=d_model, nhead=nhead, layers=layers,
                                 dim_ff=dim_ff, dropout=dropout).to(dev)
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        for k, have in (("N", N), ("cond", cond), ("d_model", d_model), ("nhead", nhead),
                        ("layers", layers), ("dim_ff", dim_ff)):
            if ck.get(k) != have:
                raise ValueError(f"--resume mismatch: checkpoint {k}={ck.get(k)} but run has {k}={have}")
        model.load_state_dict(ck["state_dict"])
        print(f"resumed from {resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.ones(L + 1, device=dev)
    w[L] = stop_weight
    policy_loss_fn = nn.CrossEntropyLoss(weight=w, reduction="none")
    value_loss_fn = nn.BCEWithLogitsLoss()

    print(f"[transformer] N={N}  cond={cond}  in_channels={ch}  classes={L+1}  "
          f"d_model={d_model} layers={layers} heads={nhead}  "
          f"params={sum(p.numel() for p in model.parameters()):,}  device={dev}  "
          f"(attention: global receptive field)")
    for rnd in range(1, rounds + 1):
        X, ym, yv, msk = generate_dataset(N, instances_per_round, cond)
        dl = DataLoader(TensorDataset(X, ym, yv, msk), batch_size=batch_size, shuffle=True)
        model.train()
        tot, correct, pol_seen, vcorrect, seen = 0.0, 0, 0, 0, 0
        for xb, ymb, yvb, mb in dl:
            xb, ymb, yvb, mb = xb.to(dev), ymb.to(dev), yvb.to(dev), mb.to(dev)
            logits, value = model(xb)
            per = policy_loss_fn(logits, ymb) * mb
            policy_loss = per.sum() / mb.sum().clamp_min(1.0)
            value_loss = value_loss_fn(value.squeeze(1), yvb)
            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            correct += ((logits.argmax(1) == ymb).float() * mb).sum().item()
            pol_seen += mb.sum().item()
            vcorrect += (((value.squeeze(1) > 0) == (yvb > 0.5)).float()).sum().item()
            seen += xb.size(0)
        if rnd == 1 or rnd % 5 == 0 or rnd == rounds:
            solve = rollout_solves(model, N, eval_instances, cond)
            print(f"round {rnd:3d}  loss {tot/seen:.4f}  move-acc {correct/max(1,pol_seen):.3f}  "
                  f"val-acc {vcorrect/seen:.3f}  greedy-solve {solve:.3f}  ({seen:,} samples)")
            if save:
                save_model(model, save, N, cond, d_model, nhead, layers, dim_ff, dropout)
    return model


def main():
    ap = argparse.ArgumentParser(description="Transformer policy/value net for the token game.")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--instances-per-round", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--cond", choices=["none", "pop", "target"], default="pop")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None, help="checkpoint path (.pt); saved each eval round")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--demo", action="store_true", help="quick small-N (N=12) run")
    args = ap.parse_args()

    random.seed(args.seed)
    if args.demo:
        train(N=12, rounds=30, instances_per_round=64, d_model=96, layers=3, nhead=6,
              dim_ff=192, device=args.device, cond=args.cond, save=args.save)
    else:
        train(N=args.N, rounds=args.rounds, instances_per_round=args.instances_per_round,
              batch_size=args.batch_size, lr=args.lr, d_model=args.d_model, nhead=args.heads,
              layers=args.layers, dim_ff=args.dim_ff, dropout=args.dropout,
              device=args.device, cond=args.cond, save=args.save, resume=args.resume)


if __name__ == "__main__":
    main()
