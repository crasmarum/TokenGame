"""
A convolutional policy network for Step A of the diagonal token game.

Setup (see diagonal-token-game.tex, sections "Algorithms" and "A worked example"):

  * A position is summarised by its DIAGONAL PROFILE: an array `n` of length 2N-1,
    where `n[i]` is the number of tokens on the diagonal of exponent `i` (i = 0..2N-2,
    diagonal d = i-(N-1), token weight 2^i). The conserved board weight is
        W' = sum_i n[i] * 2^i.
  * A DUPLICATION move "at index i" takes one token at exponent i and produces two at
    exponent i-1:  n[i] -= 1 ; n[i-1] += 2. This conserves W' (2*2^(i-1) = 2^i) and is
    exactly the game's move 4. Index 0 is never legal (no i-1).
  * Step A drives the greedy-high placement of C = p*q down to the TARGET polynomial
    n* = rho * sigma-tilde (the rectangle profile of the factorisation R x S). Reaching
    n* is the "-1 / stop" action.

Model I/O (as requested):
  * input : the current profile, an array of length 2N-1 of non-negative counts.
  * output: a distribution over 2N classes --
        classes 0 .. 2N-2  -> "play the duplication at that index",
        class   2N-1        -> "-1": the profile already equals the target polynomial.

Training data is produced by behavioural cloning of the exact (deterministic) Step A
solver: for a random N-bit semiprime C = p*q we replay the forced duplication sequence and
emit one (profile, next-move) pair per step. Because the profile losslessly encodes C
(its weight IS C), the map profile -> next-move is a well-defined function -- and computing
it is equivalent to factoring C. That is the point of the experiment.

Run:  pip install torch     (numpy is used only lightly; sympy is NOT required)
      python token_game_policy.py --demo        # quick sanity + small-N training
      python token_game_policy.py --N 64        # the real thing
"""

from __future__ import annotations
import argparse
import random
from typing import List, Tuple

# --------------------------------------------------------------------------------------
# Game / data generation  (pure Python -- no torch, no sympy)
# --------------------------------------------------------------------------------------

def capacity(N: int) -> List[int]:
    """Diagonal capacities L_d = N - |d|, indexed by exponent i (d = i-(N-1))."""
    return [N - abs(i - (N - 1)) for i in range(2 * N - 1)]


def greedy_high_profile(C: int, N: int) -> List[int]:
    """Canonical start: lay weight C as high (few tokens) as capacity allows."""
    cap = capacity(N)
    n = [0] * (2 * N - 1)
    rem = C
    for i in range(2 * N - 2, -1, -1):
        w = 1 << i
        take = min(rem // w, cap[i])
        n[i] = take
        rem -= take * w
    if rem != 0:
        raise ValueError(f"C={C} exceeds board capacity for N={N}")
    return n


def rectangle_profile(p: int, q: int, N: int) -> List[int]:
    """Target n*: rows R = bits(p), cols S = {N-1-j : bit j of q}; n*[(r-c)+N-1]++."""
    R = [r for r in range(N) if (p >> r) & 1]
    S = [N - 1 - j for j in range(N) if (q >> j) & 1]
    n = [0] * (2 * N - 1)
    for r in R:
        for c in S:
            n[(r - c) + (N - 1)] += 1
    return n


def forced_flow(n0: List[int], nstar: List[int], N: int) -> List[int]:
    """s[i] = number of duplications at exponent i, from s[i] = n0[i] + 2 s[i+1] - n*[i]."""
    s = [0] * (2 * N - 1)
    above = 0
    for i in range(2 * N - 2, -1, -1):
        s[i] = n0[i] + 2 * above - nstar[i]
        above = s[i]
    return s


def weight(n: List[int]) -> int:
    return sum(v << i for i, v in enumerate(n))


def _is_probable_prime(n: int, k: int = 20) -> bool:
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_nbit_prime(N: int) -> int:
    """A prime with exactly N bits (top and bottom bit set)."""
    while True:
        c = random.getrandbits(N) | (1 << (N - 1)) | 1
        if _is_probable_prime(c):
            return c


def make_instance(N: int) -> Tuple[List[List[int]], List[int], int, int]:
    """One semiprime -> the full (profile, move) trajectory of Step A.

    Returns (states, labels, ppop, qpop). label in 0..2N-2 is a duplication index; label
    2N-1 is STOP. ppop = popcount(p), qpop = popcount(q) are the promise (row/column counts;
    also the width/height of the target rectangle). The target profile n* is states[-1].
    Canonical order p <= q makes the target -- hence the labels -- deterministic in C.
    """
    STOP = 2 * N - 1
    p = random_nbit_prime(N)
    q = random_nbit_prime(N)
    while q == p:
        q = random_nbit_prime(N)
    if p > q:
        p, q = q, p
    C = p * q
    n0 = greedy_high_profile(C, N)
    nstar = rectangle_profile(p, q, N)
    s = forced_flow(n0, nstar, N)
    assert s[0] == 0 and all(x >= 0 for x in s), "infeasible forced flow (should not happen)"

    states: List[List[int]] = []
    labels: List[int] = []
    state = n0[:]
    for i in range(2 * N - 2, 0, -1):          # high -> low: deterministic next move
        for _ in range(s[i]):
            states.append(state[:])
            labels.append(i)
            state[i] -= 1
            state[i - 1] += 2
    assert state == nstar
    states.append(state[:])                    # terminal profile == target polynomial
    labels.append(STOP)
    return states, labels, bin(p).count("1"), bin(q).count("1")


def reachable(state: List[int], nstar: List[int], N: int) -> bool:
    """Can n* be reached from `state` by downward duplications? (Step-A feasibility.)

    Duplications only move weight down (i -> i-1), so n* is reachable iff the forced flow is
    non-negative and needs no split at index 0. This is the natural 0/1 VALUE of a profile:
    from a reachable profile optimal play scores the Step-B reward 1, otherwise 0.
    """
    s = forced_flow(state, nstar, N)
    return s[0] == 0 and all(x >= 0 for x in s)


def next_forced_move(state: List[int], nstar: List[int], N: int) -> int:
    """Deterministic Step-A move from any reachable profile: STOP if at n*, else the highest
    index with remaining forced flow. Assumes `reachable(state, nstar, N)`."""
    STOP = 2 * N - 1
    if state == nstar:
        return STOP
    s = forced_flow(state, nstar, N)
    for i in range(2 * N - 2, 0, -1):
        if s[i] > 0:
            return i
    return STOP


# --------------------------------------------------------------------------------------
# Conditioning: optional extra input channels (the promise / the target)
# --------------------------------------------------------------------------------------

_COND_CHANNELS = {"none": 1, "pop": 3, "target": 2}


def cond_channels(cond: str) -> int:
    """Number of input channels for a conditioning mode."""
    return _COND_CHANNELS[cond]


def featurize(state: List[int], cond: str, N: int, ppop: int, qpop: int,
              nstar: List[int]) -> List[List[float]]:
    """Turn one profile into a (channels, 2N-1) feature stack for the chosen mode.

      none   : [ log1p(profile) ]
      target : [ log1p(profile), log1p(n*) ]                      -- well-posed: forced flow
      pop    : [ log1p(profile), ppop/N (const), qpop/N (const) ] -- promise only, still hard
    """
    import math
    L = 2 * N - 1
    prof = [math.log1p(v) for v in state]
    if cond == "none":
        return [prof]
    if cond == "target":
        return [prof, [math.log1p(v) for v in nstar]]
    if cond == "pop":
        return [prof, [ppop / N] * L, [qpop / N] * L]
    raise ValueError(f"unknown cond {cond!r}")


# --------------------------------------------------------------------------------------
# Model + training (torch)
# --------------------------------------------------------------------------------------

def _build_torch_parts():
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, k, dilation=1):
            super().__init__()
            self.conv = nn.Conv1d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation)
            self.bn = nn.BatchNorm1d(cout)
            self.act = nn.ReLU(inplace=True)
            self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)) + self.proj(x))   # residual

    class TokenGamePolicy(nn.Module):
        """features (B, in_channels, 2N-1) -> (policy_logits (B, 2N), value_logit (B, 1)).

        policy: 2N-1 duplication indices + 1 STOP.
        value : one logit; sigmoid(value) estimates P(target n* still reachable from here) --
                the probability that optimal play from this profile scores the Step-B reward.
        """

        def __init__(self, N: int, width: int = 96, depth: int = 6, in_channels: int = 1,
                     dilated: bool = False):
            super().__init__()
            self.N = N
            self.L = 2 * N - 1
            self.in_channels = in_channels
            self.dilated = dilated
            if dilated:
                # WaveNet/TCN style: kernel 3 with exponentially growing dilation 1,2,4,...
                # (cycled), so a handful of blocks span the whole 2N-1 profile.
                cyc = 1
                while (1 << cyc) < self.L:
                    cyc += 1
                ks = [3] * depth
                dils = [1 << (j % cyc) for j in range(depth)]
            else:
                base = [7, 5, 5, 3, 3, 3]
                ks = [base[j % len(base)] for j in range(depth)]
                dils = [1] * depth
            self.receptive_field = 1 + sum((ks[j] - 1) * dils[j] for j in range(depth))
            blocks = []
            cin = in_channels
            for j in range(depth):
                blocks.append(ConvBlock(cin, width, ks[j], dilation=dils[j]))
                cin = width
            self.trunk = nn.Sequential(*blocks)
            self.move_head = nn.Conv1d(width, 1, 1)          # per-index "duplicate here"
            self.stop_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            self.value_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

        def forward(self, x):                                # x: (B, in_channels, L), featurized
            h = self.trunk(x)                                # (B, width, L)
            pooled = h.mean(dim=2)                           # (B, width) global context
            move_logits = self.move_head(h).squeeze(1)       # (B, L)
            stop_logit = self.stop_head(pooled)              # (B, 1)
            policy = torch.cat([move_logits, stop_logit], dim=1)   # (B, L+1)
            value = self.value_head(pooled)                  # (B, 1) raw logit
            return policy, value

    return torch, nn, TensorDataset, DataLoader, TokenGamePolicy


def pick_device(requested: str | None = None):
    """Auto-select the best backend: CUDA, then Apple MPS, then CPU."""
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"                      # Apple-Silicon GPU (Metal)
    return "cpu"


def legal_mask(states, N):
    """Boolean (B, 2N) mask of legal classes: index i>=1 with count>0, plus STOP.

    Built on the same device as `states` so it can mask logits without a host copy.
    """
    import torch
    B = states.shape[0]
    L = 2 * N - 1
    mask = torch.zeros(B, L + 1, dtype=torch.bool, device=states.device)
    mask[:, 1:L] = states[:, 1:L] > 0     # duplication legal only where a token sits (i>=1)
    mask[:, L] = True                     # STOP always allowed
    return mask


def generate_dataset(N: int, num_instances: int, cond: str = "none",
                     offpath_per_instance: int = 8):
    """Returns (X, y_move, y_value, policy_mask).

    On-path states (the cloned Step-A trajectory) carry their move label and value 1.
    Off-path states -- reached by a few random duplications -- teach the value head: if the
    target is still reachable the value is 1 (and we also supply the recomputed forced move),
    otherwise the value is 0 and the policy loss is masked out (there is no right move).
    """
    import torch
    STOP = 2 * N - 1
    xs: List[List[List[float]]] = []          # (M, channels, L)
    ym: List[int] = []
    yv: List[float] = []
    msk: List[float] = []
    for _ in range(num_instances):
        states, labels, ppop, qpop = make_instance(N)
        nstar = states[-1]
        pq = ppop * qpop
        for st, mv in zip(states, labels):    # on-path: cloned move, value 1
            xs.append(featurize(st, cond, N, ppop, qpop, nstar))
            ym.append(mv); yv.append(1.0); msk.append(1.0)
        onpath = states[:-1] if len(states) > 1 else states
        for _ in range(offpath_per_instance):
            st = list(random.choice(onpath))
            for _ in range(random.randint(1, 3)):
                if sum(st) >= pq:
                    break
                legal = [i for i in range(1, 2 * N - 1) if st[i] > 0]
                if not legal:
                    break
                a = random.choice(legal)
                st[a] -= 1
                st[a - 1] += 2
            feat = featurize(st, cond, N, ppop, qpop, nstar)
            if st == nstar:
                xs.append(feat); ym.append(STOP); yv.append(1.0); msk.append(1.0)
            elif reachable(st, nstar, N):
                xs.append(feat); ym.append(next_forced_move(st, nstar, N)); yv.append(1.0); msk.append(1.0)
            else:
                xs.append(feat); ym.append(0); yv.append(0.0); msk.append(0.0)   # value 0, policy masked
    X = torch.tensor(xs, dtype=torch.float32)
    return X, torch.tensor(ym, dtype=torch.long), torch.tensor(yv, dtype=torch.float32), \
        torch.tensor(msk, dtype=torch.float32)


def rollout_solves(model, N: int, num_instances: int, cond: str = "none",
                   max_steps_factor: int = 4) -> float:
    """Greedy rollout with legal masking; fraction of instances whose profile reaches n*."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    L = 2 * N - 1
    solved = 0
    with torch.no_grad():
        for _ in range(num_instances):
            states, labels, ppop, qpop = make_instance(N)
            nstar = states[-1]                       # terminal profile is the target
            cur = states[0][:]
            budget = max_steps_factor * len(states)
            for _step in range(budget):
                x = torch.tensor([featurize(cur, cond, N, ppop, qpop, nstar)],
                                 dtype=torch.float32, device=dev)      # (1, channels, L)
                raw = torch.tensor([cur], dtype=torch.float32, device=dev)  # for legality
                logits, _value = model(x)
                logits = logits.masked_fill(~legal_mask(raw, N), float("-inf"))
                a = int(logits.argmax(dim=1).item())
                if a == L:                            # STOP
                    break
                cur[a] -= 1
                cur[a - 1] += 2
            if cur == nstar:
                solved += 1
    return solved / num_instances


def save_model(model, path: str, N: int, width: int, depth: int, cond: str,
               dilated: bool = False):
    """Checkpoint the weights plus the config needed to rebuild the architecture."""
    import os
    import torch
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)          # create models/ etc. if missing
    torch.save({"state_dict": model.state_dict(), "N": N, "width": width,
                "depth": depth, "cond": cond, "dilated": dilated}, path)


def load_model(path: str, device: str | None = None):
    """Rebuild a policy from a checkpoint. Returns (model, N, cond)."""
    import torch
    dev = pick_device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    _, _, _, _, TokenGamePolicy = _build_torch_parts()
    model = TokenGamePolicy(ckpt["N"], width=ckpt["width"], depth=ckpt["depth"],
                            in_channels=cond_channels(ckpt["cond"]),
                            dilated=ckpt.get("dilated", False)).to(dev)
    # strict=False so pre-value-head checkpoints still load (value head stays untrained).
    missing, _unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if any("value_head" in k for k in missing):
        print("note: checkpoint predates the value head; value predictions are untrained.")
    model.eval()
    return model, ckpt["N"], ckpt["cond"]


def train(N: int = 64, rounds: int = 40, instances_per_round: int = 64,
          batch_size: int = 256, lr: float = 2e-3, width: int = 96, depth: int = 6,
          stop_weight: float = 20.0, eval_instances: int = 64, device: str | None = None,
          cond: str = "none", save: str | None = None, resume: str | None = None,
          dilated: bool = False):
    torch, nn, TensorDataset, DataLoader, TokenGamePolicy = _build_torch_parts()
    dev = pick_device(device)
    L = 2 * N - 1
    ch = cond_channels(cond)

    model = TokenGamePolicy(N, width=width, depth=depth, in_channels=ch, dilated=dilated).to(dev)
    if resume:                                          # warm-start from a checkpoint
        ckpt = torch.load(resume, map_location=dev, weights_only=False)
        for key, have in (("N", N), ("width", width), ("depth", depth), ("cond", cond),
                          ("dilated", dilated)):
            if ckpt.get(key, False) != have:
                raise ValueError(f"--resume mismatch: checkpoint {key}={ckpt.get(key)} "
                                 f"but this run has {key}={have}")
        model.load_state_dict(ckpt["state_dict"])
        print(f"resumed from {resume}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # STOP is ~1 of every trajectory-length samples; up-weight it so the net learns to halt.
    w = torch.ones(L + 1, device=dev)
    w[L] = stop_weight
    policy_loss_fn = nn.CrossEntropyLoss(weight=w, reduction="none")   # masked per-sample
    value_loss_fn = nn.BCEWithLogitsLoss()
    value_weight = 1.0

    rf = model.receptive_field
    print(f"N={N}  cond={cond}  in_channels={ch}  classes={L+1}  dilated={dilated}  "
          f"receptive_field={rf}{'  (< 2N-1, PARTIAL coverage)' if rf < L else ''}  "
          f"params={sum(p.numel() for p in model.parameters()):,}  device={dev}")
    for rnd in range(1, rounds + 1):
        X, ym, yv, msk = generate_dataset(N, instances_per_round, cond)
        dl = DataLoader(TensorDataset(X, ym, yv, msk), batch_size=batch_size, shuffle=True)
        model.train()
        tot, correct, pol_seen, vcorrect, seen = 0.0, 0, 0, 0, 0
        for xb, ymb, yvb, mb in dl:
            xb, ymb, yvb, mb = xb.to(dev), ymb.to(dev), yvb.to(dev), mb.to(dev)
            logits, value = model(xb)
            per = policy_loss_fn(logits, ymb) * mb                     # zero out off-path policy loss
            policy_loss = per.sum() / mb.sum().clamp_min(1.0)
            value_loss = value_loss_fn(value.squeeze(1), yvb)
            loss = policy_loss + value_weight * value_loss
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
            if save:                                    # checkpoint at every eval, not just the end
                save_model(model, save, N, width, depth, cond, dilated)
    return model


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _data_sanity(N: int = 8, instances: int = 50):
    random.seed(0)
    pairs = 0
    for _ in range(instances):
        states, labels, _ppop, _qpop = make_instance(N)
        C = weight(states[0])
        for st in states:
            assert weight(st) == C                     # every step conserves W' = C
        # replaying the labelled moves reproduces the terminal target
        cur = states[0][:]
        for a in labels[:-1]:
            cur[a] -= 1
            cur[a - 1] += 2
        assert cur == states[-1] and labels[-1] == 2 * N - 1
        pairs += len(states)
    print(f"data sanity OK (N={N}): {instances} instances, {pairs} (profile,move) pairs, "
          f"weight-conserving, replay reaches target.")


def main():
    ap = argparse.ArgumentParser(description="Conv policy net for Step A of the token game.")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--instances-per-round", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                    help="override backend; default auto-selects cuda > mps (Apple GPU) > cpu")
    ap.add_argument("--cond", choices=["none", "pop", "target"], default="none",
                    help="extra input channels: none | pop (p,q popcounts) | target (n*). "
                         "'target' makes the task well-posed; 'pop' is the promise only.")
    ap.add_argument("--demo", action="store_true", help="quick sanity + small-N (N=12) training")
    ap.add_argument("--save", type=str, default=None, help="checkpoint path (.pt); saved each eval round")
    ap.add_argument("--resume", type=str, default=None,
                    help="warm-start weights from a checkpoint (config must match N/width/depth/cond/dilated)")
    ap.add_argument("--dilated", action="store_true",
                    help="use kernel-3 blocks with exponential dilation (1,2,4,...) so a few "
                         "blocks span the whole 2N-1 profile; recommended for N>=32")
    args = ap.parse_args()

    random.seed(args.seed)
    _data_sanity()
    if args.demo:
        cfg = dict(N=12, width=64, depth=4, cond=args.cond, dilated=args.dilated)
        model = train(rounds=20, instances_per_round=64, device=args.device,
                      save=args.save, resume=args.resume, **cfg)
    else:
        cfg = dict(N=args.N, width=args.width, depth=args.depth, cond=args.cond, dilated=args.dilated)
        model = train(rounds=args.rounds, instances_per_round=args.instances_per_round,
                      batch_size=args.batch_size, lr=args.lr, device=args.device,
                      save=args.save, resume=args.resume, **cfg)
    if args.save:
        save_model(model, args.save, **cfg)             # final save (idempotent)
        print(f"saved checkpoint -> {args.save}  ({cfg})")


if __name__ == "__main__":
    main()
