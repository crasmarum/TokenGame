"""
AlphaZero-style Monte-Carlo Tree Search for the diagonal token game.

The game (see diagonal-token-game.tex and token_game_policy.py):
  * A node is a diagonal profile n (an array of length 2N-1); the ROOT lays the number m to
    be factored on the board as weight, n0 = greedy_high(m). Every node has weight
    W' = sum_i n[i] 2^i = m (conserved by every move).
  * An action is a DUPLICATION at index i (1 <= i <= 2N-2): n[i]-=1, n[i-1]+=2. It adds one
    token, so the token count sum(n) increases by exactly 1 each move.
  * A node is FINAL when it has p*q tokens (sum(n) == p*q). Because each move adds one token,
    every root-to-leaf path has the same length p*q - sum(n0) and ends at a final node.
  * At a final node we run STEP B (`step_b`): try to factor the profile's polynomial into two
    0/1 polynomials of sizes p and q. Reward = 1 if that succeeds (a factorization of m was
    found), else 0.

Search (AlphaZero flavour):
  * PUCT selection  a* = argmax_a  Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1+N(s,a)).
  * The priors P(s,a) come from the trained policy network (`token_game_policy`), softmaxed
    over the legal duplication moves -- this is the "additional weight" the model gives.
    With no model, priors are uniform (plain MCTS).
  * Leaf value: a terminal's 0/1 Step-B reward, reached by a policy-guided rollout from a
    non-terminal leaf (there is no value head; the terminal reward is the ground truth).

Usage:  .venv/bin/python token_game_mcts.py --N 6 --sims 4000
        (optionally --cond pop with a model; here priors default to uniform)
"""

from __future__ import annotations
import argparse
import math
import random
import sys
from typing import Dict, List, Optional, Tuple

from token_game_policy import (
    greedy_high_profile, weight, random_nbit_prime, rectangle_profile, featurize,
    load_model as _load_cnn, pick_device,
)


def load_model(path, device=None):
    """Load a policy/value checkpoint, dispatching on its architecture tag.

    Both nets share the same (features -> (policy, value)) contract, so the search code is
    identical; only the constructor differs. Transformer checkpoints carry arch='transformer'.
    """
    import torch
    tag = torch.load(path, map_location="cpu", weights_only=False).get("arch")
    if tag == "transformer":
        from token_game_transformer import load_model as _load_tf
        return _load_tf(path, device)
    return _load_cnn(path, device)


# --------------------------------------------------------------------------------------
# Step B: reward via 0/1-polynomial factorization of the final profile
# --------------------------------------------------------------------------------------

def step_b(P: List[int], p: int, q: int, branch_cap: int = 200_000):
    """Factor the profile polynomial sum_i P[i] x^i into rho * sigma (both 0/1),
    with |rho| = p and |sigma| = q ones. Returns (V, M, R_exps, S_exps) or None.

    Incremental low->high reconstruction: with rho(0)=sigma(0)=1, at each exponent k
        P[k] = rho[k] + sigma[k] + (known cross terms),
    so rho[k]+sigma[k] is forced to 0, 1 (a branch), or 2; anything else is infeasible.
    (For an N-bit semiprime both factors are odd, so P[0]=1 always holds at a valid target.)
    """
    n = len(P)
    if not P or P[0] != 1 or sum(P) != p * q:
        return None
    A = [0] * n
    B = [0] * n
    A[0] = B[0] = 1
    calls = [0]
    out = [None]

    def rec(k: int, ao: int, bo: int):
        if out[0] is not None:
            return
        calls[0] += 1
        if calls[0] > branch_cap:
            return
        if k == n:
            if ao == p and bo == q:
                out[0] = (A[:], B[:])
            return
        cross = 0
        for i in range(1, k):
            if A[i] and B[k - i]:
                cross += 1
        t = P[k] - cross
        if t == 0:
            opts = ((0, 0),)
        elif t == 2:
            opts = ((1, 1),)
        elif t == 1:
            opts = ((1, 0), (0, 1))
        else:
            return
        for ak, bk in opts:
            if ao + ak > p or bo + bk > q:
                continue
            A[k], B[k] = ak, bk
            rec(k + 1, ao + ak, bo + bk)
            A[k] = B[k] = 0
            if out[0] is not None:
                return

    rec(1, 1, 1)
    if out[0] is None:
        return None
    A, B = out[0]
    V = sum(1 << i for i in range(n) if A[i])
    M = sum(1 << i for i in range(n) if B[i])
    R = [i for i in range(n) if A[i]]
    S = [i for i in range(n) if B[i]]
    return V, M, R, S


# --------------------------------------------------------------------------------------
# MCTS
# --------------------------------------------------------------------------------------

class _Node:
    __slots__ = ("profile", "is_terminal", "reward", "P", "Nsa", "Wsa", "children")

    def __init__(self, profile: Tuple[int, ...]):
        self.profile = profile
        self.is_terminal: Optional[bool] = None
        self.reward: Optional[float] = None
        self.P: Dict[int, float] = {}          # priors
        self.Nsa: Dict[int, int] = {}          # edge visit counts
        self.Wsa: Dict[int, float] = {}        # edge total value
        self.children: Dict[int, "_Node"] = {}


class TokenGameMCTS:
    def __init__(self, N: int, m: int, p: int, q: int, model=None, cond: str = "none",
                 c_puct: float = 1.5, rollout: str = "policy", device: str = "cpu",
                 rng: Optional[random.Random] = None):
        assert weight(greedy_high_profile(m, N)) == m, "m does not fit the N x N board"
        self.N = N
        self.m = m
        self.p = p
        self.q = q
        self.pq = p * q
        self.L = 2 * N - 1
        self.model = model
        self.cond = cond
        self.c_puct = c_puct
        self.rollout_mode = rollout          # "policy" | "random" | "none"
        self.device = device
        self.rng = rng or random.Random()

    # ---- game mechanics -------------------------------------------------------------
    def legal_moves(self, prof) -> List[int]:
        return [i for i in range(1, self.L) if prof[i] > 0]     # split index i -> i-1

    def apply(self, prof, a) -> Tuple[int, ...]:
        n = list(prof)
        n[a] -= 1
        n[a - 1] += 2
        return tuple(n)

    def is_final(self, prof) -> bool:
        return sum(prof) == self.pq

    def reward(self, prof) -> float:
        """Step B at a final node: 1 if the profile factors m into two 0/1 polys of sizes p,q."""
        res = step_b(list(prof), self.p, self.q)
        if res is None:
            return 0.0
        V, M, _, _ = res
        return 1.0 if V * M == self.m else 0.0

    # ---- priors from the model (the "additional weight") ----------------------------
    def priors(self, prof, legal: List[int]) -> Dict[int, float]:
        if self.model is None or not legal:
            u = 1.0 / max(1, len(legal))
            return {a: u for a in legal}
        import torch
        feats = featurize(list(prof), self.cond, self.N, self.p, self.q, None)
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy, _value = self.model(x)              # (1, 2N), (1, 1)
        logits = policy[0].tolist()                     # length 2N; classes 0..2N-2 = moves
        mx = max(logits[a] for a in legal)
        exps = {a: math.exp(logits[a] - mx) for a in legal}
        z = sum(exps.values())
        return {a: e / z for a, e in exps.items()}

    def _value(self, prof) -> float:
        """Value head: estimated P(target still reachable from `prof`) in [0,1]."""
        import torch
        feats = featurize(list(prof), self.cond, self.N, self.p, self.q, None)
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _policy, value = self.model(x)
            return torch.sigmoid(value)[0, 0].item()

    # ---- rollout for leaf value -----------------------------------------------------
    def _rollout(self, prof) -> float:
        cur = prof
        while not self.is_final(cur):
            legal = self.legal_moves(cur)
            if not legal:
                return 0.0
            if self.rollout_mode == "random" or self.model is None:
                a = self.rng.choice(legal)
            else:
                pr = self.priors(cur, legal)
                r, acc = self.rng.random(), 0.0
                a = legal[-1]
                for cand, pv in pr.items():
                    acc += pv
                    if r <= acc:
                        a = cand
                        break
            cur = self.apply(cur, a)
        return self.reward(cur)

    def _leaf_value(self, node: _Node) -> float:
        if node.is_terminal:
            return node.reward
        if self.rollout_mode == "value" and self.model is not None:
            return self._value(node.profile)          # value head, no rollout
        if self.rollout_mode == "none":
            return 0.0
        return self._rollout(node.profile)

    # ---- tree ops -------------------------------------------------------------------
    def _expand(self, node: _Node):
        prof = node.profile
        if self.is_final(prof):
            node.is_terminal = True
            node.reward = self.reward(prof)
            return
        node.is_terminal = False
        legal = self.legal_moves(prof)
        node.P = self.priors(prof, legal)
        for a in legal:
            node.Nsa[a] = 0
            node.Wsa[a] = 0.0

    def _select(self, node: _Node) -> int:
        tot = sum(node.Nsa.values())
        sqrt_tot = math.sqrt(tot + 1e-8)
        best, best_a = -1e18, None
        for a, pa in node.P.items():
            na = node.Nsa[a]
            q = (node.Wsa[a] / na) if na > 0 else 0.0
            u = self.c_puct * pa * sqrt_tot / (1 + na)
            if q + u > best:
                best, best_a = q + u, a
        return best_a

    def _simulate(self, node: _Node) -> float:
        if node.is_terminal:
            return node.reward
        a = self._select(node)
        child = node.children.get(a)
        if child is None:
            child = _Node(self.apply(node.profile, a))
            node.children[a] = child
            self._expand(child)
            v = self._leaf_value(child)
        else:
            v = self._simulate(child)
        node.Nsa[a] += 1
        node.Wsa[a] += v
        return v

    def search(self, root_profile, num_simulations: int) -> _Node:
        root = _Node(tuple(root_profile))
        self._expand(root)
        for _ in range(num_simulations):
            self._simulate(root)
        return root

    # ---- driver: play greedily by visit count until a final node --------------------
    def solve(self, sims_per_move: int = 2000, verbose: bool = False):
        prof = tuple(greedy_high_profile(self.m, self.N))
        path = [prof]
        moves = 0
        while not self.is_final(prof):
            root = self.search(prof, sims_per_move)
            if not root.Nsa:
                break
            a = max(root.Nsa, key=lambda k: (root.Nsa[k], root.Wsa[k]))
            prof = self.apply(prof, a)
            path.append(prof)
            moves += 1
            if verbose:
                print(f"  move {moves}: split index {a}  tokens={sum(prof)}/{self.pq}")
        res = step_b(list(prof), self.p, self.q)
        solved = res is not None and res[0] * res[1] == self.m
        return solved, list(prof), (res[:2] if res else None), path


# --------------------------------------------------------------------------------------
# CLI / demo
# --------------------------------------------------------------------------------------

def nbit_prime_popcounts(N: int, _cache={}):
    """Histogram {popcount: #N-bit primes}. Cached; skipped (None) when 2^N is too big to sieve."""
    if N in _cache:
        return _cache[N]
    if N > 24:
        _cache[N] = None
        return None
    from collections import Counter
    n = 1 << N
    s = bytearray([1]) * n
    s[0] = s[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    _cache[N] = Counter(bin(p).count("1") for p in range(1 << (N - 1), n) if s[p])
    return _cache[N]


def main():
    ap = argparse.ArgumentParser(description="AlphaZero-style MCTS for the token game.")
    ap.add_argument("--N", type=int, default=6, help="board dimension (keep small; search is hard)")
    ap.add_argument("--sims", type=int, default=4000, help="simulations per move")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--rollout", choices=["policy", "random", "none", "value"], default="random",
                    help="leaf evaluation: value = use the model's value head (needs --model)")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for instances + rollouts; omit for a fresh random run each time "
                         "(the chosen seed is printed so you can reproduce it)")
    ap.add_argument("--model", type=str, default=None,
                    help="policy checkpoint (.pt from token_game_policy --save) for PUCT priors")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = ap.parse_args()
    sys.setrecursionlimit(10_000)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 63)
    print(f"seed = {seed}" + ("" if args.seed is not None else "  (random; pass --seed to reproduce)"))
    random.seed(seed)                      # seeds instance generation (global random)...
    rng = random.Random(seed)              # ...and gives rollouts their own stream

    model, cond, device = None, "none", "cpu"
    if args.model:
        device = pick_device(args.device)
        model, ckpt_N, cond = load_model(args.model, device)
        if ckpt_N != args.N:
            raise SystemExit(f"checkpoint is for N={ckpt_N}, but --N {args.N} was given")
        if cond == "target":
            raise SystemExit("cond=target needs n* (the answer) and cannot drive search; "
                             "train with --cond none or --cond pop")
        print(f"loaded model {args.model}  (N={ckpt_N}, cond={cond}, device={device})")

    solved_count = 0
    for t in range(1, args.instances + 1):
        # random N-bit semiprime with the promise (p,q) = popcounts of the two primes
        p1 = random_nbit_prime(args.N)
        p2 = random_nbit_prime(args.N)
        while p2 == p1:
            p2 = random_nbit_prime(args.N)
        if p1 > p2:
            p1, p2 = p2, p1
        m = p1 * p2
        p, q = bin(p1).count("1"), bin(p2).count("1")
        n0 = greedy_high_profile(m, args.N)
        depth_target = p * q - sum(n0)
        hist = nbit_prime_popcounts(args.N)
        cand = f"   candidates c[p]*c[q] = {hist[p] * hist[q]:,}" if hist else ""

        # print the instance up front (before the search runs)
        print(f"inst {t}: factor m = {m}  ({m.bit_length()} bits)   "
              f"promises (p,q) = ({p},{q})   [target {p}x{q} = {p*q} tokens, "
              f"{depth_target} duplications]{cand}", flush=True)

        mcts = TokenGameMCTS(args.N, m, p, q, model=model, cond=cond, c_puct=args.c_puct,
                             rollout=args.rollout, device=device, rng=rng)
        solved, prof, factors, path = mcts.solve(sims_per_move=args.sims)
        depth = len(path) - 1
        if solved:
            result = f"[OK] found {factors[0]} * {factors[1]} = {m}"
        else:
            result = f"[--] no factorization  (true split {p1} * {p2})"
        print(f"        {result}  (depth reached {depth}, sims/move {args.sims})", flush=True)
        solved_count += solved

    priors_desc = f"model priors ({args.model})" if model is not None else "uniform priors"
    n = args.instances
    phat = solved_count / n
    z = 1.96                                   # Wilson 95% interval
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / den
    print(f"\nsolved {solved_count}/{n} = {phat:.3f}   Wilson 95% CI "
          f"[{max(0.0, centre - half):.3f}, {min(1.0, centre + half):.3f}]   "
          f"(N={args.N}, {args.rollout} rollout, {priors_desc})")


if __name__ == "__main__":
    main()
