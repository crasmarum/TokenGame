# Searching for Primes: A Neural AlphaZero Approach to a Factoring Game

A one-player token game on an *N×N* board whose winning condition is **equivalent to integer
factoring**, together with an AlphaZero-style solver (a learned policy/value network + MCTS)
that probes the limits of neural look-ahead on a factoring-equivalent environment.

**The idea.** Tokens *slide* along diagonals or *duplicate* onto the neighbouring one to form
a combinatorial rectangle `R×S`. A conserved integer weight `W'` makes reaching a final
position **factor `W'` into two N-bit factors `V, M`** that encode the rectangle's rows and
columns — so solving the game for a balanced-semiprime target is integer factoring. Once the
target rectangle is known the solution is polynomial (a forced "chip-flow" for Step A, and a
`0/1`-polynomial factorisation for Step B, irreducible by Cohn's theorem); **all difficulty is
the initial number-theoretic split.** The popcount promise bounds the target search space,
which we attack with a dilated-CNN policy/value network and Monte-Carlo tree search.

## Repository layout

```
paper/   diagonal-token-game.tex (+ PDF and figures) — the manuscript
src/     Java reference implementation (Step A, Step B, solver, factoring test)
ml/      PyTorch: policy/value net (CNN + Transformer) and AlphaZero-style MCTS
```

## Build the paper

```bash
cd paper
pdflatex diagonal-token-game.tex && pdflatex diagonal-token-game.tex   # inline bibliography, two passes
```

## Java reference solver

`src/com/tokengame/` implements the exact algorithms: `StepA` (forced chip-flow),
`StepB` (seating by slides), `Solver` (factor-selection → A → B), and `FactorTest`
(lay `M = P1·P2` on the board and solve, recovering the factors).

```bash
cd src
javac -d out com/tokengame/*.java
java -cp out com.tokengame.FactorTest 16     # N=16: two 16-bit primes, solve = factor
```

## Neural solver (PyTorch)

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see the file for the CUDA vs CPU torch index

# 1) train the policy/value network (pop-conditioned, dilated CNN)
python token_game_policy.py --N 16 --cond pop --dilated --width 160 --depth 16 \
    --rounds 4000 --save models/pol_n16.pt

# 2) run AlphaZero-style MCTS with the trained model (reports solve rate + Wilson 95% CI)
python token_game_mcts.py --N 16 --model models/pol_n16.pt --sims 20000 --rollout value --instances 50

# (optional) attention ablation: same task/metrics, Transformer trunk
python token_game_transformer.py --N 16 --cond pop --d-model 192 --layers 6 --rounds 4000
```

Key metrics reported during training: **move-acc** (per-step imitation of Step A),
**val-acc** (target reachability, the value head), and **greedy-solve** (fraction of fresh
instances solved end-to-end). The headline empirical result: local imitation saturates while
end-to-end solving stays at the noise floor for moderate `N` — the factoring wall.

Checkpoints (`ml/models/*.pt`) and the virtualenv (`ml/.venv/`) are git-ignored.

## Citation

Marcel Crasmaru, *Searching for Primes: A Neural AlphaZero Approach to a Factoring Game*, 2026.

## License

Code under the MIT License (see `LICENSE`). The manuscript text in `paper/` is © the author.
