package com.tokengame;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Step A of the diagonal token game.
 *
 * <p>See the companion note <em>"A Diagonal Token Game, Column Patterns, and
 * Balanced Factorization"</em>. The board is {@code N x N}; a cell {@code (r,c)}
 * lies on the upper-left-to-lower-right diagonal {@code d = r - c}, which holds
 * {@code L_d = N - |d|} cells. A <b>duplicate</b> ("split") move removes one token
 * from diagonal {@code d} and places two on diagonal {@code d-1} (needing at least
 * two free cells there).
 *
 * <p>Given a start profile and a target rectangle profile {@code nStar} of the same
 * conserved weight {@code W = sum n_d 2^d}, Step A emits the forced sequence of
 * duplicate moves, or reports why it is infeasible. With {@code s_d} the tokens
 * pushed from {@code d} to {@code d-1},
 * <pre>   s_d = n0_d + 2 * s_{d+1} - nStar_d      (swept top-down, s above the board = 0)</pre>
 * and the transformation is feasible iff (i) {@code s_d >= 0}, (ii) {@code nStar_d <= L_d},
 * and (iii) {@code L_{d-1} >= 2} whenever {@code s_d > 0}; it is realized in
 * {@code sum s_d = |nStar| - |n0| <= N^2} moves. The realization runs on the concrete
 * board, so the moves are valid from any starting placement with the given profile.
 *
 * <p>Profiles are indexed by diagonal via {@code idx(d) = d + (N-1)}; arrays have
 * length {@code 2N-1} covering {@code d = -(N-1) .. (N-1)}.
 */
public final class StepA {

    /** A board cell. */
    public record Cell(int r, int c) {
        @Override public String toString() { return "(" + r + "," + c + ")"; }
    }

    /** A duplicate move: remove {@code src} on diagonal {@code d}, add {@code a,b} on {@code d-1}. */
    public record Move(int d, Cell src, Cell a, Cell b) {
        @Override public String toString() { return "dup d=" + d + ": " + src + " -> {" + a + ", " + b + "}"; }
    }

    /** Outcome of {@link #solve}. {@code reason} is {@code null} iff feasible. */
    public static final class Result {
        public final boolean feasible;
        public final String reason;
        public final List<Move> moves;
        private Result(boolean feasible, String reason, List<Move> moves) {
            this.feasible = feasible; this.reason = reason; this.moves = moves;
        }
        static Result infeasible(String reason) { return new Result(false, reason, List.of()); }
        static Result ok(List<Move> moves) { return new Result(true, null, moves); }
    }

    private final int n;

    public StepA(int n) {
        if (n < 1) throw new IllegalArgumentException("N must be >= 1");
        this.n = n;
    }

    public int N() { return n; }

    // ---- diagonal geometry -------------------------------------------------

    public int capacity(int d) { return n - Math.abs(d); }
    private int lo() { return -(n - 1); }
    private int hi() { return n - 1; }
    private int idx(int d) { return d + (n - 1); }
    private int len() { return 2 * n - 1; }

    // ---- profiles / grids --------------------------------------------------

    /** Diagonal profile of the rectangle {@code rows x cols}. */
    public int[] rectangleProfile(int[] rows, int[] cols) {
        int[] p = new int[len()];
        for (int r : rows) for (int c : cols) { requireOnBoard(r); requireOnBoard(c); p[idx(r - c)]++; }
        return p;
    }

    /** Per-diagonal token counts of a grid. */
    public int[] profileOf(boolean[][] grid) {
        int[] p = new int[len()];
        for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) if (grid[r][c]) p[idx(r - c)]++;
        return p;
    }

    /** A grid realizing {@code profile} with each diagonal's tokens top-justified. */
    public boolean[][] canonicalGrid(int[] profile) {
        checkProfile(profile, "profile");
        boolean[][] g = new boolean[n][n];
        for (int d = lo(); d <= hi(); d++) for (int t = 0; t < profile[idx(d)]; t++) fillTopFree(g, d);
        return g;
    }

    /** Conserved weight in integer form: {@code W' = sum n_d 2^(d + N - 1)}. */
    public BigInteger weightW1(int[] profile) {
        BigInteger w = BigInteger.ZERO;
        for (int d = lo(); d <= hi(); d++) {
            int c = profile[idx(d)];
            if (c != 0) w = w.add(BigInteger.valueOf(c).shiftLeft(d + n - 1));
        }
        return w;
    }

    /**
     * Lay an integer weight {@code W} on the board as {@code W' = sum n_d 2^(d+N-1)},
     * greedily filling the highest diagonals first. This uses the fewest tokens (so at
     * most {@code p*q} for any rectangle of the same weight) and maximizes every
     * {@code U_k}, so every same-weight rectangle is reachable from it by Step A.
     * Requires {@code 0 <= W <= (2^N - 1)^2}.
     */
    public int[] greedyHighProfile(BigInteger W) {
        if (W.signum() < 0) throw new IllegalArgumentException("weight must be nonnegative");
        int[] p = new int[len()];
        BigInteger rem = W;
        for (int d = hi(); d >= lo(); d--) {
            BigInteger unit = BigInteger.ONE.shiftLeft(d + n - 1);
            int take = rem.divide(unit).min(BigInteger.valueOf(capacity(d))).intValue();
            if (take > 0) { p[idx(d)] = take; rem = rem.subtract(unit.multiply(BigInteger.valueOf(take))); }
        }
        if (rem.signum() != 0)
            throw new IllegalArgumentException("weight " + W + " is not representable on a " + n + "x" + n + " board");
        return p;
    }

    /** The occupied cells of the canonical placement of {@code profile}. */
    public java.util.Set<Cell> cells(int[] profile) {
        java.util.Set<Cell> s = new java.util.LinkedHashSet<>();
        boolean[][] g = canonicalGrid(profile);
        for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) if (g[r][c]) s.add(new Cell(r, c));
        return s;
    }

    public static boolean[][] copyGrid(boolean[][] g) {
        boolean[][] h = new boolean[g.length][];
        for (int i = 0; i < g.length; i++) h[i] = g[i].clone();
        return h;
    }

    /** Replay duplicate moves on a copy of {@code grid} and return the resulting grid. */
    public boolean[][] applyDuplicates(boolean[][] grid, List<Move> moves) {
        boolean[][] g = copyGrid(grid);
        for (Move m : moves) {
            g[m.src().r()][m.src().c()] = false;
            g[m.a().r()][m.a().c()] = true;
            g[m.b().r()][m.b().c()] = true;
        }
        return g;
    }

    // ---- Step A ------------------------------------------------------------

    /** Transform the canonical placement of {@code n0} into {@code nStar}. */
    public Result solve(int[] n0, int[] nStar) {
        return solveOnGrid(canonicalGrid(n0), nStar);
    }

    /** Transform the given {@code grid} (any placement) into a grid of profile {@code nStar}. */
    public Result solveOnGrid(boolean[][] grid, int[] nStar) {
        int[] n0 = profileOf(grid);
        checkProfile(n0, "start"); checkProfile(nStar, "nStar");

        int[] s = new int[len()];
        for (int d = hi(); d >= lo(); d--) {
            int i = idx(d);
            if (nStar[i] > capacity(d))
                return Result.infeasible("target " + nStar[i] + " exceeds capacity " + capacity(d) + " on diagonal " + d);
            int sAbove = (d == hi()) ? 0 : s[idx(d + 1)];
            int present = n0[i] + 2 * sAbove;
            int sd = present - nStar[i];
            if (sd < 0)
                return Result.infeasible("deficit of " + (-sd) + " on diagonal " + d + " (weight cannot flow upward)");
            if (sd > 0) {
                int below = d - 1;
                if (below < lo() || capacity(below) < 2)
                    return Result.infeasible("must duplicate into diagonal " + below + ", which has fewer than 2 cells");
            }
            s[i] = sd;
        }
        return realize(copyGrid(grid), n0, s);
    }

    private Result realize(boolean[][] grid, int[] n0, int[] s) {
        int[] cur = n0.clone();
        int[] remaining = s.clone();
        List<Move> moves = new ArrayList<>();
        int budget = n * n + 1;
        for (int d = highestRemaining(remaining); d != NONE; d = highestRemaining(remaining)) {
            doSplit(d, grid, cur, remaining, moves);
            if (moves.size() > budget) throw new IllegalStateException("exceeded duplicate bound; scheduling bug");
        }
        return Result.ok(moves);
    }

    private static final int NONE = Integer.MIN_VALUE;

    private int highestRemaining(int[] remaining) {
        for (int d = hi(); d >= lo(); d--) if (remaining[idx(d)] > 0) return d;
        return NONE;
    }

    private void doSplit(int d, boolean[][] grid, int[] cur, int[] remaining, List<Move> moves) {
        while (capacity(d - 1) - cur[idx(d - 1)] < 2) {          // make room on d-1 first
            if (remaining[idx(d - 1)] <= 0)
                throw new IllegalStateException("no room and no pending split on diagonal " + (d - 1));
            doSplit(d - 1, grid, cur, remaining, moves);
        }
        Cell src = bottomOccupied(grid, d);
        grid[src.r()][src.c()] = false;
        cur[idx(d)]--;
        Cell a = fillTopFree(grid, d - 1);
        Cell b = fillTopFree(grid, d - 1);
        cur[idx(d - 1)] += 2;
        remaining[idx(d)]--;
        moves.add(new Move(d, src, a, b));
    }

    // ---- concrete cells on a diagonal (top = smallest row) -----------------

    private int rowTop(int d) { return Math.max(0, d); }
    private int rowBot(int d) { return Math.min(n - 1, n - 1 + d); }

    private Cell fillTopFree(boolean[][] grid, int d) {
        for (int r = rowTop(d); r <= rowBot(d); r++)
            if (!grid[r][r - d]) { grid[r][r - d] = true; return new Cell(r, r - d); }
        throw new IllegalStateException("no free cell on diagonal " + d);
    }

    private Cell bottomOccupied(boolean[][] grid, int d) {
        for (int r = rowBot(d); r >= rowTop(d); r--)
            if (grid[r][r - d]) return new Cell(r, r - d);
        throw new IllegalStateException("no token on diagonal " + d);
    }

    // ---- verification ------------------------------------------------------

    /** Replay from the canonical placement of {@code n0}; null iff legal and ends at {@code nStar}. */
    public String verify(int[] n0, int[] nStar, List<Move> moves) {
        boolean[][] grid = canonicalGrid(n0);
        for (Move m : moves) {
            int d = m.d();
            if (m.src().r() - m.src().c() != d) return "src not on diagonal " + d + ": " + m;
            if (!grid[m.src().r()][m.src().c()]) return "src empty: " + m;
            if (!onDiag(m.a(), d - 1) || !onDiag(m.b(), d - 1)) return "dest not on diagonal " + (d - 1) + ": " + m;
            if (m.a().equals(m.b())) return "duplicate dest cells: " + m;
            grid[m.src().r()][m.src().c()] = false;
            if (grid[m.a().r()][m.a().c()] || grid[m.b().r()][m.b().c()]) return "dest occupied: " + m;
            grid[m.a().r()][m.a().c()] = true;
            grid[m.b().r()][m.b().c()] = true;
        }
        int[] got = profileOf(grid);
        for (int d = lo(); d <= hi(); d++)
            if (got[idx(d)] != nStar[idx(d)])
                return "profile mismatch on diagonal " + d + ": got " + got[idx(d)] + " want " + nStar[idx(d)];
        return null;
    }

    private boolean onDiag(Cell x, int d) {
        return x.r() >= 0 && x.r() < n && x.c() >= 0 && x.c() < n && x.r() - x.c() == d;
    }

    // ---- validation --------------------------------------------------------

    private void requireOnBoard(int v) {
        if (v < 0 || v >= n) throw new IllegalArgumentException("row/col out of range: " + v);
    }

    private void checkProfile(int[] p, String name) {
        if (p.length != len()) throw new IllegalArgumentException(name + " must have length " + len());
        for (int d = lo(); d <= hi(); d++) {
            int c = p[idx(d)];
            if (c < 0 || c > capacity(d))
                throw new IllegalArgumentException(name + " diagonal " + d + " count " + c + " outside [0," + capacity(d) + "]");
        }
    }

    /** Same-weight, reachable start built by random inverse-splits (merges) of {@code nStar}. */
    int[] randomHigher(int[] nStar, Random rng, int steps) {
        int[] p = nStar.clone();
        for (int i = 0; i < steps; i++) {
            List<Integer> ok = new ArrayList<>();
            for (int d = lo() + 1; d <= hi(); d++)
                if (p[idx(d - 1)] >= 2 && p[idx(d)] < capacity(d)) ok.add(d);
            if (ok.isEmpty()) break;
            int d = ok.get(rng.nextInt(ok.size()));
            p[idx(d - 1)] -= 2; p[idx(d)] += 1;
        }
        return p;
    }

    // ---- demo / self-test --------------------------------------------------

    public static void main(String[] args) {
        int N = 4;
        StepA game = new StepA(N);
        int[] nStar = game.rectangleProfile(new int[]{0, 1}, new int[]{0, 1});
        int[] n0 = new int[2 * N - 1];
        n0[game.idx(-1)] = 1;
        n0[game.idx(1)] = 2;

        System.out.println("[StepA] N=" + N + "  W'(n0)=" + game.weightW1(n0) + "  W'(nStar)=" + game.weightW1(nStar));
        Result res = game.solve(n0, nStar);
        System.out.println("  feasible=" + res.feasible + (res.reason == null ? "" : " (" + res.reason + ")"));
        res.moves.forEach(m -> System.out.println("    " + m));
        System.out.println("  verify=" + (game.verify(n0, nStar, res.moves) == null ? "OK" : game.verify(n0, nStar, res.moves)));

        Random rng = new Random(12345);
        int trials = 5000, fails = 0;
        for (int t = 0; t < trials; t++) {
            int m = 2 + rng.nextInt(6);
            StepA g = new StepA(m);
            int[] rows = randomSubset(m, rng), cols = randomSubset(m, rng);
            int[] star = g.rectangleProfile(rows, cols);
            int[] start = g.randomHigher(star, rng, 1 + rng.nextInt(20));
            Result r = g.solve(start, star);
            String v = r.feasible ? g.verify(start, star, r.moves) : "infeasible (should be feasible)";
            if (v != null) { fails++; if (fails <= 3) System.out.println("  FAIL N=" + m + ": " + v); }
        }
        System.out.println("[StepA] stress: " + (trials - fails) + "/" + trials + " passed");
    }

    private static int[] randomSubset(int m, Random rng) {
        List<Integer> xs = new ArrayList<>();
        for (int i = 0; i < m; i++) if (rng.nextBoolean()) xs.add(i);
        if (xs.isEmpty()) xs.add(rng.nextInt(m));
        return xs.stream().mapToInt(Integer::intValue).toArray();
    }
}
