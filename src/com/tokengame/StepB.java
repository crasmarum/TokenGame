package com.tokengame;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.tokengame.StepA.Cell;

/**
 * Step B of the diagonal token game: seating by slides.
 *
 * <p>A <b>slide</b> moves a token to any empty cell on its own diagonal ({@code r-c}
 * fixed), so it never changes a token's diagonal. Once a target rectangle
 * {@code R x S} is known and the board already carries the rectangle's diagonal
 * profile (as produced by Step A), seating is polynomial: on each diagonal the
 * tokens are slid onto that diagonal's target cells
 * {@code {(r, r-d) : r in R, r-d in S}}. This is the easy case of Step B.
 *
 * <p>When the rectangle is <em>not</em> known and must be recovered from the frozen
 * profile alone, one faces the two-set turnpike / partial-digest problem; that
 * reconstruction (the "regrouping") is done in {@link Solver#reconstruct}.
 */
public final class StepB {

    /** A slide of a token from {@code from} to {@code to} along their common diagonal. */
    public record Slide(Cell from, Cell to) {
        @Override public String toString() { return "slide " + from + " -> " + to; }
    }

    private final int n;

    public StepB(int n) {
        if (n < 1) throw new IllegalArgumentException("N must be >= 1");
        this.n = n;
    }

    /**
     * Slide the tokens of {@code grid} (which must already have the profile of
     * {@code rows x cols}) onto exactly the rectangle {@code rows x cols}. Mutates a
     * copy of {@code grid}; returns the slide sequence.
     */
    public List<Slide> seat(int[] rows, int[] cols, boolean[][] grid) {
        boolean[][] g = StepA.copyGrid(grid);
        Set<Integer> R = toSet(rows), S = toSet(cols);
        List<Slide> slides = new ArrayList<>();

        for (int d = -(n - 1); d <= n - 1; d++) {
            int rTop = Math.max(0, d), rBot = Math.min(n - 1, n - 1 + d);
            List<int[]> movableSources = new ArrayList<>();  // occupied, not on a target
            List<int[]> freeTargets = new ArrayList<>();      // target cell, currently empty
            for (int r = rTop; r <= rBot; r++) {
                int c = r - d;
                boolean occupied = g[r][c];
                boolean target = R.contains(r) && S.contains(c);
                if (occupied && !target) movableSources.add(new int[]{r, c});
                if (target && !occupied) freeTargets.add(new int[]{r, c});
            }
            if (movableSources.size() != freeTargets.size())
                throw new IllegalStateException("diagonal " + d + ": profile does not match rectangle "
                        + "(" + movableSources.size() + " to move, " + freeTargets.size() + " slots)");
            for (int i = 0; i < movableSources.size(); i++) {
                int[] src = movableSources.get(i), tgt = freeTargets.get(i);
                g[src[0]][src[1]] = false;
                g[tgt[0]][tgt[1]] = true;
                slides.add(new Slide(new Cell(src[0], src[1]), new Cell(tgt[0], tgt[1])));
            }
        }
        return slides;
    }

    private static Set<Integer> toSet(int[] xs) {
        Set<Integer> s = new HashSet<>();
        for (int x : xs) s.add(x);
        return s;
    }
}
