package com.tokengame;

import java.math.BigInteger;
import java.util.Random;
import java.util.Set;

import com.tokengame.StepA.Cell;

/**
 * Factoring test for the diagonal token game.
 *
 * <p>Procedure (default grid dimension {@code N = 16}):
 * <ol>
 *   <li>Draw two distinct {@code N}-bit primes {@code P1, P2}.</li>
 *   <li>Let {@code p = popcount(P1)}, {@code q = popcount(P2)}, and {@code M = P1 * P2}.</li>
 *   <li>Lay {@code M} on the board as the conserved weight {@code W' = M} (its binary value
 *       spread across the diagonals via the greedy-high placement).</li>
 *   <li>Solve. The final rectangle {@code R x S} has {@code p * q} tokens, and the two
 *       factors read off its rows and columns must be {@code {P1, P2}} — i.e. solving the
 *       instance factors {@code M}.</li>
 * </ol>
 *
 * <p>Usage: {@code java com.tokengame.FactorTest [N] [seed]}.
 */
public final class FactorTest {

    public static void main(String[] args) {
        int N = args.length > 0 ? Integer.parseInt(args[0]) : 50;
        long seed = args.length > 1 ? Long.parseLong(args[1]) : new Random().nextLong();
        Random rng = new Random(seed);

        StepA stepA = new StepA(N);
        Solver solver = new Solver(N);

        // 1) two distinct N-bit primes
        BigInteger p1 = BigInteger.probablePrime(N, rng);
        BigInteger p2;
        do { p2 = BigInteger.probablePrime(N, rng); } while (p2.equals(p1));
        int p = p1.bitCount(), q = p2.bitCount();
        BigInteger m = p1.multiply(p2);

        System.out.println("N=" + N + "  seed=" + seed);
        System.out.println("P1 = " + p1 + "  [" + bin(p1, N) + "]  p = " + p + " set bits");
        System.out.println("P2 = " + p2 + "  [" + bin(p2, N) + "]  q = " + q + " set bits");
        System.out.println("M  = P1*P2 = " + m + "  (" + m.bitLength() + " bits)");

        // 2) lay M on the board as the weight W'
        int[] profile = stepA.greedyHighProfile(m);
        Set<Cell> start = stepA.cells(profile);
        BigInteger w = stepA.weightW1(profile);
        System.out.println("start: " + start.size() + " tokens,  W'(start) = " + w
                + (w.equals(m) ? "  (== M)" : "  (!! MISMATCH)"));

        // 3) solve under the promise (p rows, q columns)
        long t0 = System.nanoTime();
        Solver.Solution sol = solver.solve(start, p, q);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        if (!sol.solved()) {
            System.out.println("UNSOLVED (" + ms + " ms): " + sol.reason());
            System.out.println("FAIL");
            return;
        }

        // 4) read the factors off the rectangle: rows -> sum 2^r, cols -> sum 2^(N-1-c)
        BigInteger v = BigInteger.ZERO;
        for (int r : sol.rows()) v = v.setBit(r);
        BigInteger mSel = BigInteger.ZERO;
        for (int c : sol.cols()) mSel = mSel.setBit(N - 1 - c);

        // polynomial obtained after Step A: P(x) = sum_d n*_d x^(d+N-1) = rho(x) * sigmaTilde(x)
        int[] nStar = stepA.rectangleProfile(sol.rows(), sol.cols());
        System.out.println("poly after Step A:");
        System.out.println("  P(x)  = " + solver.poly(nStar));
        System.out.println("        = (" + solver.rho(sol.rows()) + ") * (" + solver.sigmaTilde(sol.cols()) + ")");
        System.out.println("  P(2)  = " + stepA.weightW1(nStar) + "   (= M)");

        String verify = solver.verify(start, sol);
        boolean product = v.multiply(mSel).equals(m);
        boolean match = (v.equals(p1) && mSel.equals(p2)) || (v.equals(p2) && mSel.equals(p1));

        System.out.println("solved in " + ms + " ms:  duplicates = " + sol.duplicates().size()
                + ",  slides = " + sol.slides().size() + ",  final tokens = " + (p * q));
        System.out.println("recovered  V = " + v + "  (rows " + java.util.Arrays.toString(sol.rows()) + ")");
        System.out.println("recovered  M_sel = " + mSel + "  (cols " + java.util.Arrays.toString(sol.cols()) + ")");
        System.out.println("check: V*M_sel = M ? " + product + "   {V,M_sel} == {P1,P2} ? " + match
                + "   moves valid & final ? " + (verify == null));
        System.out.println(verify == null && product && match ? "PASS" : "FAIL"
                + (verify == null ? "" : "  (" + verify + ")"));
    }

    private static String bin(BigInteger x, int N) {
        StringBuilder sb = new StringBuilder();
        for (int i = N - 1; i >= 0; i--) sb.append(x.testBit(i) ? '1' : '0');
        return sb.toString();
    }
}
