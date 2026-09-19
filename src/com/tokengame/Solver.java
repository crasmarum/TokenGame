package com.tokengame;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import com.tokengame.StepA.Cell;
import com.tokengame.StepA.Move;
import com.tokengame.StepB.Slide;

/**
 * Top-level solver for the diagonal token game.
 *
 * <p>Chains the three phases of the note:
 * <ol>
 *   <li><b>Factor-selection</b>: find a factorization {@code W' = V * M} with
 *       {@code 1 <= V,M < 2^N}, i.e. a target rectangle {@code R = bits(V)},
 *       {@code S = {N-1-j : bit j of M}}, whose diagonal profile is reachable from
 *       the start by Step A. This is the only super-polynomial phase (it is
 *       balanced factoring); here it is trial division.</li>
 *   <li><b>Step A</b> (duplicates): reach the rectangle's diagonal profile.</li>
 *   <li><b>Step B</b> (slides): seat the tokens onto {@code R x S}.</li>
 * </ol>
 * The returned move list (duplicates then slides) is valid from the given position.
 */
public final class Solver {

    public record Solution(boolean solved, String reason,
                           int[] rows, int[] cols,
                           List<Move> duplicates, List<Slide> slides) {}

    private final int n;
    private final StepA stepA;
    private final StepB stepB;

    public Solver(int n) {
        this.n = n;
        this.stepA = new StepA(n);
        this.stepB = new StepB(n);
    }

    public Solution solve(Set<Cell> position) { return solve(position, -1, -1); }

    /** Solve; if {@code p,q > 0} restrict to rectangles with {@code p} rows and {@code q} columns (the promise). */
    public Solution solve(Set<Cell> position, int p, int q) {
        boolean[][] start = gridOf(position);
        int[] n0 = stepA.profileOf(start);
        BigInteger w1 = stepA.weightW1(n0);
        BigInteger cap = BigInteger.ONE.shiftLeft(n).subtract(BigInteger.ONE);   // 2^N - 1

        for (BigInteger v : divisorsInRange(w1, cap)) {
            BigInteger m = w1.divide(v);
            int[] rows = bits(v), cols = colsFromM(m);
            if (rows.length == 0 || cols.length == 0) continue;
            if (p > 0 && rows.length != p) continue;
            if (q > 0 && cols.length != q) continue;

            int[] nStar = stepA.rectangleProfile(rows, cols);
            StepA.Result a = stepA.solveOnGrid(start, nStar);
            if (!a.feasible) continue;                                  // rectangle not reachable; try next factor

            boolean[][] afterA = stepA.applyDuplicates(start, a.moves);
            List<Slide> slides = stepB.seat(rows, cols, afterA);
            return new Solution(true, null, rows, cols, a.moves, slides);
        }
        return new Solution(false, "no reachable rectangle: W'=" + w1 + " has no factorization V*M with "
                + "V,M < 2^" + n + " whose rectangle Step A can reach", null, null, List.of(), List.of());
    }

    /**
     * Turnpike / partial-digest reconstruction: recover a rectangle {@code (R,S)} whose
     * diagonal profile equals {@code profile}, or {@code null}. Here it is done by
     * matching a divisor of {@code W' = weightW1(profile)} against the profile (the
     * integer-arithmetic form of splitting the {@code 0/1}-polynomial product).
     */
    public int[][] reconstruct(int[] profile) {
        BigInteger w1 = stepA.weightW1(profile);
        BigInteger cap = BigInteger.ONE.shiftLeft(n).subtract(BigInteger.ONE);
        for (BigInteger v : divisorsInRange(w1, cap)) {
            int[] rows = bits(v), cols = colsFromM(w1.divide(v));
            if (rows.length == 0 || cols.length == 0) continue;
            if (java.util.Arrays.equals(stepA.rectangleProfile(rows, cols), profile))
                return new int[][]{rows, cols};
        }
        return null;
    }

    // ---- polynomial view (Step B encoding) --------------------------------

    /** The polynomial {@code P(x) = sum_d profile_d x^(d+N-1)} read off a profile. */
    public String poly(int[] profile) {
        StringBuilder sb = new StringBuilder();
        for (int e = 2 * n - 2; e >= 0; e--) {
            int c = profile[e];
            if (c == 0) continue;
            if (sb.length() > 0) sb.append(" + ");
            appendTerm(sb, c, e);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    /** The row factor {@code rho(x) = sum_{r in R} x^r}. */
    public String rho(int[] rows) { return zeroOnePoly(rows); }

    /** The (reflected) column factor {@code sigmaTilde(x) = sum_{c in S} x^(N-1-c)}. */
    public String sigmaTilde(int[] cols) {
        int[] e = new int[cols.length];
        for (int i = 0; i < cols.length; i++) e[i] = n - 1 - cols[i];
        return zeroOnePoly(e);
    }

    private String zeroOnePoly(int[] exps) {
        int[] e = exps.clone();
        java.util.Arrays.sort(e);
        StringBuilder sb = new StringBuilder();
        for (int i = e.length - 1; i >= 0; i--) {
            if (sb.length() > 0) sb.append(" + ");
            appendTerm(sb, 1, e[i]);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    private static void appendTerm(StringBuilder sb, int coeff, int exp) {
        if (exp == 0) { sb.append(coeff); return; }
        if (coeff != 1) sb.append(coeff);
        sb.append("x");
        if (exp != 1) sb.append("^").append(exp);
    }

    // ---- factor-selection helpers -----------------------------------------

    private final Random rnd = new Random();
    private static final int[] SMALL_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};

    /**
     * Divisors {@code V} of {@code w} with {@code 1 <= V <= cap} and {@code w/V <= cap}.
     * Factors {@code w} with Pollard's rho ({@code O(w^{1/4})}), then enumerates and filters
     * divisors -- far below the old trial-division {@code O(sqrt(w))}.
     */
    private List<BigInteger> divisorsInRange(BigInteger w, BigInteger cap) {
        if (w.signum() <= 0) return List.of();
        LinkedHashSet<BigInteger> res = new LinkedHashSet<>();
        for (BigInteger v : allDivisors(factorize(w)))
            if (v.compareTo(BigInteger.ONE) >= 0 && v.compareTo(cap) <= 0
                    && w.divide(v).compareTo(cap) <= 0) res.add(v);
        return new ArrayList<>(res);
    }

    /** Full prime factorization of {@code n} (small-prime peeling + Pollard's rho + Miller-Rabin). */
    private Map<BigInteger, Integer> factorize(BigInteger n) {
        Map<BigInteger, Integer> f = new TreeMap<>();
        factorInto(n, f);
        return f;
    }

    private void factorInto(BigInteger n, Map<BigInteger, Integer> f) {
        if (n.compareTo(BigInteger.ONE) <= 0) return;
        for (int sp : SMALL_PRIMES) {
            BigInteger p = BigInteger.valueOf(sp);
            while (n.mod(p).signum() == 0) { f.merge(p, 1, Integer::sum); n = n.divide(p); }
        }
        if (n.equals(BigInteger.ONE)) return;
        if (n.isProbablePrime(40)) { f.merge(n, 1, Integer::sum); return; }
        BigInteger d = findFactor(n);
        factorInto(d, f);
        factorInto(n.divide(d), f);
    }

    // ---- factoring engine: p-1  ->  ECM  ->  Pollard-rho (fallback) --------

    private static final BigInteger THREE = BigInteger.valueOf(3);

    /** A nontrivial factor of an odd composite {@code n}: p-1 pre-stage, then ECM, then rho. */
    private BigInteger findFactor(BigInteger n) {
        BigInteger d = pMinus1(n, 2000);                 // cheap: catches smooth p-1
        if (d != null) return d;
        if (n.bitLength() >= 50) {                       // ECM worthwhile only for larger composites
            d = ecm(n, 11000, 50);                       // simple stage-1 ECM
            if (d != null) return d;
        }
        return pollardRho(n);                            // guaranteed fallback (handles balanced factors)
    }

    /** Pollard's p-1, stage 1 to bound {@code B}: finds {@code p} when {@code p-1} is B-smooth. */
    private BigInteger pMinus1(BigInteger n, int B) {
        BigInteger a = BigInteger.TWO;
        for (int e = 2; e <= B; e++) {
            a = a.modPow(BigInteger.valueOf(e), n);
            if ((e & 0x7f) == 0) {
                BigInteger g = a.subtract(BigInteger.ONE).gcd(n);
                if (g.compareTo(BigInteger.ONE) > 0 && g.compareTo(n) < 0) return g;
            }
        }
        BigInteger g = a.subtract(BigInteger.ONE).gcd(n);
        return (g.compareTo(BigInteger.ONE) > 0 && g.compareTo(n) < 0) ? g : null;
    }

    /** Lenstra ECM, stage 1: up to {@code curves} random curves with smoothness bound {@code B1}. */
    private BigInteger ecm(BigInteger n, int B1, int curves) {
        int[] primes = primesUpTo(B1);
        for (int c = 0; c < curves; c++) {
            Pt P = new Pt(randBelow(n), randBelow(n), false);   // random point fixes an implicit curve
            BigInteger a = randBelow(n);
            try {
                for (int pr : primes) {
                    long pe = pr;
                    while (pe * (long) pr <= B1) pe *= pr;        // largest pr^k <= B1
                    P = mulPoint(BigInteger.valueOf(pe), P, a, n);
                    if (P.inf) break;
                }
            } catch (FactorFound ff) {
                if (ff.factor.compareTo(BigInteger.ONE) > 0 && ff.factor.compareTo(n) < 0) return ff.factor;
            }
        }
        return null;
    }

    private Pt mulPoint(BigInteger k, Pt P, BigInteger a, BigInteger n) {
        Pt r = Pt.INF, base = P;
        while (k.signum() > 0) {
            if (k.testBit(0)) r = addPoint(r, base, a, n);
            base = addPoint(base, base, a, n);
            k = k.shiftRight(1);
        }
        return r;
    }

    private Pt addPoint(Pt p, Pt q, BigInteger a, BigInteger n) {
        if (p.inf) return q;
        if (q.inf) return p;
        BigInteger num, den;
        if (p.x.equals(q.x)) {
            if (p.y.add(q.y).mod(n).signum() == 0) return Pt.INF;         // p == -q
            num = p.x.multiply(p.x).multiply(THREE).add(a).mod(n);        // doubling
            den = p.y.add(p.y).mod(n);
        } else {
            num = q.y.subtract(p.y).mod(n);                               // chord
            den = q.x.subtract(p.x).mod(n);
        }
        BigInteger inv = invOrFactor(den, n);                            // throws FactorFound on a gcd hit
        if (inv == null) return Pt.INF;
        BigInteger lam = num.multiply(inv).mod(n);
        BigInteger x3 = lam.multiply(lam).subtract(p.x).subtract(q.x).mod(n);
        BigInteger y3 = lam.multiply(p.x.subtract(x3)).subtract(p.y).mod(n);
        return new Pt(x3, y3, false);
    }

    /** Inverse of {@code d} mod {@code n}, or throw {@link FactorFound} if gcd(d,n) is a proper factor. */
    private BigInteger invOrFactor(BigInteger d, BigInteger n) {
        d = d.mod(n);
        BigInteger g = d.gcd(n);
        if (g.equals(BigInteger.ONE)) return d.modInverse(n);
        if (g.compareTo(n) < 0) throw new FactorFound(g);
        return null;                                                     // d == 0: point at infinity
    }

    private static int[] primesUpTo(int b) {
        boolean[] comp = new boolean[b + 1];
        List<Integer> ps = new ArrayList<>();
        for (int i = 2; i <= b; i++)
            if (!comp[i]) { ps.add(i); for (long j = (long) i * i; j <= b; j += i) comp[(int) j] = true; }
        return ps.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Affine Weierstrass point mod n (curve constant b implicit); {@code inf} = identity. */
    private static final class Pt {
        static final Pt INF = new Pt(null, null, true);
        final BigInteger x, y; final boolean inf;
        Pt(BigInteger x, BigInteger y, boolean inf) { this.x = x; this.y = y; this.inf = inf; }
    }

    private static final class FactorFound extends RuntimeException {
        final BigInteger factor;
        FactorFound(BigInteger f) { super(null, null, false, false); this.factor = f; }
    }

    /** A nontrivial factor of an odd composite {@code n} via Pollard's rho (Floyd, retry on failure). */
    private BigInteger pollardRho(BigInteger n) {
        while (true) {
            BigInteger c = randBelow(n.subtract(BigInteger.ONE)).add(BigInteger.ONE);
            BigInteger x = BigInteger.TWO, y = BigInteger.TWO, d = BigInteger.ONE;
            while (d.equals(BigInteger.ONE)) {
                x = x.multiply(x).add(c).mod(n);                 // tortoise: one step
                y = y.multiply(y).add(c).mod(n);                 // hare: two steps
                y = y.multiply(y).add(c).mod(n);
                d = x.subtract(y).abs().gcd(n);
            }
            if (!d.equals(n)) return d;                          // else retry with a fresh c
        }
    }

    private BigInteger randBelow(BigInteger bound) {
        BigInteger r;
        do { r = new BigInteger(bound.bitLength(), rnd); } while (r.compareTo(bound) >= 0);
        return r;
    }

    /** All positive divisors from a prime factorization. */
    private List<BigInteger> allDivisors(Map<BigInteger, Integer> f) {
        List<BigInteger> divs = new ArrayList<>();
        divs.add(BigInteger.ONE);
        for (Map.Entry<BigInteger, Integer> e : f.entrySet()) {
            int sz = divs.size();
            BigInteger pk = BigInteger.ONE;
            for (int k = 1; k <= e.getValue(); k++) {
                pk = pk.multiply(e.getKey());
                for (int i = 0; i < sz; i++) divs.add(divs.get(i).multiply(pk));
            }
        }
        return divs;
    }

    /** Set-bit positions of {@code v} (the rows {@code R}); {@code v < 2^N} so bits are in [0,N). */
    private int[] bits(BigInteger v) {
        List<Integer> b = new ArrayList<>();
        for (int j = 0; j < n; j++) if (v.testBit(j)) b.add(j);
        return b.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Columns {@code S} from the selector {@code M}: bit {@code j} set means column {@code N-1-j}. */
    private int[] colsFromM(BigInteger m) {
        List<Integer> cols = new ArrayList<>();
        for (int j = 0; j < n; j++) if (m.testBit(j)) cols.add(n - 1 - j);
        Collections.sort(cols);
        return cols.stream().mapToInt(Integer::intValue).toArray();
    }

    private boolean[][] gridOf(Set<Cell> pos) {
        boolean[][] g = new boolean[n][n];
        for (Cell c : pos) {
            if (c.r() < 0 || c.r() >= n || c.c() < 0 || c.c() >= n)
                throw new IllegalArgumentException("cell off board: " + c);
            g[c.r()][c.c()] = true;
        }
        return g;
    }

    // ---- end-to-end verification ------------------------------------------

    /** Replay duplicates then slides from {@code position}; null iff legal and the final board is final. */
    public String verify(Set<Cell> position, Solution sol) {
        boolean[][] g = gridOf(position);
        for (Move mv : sol.duplicates()) {
            if (mv.src().r() - mv.src().c() != mv.d()) return "dup src off diagonal: " + mv;
            if (!g[mv.src().r()][mv.src().c()]) return "dup src empty: " + mv;
            if (mv.a().equals(mv.b())) return "dup dest cells equal: " + mv;
            g[mv.src().r()][mv.src().c()] = false;
            if (g[mv.a().r()][mv.a().c()] || g[mv.b().r()][mv.b().c()]) return "dup dest occupied: " + mv;
            g[mv.a().r()][mv.a().c()] = true; g[mv.b().r()][mv.b().c()] = true;
        }
        for (Slide sl : sol.slides()) {
            if (sl.from().r() - sl.from().c() != sl.to().r() - sl.to().c()) return "slide changes diagonal: " + sl;
            if (!g[sl.from().r()][sl.from().c()]) return "slide src empty: " + sl;
            if (g[sl.to().r()][sl.to().c()]) return "slide dest occupied: " + sl;
            g[sl.from().r()][sl.from().c()] = false; g[sl.to().r()][sl.to().c()] = true;
        }
        return isFinal(g) ? null : "final board is not final (columns differ)";
    }

    /** All non-empty columns share the same pattern. */
    public boolean isFinal(boolean[][] g) {
        Set<Integer> ref = null;
        for (int c = 0; c < n; c++) {
            Set<Integer> col = new java.util.TreeSet<>();
            for (int r = 0; r < n; r++) if (g[r][c]) col.add(r);
            if (col.isEmpty()) continue;
            if (ref == null) ref = col; else if (!ref.equals(col)) return false;
        }
        return true;
    }

    // ---- demo / stress -----------------------------------------------------

    public static void main(String[] args) {
        // Factoring-engine self-check (exercises p-1 / ECM / rho on balanced & unbalanced inputs).
        Solver fe = new Solver(2);
        long[][] pairs = {
            {104729L, 104723L},          // balanced ~17-bit primes  -> rho
            {97L, 2147483647L},          // small factor (p-1 = 96 smooth) -> p-1
            {1048583L, 1099511627791L},  // ~21-bit x ~40-bit         -> ECM/rho
            {6700417L, 4294967291L},     // ~23-bit x ~32-bit         -> ECM/rho
        };
        for (long[] pr : pairs) {
            BigInteger prod = BigInteger.valueOf(pr[0]).multiply(BigInteger.valueOf(pr[1]));
            Map<BigInteger, Integer> f = fe.factorize(prod);
            BigInteger recon = BigInteger.ONE;
            for (Map.Entry<BigInteger, Integer> e : f.entrySet()) recon = recon.multiply(e.getKey().pow(e.getValue()));
            System.out.println("[factor] " + prod + " = " + f + (recon.equals(prod) ? "  ok" : "  WRONG"));
        }
        System.out.println();

        // Worked example: start = {(1,0),(2,1),(0,1)} on a 4x4 board (W'=36).
        int N = 4;
        Solver solver = new Solver(N);
        Set<Cell> P0 = new LinkedHashSet<>(List.of(new Cell(1, 0), new Cell(2, 1), new Cell(0, 1)));
        Solution sol = solver.solve(P0);
        System.out.println("[Solver] start=" + P0);
        System.out.println("  solved=" + sol.solved()
                + "  R=" + java.util.Arrays.toString(sol.rows())
                + "  S=" + java.util.Arrays.toString(sol.cols()));
        sol.duplicates().forEach(m -> System.out.println("    " + m));
        sol.slides().forEach(s -> System.out.println("    " + s));
        System.out.println("  verify=" + (solver.verify(P0, sol) == null ? "OK" : solver.verify(P0, sol)));

        // reconstruct() demo: recover (R,S) from a frozen rectangle profile.
        int[] prof = new StepA(N).rectangleProfile(new int[]{0, 1}, new int[]{0, 1});
        int[][] rs = solver.reconstruct(prof);
        System.out.println("[Solver] reconstruct(profile) -> R=" + java.util.Arrays.toString(rs[0])
                + " S=" + java.util.Arrays.toString(rs[1]));

        // End-to-end stress test: random rectangle, random reachable + scattered start.
        Random rng = new Random(2024);
        int trials = 3000, fails = 0, unsolved = 0;
        for (int t = 0; t < trials; t++) {
            int m = 2 + rng.nextInt(6);                       // board 2..7
            Solver s = new Solver(m);
            StepA g = new StepA(m);
            int[] rows = randomSubset(m, rng), cols = randomSubset(m, rng);
            int[] star = g.rectangleProfile(rows, cols);
            int[] startProfile = g.randomHigher(star, rng, 1 + rng.nextInt(20));
            Set<Cell> pos = scatter(startProfile, m, rng);
            Solution r = s.solve(pos);
            if (!r.solved()) { unsolved++; continue; }        // some factorizations may not be reachable
            String v = s.verify(pos, r);
            if (v != null) { fails++; if (fails <= 3) System.out.println("  FAIL N=" + m + ": " + v); }
        }
        System.out.println("[Solver] stress: " + (trials - fails) + "/" + trials + " verified ("
                + unsolved + " unsolved), failures=" + fails);
    }

    /** Place a profile at random (any) cells on each diagonal. */
    private static Set<Cell> scatter(int[] profile, int m, Random rng) {
        Set<Cell> pos = new LinkedHashSet<>();
        for (int d = -(m - 1); d <= m - 1; d++) {
            int cnt = profile[d + (m - 1)];
            List<Cell> cells = new ArrayList<>();
            for (int r = Math.max(0, d); r <= Math.min(m - 1, m - 1 + d); r++) cells.add(new Cell(r, r - d));
            Collections.shuffle(cells, rng);
            for (int i = 0; i < cnt; i++) pos.add(cells.get(i));
        }
        return pos;
    }

    private static int[] randomSubset(int m, Random rng) {
        List<Integer> xs = new ArrayList<>();
        for (int i = 0; i < m; i++) if (rng.nextBoolean()) xs.add(i);
        if (xs.isEmpty()) xs.add(rng.nextInt(m));
        return xs.stream().mapToInt(Integer::intValue).toArray();
    }
}
