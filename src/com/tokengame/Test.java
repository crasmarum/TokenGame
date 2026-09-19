package com.tokengame;

import java.math.BigInteger;
import java.util.Random;

/**
 * Greatest common divisor (Euclid's algorithm) plus a small self-test.
 *
 * <p>Results are non-negative; inputs may be negative; {@code gcd(0,0)=0}.
 * (The only value the {@code long} version cannot represent is {@code gcd(Long.MIN_VALUE, 0) = 2^63},
 * which has no positive {@code long}; use the {@link BigInteger} overload there.)
 */
public final class Test {

    private Test() {}

    /** GCD of two longs via the iterative Euclidean algorithm. */
    public static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;   // Java's % takes the sign of the dividend; Euclid still converges
            a = b;
            b = t;
        }
        return a < 0 ? -a : a;
    }

    /** GCD of two BigIntegers via the iterative Euclidean algorithm. */
    public static BigInteger gcd(BigInteger a, BigInteger b) {
        BigInteger x = a.abs(), y = b.abs();
        while (y.signum() != 0) {
            BigInteger t = x.mod(y);   // y > 0 here, so mod is well-defined
            x = y;
            y = t;
        }
        return x;
    }
    
    /**
     * GCD of two polynomials over GF(2): each BigInteger is read as a bit-vector of
     * coefficients (bit {@code i} = coefficient of {@code x^i}), so arithmetic is
     * carry-less -- addition/subtraction is XOR. The returned BigInteger's bits are
     * the coefficients of the gcd polynomial. {@code gcdPoly(0,0) = 0}.
     */
    public static BigInteger gcdPoly(BigInteger a, BigInteger b) {
        a = a.abs();   // only the bit pattern matters
        b = b.abs();
        while (b.signum() != 0) {
            BigInteger t = polyModGF2(a, b);
            a = b;
            b = t;
        }
        return a;
    }

    /** Remainder of {@code a} divided by {@code b} in GF(2)[x] ({@code b != 0}), by XOR-and-shift. */
    private static BigInteger polyModGF2(BigInteger a, BigInteger b) {
        int degB = b.bitLength() - 1;
        for (int degA = a.bitLength() - 1; degA >= degB; degA = a.bitLength() - 1) {
            a = a.xor(b.shiftLeft(degA - degB));   // kill a's leading term with x^(degA-degB)*b
        }
        return a;
    }

    public static void main(String[] args) {
    	Random rand = new Random();
		int N = 16;
		
		BigInteger big1 = BigInteger.probablePrime(N , rand);
		BigInteger big2 = BigInteger.probablePrime(N, rand);
		BigInteger prod = big1.multiply(big2);
		
		BigInteger W = (BigInteger.TWO.pow(N)).subtract(BigInteger.ONE);
		BigInteger W2 = W.multiply(W);
		
		BigInteger first = prod.mod(W);
		BigInteger second = (W2.subtract(prod)).mod(W);
		
		System.out.println(gcdPoly(first, second));
		
    }

}
