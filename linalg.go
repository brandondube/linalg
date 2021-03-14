// Package linalg is a pure-Go implementation of basic linear algebra operations.
//
// It has no facilities for higher order tensor algebra, only 1D vectors and 2D
// matrices.  It includes algorithms only for doubles.
//
// This is not a competitor to Eigen, Boost, or other C++ libraries and will
// perform somewhat worse.  It will be substantially faster for small matrices
// and vectors.
//
// This is also not a competitor to Gonum, and exists to be usable in places
// where Gonum is not possible to use, or is uneconomic (small platforms/cross-compilation).
//
// The user is, in general, able to eliminate allocation for all algorithms by
// passing in output or scratch buffers.  The user is also able to control the
// level of data locality between rows of a matrix.
//
// The algorithms are not necessarily state of the art.  This library was written
// with real-time operations on small data in mind, as for state-space control,
// Kalman filtering, and so forth.
//
// This library defines no types and operates only on []float64 and [][]float64
// for user convenience.
//
//
//
// ## out and scratch parameters
// Several functions take out and/or scratch as parameters.  These are to allow
// the user to avoid allocation if the desire.  nil means "allocate and return"
// non-nil means "populate this buffer and return."  The call
//
// var (
//   a [][]float64
//   b [][]float64
//   c [][]float64
// c = linalg.MatMul(a,b)
//
// and
//
// linalg.MatMul(a,b,c)
//
// do the same thing, storing the matrix-matrix multiplication of a,b in c.
// the former case does not require a predefinition of C, the latter requires
// C to be pre-allocated
//
// when the buffer must be zerored before use, this is documented in the function
// docstring
package linalg

// JacobiSVD computes the Jacobi SVD of A, returning U, Sigma, V^T
// func JacobiSVD(A [][]float64) ([][]float64, [][]float64, [][]float64) {
// 	const tol = 1e-14
// 	var rots int = 1
// 	nrow, ncol := Shape(A)
// 	sigma := make([]float64, ncol)
// 	// populate sigma with the columnwise squared sum of A
// 	for i := 0; i < nrow; i++ {
// 		for j := 0; j < ncol; j++ {
// 			sigma[j] += A[i][j] * A[i][j] // A*A = A^2
// 		}
// 	}
// 	// populate V as the identity matrix
// 	V := Eye(nrow)
// 	var i, j, iters int
// 	tolsigma = tol * MatNormL2(A)
// 	for rots >= 1 {
// 		i++
// 		rots = 0
// 		for p := 0; p < ncol; p++ {
// 			k := VectorArgmax(sigma[p:ncol])
// 			k = k + p - 1
// 			if k != p {
// 				sigma[k], sigma[p] = sigma[p], sigma[k]
// 				// want to express the matlab syntax A(:, [k, p]) = A(:, [p, k])
// 				// this swaps two columns of A
// 				// do the same in V

// 				// https://github.com/zlliang/jacobi-svd/blob/master/jacobi_svd.m
// 			}
// 		}
// 	}
// }

// NashSVD is based on the procedure of the same name from the book
// "Compact Numerical Methods", 2nd ed, John C. Nash
// func NashSVD(A [][]float64) [][]float64 {

// }
