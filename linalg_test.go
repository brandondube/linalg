package linalg

import "math"

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol

}

func matrixEqualTol(A, B [][]float64, tol float64) bool {
	m, n := Shape(A)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if !almostEqual(A[i][j], B[i][j], tol) {
				return false
			}
		}
	}
	return true
}

func matrixEqual(A, B [][]float64) bool {
	return matrixEqualTol(A, B, 0)
}
