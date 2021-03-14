package linalg

import "math"

// MatVecProd computes the matrix-vector product Ax for (nxm) matrix A and (1xm)
// vector B, storing it in out
func MatVecProd(A [][]float64, x []float64, out []float64) []float64 {
	n := len(x)
	m := len(A)
	if out == nil {
		out = make([]float64, m)
	}
	for i := 0; i < m; i++ {
		out[i] = 0
		for j := 0; j < n; j++ {
			out[i] += A[i][j] * x[j]
		}
	}
	return out
}

// VectorAdd computes the elementwise sum a+b and stores the result in c
func VectorAdd(a []float64, b []float64, c []float64) []float64 {
	n := len(a)
	if c == nil {
		c = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		c[i] = a[i] + b[i]
	}
	return c
}

// VectorSub computes the elementwise difference a-b and stores the result in c
func VectorSub(a []float64, b []float64, c []float64) []float64 {
	n := len(a)
	if c == nil {
		c = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		c[i] = a[i] - b[i]
	}
	return c
}

// VectorArgmax returns the index of the largest element of a
func VectorArgmax(a []float64) int {
	max := math.Inf(-1)
	var imax int
	for i := 0; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
			imax = i
		}
	}
	return imax
}
