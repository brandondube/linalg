package linalg

// NewDenseMatrix produces a new (rxc) matrix backed by contiguous data.
// this function produces superior memory access patterns and prevents the rows
// of the output from being scattered in memory.
//
// data may be nil, in which case an array of zeros is returned
func NewDenseMatrix(r, c int, data []float64) [][]float64 {
	if data == nil {
		data = make([]float64, r*c)
	}
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		// c is length of a row, offset by i rows and index up to the next one
		row := data[i*c : (i+1)*c]
		out[i] = row
	}
	return out
}

// MatCopy produces a copy of A with no overlapping memory
func MatCopy(A [][]float64) [][]float64 {
	m, n := Shape(A)
	out := NewDenseMatrix(m, n, nil)
	MatCopyTo(A, out)
	return out
}

// MatCopyTo copies all elements of A into B
func MatCopyTo(A, B [][]float64) {
	m, n := Shape(A)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			B[i][j] = A[i][j]
		}
	}
	return
}

// Eye is the identity matrix of size N
func Eye(n int) [][]float64 {
	out := NewDenseMatrix(n, n, nil)
	for i := 0; i < n; i++ {
		out[i][i] = 1
	}
	return out
}

// Shape is the number of rows, cols in A
func Shape(A [][]float64) (int, int) {
	return len(A), len(A[0])
}

// MatNormL2 computes the "Euclidian Norm" or the ||2 norm of a matrix.  It is
// more properly known as the Frobenius norm.
func MatNormL2(A [][]float64) float64 {
	m, n := Shape(A)
	var out float64
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out += A[i][j] * A[i][j]
		}
	}
	// abs(A)^2 == A*A, 2 is an even power
	return out
}

// MatTranspose transposes (nxp) matrix input and stores the result in (pxn) matrix out
func MatTranspose(A [][]float64, out [][]float64) [][]float64 {
	m, n := Shape(A)
	if out == nil {
		out = NewDenseMatrix(n, m, nil)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[j][i] = A[i][j]
		}
	}
	return out
}

// MatInvertSquare inverts square matrix input, storing the result in out
// scratch has the same nil rules as out, but must be n x 2n instead of n x n
//
// Gauss-Jordan elimination is used to perform the inversion, A must be non-singular.
func MatrixInvertSquare(A [][]float64, scratch [][]float64, out [][]float64) [][]float64 {
	n := len(A)
	var n2 = 2 * n
	if out == nil {
		out = NewDenseMatrix(n, n, nil)
	}
	if scratch == nil {
		scratch = NewDenseMatrix(n, n2, nil)
	}

	// make scratch into the augmenting identity matrix
	for i := 0; i < n; i++ {
		for j := 0; j < n2; j++ {
			if j < n {
				scratch[i][j] = A[i][j]
			}
			if j == (i + n) {
				scratch[i][j] = 1
			} else {
				// would not need this if we alloced (guaranteed zero)
				// but to be safe, zero here
				scratch[i][j] = 0
			}
		}
	}
	// exchange rows of the matrix, bottom-up
	for i := n - 1; i > 0; i-- {
		if scratch[i-1][0] < scratch[i][0] {
			SwapRows(scratch, i, i-1, false)
		}
	}

	// replace each row by sum of itself and a constant times another row
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				tmp := scratch[j][i] / scratch[i][i]
				for k := 0; k < n2; k++ {
					scratch[j][k] -= scratch[i][k] * tmp
				}
			}
		}
	}

	// mul each row by a nonzero integer and divide each row by the diagonal
	for i := 0; i < n; i++ {
		tmp := scratch[i][i]
		for j := 0; j < n2; j++ {
			scratch[i][j] = scratch[i][j] / tmp
		}
	}
	// scratch now contains the inverse of input in its lefthand half
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i][j] = scratch[i][j]
		}
	}
	// it could be faster to use copy here, should bench.
	return out
}

// MatMul computes the matrix-matrix product C = AB for (nxm) matrix A and (mxp)
// matrix B, storing the result in (nxp) matrix C.
func MatMul(A [][]float64, B [][]float64, C [][]float64) [][]float64 {
	n := len(A)
	m := len(A[0])
	p := len(B[0])
	if C == nil {
		C = NewDenseMatrix(n, p, nil)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			C[i][j] = 0
			for k := 0; k < m; k++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

// MatAdd produces the elementwise sum of A and B and stores it in C.
func MatAdd(A [][]float64, B [][]float64, C [][]float64) [][]float64 {
	m, n := Shape(A)
	if C == nil {
		C = NewDenseMatrix(m, n, nil)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			C[i][j] = A[i][j] + B[i][j]
		}
	}
	return C
}

// MatSub produces the elementwise difference A-B and stores it in C.
func MatSub(A [][]float64, B [][]float64, C [][]float64) [][]float64 {
	m, n := Shape(A)
	if C == nil {
		C = NewDenseMatrix(m, n, nil)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			C[i][j] = A[i][j] - B[i][j]
		}
	}
	return C
}

// SwapRows swaps rows i and j of A in-place.
//
// If copy is false, this function causes minor "memory fragmentation" --
// the swap will be done by pointer juggling instead of by copying data.
func SwapRows(A [][]float64, i, j int, copy bool) {
	if !copy {
		A[i], A[j] = A[j], A[i]
	} else {
		// k = ncols, slightly faster than calling Shape
		for k := 0; k < len(A[0]); k++ {
			A[i][k], A[j][k] = A[j][k], A[i][k]
		}
	}
	return
}

// SwapCols swaps columns i and j of A in-place.
func SwapCols(A [][]float64, i, j int) {
	for k := 0; k < len(A); k++ {
		A[k][i], A[k][j] = A[k][j], A[k][i]
	}
	return
}
