package linalg

import (
	"reflect"
	"testing"
	"unsafe"
)

const almostEps = 1e-16

func TestNewDenseMatrix(t *testing.T) {
	n := 4
	m := 6
	// will panic if new dense accesses invalid memory
	mat := NewDenseMatrix(n, m, nil)
	for i := 0; i < n; i++ {
		col := mat[i]
		if len(col) != m {
			t.Errorf("matrix row %d was expected to have length %d, had length %d", i, len(col), m)
		}
	}
}

func TestSwapRows_Copyless(t *testing.T) {
	A := NewDenseMatrix(2, 3, []float64{
		0, 1, 2,
		3, 4, 5,
	})
	B := MatCopy(A)
	SwapRows(B, 0, 1, false)
	for i := 0; i < 3; i++ {
		if B[0][i] != float64(i+3) {
			t.Errorf("row 0 at position %d had value %.0f, expected %d", i, B[0][i], i+3)
		}
	}
}

func TestSwapRows_Copy(t *testing.T) {
	A := NewDenseMatrix(2, 3, []float64{
		0, 1, 2,
		3, 4, 5,
	})
	B := MatCopy(A)
	SwapRows(B, 0, 1, true)
	for i := 0; i < 3; i++ {
		if B[0][i] != float64(i+3) {
			t.Errorf("row 0 at position %d had value %.0f, expected %d", i, B[0][i], i+3)
		}
	}
}

func TestShape(t *testing.T) {
	A := NewDenseMatrix(3, 4, nil)
	m, n := Shape(A)
	if m != 3 || n != 4 {
		t.Errorf("matrix had shape (%d,%d), expected (3,4)", m, n)
	}
}

func TestEye(t *testing.T) {
	A := Eye(3)
	var expectation float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i == j {
				expectation = 1
			} else {
				expectation = 0
			}
			if A[i][j] != expectation {
				t.Errorf("at r,c (%d,%d) identity matrix had value %f, expected %f", i, j, A[i][j], expectation)
			}
		}
	}
}

func TestMatCopy(t *testing.T) {
	// MatCopy produces a copy of A with no overlapping memory, this tests that
	// the pointers truly do not overlap.  TestMatCopyTo verifies that the
	// elements are the same.  This is fragile, in that we assume they share
	// an implementation.  _shrug_
	A := NewDenseMatrix(4, 4, nil)
	B := MatCopy(A)
	for i := 0; i < len(A); i++ {
		dataptrA := (*reflect.SliceHeader)(unsafe.Pointer(&A)).Data
		dataptrB := (*reflect.SliceHeader)(unsafe.Pointer(&B)).Data
		if dataptrA == dataptrB {
			t.Errorf("row %d had same data pointer, expected to be different", i)
		}
	}
}

func TestMatCopyTo(t *testing.T) {
	A := NewDenseMatrix(4, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16})
	B := NewDenseMatrix(4, 4, nil)
	MatCopyTo(A, B)
	m, n := Shape(A)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a := A[i][j]
			b := B[i][j]
			if a != b {
				t.Errorf("at (%d,%d), expected A==B, got %f and %f", i, j, a, b)
			}
		}
	}
}

func TestMatL2Norm(t *testing.T) {
	data := []float64{
		1, 2, 3,
		4, 5, 6}
	A := NewDenseMatrix(2, 3, data)
	var expectation float64
	for _, v := range data {
		expectation += v * v
	}
	norm := MatNormL2(A)
	if !almostEqual(expectation, norm, almostEps) {
		t.Errorf("expected %f, got %f", expectation, norm)
	}
}

func TestMatTranspose(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6})
	exp := NewDenseMatrix(3, 2, []float64{
		1, 4,
		2, 5,
		3, 6})
	inpT := MatTranspose(inp, nil)
	t.Log(exp)
	t.Log(inpT)
	if !matrixEqual(exp, inpT) {
		t.Error("matrix transpose did not match expectation")
	}
}

func TestMatAdd(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	sub := NewDenseMatrix(2, 3, []float64{
		2, 3, 4,
		5, 6, 7,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		3, 5, 7,
		9, 11, 13,
	})
	out := MatAdd(inp, sub, nil)
	if !matrixEqual(exp, out) {
		t.Error("matrix sub did not match expectation")
	}
}

func TestMatSub(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	sub := NewDenseMatrix(2, 3, []float64{
		2, 3, 4,
		5, 6, 7,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		-1, -1, -1,
		-1, -1, -1,
	})
	out := MatSub(inp, sub, nil)
	if !matrixEqual(exp, out) {
		t.Error("matrix sub did not match expectation")
	}
}

func TestMatSwapRows(t *testing.T) {
	inp1 := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	inp2 := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		4, 5, 6,
		1, 2, 3,
	})
	SwapRows(inp1, 0, 1, false)
	SwapRows(inp2, 0, 1, true)
	if !matrixEqual(exp, inp1) {
		t.Error("matrix swap rows non copying did not match expectation")
	}
	if !matrixEqual(exp, inp2) {
		t.Error("matrix swap rows copying did not match expectation")
	}
}

func TestMatSwapCols(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		2, 1, 3,
		5, 4, 6,
	})
	SwapCols(inp, 0, 1)
	if !matrixEqual(exp, inp) {
		t.Error("matrix swap cols did not match expectation")
	}
}

func TestSquareMatrixInvert(t *testing.T) {
	inp := NewDenseMatrix(2, 2, []float64{
		1, 2,
		3, 4,
	})
	exp := NewDenseMatrix(2, 2, []float64{
		-2, 1,
		1.5, -0.5,
	})
	inv := MatrixInvertSquare(inp, nil, nil)
	t.Log(inv)
	if !matrixEqual(exp, inv) {
		t.Error("matrix inversion did not match expectation")
	}
}
