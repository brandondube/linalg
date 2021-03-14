package linalg

import (
	"reflect"
	"testing"
	"unsafe"
)

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
	A := NewDenseMatrix(4, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
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
