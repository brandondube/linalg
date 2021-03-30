package linalg

import "testing"

// MatVecProd
// VectorAdd
// VectorSub
// VectorArgmax

func TestVectorAdd(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{5, 4, 3, 2, 1}
	exp := []float64{6, 6, 6, 6, 6}
	res := VectorAdd(a, b, nil)
	if !vectorEqualTol(exp, res, 1e-16) {
		t.Error("vector addition failed to produce the expected result")
	}
}

func TestVectorSub(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{5, 4, 3, 2, 1}
	exp := []float64{-4, -2, 0, 2, 4}
	res := VectorSub(a, b, nil)
	if !vectorEqualTol(exp, res, 1e-16) {
		t.Error("vector addition failed to produce the expected result")
	}
}

func TestVectorArgmax(t *testing.T) {
	// 100 is in slot 5
	a := []float64{0, 1, 2, 3, 4, 100, 6, 7, 8, 9, 10}
	exp := 5
	res := VectorArgmax(a)
	if res != exp {
		t.Errorf("vector argmax found max element in slot %d, expected %d", res, exp)
	}
}

func TestMatVecProd(t *testing.T) {
	A := NewDenseMatrix(2, 2, []float64{
		0, 1,
		2, 3,
	})
	b := []float64{9, 8}
	exp := []float64{8, 42}
	res := MatVecProd(A, b, nil)
	if !vectorEqual(exp, res) {
		t.Log("expected", exp)
		t.Log("got", res)
		t.Error("matrix-vector product gave incorrect result")

	}
}
