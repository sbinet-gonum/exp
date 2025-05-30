// Code generated by 'github.com/containous/yaegi/extract gonum.org/v1/gonum/blas/blas64'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15 && !go1.16
// +build go1.15,!go1.16

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/blas/blas64"
)

func init() {
	Symbols["gonum.org/v1/gonum/blas/blas64"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Asum":           reflect.ValueOf(blas64.Asum),
		"Axpy":           reflect.ValueOf(blas64.Axpy),
		"Copy":           reflect.ValueOf(blas64.Copy),
		"Dot":            reflect.ValueOf(blas64.Dot),
		"Gbmv":           reflect.ValueOf(blas64.Gbmv),
		"Gemm":           reflect.ValueOf(blas64.Gemm),
		"Gemv":           reflect.ValueOf(blas64.Gemv),
		"Ger":            reflect.ValueOf(blas64.Ger),
		"Iamax":          reflect.ValueOf(blas64.Iamax),
		"Implementation": reflect.ValueOf(blas64.Implementation),
		"Nrm2":           reflect.ValueOf(blas64.Nrm2),
		"Rot":            reflect.ValueOf(blas64.Rot),
		"Rotg":           reflect.ValueOf(blas64.Rotg),
		"Rotm":           reflect.ValueOf(blas64.Rotm),
		"Rotmg":          reflect.ValueOf(blas64.Rotmg),
		"Sbmv":           reflect.ValueOf(blas64.Sbmv),
		"Scal":           reflect.ValueOf(blas64.Scal),
		"Spmv":           reflect.ValueOf(blas64.Spmv),
		"Spr":            reflect.ValueOf(blas64.Spr),
		"Spr2":           reflect.ValueOf(blas64.Spr2),
		"Swap":           reflect.ValueOf(blas64.Swap),
		"Symm":           reflect.ValueOf(blas64.Symm),
		"Symv":           reflect.ValueOf(blas64.Symv),
		"Syr":            reflect.ValueOf(blas64.Syr),
		"Syr2":           reflect.ValueOf(blas64.Syr2),
		"Syr2k":          reflect.ValueOf(blas64.Syr2k),
		"Syrk":           reflect.ValueOf(blas64.Syrk),
		"Tbmv":           reflect.ValueOf(blas64.Tbmv),
		"Tbsv":           reflect.ValueOf(blas64.Tbsv),
		"Tpmv":           reflect.ValueOf(blas64.Tpmv),
		"Tpsv":           reflect.ValueOf(blas64.Tpsv),
		"Trmm":           reflect.ValueOf(blas64.Trmm),
		"Trmv":           reflect.ValueOf(blas64.Trmv),
		"Trsm":           reflect.ValueOf(blas64.Trsm),
		"Trsv":           reflect.ValueOf(blas64.Trsv),
		"Use":            reflect.ValueOf(blas64.Use),

		// type definitions
		"Band":               reflect.ValueOf((*blas64.Band)(nil)),
		"BandCols":           reflect.ValueOf((*blas64.BandCols)(nil)),
		"General":            reflect.ValueOf((*blas64.General)(nil)),
		"GeneralCols":        reflect.ValueOf((*blas64.GeneralCols)(nil)),
		"Symmetric":          reflect.ValueOf((*blas64.Symmetric)(nil)),
		"SymmetricBand":      reflect.ValueOf((*blas64.SymmetricBand)(nil)),
		"SymmetricBandCols":  reflect.ValueOf((*blas64.SymmetricBandCols)(nil)),
		"SymmetricCols":      reflect.ValueOf((*blas64.SymmetricCols)(nil)),
		"SymmetricPacked":    reflect.ValueOf((*blas64.SymmetricPacked)(nil)),
		"Triangular":         reflect.ValueOf((*blas64.Triangular)(nil)),
		"TriangularBand":     reflect.ValueOf((*blas64.TriangularBand)(nil)),
		"TriangularBandCols": reflect.ValueOf((*blas64.TriangularBandCols)(nil)),
		"TriangularCols":     reflect.ValueOf((*blas64.TriangularCols)(nil)),
		"TriangularPacked":   reflect.ValueOf((*blas64.TriangularPacked)(nil)),
		"Vector":             reflect.ValueOf((*blas64.Vector)(nil)),
	}
}
