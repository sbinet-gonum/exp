// Code generated by 'github.com/containous/yaegi/extract gonum.org/v1/gonum/lapack/lapack64'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15 && !go1.16
// +build go1.15,!go1.16

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/lapack/lapack64"
)

func init() {
	Symbols["gonum.org/v1/gonum/lapack/lapack64"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Gecon":  reflect.ValueOf(lapack64.Gecon),
		"Geev":   reflect.ValueOf(lapack64.Geev),
		"Gelqf":  reflect.ValueOf(lapack64.Gelqf),
		"Gels":   reflect.ValueOf(lapack64.Gels),
		"Geqrf":  reflect.ValueOf(lapack64.Geqrf),
		"Gesvd":  reflect.ValueOf(lapack64.Gesvd),
		"Getrf":  reflect.ValueOf(lapack64.Getrf),
		"Getri":  reflect.ValueOf(lapack64.Getri),
		"Getrs":  reflect.ValueOf(lapack64.Getrs),
		"Ggsvd3": reflect.ValueOf(lapack64.Ggsvd3),
		"Lange":  reflect.ValueOf(lapack64.Lange),
		"Lansy":  reflect.ValueOf(lapack64.Lansy),
		"Lantr":  reflect.ValueOf(lapack64.Lantr),
		"Lapmt":  reflect.ValueOf(lapack64.Lapmt),
		"Ormlq":  reflect.ValueOf(lapack64.Ormlq),
		"Ormqr":  reflect.ValueOf(lapack64.Ormqr),
		"Pbtrf":  reflect.ValueOf(lapack64.Pbtrf),
		"Pbtrs":  reflect.ValueOf(lapack64.Pbtrs),
		"Pocon":  reflect.ValueOf(lapack64.Pocon),
		"Potrf":  reflect.ValueOf(lapack64.Potrf),
		"Potri":  reflect.ValueOf(lapack64.Potri),
		"Potrs":  reflect.ValueOf(lapack64.Potrs),
		"Syev":   reflect.ValueOf(lapack64.Syev),
		"Trcon":  reflect.ValueOf(lapack64.Trcon),
		"Trtri":  reflect.ValueOf(lapack64.Trtri),
		"Trtrs":  reflect.ValueOf(lapack64.Trtrs),
		"Use":    reflect.ValueOf(lapack64.Use),
	}
}
