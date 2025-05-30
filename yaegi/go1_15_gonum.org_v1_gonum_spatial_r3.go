// Code generated by 'github.com/containous/yaegi/extract gonum.org/v1/gonum/spatial/r3'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15 && !go1.16
// +build go1.15,!go1.16

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/spatial/r3"
)

func init() {
	Symbols["gonum.org/v1/gonum/spatial/r3"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Cos":   reflect.ValueOf(r3.Cos),
		"Norm":  reflect.ValueOf(r3.Norm),
		"Norm2": reflect.ValueOf(r3.Norm2),
		"Unit":  reflect.ValueOf(r3.Unit),

		// type definitions
		"Box": reflect.ValueOf((*r3.Box)(nil)),
		"Vec": reflect.ValueOf((*r3.Vec)(nil)),
	}
}
