// Code generated by 'goexports gonum.org/v1/gonum/stat/card'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/stat/card"
)

func init() {
	Symbols["gonum.org/v1/gonum/stat/card"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"NewHyperLogLog32": reflect.ValueOf(card.NewHyperLogLog32),
		"NewHyperLogLog64": reflect.ValueOf(card.NewHyperLogLog64),
		"RegisterHash":     reflect.ValueOf(card.RegisterHash),

		// type definitions
		"HyperLogLog32": reflect.ValueOf((*card.HyperLogLog32)(nil)),
		"HyperLogLog64": reflect.ValueOf((*card.HyperLogLog64)(nil)),
	}
}
