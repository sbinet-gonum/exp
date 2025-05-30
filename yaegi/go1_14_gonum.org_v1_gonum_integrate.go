// Code generated by 'goexports gonum.org/v1/gonum/integrate'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/integrate"
)

func init() {
	Symbols["gonum.org/v1/gonum/integrate"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Romberg":     reflect.ValueOf(integrate.Romberg),
		"Simpsons":    reflect.ValueOf(integrate.Simpsons),
		"Trapezoidal": reflect.ValueOf(integrate.Trapezoidal),
	}
}
