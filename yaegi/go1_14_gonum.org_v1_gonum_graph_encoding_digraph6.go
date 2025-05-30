// Code generated by 'goexports gonum.org/v1/gonum/graph/encoding/digraph6'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/graph/encoding/digraph6"
)

func init() {
	Symbols["gonum.org/v1/gonum/graph/encoding/digraph6"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Encode":  reflect.ValueOf(digraph6.Encode),
		"IsValid": reflect.ValueOf(digraph6.IsValid),

		// type definitions
		"Graph": reflect.ValueOf((*digraph6.Graph)(nil)),
	}
}
