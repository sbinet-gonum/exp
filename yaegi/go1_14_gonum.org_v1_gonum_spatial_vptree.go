// Code generated by 'goexports gonum.org/v1/gonum/spatial/vptree'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/spatial/vptree"
)

func init() {
	Symbols["gonum.org/v1/gonum/spatial/vptree"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"New":           reflect.ValueOf(vptree.New),
		"NewDistKeeper": reflect.ValueOf(vptree.NewDistKeeper),
		"NewNKeeper":    reflect.ValueOf(vptree.NewNKeeper),

		// type definitions
		"Comparable":     reflect.ValueOf((*vptree.Comparable)(nil)),
		"ComparableDist": reflect.ValueOf((*vptree.ComparableDist)(nil)),
		"DistKeeper":     reflect.ValueOf((*vptree.DistKeeper)(nil)),
		"Heap":           reflect.ValueOf((*vptree.Heap)(nil)),
		"Keeper":         reflect.ValueOf((*vptree.Keeper)(nil)),
		"NKeeper":        reflect.ValueOf((*vptree.NKeeper)(nil)),
		"Node":           reflect.ValueOf((*vptree.Node)(nil)),
		"Operation":      reflect.ValueOf((*vptree.Operation)(nil)),
		"Point":          reflect.ValueOf((*vptree.Point)(nil)),
		"Tree":           reflect.ValueOf((*vptree.Tree)(nil)),

		// interface wrapper definitions
		"_Comparable": reflect.ValueOf((*_gonum_org_v1_gonum_spatial_vptree_Comparable)(nil)),
		"_Keeper":     reflect.ValueOf((*_gonum_org_v1_gonum_spatial_vptree_Keeper)(nil)),
	}
}

// _gonum_org_v1_gonum_spatial_vptree_Comparable is an interface wrapper for Comparable type
type _gonum_org_v1_gonum_spatial_vptree_Comparable struct {
	WDistance func(a0 vptree.Comparable) float64
}

func (W _gonum_org_v1_gonum_spatial_vptree_Comparable) Distance(a0 vptree.Comparable) float64 {
	return W.WDistance(a0)
}

// _gonum_org_v1_gonum_spatial_vptree_Keeper is an interface wrapper for Keeper type
type _gonum_org_v1_gonum_spatial_vptree_Keeper struct {
	WKeep func(a0 vptree.ComparableDist)
	WLen  func() int
	WLess func(i int, j int) bool
	WMax  func() vptree.ComparableDist
	WPop  func() interface{}
	WPush func(x interface{})
	WSwap func(i int, j int)
}

func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Keep(a0 vptree.ComparableDist) { W.WKeep(a0) }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Len() int                      { return W.WLen() }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Less(i int, j int) bool        { return W.WLess(i, j) }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Max() vptree.ComparableDist    { return W.WMax() }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Pop() interface{}              { return W.WPop() }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Push(x interface{})            { W.WPush(x) }
func (W _gonum_org_v1_gonum_spatial_vptree_Keeper) Swap(i int, j int)             { W.WSwap(i, j) }
