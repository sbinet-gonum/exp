// Code generated by 'github.com/containous/yaegi/extract gonum.org/v1/gonum/spatial/barneshut'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15 && !go1.16
// +build go1.15,!go1.16

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/spatial/barneshut"
	"gonum.org/v1/gonum/spatial/r2"
	"gonum.org/v1/gonum/spatial/r3"
)

func init() {
	Symbols["gonum.org/v1/gonum/spatial/barneshut"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"Gravity2":  reflect.ValueOf(barneshut.Gravity2),
		"Gravity3":  reflect.ValueOf(barneshut.Gravity3),
		"NewPlane":  reflect.ValueOf(barneshut.NewPlane),
		"NewVolume": reflect.ValueOf(barneshut.NewVolume),

		// type definitions
		"Force2":    reflect.ValueOf((*barneshut.Force2)(nil)),
		"Force3":    reflect.ValueOf((*barneshut.Force3)(nil)),
		"Particle2": reflect.ValueOf((*barneshut.Particle2)(nil)),
		"Particle3": reflect.ValueOf((*barneshut.Particle3)(nil)),
		"Plane":     reflect.ValueOf((*barneshut.Plane)(nil)),
		"Volume":    reflect.ValueOf((*barneshut.Volume)(nil)),

		// interface wrapper definitions
		"_Particle2": reflect.ValueOf((*_gonum_org_v1_gonum_spatial_barneshut_Particle2)(nil)),
		"_Particle3": reflect.ValueOf((*_gonum_org_v1_gonum_spatial_barneshut_Particle3)(nil)),
	}
}

// _gonum_org_v1_gonum_spatial_barneshut_Particle2 is an interface wrapper for Particle2 type
type _gonum_org_v1_gonum_spatial_barneshut_Particle2 struct {
	WCoord2 func() r2.Vec
	WMass   func() float64
}

func (W _gonum_org_v1_gonum_spatial_barneshut_Particle2) Coord2() r2.Vec { return W.WCoord2() }
func (W _gonum_org_v1_gonum_spatial_barneshut_Particle2) Mass() float64  { return W.WMass() }

// _gonum_org_v1_gonum_spatial_barneshut_Particle3 is an interface wrapper for Particle3 type
type _gonum_org_v1_gonum_spatial_barneshut_Particle3 struct {
	WCoord3 func() r3.Vec
	WMass   func() float64
}

func (W _gonum_org_v1_gonum_spatial_barneshut_Particle3) Coord3() r3.Vec { return W.WCoord3() }
func (W _gonum_org_v1_gonum_spatial_barneshut_Particle3) Mass() float64  { return W.WMass() }
