// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rings

import (
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/plot/vg"
)

// LengthDist generates a random value in the range [Length*Min, Length*Max), depending on a
// provided random factor.
type LengthDist struct {
	Length   vg.Length
	Min, Max *float64 // A nil value is interpreted as 1.
}

// Perturb returns a perturbed vg.Length value. Calling Perturb on a nil LengthDist will panic.
func (p *LengthDist) Perturb(f float64) vg.Length {
	if p.Min == nil && p.Max == nil {
		return p.Length
	}
	var min, max = 1., 1.
	if p.Min != nil {
		min = *p.Min
	}
	if p.Max != nil {
		max = *p.Max
	}
	return p.Length * vg.Length(min+(max-min)*f)
}

// FactorDist generates a random value in the range [Length*Min, Length*Max), depending on a
// provided random factor.
type FactorDist struct {
	Factor   float64
	Min, Max *float64 // A nil value is interpreted as 1.
}

// Perturb returns a perturbed float value. Calling Perturb on a nil FactorDist will panic.
func (p *FactorDist) Perturb(f float64) float64 {
	if p.Min == nil && p.Max == nil {
		return p.Factor
	}
	var min, max = 1., 1.
	if p.Min != nil {
		min = *p.Min
	}
	if p.Max != nil {
		max = *p.Max
	}
	return p.Factor * (min + (max-min)*f)
}

// Bezier defines Bézier control points for a link between features represented by Links and Ribbons.
type Bezier struct {
	// Segments defines the number of segments to draw when rendering the curve.
	Segments int

	// Radius, Crest and Purity define aspects of Bézier geometry.
	//
	// See http://circos.ca/documentation/tutorials/links/geometry/images for a detailed explanation
	// of radius, crest and purity.
	//
	// Radius specifies the Bézier radius of a curve generated by the Bezier.
	Radius LengthDist
	// Crest and Purity specify the crest and purity behaviour of a curve generated by the Bezier.
	// If nil, these values are not used.
	Crest  *FactorDist
	Purity *FactorDist
}

// ControlPoints returns a set of Bézier curve control points defining the path between the points defined
// by the parameters and the Bezier's Radius, Crest and Purity fields.
func (b *Bezier) ControlPoints(a [2]Angle, rad [2]vg.Length) []vg.Point {
	var p [2]vg.Point
	for i := range a {
		p[i] = Rectangular(a[i], rad[i])
	}

	var radius = b.Radius
	if b.Purity != nil {
		bisectRadius := vg.Length(math.Hypot(float64(p[0].X+p[1].X)/2, float64(p[0].Y+p[1].Y)/2))
		radius.Length += vg.Length(b.Purity.Perturb(rand.Float64())-1) * (radius.Length - bisectRadius)
	}

	var bisect Angle
	if math.Abs(float64(a[1]-a[0])) > math.Pi {
		bisect = (a[0]+a[1]+Angle(2*math.Pi))/2 - Angle(2*math.Pi)
	} else {
		bisect = (a[1] + a[0]) / 2
	}
	mid := Rectangular(bisect, radius.Perturb(rand.Float64()))

	if b.Crest != nil {
		points := []vg.Point{0: p[0], 2: mid, 4: p[1]}
		c := b.Crest.Perturb(rand.Float64())

		for i, r := range rad {
			points[2*i+1] = Rectangular(a[i], r-(r-radius.Length)*vg.Length(c))
		}
		return points
	}

	return []vg.Point{p[0], mid, p[1]}
}