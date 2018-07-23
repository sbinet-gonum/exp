// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linsolve

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/floats"
)

// BCG implements the BiConjugate Gradient method with
// preconditioning for solving systems of linear equations
//  A*x = b,
// where A is a square, generally nonsymmetric matrix.
//
// References:
//  - Barrett, R. et al. (1994). Section 2.3.5 BiConjugate Gradient (BiCG).
//    In Templates for the Solution of Linear Systems: Building Blocks
//    for Iterative Methods (2nd ed.) (pp. 19-20). Philadelphia, PA: SIAM.
//    Retrieved from http://www.netlib.org/templates/templates.pdf
type BCG struct {
	first  bool
	resume int

	rho, rhoPrev float64
	alpha        float64

	rt    []float64
	z, zt []float64
	p, pt []float64
}

// Init implements the Method interface.
func (b *BCG) Init(dim int) {
	if dim <= 0 {
		panic("bcg: dimension not positive")
	}

	b.rt = reuse(b.rt, dim)
	b.z = reuse(b.z, dim)
	b.zt = reuse(b.zt, dim)
	b.p = reuse(b.p, dim)
	b.pt = reuse(b.pt, dim)

	b.first = true
	b.resume = 1
}

// Iterate implements the Method interface. It will command the following
// operations:
//  MulVec
//  MulVec|Trans
//  PreconSolve
//  PreconSolve|Trans
//  CheckResidual
//  MajorIteration
func (b *BCG) Iterate(ctx *Context) (Operation, error) {
	switch b.resume {
	case 1:
		if b.first {
			copy(b.rt, ctx.Residual)
		}
		copy(ctx.Src, ctx.Residual)
		b.resume = 2
		// Solve M^{-1} * r_{i-1}.
		return PreconSolve, nil
	case 2:
		copy(b.z, ctx.Dst)
		copy(ctx.Src, b.rt)
		b.resume = 3
		// Solve M^{-T} * rt_{i-1}.
		return PreconSolve | Trans, nil
	case 3:
		copy(b.zt, ctx.Dst)
		b.rho = floats.Dot(b.z, b.rt)
		if math.Abs(b.rho) < rhoBreakdownTol {
			b.resume = 0
			return NoOperation, errors.New("bcg: rho breakdown")
		}
		if b.first {
			copy(b.p, b.z)
			copy(b.pt, b.zt)
		} else {
			beta := b.rho / b.rhoPrev
			floats.AddScaledTo(b.p, b.z, beta, b.p)
			floats.AddScaledTo(b.pt, b.zt, beta, b.pt)
		}
		copy(ctx.Src, b.p)
		b.resume = 4
		// Compute A * p.
		return MulVec, nil
	case 4:
		// z is overwritten and reused.
		copy(b.z, ctx.Dst)
		copy(ctx.Src, b.pt)
		b.resume = 5
		// Compute A^T * pt.
		return MulVec | Trans, nil
	case 5:
		// zt is overwritten and reused.
		copy(b.zt, ctx.Dst)
		b.alpha = b.rho / floats.Dot(b.pt, b.z)
		floats.AddScaled(ctx.X, b.alpha, b.p)
		floats.AddScaled(ctx.Residual, -b.alpha, b.z)
		b.resume = 6
		return CheckResidual, nil
	case 6:
		// Prepare for the next iteration.
		floats.AddScaled(b.rt, -b.alpha, b.zt)
		b.rhoPrev = b.rho
		b.first = false
		b.resume = 1
		return MajorIteration, nil

	default:
		panic("bcg: Init not called")
	}
}
