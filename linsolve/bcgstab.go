// Copyright ©2017 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linsolve

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
)

// BCGSTAB implements the BiConjugate Gradient Stabilized method with
// preconditioning for solving systems of linear equations
//  A*x = b,
// where A is a square, generally nonsymmetric matrix.
//
//
// References:
//  - Barrett, R. et al. (1994). Section 2.3.8 BiConjugate Gradient Stabilized (Bi-CGSTAB).
//    In Templates for the Solution of Linear Systems: Building Blocks
//    for Iterative Methods (2nd ed.) (pp. 24-25). Philadelphia, PA: SIAM.
//    Retrieved from http://www.netlib.org/templates/templates.pdf
type BCGSTAB struct {
	first  bool
	resume int

	rho, rhoPrev float64
	alpha        float64
	omega        float64

	rt   []float64
	p    []float64
	v    []float64
	t    []float64
	phat []float64
	shat []float64
}

// Init implements the Method interface.
func (b *BCGSTAB) Init(dim int) {
	if dim <= 0 {
		panic("bcgstab: dimension not positive")
	}

	b.rt = reuse(b.rt, dim)
	b.p = reuse(b.p, dim)
	b.v = reuse(b.v, dim)
	b.t = reuse(b.t, dim)
	b.phat = reuse(b.phat, dim)
	b.shat = reuse(b.shat, dim)

	b.first = true
	b.resume = 1
}

// Iterate implements the Method interface. It will command the following
// operations:
//  MulVec
//  PreconSolve
//  CheckResidual
//  MajorIteration
func (b *BCGSTAB) Iterate(ctx *Context) (Operation, error) {
	switch b.resume {
	case 1:
		if b.first {
			copy(b.rt, ctx.Residual)
			copy(b.p, ctx.Residual)
		}
		b.rho = floats.Dot(b.rt, ctx.Residual)
		fmt.Println("rho", b.rho)
		if math.Abs(b.rho) < rhoBreakdownTol {
			b.resume = 0
			fmt.Println("rho", b.rho, rhoBreakdownTol, floats.Norm(ctx.Residual, 2))
			return NoOperation, errors.New("bcgstab: rho breakdown")
		}
		if !b.first {
			beta := (b.rho / b.rhoPrev) * (b.alpha / b.omega)
			floats.AddScaled(b.p, -b.omega, b.v)
			floats.Scale(beta, b.p)
			floats.Add(b.p, ctx.Residual)
		}
		b.first = false
		copy(ctx.Src, b.p)
		b.resume = 2
		// Solve M^{-1} * p_i.
		return PreconSolve, nil
	case 2:
		copy(b.phat, ctx.Dst)
		copy(ctx.Src, b.phat)
		b.resume = 3
		// Compute A * p^_i.
		return MulVec, nil
	case 3:
		copy(b.v, ctx.Dst)
		b.alpha = b.rho / floats.Dot(b.rt, b.v)
		// Form the residual so that we can check for tolerance early.
		floats.AddScaled(ctx.Residual, -b.alpha, b.v)
		b.resume = 4
		return CheckResidual, nil
	case 4:
		if ctx.Converged {
			floats.AddScaled(ctx.X, b.alpha, b.phat)
			b.resume = 5
			return MajorIteration, nil
		}
		fallthrough
	case 5:
		copy(ctx.Src, ctx.Residual)
		b.resume = 6
		// Solve M^{-1} * r_i.
		return PreconSolve, nil
	case 6:
		copy(b.shat, ctx.Dst)
		copy(ctx.Src, b.shat)
		b.resume = 7
		// Compute A * s^_i.
		return MulVec, nil
	case 7:
		copy(b.t, ctx.Dst)
		b.omega = floats.Dot(b.t, ctx.Residual) / floats.Dot(b.t, b.t)
		floats.AddScaled(ctx.X, b.alpha, b.phat)
		floats.AddScaled(ctx.X, b.omega, b.shat)
		floats.AddScaled(ctx.Residual, -b.omega, b.t)
		b.resume = 8
		return CheckResidual, nil
	case 8:
		if !ctx.Converged && math.Abs(b.omega) < omegaBreakdownTol {
			b.resume = 0
			fmt.Println("omega", b.omega, omegaBreakdownTol, floats.Norm(ctx.Residual, 2))
			return NoOperation, errors.New("bcgstab: omega breakdown")
		}
		b.rhoPrev = b.rho
		b.resume = 1
		return MajorIteration, nil

	default:
		panic("bcgstab: Init not called")
	}
}
