// Code generated by 'github.com/containous/yaegi/extract gonum.org/v1/gonum/stat/distmv'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15 && !go1.16
// +build go1.15,!go1.16

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/stat/distmv"
)

func init() {
	Symbols["gonum.org/v1/gonum/stat/distmv"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"NewDirichlet":       reflect.ValueOf(distmv.NewDirichlet),
		"NewNormal":          reflect.ValueOf(distmv.NewNormal),
		"NewNormalChol":      reflect.ValueOf(distmv.NewNormalChol),
		"NewNormalPrecision": reflect.ValueOf(distmv.NewNormalPrecision),
		"NewStudentsT":       reflect.ValueOf(distmv.NewStudentsT),
		"NewUniform":         reflect.ValueOf(distmv.NewUniform),
		"NewUnitUniform":     reflect.ValueOf(distmv.NewUnitUniform),
		"NormalLogProb":      reflect.ValueOf(distmv.NormalLogProb),
		"NormalRand":         reflect.ValueOf(distmv.NormalRand),

		// type definitions
		"Bhattacharyya":   reflect.ValueOf((*distmv.Bhattacharyya)(nil)),
		"CrossEntropy":    reflect.ValueOf((*distmv.CrossEntropy)(nil)),
		"Dirichlet":       reflect.ValueOf((*distmv.Dirichlet)(nil)),
		"Hellinger":       reflect.ValueOf((*distmv.Hellinger)(nil)),
		"KullbackLeibler": reflect.ValueOf((*distmv.KullbackLeibler)(nil)),
		"LogProber":       reflect.ValueOf((*distmv.LogProber)(nil)),
		"Normal":          reflect.ValueOf((*distmv.Normal)(nil)),
		"Quantiler":       reflect.ValueOf((*distmv.Quantiler)(nil)),
		"RandLogProber":   reflect.ValueOf((*distmv.RandLogProber)(nil)),
		"Rander":          reflect.ValueOf((*distmv.Rander)(nil)),
		"Renyi":           reflect.ValueOf((*distmv.Renyi)(nil)),
		"StudentsT":       reflect.ValueOf((*distmv.StudentsT)(nil)),
		"Uniform":         reflect.ValueOf((*distmv.Uniform)(nil)),
		"Wasserstein":     reflect.ValueOf((*distmv.Wasserstein)(nil)),

		// interface wrapper definitions
		"_LogProber":     reflect.ValueOf((*_gonum_org_v1_gonum_stat_distmv_LogProber)(nil)),
		"_Quantiler":     reflect.ValueOf((*_gonum_org_v1_gonum_stat_distmv_Quantiler)(nil)),
		"_RandLogProber": reflect.ValueOf((*_gonum_org_v1_gonum_stat_distmv_RandLogProber)(nil)),
		"_Rander":        reflect.ValueOf((*_gonum_org_v1_gonum_stat_distmv_Rander)(nil)),
	}
}

// _gonum_org_v1_gonum_stat_distmv_LogProber is an interface wrapper for LogProber type
type _gonum_org_v1_gonum_stat_distmv_LogProber struct {
	WLogProb func(x []float64) float64
}

func (W _gonum_org_v1_gonum_stat_distmv_LogProber) LogProb(x []float64) float64 { return W.WLogProb(x) }

// _gonum_org_v1_gonum_stat_distmv_Quantiler is an interface wrapper for Quantiler type
type _gonum_org_v1_gonum_stat_distmv_Quantiler struct {
	WQuantile func(x []float64, p []float64) []float64
}

func (W _gonum_org_v1_gonum_stat_distmv_Quantiler) Quantile(x []float64, p []float64) []float64 {
	return W.WQuantile(x, p)
}

// _gonum_org_v1_gonum_stat_distmv_RandLogProber is an interface wrapper for RandLogProber type
type _gonum_org_v1_gonum_stat_distmv_RandLogProber struct {
	WLogProb func(x []float64) float64
	WRand    func(x []float64) []float64
}

func (W _gonum_org_v1_gonum_stat_distmv_RandLogProber) LogProb(x []float64) float64 {
	return W.WLogProb(x)
}
func (W _gonum_org_v1_gonum_stat_distmv_RandLogProber) Rand(x []float64) []float64 { return W.WRand(x) }

// _gonum_org_v1_gonum_stat_distmv_Rander is an interface wrapper for Rander type
type _gonum_org_v1_gonum_stat_distmv_Rander struct {
	WRand func(x []float64) []float64
}

func (W _gonum_org_v1_gonum_stat_distmv_Rander) Rand(x []float64) []float64 { return W.WRand(x) }
