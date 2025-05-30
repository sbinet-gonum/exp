// Code generated by 'goexports gonum.org/v1/gonum/stat/samplemv'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/samplemv"
)

func init() {
	Symbols["gonum.org/v1/gonum/stat/samplemv"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"ErrRejection":      reflect.ValueOf(&samplemv.ErrRejection).Elem(),
		"NewProposalNormal": reflect.ValueOf(samplemv.NewProposalNormal),
		"Owen":              reflect.ValueOf(samplemv.Owen),

		// type definitions
		"Halton":                reflect.ValueOf((*samplemv.Halton)(nil)),
		"HaltonKind":            reflect.ValueOf((*samplemv.HaltonKind)(nil)),
		"IID":                   reflect.ValueOf((*samplemv.IID)(nil)),
		"Importance":            reflect.ValueOf((*samplemv.Importance)(nil)),
		"LatinHypercube":        reflect.ValueOf((*samplemv.LatinHypercube)(nil)),
		"MHProposal":            reflect.ValueOf((*samplemv.MHProposal)(nil)),
		"MetropolisHastingser":  reflect.ValueOf((*samplemv.MetropolisHastingser)(nil)),
		"ProposalNormal":        reflect.ValueOf((*samplemv.ProposalNormal)(nil)),
		"Rejection":             reflect.ValueOf((*samplemv.Rejection)(nil)),
		"SampleUniformWeighted": reflect.ValueOf((*samplemv.SampleUniformWeighted)(nil)),
		"Sampler":               reflect.ValueOf((*samplemv.Sampler)(nil)),
		"WeightedSampler":       reflect.ValueOf((*samplemv.WeightedSampler)(nil)),

		// interface wrapper definitions
		"_MHProposal":      reflect.ValueOf((*_gonum_org_v1_gonum_stat_samplemv_MHProposal)(nil)),
		"_Sampler":         reflect.ValueOf((*_gonum_org_v1_gonum_stat_samplemv_Sampler)(nil)),
		"_WeightedSampler": reflect.ValueOf((*_gonum_org_v1_gonum_stat_samplemv_WeightedSampler)(nil)),
	}
}

// _gonum_org_v1_gonum_stat_samplemv_MHProposal is an interface wrapper for MHProposal type
type _gonum_org_v1_gonum_stat_samplemv_MHProposal struct {
	WConditionalLogProb func(x []float64, y []float64) (prob float64)
	WConditionalRand    func(x []float64, y []float64) []float64
}

func (W _gonum_org_v1_gonum_stat_samplemv_MHProposal) ConditionalLogProb(x []float64, y []float64) (prob float64) {
	return W.WConditionalLogProb(x, y)
}
func (W _gonum_org_v1_gonum_stat_samplemv_MHProposal) ConditionalRand(x []float64, y []float64) []float64 {
	return W.WConditionalRand(x, y)
}

// _gonum_org_v1_gonum_stat_samplemv_Sampler is an interface wrapper for Sampler type
type _gonum_org_v1_gonum_stat_samplemv_Sampler struct {
	WSample func(batch *mat.Dense)
}

func (W _gonum_org_v1_gonum_stat_samplemv_Sampler) Sample(batch *mat.Dense) { W.WSample(batch) }

// _gonum_org_v1_gonum_stat_samplemv_WeightedSampler is an interface wrapper for WeightedSampler type
type _gonum_org_v1_gonum_stat_samplemv_WeightedSampler struct {
	WSampleWeighted func(batch *mat.Dense, weights []float64)
}

func (W _gonum_org_v1_gonum_stat_samplemv_WeightedSampler) SampleWeighted(batch *mat.Dense, weights []float64) {
	W.WSampleWeighted(batch, weights)
}
