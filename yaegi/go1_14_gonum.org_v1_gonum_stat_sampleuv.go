// Code generated by 'goexports gonum.org/v1/gonum/stat/sampleuv'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/stat/sampleuv"
)

func init() {
	Symbols["gonum.org/v1/gonum/stat/sampleuv"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"ErrRejection":       reflect.ValueOf(&sampleuv.ErrRejection).Elem(),
		"NewWeighted":        reflect.ValueOf(sampleuv.NewWeighted),
		"WithoutReplacement": reflect.ValueOf(sampleuv.WithoutReplacement),

		// type definitions
		"IIDer":                 reflect.ValueOf((*sampleuv.IIDer)(nil)),
		"Importance":            reflect.ValueOf((*sampleuv.Importance)(nil)),
		"LatinHypercube":        reflect.ValueOf((*sampleuv.LatinHypercube)(nil)),
		"MHProposal":            reflect.ValueOf((*sampleuv.MHProposal)(nil)),
		"MetropolisHastings":    reflect.ValueOf((*sampleuv.MetropolisHastings)(nil)),
		"Rejection":             reflect.ValueOf((*sampleuv.Rejection)(nil)),
		"SampleUniformWeighted": reflect.ValueOf((*sampleuv.SampleUniformWeighted)(nil)),
		"Sampler":               reflect.ValueOf((*sampleuv.Sampler)(nil)),
		"Weighted":              reflect.ValueOf((*sampleuv.Weighted)(nil)),
		"WeightedSampler":       reflect.ValueOf((*sampleuv.WeightedSampler)(nil)),

		// interface wrapper definitions
		"_MHProposal":      reflect.ValueOf((*_gonum_org_v1_gonum_stat_sampleuv_MHProposal)(nil)),
		"_Sampler":         reflect.ValueOf((*_gonum_org_v1_gonum_stat_sampleuv_Sampler)(nil)),
		"_WeightedSampler": reflect.ValueOf((*_gonum_org_v1_gonum_stat_sampleuv_WeightedSampler)(nil)),
	}
}

// _gonum_org_v1_gonum_stat_sampleuv_MHProposal is an interface wrapper for MHProposal type
type _gonum_org_v1_gonum_stat_sampleuv_MHProposal struct {
	WConditionalLogProb func(x float64, y float64) (prob float64)
	WConditionalRand    func(y float64) (x float64)
}

func (W _gonum_org_v1_gonum_stat_sampleuv_MHProposal) ConditionalLogProb(x float64, y float64) (prob float64) {
	return W.WConditionalLogProb(x, y)
}
func (W _gonum_org_v1_gonum_stat_sampleuv_MHProposal) ConditionalRand(y float64) (x float64) {
	return W.WConditionalRand(y)
}

// _gonum_org_v1_gonum_stat_sampleuv_Sampler is an interface wrapper for Sampler type
type _gonum_org_v1_gonum_stat_sampleuv_Sampler struct {
	WSample func(batch []float64)
}

func (W _gonum_org_v1_gonum_stat_sampleuv_Sampler) Sample(batch []float64) { W.WSample(batch) }

// _gonum_org_v1_gonum_stat_sampleuv_WeightedSampler is an interface wrapper for WeightedSampler type
type _gonum_org_v1_gonum_stat_sampleuv_WeightedSampler struct {
	WSampleWeighted func(batch []float64, weights []float64)
}

func (W _gonum_org_v1_gonum_stat_sampleuv_WeightedSampler) SampleWeighted(batch []float64, weights []float64) {
	W.WSampleWeighted(batch, weights)
}
