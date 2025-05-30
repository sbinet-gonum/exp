// Code generated by 'goexports gonum.org/v1/gonum/graph/topo'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/topo"
)

func init() {
	Symbols["gonum.org/v1/gonum/graph/topo"] = map[string]reflect.Value{
		// function, constant and variable definitions
		"BronKerbosch":        reflect.ValueOf(topo.BronKerbosch),
		"CliqueGraph":         reflect.ValueOf(topo.CliqueGraph),
		"ConnectedComponents": reflect.ValueOf(topo.ConnectedComponents),
		"DegeneracyOrdering":  reflect.ValueOf(topo.DegeneracyOrdering),
		"DirectedCyclesIn":    reflect.ValueOf(topo.DirectedCyclesIn),
		"Equal":               reflect.ValueOf(topo.Equal),
		"IsPathIn":            reflect.ValueOf(topo.IsPathIn),
		"KCore":               reflect.ValueOf(topo.KCore),
		"PathExistsIn":        reflect.ValueOf(topo.PathExistsIn),
		"Sort":                reflect.ValueOf(topo.Sort),
		"SortStabilized":      reflect.ValueOf(topo.SortStabilized),
		"TarjanSCC":           reflect.ValueOf(topo.TarjanSCC),
		"UndirectedCyclesIn":  reflect.ValueOf(topo.UndirectedCyclesIn),

		// type definitions
		"Builder":         reflect.ValueOf((*topo.Builder)(nil)),
		"Clique":          reflect.ValueOf((*topo.Clique)(nil)),
		"CliqueGraphEdge": reflect.ValueOf((*topo.CliqueGraphEdge)(nil)),
		"Unorderable":     reflect.ValueOf((*topo.Unorderable)(nil)),

		// interface wrapper definitions
		"_Builder": reflect.ValueOf((*_gonum_org_v1_gonum_graph_topo_Builder)(nil)),
	}
}

// _gonum_org_v1_gonum_graph_topo_Builder is an interface wrapper for Builder type
type _gonum_org_v1_gonum_graph_topo_Builder struct {
	WAddNode func(a0 graph.Node)
	WSetEdge func(a0 graph.Edge)
}

func (W _gonum_org_v1_gonum_graph_topo_Builder) AddNode(a0 graph.Node) { W.WAddNode(a0) }
func (W _gonum_org_v1_gonum_graph_topo_Builder) SetEdge(a0 graph.Edge) { W.WSetEdge(a0) }
