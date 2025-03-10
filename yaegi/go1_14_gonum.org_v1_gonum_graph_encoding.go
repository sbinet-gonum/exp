// Code generated by 'goexports gonum.org/v1/gonum/graph/encoding'. DO NOT EDIT.

// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.14 && !go1.15
// +build go1.14,!go1.15

package yaegi

import (
	"reflect"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

func init() {
	Symbols["gonum.org/v1/gonum/graph/encoding"] = map[string]reflect.Value{
		// type definitions
		"Attribute":       reflect.ValueOf((*encoding.Attribute)(nil)),
		"AttributeSetter": reflect.ValueOf((*encoding.AttributeSetter)(nil)),
		"Attributer":      reflect.ValueOf((*encoding.Attributer)(nil)),
		"Builder":         reflect.ValueOf((*encoding.Builder)(nil)),
		"MultiBuilder":    reflect.ValueOf((*encoding.MultiBuilder)(nil)),

		// interface wrapper definitions
		"_AttributeSetter": reflect.ValueOf((*_gonum_org_v1_gonum_graph_encoding_AttributeSetter)(nil)),
		"_Attributer":      reflect.ValueOf((*_gonum_org_v1_gonum_graph_encoding_Attributer)(nil)),
		"_Builder":         reflect.ValueOf((*_gonum_org_v1_gonum_graph_encoding_Builder)(nil)),
		"_MultiBuilder":    reflect.ValueOf((*_gonum_org_v1_gonum_graph_encoding_MultiBuilder)(nil)),
	}
}

// _gonum_org_v1_gonum_graph_encoding_AttributeSetter is an interface wrapper for AttributeSetter type
type _gonum_org_v1_gonum_graph_encoding_AttributeSetter struct {
	WSetAttribute func(a0 encoding.Attribute) error
}

func (W _gonum_org_v1_gonum_graph_encoding_AttributeSetter) SetAttribute(a0 encoding.Attribute) error {
	return W.WSetAttribute(a0)
}

// _gonum_org_v1_gonum_graph_encoding_Attributer is an interface wrapper for Attributer type
type _gonum_org_v1_gonum_graph_encoding_Attributer struct {
	WAttributes func() []encoding.Attribute
}

func (W _gonum_org_v1_gonum_graph_encoding_Attributer) Attributes() []encoding.Attribute {
	return W.WAttributes()
}

// _gonum_org_v1_gonum_graph_encoding_Builder is an interface wrapper for Builder type
type _gonum_org_v1_gonum_graph_encoding_Builder struct {
	WAddNode        func(a0 graph.Node)
	WEdge           func(uid int64, vid int64) graph.Edge
	WFrom           func(id int64) graph.Nodes
	WHasEdgeBetween func(xid int64, yid int64) bool
	WNewEdge        func(from graph.Node, to graph.Node) graph.Edge
	WNewNode        func() graph.Node
	WNode           func(id int64) graph.Node
	WNodes          func() graph.Nodes
	WSetEdge        func(e graph.Edge)
}

func (W _gonum_org_v1_gonum_graph_encoding_Builder) AddNode(a0 graph.Node) { W.WAddNode(a0) }
func (W _gonum_org_v1_gonum_graph_encoding_Builder) Edge(uid int64, vid int64) graph.Edge {
	return W.WEdge(uid, vid)
}
func (W _gonum_org_v1_gonum_graph_encoding_Builder) From(id int64) graph.Nodes { return W.WFrom(id) }
func (W _gonum_org_v1_gonum_graph_encoding_Builder) HasEdgeBetween(xid int64, yid int64) bool {
	return W.WHasEdgeBetween(xid, yid)
}
func (W _gonum_org_v1_gonum_graph_encoding_Builder) NewEdge(from graph.Node, to graph.Node) graph.Edge {
	return W.WNewEdge(from, to)
}
func (W _gonum_org_v1_gonum_graph_encoding_Builder) NewNode() graph.Node      { return W.WNewNode() }
func (W _gonum_org_v1_gonum_graph_encoding_Builder) Node(id int64) graph.Node { return W.WNode(id) }
func (W _gonum_org_v1_gonum_graph_encoding_Builder) Nodes() graph.Nodes       { return W.WNodes() }
func (W _gonum_org_v1_gonum_graph_encoding_Builder) SetEdge(e graph.Edge)     { W.WSetEdge(e) }

// _gonum_org_v1_gonum_graph_encoding_MultiBuilder is an interface wrapper for MultiBuilder type
type _gonum_org_v1_gonum_graph_encoding_MultiBuilder struct {
	WAddNode        func(a0 graph.Node)
	WFrom           func(id int64) graph.Nodes
	WHasEdgeBetween func(xid int64, yid int64) bool
	WLines          func(uid int64, vid int64) graph.Lines
	WNewLine        func(from graph.Node, to graph.Node) graph.Line
	WNewNode        func() graph.Node
	WNode           func(id int64) graph.Node
	WNodes          func() graph.Nodes
	WSetLine        func(l graph.Line)
}

func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) AddNode(a0 graph.Node) { W.WAddNode(a0) }
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) From(id int64) graph.Nodes {
	return W.WFrom(id)
}
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) HasEdgeBetween(xid int64, yid int64) bool {
	return W.WHasEdgeBetween(xid, yid)
}
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) Lines(uid int64, vid int64) graph.Lines {
	return W.WLines(uid, vid)
}
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) NewLine(from graph.Node, to graph.Node) graph.Line {
	return W.WNewLine(from, to)
}
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) NewNode() graph.Node { return W.WNewNode() }
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) Node(id int64) graph.Node {
	return W.WNode(id)
}
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) Nodes() graph.Nodes   { return W.WNodes() }
func (W _gonum_org_v1_gonum_graph_encoding_MultiBuilder) SetLine(l graph.Line) { W.WSetLine(l) }
