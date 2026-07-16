#!/usr/bin/env python3
"""
flat_interpreter.py – Convert flat action language to STEP CAD model.
Now handles hole placement correctly on any sketch plane.
"""

import re
import math
import sys
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from OCC.Core.gp import (gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Vec, gp_Trsf, gp_Circ)
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeWire, BRepBuilderAPI_Copy)
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol,
                                  BRepPrimAPI_MakeCylinder)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID
from OCC.Core.BRep import BRep_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopLoc import TopLoc_Location 

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def ensure_3d_vector(v, default=None):
    if default is None:
        default = np.array([0.0, 0.0, 1.0])
    if isinstance(v, str):
        v = [float(x) for x in v.split(',')]
    v = np.asarray(v, dtype=float).flatten()
    if len(v) < 3:
        v = np.pad(v, (0, 3 - len(v)), mode='constant')
    else:
        v = v[:3]
    norm = np.linalg.norm(v)
    if norm > 1e-9:
        v = v / norm
    else:
        v = default
    return v

def parse_vector(s):
    return [float(x) for x in s.split(',')]

# ----------------------------------------------------------------------
# Flat CAD Builder (with sketch plane storage)
# ----------------------------------------------------------------------
class FlatCADBuilder:
    def __init__(self):
        self.current_shape = None
        self.sketch_face = None
        self.sketch_plane_origin = None   # 3D point
        self.sketch_plane_normal = None   # unit normal
        self.sketch_u = None              # unit vector along U
        self.sketch_v = None              # unit vector along V
        self.builder = BRep_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)

    def _fuse(self, shape):
        if self.current_shape is None:
            self.current_shape = shape
        else:
            fuse = BRepAlgoAPI_Fuse(self.current_shape, shape)
            if fuse.IsDone():
                self.current_shape = fuse.Shape()

    def _cut(self, shape):
        if self.current_shape is None:
            return
        cut = BRepAlgoAPI_Cut(self.current_shape, shape)
        if cut.IsDone():
            self.current_shape = cut.Shape()
        else:
            print("Warning: Cut operation failed – hole may not intersect the body")

    # ------------------------------------------------------------------
    # Sketch creation (stores plane transform)
    # ------------------------------------------------------------------
    def create_sketch(self, params: Dict[str, str]):
        plane_normal = ensure_3d_vector(parse_vector(params['plane_normal']))
        plane_origin = np.array(parse_vector(params['plane_origin']), dtype=float)
        profile_type = params.get('profile', 'rect')

        # Store sketch plane data for later hole placement
        self.sketch_plane_origin = plane_origin.copy()
        self.sketch_plane_normal = plane_normal.copy()
        # Build local 2D basis
        n = plane_normal
        if abs(n[0]) < 0.9:
            u = np.cross(n, [1, 0, 0])
        else:
            u = np.cross(n, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        self.sketch_u = u
        self.sketch_v = v

        def uv_to_3d(uv):
            p = plane_origin + uv[0]*u + uv[1]*v
            return gp_Pnt(p[0], p[1], p[2])

        edges = []
        if profile_type == 'rect':
            w = float(params['width'])
            h = float(params['height'])
            half_w, half_h = w/2, h/2
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            for i in range(4):
                p1 = uv_to_3d(corners[i])
                p2 = uv_to_3d(corners[(i+1)%4])
                edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        elif profile_type == 'circle':
            r = float(params['radius'])
            center = uv_to_3d((0.0, 0.0))
            ax2 = gp_Ax2(center, gp_Dir(n[0], n[1], n[2]))
            circ = gp_Circ(ax2, r)
            edges.append(BRepBuilderAPI_MakeEdge(circ).Edge())
        elif profile_type == 'poly':
            points_str = params['points']
            pts = []
            for pt in points_str.split():
                xy = parse_vector(pt)
                pts.append(uv_to_3d((xy[0], xy[1])))
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i+1)%len(pts)]
                edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        elif profile_type == 'slot':
            w = float(params['width'])
            h = float(params['height'])
            half_w, half_h = w/2, h/2
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            for i in range(4):
                p1 = uv_to_3d(corners[i])
                p2 = uv_to_3d(corners[(i+1)%4])
                edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

        wire_builder = BRepBuilderAPI_MakeWire()
        for e in edges:
            wire_builder.Add(e)
        if not wire_builder.IsDone():
            raise RuntimeError("Failed to build sketch wire")
        wire = wire_builder.Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        self.sketch_face = face
        return face

    # ------------------------------------------------------------------
    # Extrude
    # ------------------------------------------------------------------
    def extrude(self, params: Dict[str, str], profile_shape: TopoDS_Shape = None):
        if profile_shape is None:
            profile_shape = self.sketch_face
        if profile_shape is None:
            raise ValueError("No sketch available for extrusion")
        direction = ensure_3d_vector(parse_vector(params['direction']))
        distance = float(params['distance'])
        if distance <= 0:
            distance = 1.0
        vec = gp_Vec(direction[0]*distance, direction[1]*distance, direction[2]*distance)
        prism = BRepPrimAPI_MakePrism(profile_shape, vec, False, False)
        if not prism.IsDone():
            raise RuntimeError("Extrusion failed")
        return prism.Shape()

    # ------------------------------------------------------------------
    # Add hole (uses stored sketch plane to transform 2D center)
    # ------------------------------------------------------------------
    def add_hole(self, params: Dict[str, str]):
        center_uv = parse_vector(params['center'])   # [x,y] in sketch 2D coordinates
        radius = float(params['radius'])
        depth = float(params['depth'])
        axis = ensure_3d_vector(parse_vector(params['axis']))

        # Transform 2D hole center to 3D using stored sketch plane
        if self.sketch_plane_origin is None:
            raise RuntimeError("No sketch plane defined; cannot place hole")
        center_3d = self.sketch_plane_origin + center_uv[0]*self.sketch_u + center_uv[1]*self.sketch_v

        # Create cylinder along the hole axis
        height = depth * 1.1   # make slightly longer to ensure through cut
        # Base center: offset along axis by half height so cylinder extends both sides
        base_center = center_3d - axis * (height/2)
        cyl_axis = gp_Ax2(gp_Pnt(base_center[0], base_center[1], base_center[2]),
                          gp_Dir(axis[0], axis[1], axis[2]))
        cylinder = BRepPrimAPI_MakeCylinder(cyl_axis, radius, height).Shape()
        return cylinder

    # ------------------------------------------------------------------
    # Fillet
    # ------------------------------------------------------------------
    def fillet(self, params: Dict[str, str]):
        radius = float(params['radius'])
        if self.current_shape is None:
            return
        fillet = BRepFilletAPI_MakeFillet(self.current_shape)
        explorer = TopExp_Explorer(self.current_shape, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            try:
                fillet.Add(radius, edge)
            except:
                pass
            explorer.Next()
        if fillet.IsDone():
            self.current_shape = fillet.Shape()

    # ------------------------------------------------------------------
    # Circular pattern (simplified)
    # ------------------------------------------------------------------
    def circular_pattern(self, params: Dict[str, str]):
        if self.current_shape is None:
            return
        count = int(params['count'])
        center = np.array(parse_vector(params['center']), dtype=float)
        axis = ensure_3d_vector(parse_vector(params['axis']))

        angle_step = 2 * math.pi / count
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, self.current_shape)

        for i in range(1, count):
            rot_angle = i * angle_step
            trsf = gp_Trsf()
            ax = gp_Ax1(gp_Pnt(center[0], center[1], center[2]),
                    gp_Dir(axis[0], axis[1], axis[2]))
            trsf.SetRotation(ax, rot_angle)
            location = TopLoc_Location(trsf)   # convert to Location
            copy = BRepBuilderAPI_Copy(self.current_shape).Shape()
            copy.Move(location)                # now works
            builder.Add(compound, copy)

        # Fuse all copies
        result = None
        explorer = TopExp_Explorer(compound, TopAbs_SOLID)
        while explorer.More():
            solid = explorer.Current()
            if result is None:
                result = solid
            else:
                fuse = BRepAlgoAPI_Fuse(result, solid)
                if fuse.IsDone():
                    result = fuse.Shape()
            explorer.Next()
        self.current_shape = result

    # ------------------------------------------------------------------
    # Parse and execute action string
    # ------------------------------------------------------------------
    def execute_actions(self, action_str: str):
        # Remove trailing <EOS> and split by '|'
        action_str = re.sub(r'\s*<EOS>\s*$', '', action_str.strip())
        parts = action_str.split('|')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            if not tokens:
                continue
            action = tokens[0]
            params = {}
            for tok in tokens[1:]:
                if '=' not in tok:
                    continue
                key, val = tok.split('=', 1)
                params[key] = val
            if action == 'CREATE_SKETCH':
                self.create_sketch(params)
            elif action == 'EXTRUDE':
                solid = self.extrude(params)
                if params.get('operation', 'join') == 'cut':
                    self._cut(solid)
                else:
                    self._fuse(solid)
            elif action == 'ADD_HOLE':
                hole = self.add_hole(params)
                self._cut(hole)
            elif action == 'FILLET':
                self.fillet(params)
            elif action == 'CIRCULAR_PATTERN':
                self.circular_pattern(params)
            else:
                print(f"Warning: Unknown action '{action}' ignored")
        return self.current_shape

# ----------------------------------------------------------------------
# STEP saving and main
# ----------------------------------------------------------------------
def save_step(shape: TopoDS_Shape, filename: str) -> bool:
    if shape is None:
        print("Error: No shape to save.")
        return False
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filename)
    if status != IFSelect_RetDone:
        print("Error: STEP write failed.")
        return False
    print(f"Saved STEP file: {filename}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert flat action string to STEP model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--action", type=str, help="Flat action string directly")
    group.add_argument("--input", type=str, help="File containing action string")
    parser.add_argument("--output", type=str, default="output.step", help="Output STEP file path")
    args = parser.parse_args()

    if args.action:
        action_str = args.action
    else:
        with open(args.input, 'r') as f:
            action_str = f.read().strip()

    builder = FlatCADBuilder()
    shape = builder.execute_actions(action_str)
    if shape is None:
        print("No shape generated.")
        sys.exit(1)
    save_step(shape, args.output)

if __name__ == "__main__":
    main()
