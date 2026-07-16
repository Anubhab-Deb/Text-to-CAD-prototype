#!/usr/bin/env python3
"""
PHASE 7: CAD Interpreter – Execute action sequences to reconstruct 3D geometry.
Uses pythonOCC to build a solid from parametric operations.
(With robust JSON repair and rectangle support)
"""

import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# pythonOCC imports
from OCC.Core.gp import (gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Ax3,
                         gp_Vec, gp_Trsf, gp_Circ)
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeVertex,
                                     BRepBuilderAPI_Copy)
from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakePrism,
                                  BRepPrimAPI_MakeRevol,
                                  BRepPrimAPI_MakeCylinder)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.TopoDS import (TopoDS_Shape, TopoDS_Solid, TopoDS_Compound)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID
from OCC.Core.BRep import BRep_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

# ----------------------------------------------------------------------
# Helper: Normalise vector to length 3, replace zero vector with default
# ----------------------------------------------------------------------
def ensure_3d_vector(v, default=None):
    """Convert v to a 3D numpy array, pad/truncate to length 3, and normalise.
    If the result is zero, use default (or [0,0,1] if default not given)."""
    if default is None:
        default = np.array([0.0, 0.0, 1.0])
    v = np.asarray(v, dtype=float).flatten()
    if len(v) < 3:
        v = np.pad(v, (0, 3 - len(v)), mode='constant')
    elif len(v) > 3:
        v = v[:3]
    norm = np.linalg.norm(v)
    if norm > 1e-9:
        v = v / norm
    else:
        v = default
    return v

# ----------------------------------------------------------------------
# JSON repair helper (stronger version)
# ----------------------------------------------------------------------
def repair_json_like_string(s: str) -> str:
    """Repair malformed JSON-like strings that are missing colons/braces."""
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        return s
    inner = s[1:-1].strip()
    # Case 1: No braces, likely flat object missing braces
    if '{' not in inner and '}' not in inner:
        return json.dumps(json.loads('{' + inner + '}'))
    # Case 2: Contains braces, fix missing colons/commas
    inner = re.sub(r'\}\s*\{', '}, {', inner)
    brace_level = 0
    start = 0
    parts = []
    for i, ch in enumerate(inner):
        if ch == '{':
            brace_level += 1
        elif ch == '}':
            brace_level -= 1
            if brace_level == 0:
                parts.append(inner[start:i+1].strip())
                start = i+2  # skip comma
    if not parts:
        parts.append(inner)
    repaired_parts = []
    for part in parts:
        part = part.strip()
        if part.startswith('{'):
            part = part[1:]
        if part.endswith('}'):
            part = part[:-1]
        # Fix missing colons: convert '"key""value"' to '"key":"value"'
        part = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*"', r'"\1":"', part)
        repaired_parts.append('{' + part + '}')
    return '[' + ','.join(repaired_parts) + ']'

def repair_profile_string(s: str) -> str:
    """Convert malformed profile like '["type":"rectangle",...]' to '[{"type":"rectangle",...}]'."""
    s = s.strip()
    if not s.startswith('[') or not s.endswith(']'):
        return s
    inner = s[1:-1].strip()
    # If the inner part does not start with '{', assume missing braces around the whole object
    if inner and not inner.startswith('{'):
        return '[{' + inner + '}]'
    return s

# ----------------------------------------------------------------------
# Action parsing with robust parameter extraction
# ----------------------------------------------------------------------
def parse_actions(action_str: str) -> List[Dict]:
    actions = []
    parts = action_str.split("<SEP>")
    for part in parts:
        part = part.strip()
        if not part or part == "<EOS>":
            continue
        end_tag = part.find(">")
        if end_tag == -1:
            continue
        action_name = part[1:end_tag]
        params_str = part[end_tag+1:].strip()
        params = {}
        if params_str:
            # Split key=value pairs by '|' that are not inside quotes? Simple split is fine.
            for pair in params_str.split("|"):
                if "=" not in pair:
                    continue
                key, val = pair.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Special handling for profile
                if key == "profile":
                    # Try to repair the string first
                    val = repair_profile_string(val)
                # Try to parse as JSON
                try:
                    parsed = json.loads(val)
                    params[key] = parsed
                except:
                    # If fails, try repair
                    try:
                        repaired = repair_json_like_string(val)
                        parsed = json.loads(repaired)
                        # If val started with '[' but parsed becomes dict, wrap in list
                        if isinstance(parsed, dict) and val.startswith('['):
                            parsed = [parsed]
                        params[key] = parsed
                    except:
                        # Fallback: treat as number or string
                        try:
                            if '.' in val or 'e' in val.lower():
                                params[key] = float(val)
                            else:
                                params[key] = int(val)
                        except:
                            params[key] = val
            # Post-process vectors: ensure direction, plane_normal, etc. are lists of floats
            for vect_key in ['direction', 'plane_normal', 'axis']:
                if vect_key in params and isinstance(params[vect_key], list):
                    params[vect_key] = [float(x) for x in params[vect_key]]
        actions.append({"action": action_name, "params": params})
    return actions

# ----------------------------------------------------------------------
# CAD Builder
# ----------------------------------------------------------------------
class CADBuilder:
    def __init__(self):
        self.current_shape = None
        self.builder = BRep_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)
        self.sketch_profile = None

    def _fuse_to_current(self, shape: TopoDS_Shape):
        if self.current_shape is None:
            self.current_shape = shape
        else:
            fuse = BRepAlgoAPI_Fuse(self.current_shape, shape)
            if fuse.IsDone():
                self.current_shape = fuse.Shape()

    def _cut_from_current(self, shape: TopoDS_Shape):
        if self.current_shape is None:
            return
        cut = BRepAlgoAPI_Cut(self.current_shape, shape)
        if cut.IsDone():
            self.current_shape = cut.Shape()

    def create_sketch(self, params: Dict) -> TopoDS_Shape:
        plane_normal = ensure_3d_vector(params.get("plane_normal", [0, 0, 1]))
        plane_origin = np.array(params.get("plane_origin", [0, 0, 0]), dtype=float)
        if len(plane_origin) < 3:
            plane_origin = np.pad(plane_origin, (0, 3 - len(plane_origin)), mode='constant')
        else:
            plane_origin = plane_origin[:3]

        profile = params.get("profile", [])
        if not profile:
            raise ValueError("No profile defined in CreateSketch.")

        # Build local 2D basis
        n = plane_normal
        if abs(n[0]) < 0.9:
            u = np.cross(n, [1, 0, 0])
        else:
            u = np.cross(n, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        def uv_to_3d(uv_point: Tuple[float, float]) -> gp_Pnt:
            p = plane_origin + uv_point[0] * u + uv_point[1] * v
            return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))

        edges = []
        for prim in profile:
            ptype = prim.get("type")
            if ptype == "line":
                p1 = uv_to_3d((float(prim["start"][0]), float(prim["start"][1])))
                p2 = uv_to_3d((float(prim["end"][0]), float(prim["end"][1])))
                edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
            elif ptype == "arc":
                center_uv = prim["center"]
                center = uv_to_3d((float(center_uv[0]), float(center_uv[1])))
                radius = float(prim["radius"])
                start_angle = float(prim.get("start_angle", 0.0))
                end_angle = float(prim.get("end_angle", math.pi))
                ax2 = gp_Ax2(center, gp_Dir(float(n[0]), float(n[1]), float(n[2])))
                circ = gp_Circ(ax2, radius)
                edges.append(BRepBuilderAPI_MakeEdge(circ, start_angle, end_angle).Edge())
            elif ptype == "circle":
                center_uv = prim["center"]
                center = uv_to_3d((float(center_uv[0]), float(center_uv[1])))
                radius = float(prim["radius"])
                ax2 = gp_Ax2(center, gp_Dir(float(n[0]), float(n[1]), float(n[2])))
                circ = gp_Circ(ax2, radius)
                edges.append(BRepBuilderAPI_MakeEdge(circ).Edge())
            elif ptype == "rectangle":
                w = float(prim["width"]) / 2.0
                h = float(prim["height"]) / 2.0
                corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
                for i in range(4):
                    p1 = uv_to_3d(corners[i])
                    p2 = uv_to_3d(corners[(i+1)%4])
                    edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
            else:
                # Fallback: treat as line if start/end present
                if "start" in prim and "end" in prim:
                    p1 = uv_to_3d((float(prim["start"][0]), float(prim["start"][1])))
                    p2 = uv_to_3d((float(prim["end"][0]), float(prim["end"][1])))
                    edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
                else:
                    continue

        if not edges:
            raise ValueError("No valid edges created for sketch.")

        wire_builder = BRepBuilderAPI_MakeWire()
        for e in edges:
            wire_builder.Add(e)
        if not wire_builder.IsDone():
            raise RuntimeError("Failed to build wire.")
        wire = wire_builder.Wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        return face

    def extrude(self, params: Dict, profile_shape: TopoDS_Shape) -> TopoDS_Shape:
        direction = ensure_3d_vector(params.get("direction", [0, 0, 1]))
        distance = float(params.get("distance", 10.0))
        vec = gp_Vec(float(direction[0]*distance),
                     float(direction[1]*distance),
                     float(direction[2]*distance))
        prism = BRepPrimAPI_MakePrism(profile_shape, vec, False, False)
        if not prism.IsDone():
            raise RuntimeError("Extrusion failed.")
        return prism.Shape()

    def add_hole(self, params: Dict) -> TopoDS_Shape:
        center = np.array(params.get("center", [0, 0, 0]), dtype=float)
        if len(center) < 3:
            center = np.pad(center, (0, 3 - len(center)), mode='constant')
        radius = float(params.get("radius", 1.0))
        depth = float(params.get("depth", 20.0))
        axis = ensure_3d_vector(params.get("axis", [0, 0, 1]))
        height = depth * 1.1
        base_center = center - axis * (height / 2)
        cyl_axis = gp_Ax2(
            gp_Pnt(float(base_center[0]), float(base_center[1]), float(base_center[2])),
            gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
        )
        return BRepPrimAPI_MakeCylinder(cyl_axis, radius, height).Shape()

    def fillet(self, params: Dict):
        radius = float(params.get("radius", 1.0))
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

    def circular_pattern(self, params: Dict):
        if self.current_shape is None:
            return
        count = int(params.get("count", 4))
        center = np.array(params.get("center", [0, 0, 0]), dtype=float)
        if len(center) < 3:
            center = np.pad(center, (0, 3 - len(center)), mode='constant')
        axis = ensure_3d_vector(params.get("axis", [0, 0, 1]))
        angle = 2 * math.pi / count

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, self.current_shape)

        for i in range(1, count):
            rot_angle = i * angle
            trsf = gp_Trsf()
            ax = gp_Ax1(
                gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
                gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
            )
            trsf.SetRotation(ax, rot_angle)
            copy = BRepBuilderAPI_Copy(self.current_shape).Shape()
            copy.Move(trsf)
            builder.Add(compound, copy)

        result = None
        explorer = TopExp_Explorer(compound, TopAbs_SOLID)
        first = True
        while explorer.More():
            solid = explorer.Current()
            if first:
                result = solid
                first = False
            else:
                fuse = BRepAlgoAPI_Fuse(result, solid)
                if fuse.IsDone():
                    result = fuse.Shape()
            explorer.Next()
        if result is not None:
            self.current_shape = result

    def revolve(self, params: Dict, profile_shape: TopoDS_Shape):
        axis_dir = ensure_3d_vector(params.get("axis", [0, 0, 1]))
        axis_point = np.array(params.get("origin", [0, 0, 0]), dtype=float)
        if len(axis_point) < 3:
            axis_point = np.pad(axis_point, (0, 3 - len(axis_point)), mode='constant')
        angle = float(params.get("angle", 360.0))
        angle_rad = math.radians(angle)
        ax = gp_Ax1(
            gp_Pnt(float(axis_point[0]), float(axis_point[1]), float(axis_point[2])),
            gp_Dir(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]))
        )
        rev = BRepPrimAPI_MakeRevol(profile_shape, ax, angle_rad, False)
        if rev.IsDone():
            return rev.Shape()
        else:
            raise RuntimeError("Revolve failed.")

    def apply_action(self, action: Dict) -> Optional[TopoDS_Shape]:
        aname = action["action"]
        params = action.get("params", {})

        if aname == "CreateSketch":
            self.sketch_profile = self.create_sketch(params)
            return self.sketch_profile
        elif aname == "Extrude":
            profile = params.get("sketch_profile", self.sketch_profile)
            if profile is None:
                raise ValueError("No sketch available for extrusion.")
            operation = params.get("operation", "join")
            solid = self.extrude(params, profile)
            if operation == "join":
                self._fuse_to_current(solid)
            elif operation == "cut":
                self._cut_from_current(solid)
            else:
                self._fuse_to_current(solid)
            return solid
        elif aname == "Revolve":
            profile = params.get("sketch_profile", self.sketch_profile)
            if profile is None:
                raise ValueError("No sketch for revolve.")
            operation = params.get("operation", "join")
            solid = self.revolve(params, profile)
            if operation == "join":
                self._fuse_to_current(solid)
            elif operation == "cut":
                self._cut_from_current(solid)
            else:
                self._fuse_to_current(solid)
            return solid
        elif aname == "AddHole":
            hole = self.add_hole(params)
            self._cut_from_current(hole)
            return hole
        elif aname == "Fillet":
            self.fillet(params)
            return None
        elif aname == "CircularPattern":
            self.circular_pattern(params)
            return None
        else:
            print(f"Warning: Unsupported action '{aname}' ignored.")
            return None

    def build_from_actions(self, actions: List[Dict]) -> TopoDS_Shape:
        self.sketch_profile = None
        for act in actions:
            self.apply_action(act)
        return self.current_shape

# ----------------------------------------------------------------------
# Save STEP and main test
# ----------------------------------------------------------------------
def save_step(shape: TopoDS_Shape, filename: str):
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

def build_shape_from_action_string(action_str: str, output_step: str) -> bool:
    actions = parse_actions(action_str)
    builder = CADBuilder()
    shape = builder.build_from_actions(actions)
    if shape is None:
        print("No shape generated. Check actions.")
        return False
    return save_step(shape, output_step)

if __name__ == "__main__":
    sample_actions = (
        '<CreateSketch> plane_normal=[0,0,1]|plane_origin=[0,0,0]|profile=[{"type":"circle","center":[0,0],"radius":5}] '
        '<SEP> <Extrude> distance=10|direction=[0,0,1]|operation=join '
        '<SEP> <AddHole> center=[0,0,5]|radius=2|depth=10|axis=[0,0,1] '
        '<SEP> <Fillet> radius=1 '
        '<SEP> <EOS>'
    )
