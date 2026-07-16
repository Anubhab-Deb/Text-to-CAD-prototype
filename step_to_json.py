import os
import json
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TCollection import TCollection_AsciiString
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

class STEPAP214ToGraphConverter:
    def __init__(self, min_edge_length=0.1, min_face_area=0.01, angle_tolerance=0.1):
        self.face_id_counter = 0
        self.edge_id_counter = 0
        self.vertex_id_counter = 0
        self.product_id_counter = 0

        # Tolerance and filtering parameters
        self.min_edge_length = min_edge_length
        self.min_face_area = min_face_area
        self.angle_tolerance = angle_tolerance
        self.model_scale = 1.0
        self.step_reader = None
        
        self.max_faces = 500      # Maximum faces per model
        self.max_edges = 2000     # Maximum edges per model  
        self.max_vertices = 2000  # Maximum vertices per model
        self.max_file_size_mb = 50  # Maximum STEP file size in MB
        self.max_adjacencies = 50000  # Maximum total adjacency connections
        
    def calculate_model_scale(self, shape):
        """Calculate model scale for normalization"""
        if shape.IsNull():
            return 1.0
            
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        if bbox.IsVoid():
            return 1.0
            
        diagonal = np.sqrt(
            (bbox.CornerMax().X() - bbox.CornerMin().X())**2 +
            (bbox.CornerMax().Y() - bbox.CornerMin().Y())**2 +
            (bbox.CornerMax().Z() - bbox.CornerMin().Z())**2
        )
        self.model_scale = diagonal if diagonal > 0 else 1.0
        return self.model_scale

    def normalize_vector(self, vector):
        """Normalize vector coordinates to unit scale"""
        if self.model_scale == 0:
            return vector
        return [v / self.model_scale for v in vector]

    def normalize_point(self, point):
        """Normalize point coordinates to unit bounding box"""
        return self.normalize_vector(point)

    def read_step_file(self, filepath):
        """Read STEP file and handle AP214 specific data"""
        self.step_reader = STEPControl_Reader()
        status = self.step_reader.ReadFile(filepath)

        if status != IFSelect_RetDone:
            raise ValueError(f"Failed to read STEP file: {filepath}")

        # Transfer all roots (important for AP214 assemblies)
        self.step_reader.TransferRoots()
        shape = self.step_reader.OneShape()

        # Calculate model scale if we have valid geometry
        if not shape.IsNull():
            self.calculate_model_scale(shape)
            
        return shape

    def extract_ap214_product_structure(self):
        """Extract product structure and assembly information from AP214 file"""
        products = []
        assemblies = []
    
        if not self.step_reader:
            return products, assemblies

        try:
            # Get the STEP model and graph for traversing relationships
            ws = self.step_reader.WS()
            if not ws:
                return products, assemblies
            
            model = ws.Model()
            graph = ws.Graph()
        
            if not model:
                return products, assemblies

            # Iterate through entities to find product definitions
            for i in range(1, model.NbEntities() + 1):
                try:
                    entity = model.Value(i)
                    entity_type = entity.DynamicType().Name()
                
                    # Extract different AP214 entity types with proper handling
                    if "ProductDefinition" in entity_type:
                        product_data = self.extract_product_definition(entity, graph)
                        if product_data:
                            products.append(product_data)
                        
                    elif "ProductDefinitionShape" in entity_type:
                        shape_data = self.extract_product_definition_shape(entity, graph)
                        if shape_data:
                            products.append(shape_data)
                        
                    elif "NextAssemblyUsageOccurrence" in entity_type:
                        assembly_data = self.extract_assembly_occurrence(entity, graph)
                        if assembly_data:
                            assemblies.append(assembly_data)
                        
                    elif "ShapeDefinitionRepresentation" in entity_type:
                        rep_data = self.extract_shape_representation(entity, graph)
                        if rep_data:
                            products.append(rep_data)
                        
                except Exception as e:
                    print(f"Warning: Could not process entity {i}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error extracting AP214 product structure: {e}")
        
        return products, assemblies

    def extract_product_definition(self, product_entity, graph):
        """Extract product definition information with proper attribute extraction"""
        try:
            # Get basic product definition attributes
            product_id = getattr(product_entity, 'Id', lambda: f"PD_{self.product_id_counter}")()
            product_name = getattr(product_entity, 'Name', lambda: 'Unknown Product')()
            product_description = getattr(product_entity, 'Description', lambda: '')()
        
            # Get frame of reference if available
            frame_of_reference = ""
            try:
                if hasattr(product_entity, 'FrameOfReference'):
                    frame_ref = product_entity.FrameOfReference()
                    if frame_ref:
                        frame_of_reference = getattr(frame_ref, 'Name', lambda: '')()
            except:
                pass

            product_data = {
                'id': self.product_id_counter,
                'type': 'PRODUCT_DEFINITION',
                'name': str(product_name),
                'description': str(product_description),
                'identifier': str(product_id),
                'frame_of_reference': str(frame_of_reference),
                'entity_handle': hash(product_entity)  # For tracking relationships
            }
        
            self.product_id_counter += 1
            return product_data
        
        except Exception as e:
            print(f"Error extracting product definition: {e}")
            return None

    def extract_product_definition_shape(self, shape_entity, graph):
        """Extract product definition shape with relationships"""
        try:
            shape_name = getattr(shape_entity, 'Name', lambda: 'Product Definition Shape')()
            shape_description = getattr(shape_entity, 'Description', lambda: '')()
        
            # Extract related product definition if available
            related_products = []
            try:
                if hasattr(shape_entity, 'Definition'):
                    definition = shape_entity.Definition()
                    if definition:
                        product_ref = getattr(definition, 'Product', None)
                        if product_ref:
                            product_id = getattr(product_ref, 'Id', lambda: 'Unknown')()
                            related_products.append(str(product_id))
            except:
                pass

            product_data = {
                'id': self.product_id_counter,
                'type': 'PRODUCT_DEFINITION_SHAPE',
                'name': str(shape_name),
                'description': str(shape_description),
                'identifier': f"PDS_{self.product_id_counter}",
                'related_products': related_products,
                'entity_handle': hash(shape_entity)
            }
        
            self.product_id_counter += 1
            return product_data
        
        except Exception as e:
            print(f"Error extracting product definition shape: {e}")
            return None

    def extract_assembly_occurrence(self, assembly_entity, graph):
        """Extract assembly occurrence data with component relationships"""
        try:
            occurrence_id = getattr(assembly_entity, 'Id', lambda: f"ASSY_{self.product_id_counter}")()
            occurrence_name = getattr(assembly_entity, 'Name', lambda: 'Assembly Occurrence')()
        
            related_products = []
        
            # Extract relating product (parent assembly)
            try:
                if hasattr(assembly_entity, 'RelatingProductDefinition'):
                    relating_product = assembly_entity.RelatingProductDefinition()
                    if relating_product:
                        product_ref = getattr(relating_product, 'Product', None)
                        if product_ref:
                            product_id = getattr(product_ref, 'Id', lambda: 'Unknown')()
                            related_products.append(f"parent:{product_id}")
            except:
                pass
            
            # Extract related product (child component)
            try:
                if hasattr(assembly_entity, 'RelatedProductDefinition'):
                    related_product = assembly_entity.RelatedProductDefinition()
                    if related_product:
                        product_ref = getattr(related_product, 'Product', None)
                        if product_ref:
                            product_id = getattr(product_ref, 'Id', lambda: 'Unknown')()
                            related_products.append(f"child:{product_id}")
            except:
                pass

            assembly_data = {
                'id': self.product_id_counter,
                'type': 'ASSEMBLY_OCCURRENCE',
                'name': str(occurrence_name),
                'description': f"Assembly occurrence {occurrence_id}",
                'identifier': str(occurrence_id),
                'related_products': related_products,
                'entity_handle': hash(assembly_entity)
            }
        
            self.product_id_counter += 1
            return assembly_data
        
        except Exception as e:
            print(f"Error extracting assembly occurrence: {e}")
            return None

    def extract_shape_representation(self, rep_entity, graph):
        """Extract shape representation with items"""
        try:
            rep_name = getattr(rep_entity, 'Name', lambda: 'Shape Representation')()
        
            # Extract representation items
            items = []
            try:
                if hasattr(rep_entity, 'Items'):
                    rep_items = rep_entity.Items()
                    if rep_items:
                        for j in range(1, rep_items.Length() + 1):
                            item = rep_items.Value(j)
                            item_type = item.DynamicType().Name()
                            items.append({
                                'type': str(item_type),
                                'handle': hash(item)
                            })
            except:
                pass

            rep_data = {
                'id': self.product_id_counter,
                'type': 'SHAPE_REPRESENTATION',
                'name': str(rep_name),
                'description': 'Shape representation with geometric items',
                'identifier': f"REP_{self.product_id_counter}",
                'items': items,
                'entity_handle': hash(rep_entity)
            }
        
            self.product_id_counter += 1
            return rep_data
        
        except Exception as e:
            print(f"Error extracting shape representation: {e}")
            return None

    def get_face_surface_type(self, face):
        """Determine surface type with enhanced geometric analysis"""
        try:
            surf = BRepAdaptor_Surface(face)
            surf_type = surf.GetType()  # This returns GeomAbs_SurfaceType enum

            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()

            # Filter tiny faces early
            if area < self.min_face_area:
                return None

            centroid = props.CentreOfMass()
            normalized_centroid = self.normalize_point([centroid.X(), centroid.Y(), centroid.Z()])

            face_data = {
                'surface_type': 'UNKNOWN',
                'parameters': {},
                'area': area,
                'centroid': normalized_centroid,
                'normal': [0.0, 0.0, 1.0]
            }

            # GeomAbs_Plane = 0, GeomAbs_Cylinder = 1, GeomAbs_Cone = 2, GeomAbs_Sphere = 3, 
            # GeomAbs_Torus = 4, GeomAbs_BezierSurface = 5, GeomAbs_BSplineSurface = 6, 
            # GeomAbs_SurfaceOfRevolution = 7, GeomAbs_SurfaceOfExtrusion = 8, 
            # GeomAbs_OffsetSurface = 9, GeomAbs_OtherSurface = 10

            if surf_type == 0:  # Plane
                try:
                    plane = surf.Plane()
                    location = plane.Location()
                    normal = plane.Axis().Direction()
                    normalized_normal = [normal.X(), normal.Y(), normal.Z()]

                    face_data['surface_type'] = 'PLANE'
                    face_data['parameters'] = {
                        'normal': normalized_normal,
                        'point': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                    face_data['normal'] = normalized_normal
                except:
                    face_data['surface_type'] = 'PLANE_SIMPLE'

            elif surf_type == 1:  # Cylinder
                try:
                    cylinder = surf.Cylinder()
                    location = cylinder.Location()
                    axis = cylinder.Axis().Direction()
                    axis_vector = [axis.X(), axis.Y(), axis.Z()]

                    face_data['surface_type'] = 'CYLINDER'
                    face_data['parameters'] = {
                        'axis': axis_vector,
                        'radius': cylinder.Radius() / self.model_scale,
                        'center': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                    face_data['normal'] = axis_vector
                except:
                    face_data['surface_type'] = 'CYLINDER_SIMPLE'

            elif surf_type == 2:  # Cone
                try:
                    cone = surf.Cone()
                    location = cone.Location()
                    axis = cone.Axis().Direction()

                    face_data['surface_type'] = 'CONE'
                    face_data['parameters'] = {
                        'axis': [axis.X(), axis.Y(), axis.Z()],
                        'angle': cone.SemiAngle(),
                        'radius': cone.RefRadius() / self.model_scale,
                        'center': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                    face_data['normal'] = [axis.X(), axis.Y(), axis.Z()]
                except:
                    face_data['surface_type'] = 'CONE_SIMPLE'

            elif surf_type == 3:  # Sphere
                try:
                    sphere = surf.Sphere()
                    location = sphere.Location()

                    face_data['surface_type'] = 'SPHERE'
                    face_data['parameters'] = {
                        'radius': sphere.Radius() / self.model_scale,
                        'center': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                except:
                    face_data['surface_type'] = 'SPHERE_SIMPLE'

            elif surf_type in [5, 6]:  # Bezier or BSpline surface
                face_data['surface_type'] = 'SPLINE_SURFACE'
                try:
                    face_data['parameters'] = {
                        'u_degree': getattr(surf, 'UDegree', lambda: 0)(),
                        'v_degree': getattr(surf, 'VDegree', lambda: 0)(),
                        'nb_u_poles': getattr(surf, 'NbUPoles', lambda: 0)(),
                        'nb_v_poles': getattr(surf, 'NbVPoles', lambda: 0)()
                    }
                except:
                    pass
    
            else:
                face_data['surface_type'] = f'SURFACE_TYPE_{surf_type}'

            return face_data

        except Exception as e:
            print(f"Error processing face surface: {e}")
            # Return minimal data
            try:
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face, props)
                area = props.Mass()
            
                if area < self.min_face_area:
                    return None
                
                centroid = props.CentreOfMass()
                return {
                    'surface_type': 'PROCESSING_ERROR',
                    'parameters': {'error': str(e)},
                    'area': area,
                    'centroid': self.normalize_point([centroid.X(), centroid.Y(), centroid.Z()]),
                    'normal': [0, 0, 1]
                }
            except:
                return None

    def get_edge_curve_type(self, edge):
        """Determine curve type with enhanced geometric analysis"""
        try:
            curve = BRepAdaptor_Curve(edge)
            curve_type = curve.GetType()
    
            props = GProp_GProps()
            brepgprop.LinearProperties(edge, props)
            length = props.Mass()

            # Filter tiny edges early
            if length < self.min_edge_length:
                return None

            edge_data = {
                'curve_type': 'UNKNOWN_CURVE',
                'parameters': {},
                'length': length / self.model_scale,
                'start_point': [0.0, 0.0, 0.0],
                'end_point': [0.0, 0.0, 0.0],
                'direction': [0.0, 0.0, 1.0]
            }

            # Get start and end points
            try:
                start_point = curve.Value(curve.FirstParameter())
                end_point = curve.Value(curve.LastParameter())
                edge_data['start_point'] = self.normalize_point([start_point.X(), start_point.Y(), start_point.Z()])
                edge_data['end_point'] = self.normalize_point([end_point.X(), end_point.Y(), end_point.Z()])
            except Exception as e:
                print(f"Warning: Could not get edge endpoints: {e}")

            # CORRECTED: Use proper GeomAbs_CurveType enum values
            # GeomAbs_Line = 0, GeomAbs_Circle = 1, GeomAbs_Ellipse = 2, GeomAbs_Hyperbola = 3, 
            # GeomAbs_Parabola = 4, GeomAbs_BezierCurve = 5, GeomAbs_BSplineCurve = 6, 
            # GeomAbs_OtherCurve = 7
        
            if curve_type == 0:  # GeomAbs_Line
                try:
                    line = curve.Line()
                    direction = line.Direction()
                    dir_vector = [direction.X(), direction.Y(), direction.Z()]

                    edge_data['curve_type'] = 'LINE'
                    edge_data['parameters'] = {
                        'direction': dir_vector,
                        'point': edge_data['start_point']
                    }
                    edge_data['direction'] = dir_vector
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    edge_data['curve_type'] = 'LINE_SIMPLE'
                    edge_data['parameters'] = {'simple_line': True}

            elif curve_type == 1:  # GeomAbs_Circle
                try:
                    circle = curve.Circle()
                    location = circle.Location()
                    axis = circle.Axis().Direction()
                    axis_vector = [axis.X(), axis.Y(), axis.Z()]

                    edge_data['curve_type'] = 'CIRCLE'
                    edge_data['parameters'] = {
                        'axis': axis_vector,
                        'radius': circle.Radius() / self.model_scale,
                        'center': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                    edge_data['direction'] = axis_vector
                except Exception as e:
                    print(f"Warning: Error processing circle: {e}")
                    edge_data['curve_type'] = 'CIRCLE_SIMPLE'

            elif curve_type == 2:  # GeomAbs_Ellipse
                try:
                    ellipse = curve.Ellipse()
                    location = ellipse.Location()
                    axis = ellipse.Axis().Direction()

                    edge_data['curve_type'] = 'ELLIPSE'
                    edge_data['parameters'] = {
                        'axis': [axis.X(), axis.Y(), axis.Z()],
                        'major_radius': ellipse.MajorRadius() / self.model_scale,
                        'minor_radius': ellipse.MinorRadius() / self.model_scale,
                        'center': self.normalize_point([location.X(), location.Y(), location.Z()])
                    }
                    edge_data['direction'] = [axis.X(), axis.Y(), axis.Z()]
                except Exception as e:
                    print(f"Warning: Error processing ellipse: {e}")
                    edge_data['curve_type'] = 'ELLIPSE_SIMPLE'

            elif curve_type in [5, 6]:  # GeomAbs_BezierCurve or GeomAbs_BSplineCurve
                edge_data['curve_type'] = 'SPLINE'
                try:
                    # Try to get approximate shape info
                    points = []
                    num_samples = min(10, int(curve.LastParameter() - curve.FirstParameter()))
                    for i in range(num_samples):
                        t = curve.FirstParameter() + i * (curve.LastParameter() - curve.FirstParameter()) / (num_samples - 1)
                        point = curve.Value(t)
                        points.append(self.normalize_point([point.X(), point.Y(), point.Z()]))
                    
                    edge_data['parameters'] = {
                        'sample_points': points,
                        'is_rational': hasattr(curve, 'IsRational') and curve.IsRational(),
                        'degree': getattr(curve, 'Degree', lambda: 0)() if hasattr(curve, 'Degree') else 0
                    }
                except:
                    edge_data['parameters'] = {'spline_type': 'complex'}

            else:
                # For any other curve type, try to at least identify it
                edge_data['curve_type'] = f'CURVE_TYPE_{curve_type}'
                edge_data['parameters'] = {'original_type': str(curve_type)}

            return edge_data

        except Exception as e:
            print(f"Error processing edge curve: {e}")
            # Return minimal data instead of None to preserve the edge
            try:
                # Try to get at least length and endpoints
                props = GProp_GProps()
                brepgprop.LinearProperties(edge, props)
                length = props.Mass()
            
                if length < self.min_edge_length:
                    return None
                
                return {
                    'curve_type': 'PROCESSING_ERROR',
                    'parameters': {'error': str(e)},
                    'length': length / self.model_scale,
                    'start_point': [0, 0, 0],
                    'end_point': [0, 0, 0],
                    'direction': [0, 0, 1]
                }
            except:
                return None

    def extract_vertices(self, shape):
        """Extract all vertices from the shape"""
        vertices = []
        
        if shape.IsNull():
            return vertices

        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

        while vertex_explorer.More():
            vertex = topods_Vertex(vertex_explorer.Current())
            vertex_data = self.get_vertex_properties(vertex)

            if vertex_data is not None:
                vertex_data['id'] = self.vertex_id_counter
                vertices.append(vertex_data)
                self.vertex_id_counter += 1

            vertex_explorer.Next()

        return vertices

    def get_vertex_properties(self, vertex):
        """Extract properties for a vertex node"""
        try:
            point = BRep_Tool.Pnt(vertex)
            normalized_point = self.normalize_point([point.X(), point.Y(), point.Z()])

            vertex_data = {
                'type': 'VERTEX',
                'point': normalized_point,
                'x': normalized_point[0],
                'y': normalized_point[1],
                'z': normalized_point[2]
            }
            return vertex_data
        except Exception as e:
            print(f"Error processing vertex: {e}")
            return None

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def find_closest_vertex_by_coords(self, point, vertices, tolerance=0.001):
        """Find vertex closest to given coordinates with better matching"""
        min_distance = float('inf')
        closest_vertex_id = None
    
        for vertex in vertices:
            distance = self.calculate_distance(point, vertex['point'])
            if distance < min_distance:
                min_distance = distance
                closest_vertex_id = vertex['id']
    
        # Be more generous with tolerance for edge endpoints
        return closest_vertex_id if min_distance < tolerance else None

    def build_complete_topology_graph(self, faces, edges, vertices, shape):
        """Build complete topology graph using OpenCASCADE topological analysis"""
        topology = {
            'face_edge_adjacency': [[], []],
            'edge_vertex_adjacency': [[], []],
            'vertex_face_adjacency': [[], []],
            'edge_edge_adjacency': [[], []],
            'face_face_adjacency': [[], []]
        }

        print("Building topology using OpenCASCADE topological analysis...")

        # 1. Build edge-vertex adjacency using OpenCASCADE
        print(f"Building edge-vertex adjacency for {len(edges)} edges and {len(vertices)} vertices")
        edge_vertex_map = self._build_edge_vertex_topology(shape, edges, vertices)
        for edge_id, vertex_ids in edge_vertex_map.items():
            for vertex_id in vertex_ids:
                topology['edge_vertex_adjacency'][0].append(edge_id)
                topology['edge_vertex_adjacency'][1].append(vertex_id)

        print(f"  Edge-vertex: {len(topology['edge_vertex_adjacency'][0])} connections")

        # 2. Build face-edge adjacency using OpenCASCADE
        print(f"Building face-edge adjacency for {len(faces)} faces and {len(edges)} edges")
        face_edge_map = self._build_face_edge_topology(shape, faces, edges)
        for face_id, edge_ids in face_edge_map.items():
            for edge_id in edge_ids:
                topology['face_edge_adjacency'][0].append(face_id)
                topology['face_edge_adjacency'][1].append(edge_id)

        print(f"  Face-edge: {len(topology['face_edge_adjacency'][0])} connections")

        # 3. Build vertex-face adjacency using OpenCASCADE (REPLACED geometric proximity)
        print(f"Building vertex-face adjacency for {len(vertices)} vertices and {len(faces)} faces")
        vertex_face_map = self._build_vertex_face_topology(shape, vertices, faces)
        for vertex_id, face_ids in vertex_face_map.items():
            for face_id in face_ids:
                topology['vertex_face_adjacency'][0].append(vertex_id)
                topology['vertex_face_adjacency'][1].append(face_id)

        print(f"  Vertex-face: {len(topology['vertex_face_adjacency'][0])} connections")

        # 4. Build face-face adjacency (faces sharing edges) - improved
        print(f"Building face-face adjacency for {len(faces)} faces")
        face_face_connections = self._build_face_face_adjacency_topology(shape, faces)
        topology['face_face_adjacency'] = face_face_connections
        print(f"  Face-face: {len(face_face_connections[0])} connections")

        # 5. Build edge-edge adjacency (edges sharing vertices) - improved
        print(f"Building edge-edge adjacency")
        edge_edge_connections = self._build_edge_edge_adjacency_topology(shape, edges)
        topology['edge_edge_adjacency'] = edge_edge_connections
        print(f"  Edge-edge: {len(edge_edge_connections[0])} connections")

        # Final summary
        print(f"Topology construction complete:")
        print(f"  face_edge: {len(topology['face_edge_adjacency'][0])} connections")
        print(f"  edge_vertex: {len(topology['edge_vertex_adjacency'][0])} connections") 
        print(f"  vertex_face: {len(topology['vertex_face_adjacency'][0])} connections")
        print(f"  face_face: {len(topology['face_face_adjacency'][0])} connections")
        print(f"  edge_edge: {len(topology['edge_edge_adjacency'][0])} connections")

        return topology
    
    def _build_vertex_face_topology(self, shape, vertices, faces):
        """Build vertex-face topology using OpenCASCADE topological exploration"""
        vertex_face_map = {}
    
        # Explore vertices in the shape
        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        vertex_counter = 0
    
        while vertex_explorer.More():
            occ_vertex = topods_Vertex(vertex_explorer.Current())
            vertex_point = BRep_Tool.Pnt(occ_vertex)
            normalized_point = self.normalize_point([vertex_point.X(), vertex_point.Y(), vertex_point.Z()])
        
            # Find matching vertex in our list
            vertex_id = self._find_matching_vertex(normalized_point, vertices)
        
            if vertex_id is not None:
                vertex_faces = []
            
                # Explore all faces in the shape to find which ones contain this vertex
                face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
                face_counter = 0
            
                while face_explorer.More():
                    occ_face = topods_Face(face_explorer.Current())
                
                    # Check if vertex belongs to this face using OpenCASCADE topology
                    if self._vertex_belongs_to_face(occ_vertex, occ_face):
                        # Find matching face in our list
                        face_centroid = self._get_face_centroid(occ_face)
                        face_id = self._find_matching_face_by_centroid(face_centroid, faces)
                        if face_id is not None:
                            vertex_faces.append(face_id)
                
                    face_counter += 1
                    face_explorer.Next()
            
                if vertex_faces:
                    vertex_face_map[vertex_id] = vertex_faces
        
            vertex_counter += 1
            vertex_explorer.Next()
    
        return vertex_face_map
    
    def _vertex_belongs_to_face(self, vertex, face):
        """Check if a vertex belongs to a face using OpenCASCADE topology"""
        try:
            # Method 1: Check if vertex is in face's vertices (topological check)
            vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
            while vertex_explorer.More():
                face_vertex = topods_Vertex(vertex_explorer.Current())
                if face_vertex.IsSame(vertex):
                    return True
                vertex_explorer.Next()
    
            # Method 2: Check if vertex is geometrically on the face using BRepExtrema
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
        
            dist_checker = BRepExtrema_DistShapeShape(vertex, face)
            if dist_checker.IsDone() and dist_checker.NbSolution() > 0:
                distance = dist_checker.Value()
                return distance < 1e-6  # Very small tolerance
        
            return False
    
        except Exception as e:
            print(f"Error checking vertex-face membership: {e}")
            return False
    
    def _get_face_centroid(self, face):
        """Calculate centroid of a face using OpenCASCADE"""
        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            centroid = props.CentreOfMass()
            return self.normalize_point([centroid.X(), centroid.Y(), centroid.Z()])
        except:
            return [0, 0, 0]
    
    def _find_matching_face_by_centroid(self, centroid, faces, tolerance=0.001):
        """Find face matching given centroid coordinates"""
        for face in faces:
            distance = self.calculate_distance(centroid, face['centroid'])
            if distance < tolerance:
                return face['id']
        return None
    
    def _build_edge_vertex_topology(self, shape, edges, vertices):
        """Build edge-vertex topology using OpenCASCADE"""
        edge_vertex_map = {}
    
        # Explore edges in the shape
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_counter = 0
    
        while edge_explorer.More():
            occ_edge = topods_Edge(edge_explorer.Current())
        
            # Get vertices from this edge
            vertex_explorer = TopExp_Explorer(occ_edge, TopAbs_VERTEX)
            edge_vertices = []
        
            while vertex_explorer.More():
                occ_vertex = topods_Vertex(vertex_explorer.Current())
                vertex_point = BRep_Tool.Pnt(occ_vertex)
                normalized_point = self.normalize_point([vertex_point.X(), vertex_point.Y(), vertex_point.Z()])
            
                # Find matching vertex in our list
                vertex_id = self._find_matching_vertex(normalized_point, vertices)
                if vertex_id is not None:
                    edge_vertices.append(vertex_id)
            
                vertex_explorer.Next()
        
            if edge_counter < len(edges) and edge_vertices:
                edge_vertex_map[edge_counter] = edge_vertices
        
            edge_counter += 1
            edge_explorer.Next()
    
        return edge_vertex_map

    def _build_face_edge_topology(self, shape, faces, edges):
        """Build face-edge topology using OpenCASCADE"""
        face_edge_map = {}
    
        # Explore faces in the shape
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_counter = 0
    
        while face_explorer.More():
            occ_face = topods_Face(face_explorer.Current())
            face_edges = []
        
            # Get edges from this face
            edge_explorer = TopExp_Explorer(occ_face, TopAbs_EDGE)
        
            while edge_explorer.More():
                occ_edge = topods_Edge(edge_explorer.Current())
            
                # Get edge curve for matching
                try:
                    curve = BRepAdaptor_Curve(occ_edge)
                    start_point = curve.Value(curve.FirstParameter())
                    end_point = curve.Value(curve.LastParameter())
                
                    start_normalized = self.normalize_point([start_point.X(), start_point.Y(), start_point.Z()])
                    end_normalized = self.normalize_point([end_point.X(), end_point.Y(), end_point.Z()])
                
                    # Find matching edge in our list
                    edge_id = self._find_matching_edge(start_normalized, end_normalized, edges)
                    if edge_id is not None:
                        face_edges.append(edge_id)
                    
                except:
                    pass
            
                edge_explorer.Next()
        
            if face_counter < len(faces) and face_edges:
                face_edge_map[face_counter] = face_edges
        
            face_counter += 1
            face_explorer.Next()
    
        return face_edge_map

    def _build_face_face_adjacency_topology(self, shape, faces):
        """Build face-face adjacency using OpenCASCADE topological analysis"""
        face_face_pairs = [[], []]
    
        # Create a map from face centroids to face IDs for lookup
        centroid_to_face = {}
        for face in faces:
            centroid_to_face[tuple(face['centroid'])] = face['id']
    
        # Explore faces and find adjacent faces through shared edges
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_counter = 0
    
        while face_explorer.More():
            occ_face1 = topods_Face(face_explorer.Current())
            centroid1 = self._get_face_centroid(occ_face1)
            face1_id = self._find_matching_face_by_centroid(centroid1, faces)
        
            if face1_id is not None:
                # Find faces that share edges with this face
                adjacent_faces = self._get_adjacent_faces(occ_face1, shape)
            
                for occ_face2 in adjacent_faces:
                    centroid2 = self._get_face_centroid(occ_face2)
                    face2_id = self._find_matching_face_by_centroid(centroid2, faces)
                
                    if face2_id is not None and face1_id != face2_id:
                        face_face_pairs[0].append(face1_id)
                        face_face_pairs[1].append(face2_id)
        
            face_counter += 1
            face_explorer.Next()
    
        return face_face_pairs

    def _get_adjacent_faces(self, face, shape):
        """Get faces adjacent to given face through shared edges using OpenCASCADE"""
        adjacent_faces = []
    
        try:
            # Get all edges of the face
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            edges = []
            while edge_explorer.More():
                edges.append(topods_Edge(edge_explorer.Current()))
                edge_explorer.Next()
        
            # For each edge, find other faces that share it
            for edge in edges:
                face_explorer_shape = TopExp_Explorer(shape, TopAbs_FACE)
                while face_explorer_shape.More():
                    other_face = topods_Face(face_explorer_shape.Current())
                    if not other_face.IsSame(face):
                        # Check if this face also contains the edge
                        edge_explorer_other = TopExp_Explorer(other_face, TopAbs_EDGE)
                        while edge_explorer_other.More():
                            other_edge = topods_Edge(edge_explorer_other.Current())
                            if other_edge.IsSame(edge):
                                adjacent_faces.append(other_face)
                                break
                            edge_explorer_other.Next()
                    face_explorer_shape.Next()
                
        except Exception as e:
            print(f"Error finding adjacent faces: {e}")
    
        return adjacent_faces

    def _build_edge_edge_adjacency_topology(self, shape, edges):
        """Build edge-edge adjacency using OpenCASCADE topological analysis"""
        edge_edge_pairs = [[], []]
    
        # Explore edges in the shape
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edge_counter = 0
    
        while edge_explorer.More():
            occ_edge1 = topods_Edge(edge_explorer.Current())
        
            # Get start and end points for matching
            try:
                curve1 = BRepAdaptor_Curve(occ_edge1)
                start1 = curve1.Value(curve1.FirstParameter())
                end1 = curve1.Value(curve1.LastParameter())
                start_normalized1 = self.normalize_point([start1.X(), start1.Y(), start1.Z()])
                end_normalized1 = self.normalize_point([end1.X(), end1.Y(), end1.Z()])
            
                edge1_id = self._find_matching_edge(start_normalized1, end_normalized1, edges)
            
                if edge1_id is not None:
                    # Find edges that share vertices with this edge
                    adjacent_edges = self._get_adjacent_edges(occ_edge1, shape)
                
                    for occ_edge2 in adjacent_edges:
                        curve2 = BRepAdaptor_Curve(occ_edge2)
                        start2 = curve2.Value(curve2.FirstParameter())
                        end2 = curve2.Value(curve2.LastParameter())
                        start_normalized2 = self.normalize_point([start2.X(), start2.Y(), start2.Z()])
                        end_normalized2 = self.normalize_point([end2.X(), end2.Y(), end2.Z()])
                    
                        edge2_id = self._find_matching_edge(start_normalized2, end_normalized2, edges)
                    
                        if edge2_id is not None and edge1_id != edge2_id:
                            edge_edge_pairs[0].append(edge1_id)
                            edge_edge_pairs[1].append(edge2_id)
                        
            except Exception as e:
                print(f"Error processing edge {edge_counter}: {e}")
        
            edge_counter += 1
            edge_explorer.Next()
    
        return edge_edge_pairs

    def _get_adjacent_edges(self, edge, shape):
        """Get edges adjacent to given edge through shared vertices using OpenCASCADE"""
        adjacent_edges = []
    
        try:
            # Get vertices of the edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            vertices = []
            while vertex_explorer.More():
                vertices.append(topods_Vertex(vertex_explorer.Current()))
                vertex_explorer.Next()
        
            # For each vertex, find other edges that share it
            for vertex in vertices:
                edge_explorer_shape = TopExp_Explorer(shape, TopAbs_EDGE)
                while edge_explorer_shape.More():
                    other_edge = topods_Edge(edge_explorer_shape.Current())
                    if not other_edge.IsSame(edge):
                        # Check if this edge also contains the vertex
                        vertex_explorer_other = TopExp_Explorer(other_edge, TopAbs_VERTEX)
                        while vertex_explorer_other.More():
                            other_vertex = topods_Vertex(vertex_explorer_other.Current())
                            if other_vertex.IsSame(vertex):
                                adjacent_edges.append(other_edge)
                                break
                            vertex_explorer_other.Next()
                    edge_explorer_shape.Next()
                
        except Exception as e:
            print(f"Error finding adjacent edges: {e}")
    
        return adjacent_edges

    def _find_matching_vertex(self, point, vertices, tolerance=0.001):
        """Find vertex matching given coordinates"""
        for vertex in vertices:
            distance = self.calculate_distance(point, vertex['point'])
            if distance < tolerance:
                return vertex['id']
        return None

    def _find_matching_edge(self, start_point, end_point, edges, tolerance=0.001):
        """Find edge matching given start and end points"""
        for edge in edges:
            start_dist = self.calculate_distance(start_point, edge['start_point'])
            end_dist = self.calculate_distance(end_point, edge['end_point'])
        
            # Check both directions (edge could be reversed)
            start_dist_rev = self.calculate_distance(start_point, edge['end_point'])
            end_dist_rev = self.calculate_distance(end_point, edge['start_point'])
        
            if (start_dist < tolerance and end_dist < tolerance) or (start_dist_rev < tolerance and end_dist_rev < tolerance):
                return edge['id']
        return None

    def extract_geometry_only(self, shape):
        """Extract faces, edges, vertices without building topology (fast)."""
        faces = []
        edges = []
        vertices = []

        if shape.IsNull():
            return faces, edges, vertices

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods_Face(face_explorer.Current())
            face_data = self.get_face_surface_type(face)
            if face_data is not None:
                face_data['id'] = self.face_id_counter
                faces.append(face_data)
                self.face_id_counter += 1
            face_explorer.Next()

        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods_Edge(edge_explorer.Current())
            edge_data = self.get_edge_curve_type(edge)
            if edge_data is not None:
                edge_data['id'] = self.edge_id_counter
                edges.append(edge_data)
                self.edge_id_counter += 1
            edge_explorer.Next()

        vertices = self.extract_vertices(shape)

        return faces, edges, vertices

    def extract_brep_geometry(self, shape, build_topology=True):
        """Extract B-Rep geometry. If build_topology=False, skip topology construction."""
        faces, edges, vertices = self.extract_geometry_only(shape)

        if not build_topology:
            return faces, edges, vertices, {}

        # Build topology only if requested
        topology_graph = self.build_complete_topology_graph(faces, edges, vertices, shape)
        return faces, edges, vertices, topology_graph

    def convert_step_to_json(self, step_file_path, output_dir):
        try:
            print(f"Processing: {step_file_path}")

            # File size check
            file_size_mb = os.path.getsize(step_file_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                print(f"  ⚠ SKIPPING: File too large ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB)")
                return False

            # Reset counters
            self.face_id_counter = 0
            self.edge_id_counter = 0
            self.vertex_id_counter = 0
            self.product_id_counter = 0

            # Read STEP file
            shape = self.read_step_file(step_file_path)
        
            # Extract AP214 specific data
            products, assemblies = self.extract_ap214_product_structure()
        
            # Extract B-Rep geometry if available
            if not shape.IsNull():
                # FIRST: extract geometry without topology (cheap)
                faces, edges, vertices = self.extract_geometry_only(shape)
            
                # Check complexity using counts
                if len(faces) > self.max_faces or len(edges) > self.max_edges or len(vertices) > self.max_vertices:
                    print(f"  ⚠ SKIPPING: Model too complex (F:{len(faces)}, E:{len(edges)}, V:{len(vertices)})")
                    return False

                # SECOND: now build the topology graph (expensive, but only if counts are acceptable)
                topology_graph = self.build_complete_topology_graph(faces, edges, vertices, shape)

                # Optional: also check adjacency count after building
                total_adj = (len(topology_graph['face_edge_adjacency'][0]) +
                             len(topology_graph['edge_vertex_adjacency'][0]) +
                             len(topology_graph['vertex_face_adjacency'][0]) +
                             len(topology_graph['edge_edge_adjacency'][0]) +
                             len(topology_graph['face_face_adjacency'][0]))
                if total_adj > self.max_adjacencies:
                    print(f"  ⚠ SKIPPING: Too many adjacencies ({total_adj} > {self.max_adjacencies})")
                    return False
            else:
                faces, edges, vertices, topology_graph = [], [], [], {}

            # Create AP214 output structure
            output_data = {
                'source_file': os.path.basename(step_file_path),
                'step_protocol': 'AP214',
                'metadata': {
                    'total_products': len(products),
                    'total_assemblies': len(assemblies),
                    'total_faces': len(faces),
                    'total_edges': len(edges),
                    'total_vertices': len(vertices),
                    'has_geometry': not shape.IsNull(),
                    'model_scale': self.model_scale
                },
                'product_structure': {
                    'products': products,
                    'assemblies': assemblies
                },
                'geometry_data': {
                    'faces': faces,
                    'edges': edges,
                    'vertices': vertices
                },
                'topology_graph': topology_graph,
                'processing_info': {
                    'conversion_timestamp': np.datetime64('now').astype(str)
                }
            }

            # Save to JSON file
            base_name = os.path.splitext(os.path.basename(step_file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}.json")

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=self.json_serializer)

            print(f"Successfully converted AP214: {base_name} - {len(products)} products, {len(faces)} faces")
            return True

        except Exception as e:
            print(f"Error converting {step_file_path}: {e}")
            return False

    def json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (gp_Pnt, gp_Dir)):
            return [obj.X(), obj.Y(), obj.Z()]
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def batch_convert_step_files(self, input_dir, output_dir):
        """Convert all STEP files in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        step_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.step') or f.lower().endswith('.stp')]

        print(f"Found {len(step_files)} STEP files to convert")
        print(f"Processing as STEP AP214 files")

        success_count = 0
        failed_files = []

        for step_file in step_files:
            step_path = os.path.join(input_dir, step_file)
            success = self.convert_step_to_json(step_path, output_dir)

            if success:
                success_count += 1
            else:
                failed_files.append(step_file)

        print(f"\nAP214 Conversion Summary:")
        print(f"Successfully converted: {success_count}/{len(step_files)}")
        if failed_files:
            print(f"Failed files: {failed_files}")

def main():
    INPUT_DIR = "/media/anubhab/External/Project/input_dir"
    OUTPUT_DIR = "/media/anubhab/External/Project/graph_data"

    converter = STEPAP214ToGraphConverter(
        min_edge_length=0.1,
        min_face_area=0.01,
        angle_tolerance=0.1
    )
    converter.max_faces = 500
    converter.max_edges = 1000
    converter.max_vertices = 2000
    converter.max_file_size_mb = 20
    converter.max_adjacencies = 50000   # reject if total adjacency connections exceed 50000
    
    converter.batch_convert_step_files(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
