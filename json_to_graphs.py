"""
PHASE 2: Hybrid Graph Construction - AP214 COMPATIBLE VERSION
Modified to work with AP214 Phase 1 output structure
"""
import gc
import json
import numpy as np
import torch
from torch_geometric.data import Data
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

class HybridGraphBuilder:
    def __init__(self, angle_tolerance=1.0, distance_tolerance=0.01):
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance

        # Enhanced feature dimensions for hybrid graph
        self.face_feature_dim = 32
        self.edge_feature_dim = 32
        self.vertex_feature_dim = 20
        self.relationship_feature_dim = 16
    def estimate_total_edges(self, entities, topology_graph):
        """Estimate total edges to prevent memory explosion"""
        total_edges = 0
    
        # Count from each topology section
        for section in ['face_edge_adjacency', 'edge_vertex_adjacency', 
                    'vertex_face_adjacency', 'edge_edge_adjacency', 'face_face_adjacency']:
            if section in topology_graph:
                # Each adjacency pair creates 2 edges (undirected)
                total_edges += len(topology_graph[section][0]) * 2
    
        return total_edges
    def create_minimal_graph(self, phase1_data):
        """Create minimal valid graph for skipped files"""
        # Create a tiny valid graph to maintain pipeline
        x = torch.tensor([[0.0] * 32], dtype=torch.float)  # 1 node with 32 features
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop
        edge_attr = torch.tensor([[0.0] * 16], dtype=torch.float)  # 1 edge with 16 features
    
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_faces=torch.tensor([0]),
            num_edges=torch.tensor([0]),
            num_vertices=torch.tensor([1]),
            node_entity_mapping=torch.tensor([[0, 0, 1, 0]]),  # vertex type
            edge_type_mapping=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),  # edge_edge type
            node_type_mapping_str=json.dumps(['vertex']),
            edge_type_mapping_str=json.dumps(['edge_edge']),
            source_file=phase1_data.get('source_file', 'unknown'),
            skipped=True,
            skip_reason="Graph too large for memory"
        )
    
        return graph_data
    def should_skip_graph(self, entities, topology_graph):
        """Determine if graph should be skipped due to size"""
        total_entities = len(entities['faces']) + len(entities['edges']) + len(entities['vertices'])
        estimated_edges = self.estimate_total_edges(entities, topology_graph)
    
        # Skip criteria
        if total_entities > 8000:
            return True, f"Too many entities ({total_entities})"
        if estimated_edges > 50000:
            return True, f"Too many edges (estimated {estimated_edges})"
    
        return False, "OK"
        
    def debug_all_feature_dimensions(self, entities):
        """Debug method to check ALL feature dimensions"""
        print("=== DEBUGGING ALL FEATURE DIMENSIONS ===")

        face_dims = []
        for i, face in enumerate(entities['faces']):
            features = self.build_face_node_features(face)
            face_dims.append(len(features))
            if i < 5:  # Print first 5
                print(f"Face {i}: {len(features)} features")

        edge_dims = []
        for i, edge in enumerate(entities['edges']):
            features = self.build_edge_node_features(edge)
            edge_dims.append(len(features))
            if i < 5:  # Print first 5
                print(f"Edge {i}: {len(features)} features")

        vertex_dims = []
        for i, vertex in enumerate(entities['vertices']):
            # Use default/empty values for debugging since we don't have the full context
            features = self.build_vertex_node_features(
                vertex, 
                vertex_idx=i,
                entities=entities,
                topology_graph={'edge_vertex_adjacency': [[], []], 'vertex_face_adjacency': [[], []]},  # Empty topology for debug
                valency=0,  # Default valency
                normal=[0, 0, 1]  # Default normal
            )
            vertex_dims.append(len(features))
            if i < 5:  # Print first 5
                print(f"Vertex {i}: {len(features)} features")
    
        print(f"Face dimensions: {set(face_dims)}")
        print(f"Edge dimensions: {set(edge_dims)}")
        print(f"Vertex dimensions: {set(vertex_dims)}")
        print("=== END DEBUG ===")
    
    def load_phase1_data(self, json_file_path):
        """Load Phase 1 JSON data - AP214 COMPATIBLE"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data

    def extract_entities_from_ap214(self, phase1_data):
        """Extract entities from AP214 data structure"""
        # Handle both AP214 structure and standard structure
        if 'geometry_data' in phase1_data:
            # AP214 structure
            entities = phase1_data['geometry_data']
        elif 'entities' in phase1_data:
            # Standard structure
            entities = phase1_data['entities']
        else:
            # Fallback - try to find entities directly
            entities = {
                'faces': phase1_data.get('faces', []),
                'edges': phase1_data.get('edges', []),
                'vertices': phase1_data.get('vertices', [])
            }
        
        return entities

    def extract_topology_from_ap214(self, phase1_data):
        """Extract topology from AP214 data structure"""
        topology = phase1_data.get('topology_graph', {})
        
        # Ensure all required topology sections exist
        required_sections = ['face_edge_adjacency', 'edge_vertex_adjacency', 
                           'vertex_face_adjacency', 'edge_edge_adjacency', 'face_face_adjacency']
        
        for section in required_sections:
            if section not in topology:
                topology[section] = [[], []]
        
        return topology

    def encode_surface_type(self, surface_type):
        """One-hot encode surface types - UPDATED for new types"""
        types = [
            'PLANE', 'PLANE_SIMPLE',           # Plane types
            'CYLINDER', 'CYLINDER_SIMPLE',      # Cylinder types  
            'CONE', 'CONE_SIMPLE',              # Cone types
            'SPHERE', 'SPHERE_SIMPLE',          # Sphere types
            'SPLINE_SURFACE',                   # Spline surfaces
            'PROCESSING_ERROR',                  # Error cases
            'UNKNOWN'                           # Default unknown
        ]
    
        # Handle numbered surface types (SURFACE_TYPE_X)
        if surface_type.startswith('SURFACE_TYPE_'):
            try:
                type_num = int(surface_type.split('_')[-1])
                # Map to UNKNOWN for now, but could add more specific handling
                encoding = [0] * len(types)
                encoding[-1] = 1  # UNKNOWN
                return encoding
            except:
                pass
    
        encoding = [0] * len(types)
        if surface_type in types:
            encoding[types.index(surface_type)] = 1
        else:
            encoding[-1] = 1  # UNKNOWN
        return encoding

    def encode_curve_type(self, curve_type):
        """One-hot encode curve types - UPDATED for new types"""
        types = [
            'LINE', 'LINE_SIMPLE',               # Line types
            'CIRCLE', 'CIRCLE_SIMPLE',           # Circle types
            'ELLIPSE', 'ELLIPSE_SIMPLE',         # Ellipse types
            'SPLINE',                             # Spline curves
            'PROCESSING_ERROR',                    # Error cases
            'UNKNOWN_CURVE'                        # Default unknown
        ]
    
        # Handle numbered curve types (CURVE_TYPE_X)
        if curve_type.startswith('CURVE_TYPE_'):
            try:
                type_num = int(curve_type.split('_')[-1])
                # Map to UNKNOWN for now
                encoding = [0] * len(types)
                encoding[-1] = 1  # UNKNOWN_CURVE
                return encoding
            except:
                pass
    
        encoding = [0] * len(types)
        if curve_type in types:
            encoding[types.index(curve_type)] = 1
        else:
            encoding[-1] = 1  # UNKNOWN_CURVE
        return encoding

    def calculate_angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1 = np.array(v1)
        v2 = np.array(v2)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0

        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosine = np.clip(dot_product / norms, -1.0, 1.0)
        angle_rad = np.arccos(cosine)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def build_face_node_features(self, face_data):
        """Build feature vector for face nodes - UPDATED for new types"""
        features = []
        features.extend([1, 0, 0])  # [IS_FACE, IS_EDGE, IS_VERTEX]
    
        # Surface type one-hot encoding (now with more types)
        surface_type_encoding = self.encode_surface_type(face_data['surface_type'])
        features.extend(surface_type_encoding)

        # Safe parameter extraction with defaults
        params = face_data.get('parameters', {})
    
        # Geometric parameters based on surface type
        surface_type = face_data['surface_type']
    
        if surface_type.startswith('PLANE'):
            normal = params.get('normal', [0, 0, 1])
            point = params.get('point', [0, 0, 0])
            features.extend(normal)  # normal_x, normal_y, normal_z
            features.extend(point)  
            features.extend([0, 0])

        elif surface_type.startswith('CYLINDER'):
            axis = params.get('axis', [0, 0, 1])
            radius = params.get('radius', 0)
            center = params.get('center', [0, 0, 0])
            features.extend(axis)  # axis_x, axis_y, axis_z
            features.append(radius)
            features.extend(center)
            features.append(0)

        elif surface_type.startswith('CONE'):
            axis = params.get('axis', [0, 0, 1])
            angle = params.get('angle', 0)
            radius = params.get('radius', 0)
            center = params.get('center', [0, 0, 0])
            features.extend(axis)  # axis_x, axis_y, axis_z
            features.append(angle)
            features.append(radius)
            features.extend(center)  # Use first component of center

        elif surface_type.startswith('SPHERE'):
            radius = params.get('radius', 0)
            center = params.get('center', [0, 0, 0])
            features.append(radius)
            features.extend(center)  # center_x, center_y, center_z
            features.extend([0, 0, 0, 0])  # Padding

        elif surface_type == 'SPLINE_SURFACE':
            # For spline surfaces, use degree info if available
            u_degree = params.get('u_degree', 0)
            v_degree = params.get('v_degree', 0)
            features.append(u_degree)
            features.append(v_degree)
            features.extend([0] * 6)  # Padding

        else:  # PROCESSING_ERROR, UNKNOWN, etc.
            features.extend([0] * 8)  # Placeholder

        # Common features for all face types
        features.append(face_data.get('area', 0))
        centroid = face_data.get('centroid', [0, 0, 0])
        features.extend(centroid)  # centroid_x, centroid_y, centroid_z

        normal = face_data.get('normal', [0, 0, 1])
        features.extend(normal)  # normal_x, normal_y, normal_z

        # Pad or truncate to fixed dimension
        if len(features) < self.face_feature_dim:
            features.extend([0] * (self.face_feature_dim - len(features)))
        else:
            features = features[:self.face_feature_dim]

        return features

    def build_edge_node_features(self, edge_data):
        """Build feature vector for edge nodes - UPDATED for new types"""
        features = []
        features.extend([0, 1, 0])  # [IS_FACE, IS_EDGE, IS_VERTEX]
    
        # Curve type one-hot encoding (now with more types)
        curve_type_encoding = self.encode_curve_type(edge_data['curve_type'])
        features.extend(curve_type_encoding)

        # Safe parameter extraction
        params = edge_data.get('parameters', {})
        curve_type = edge_data['curve_type']

        # Geometric parameters based on curve type
        if curve_type.startswith('LINE'):
            direction = params.get('direction', [1, 0, 0])
            point = params.get('point', [0, 0, 0])
            features.extend(direction)  # direction_x, direction_y, direction_z
            features.extend(point[:2])  # First two components of point

        elif curve_type.startswith('CIRCLE'):
            axis = params.get('axis', [0, 0, 1])
            radius = params.get('radius', 0)
            center = params.get('center', [0, 0, 0])
            features.extend(axis)  # axis_x, axis_y, axis_z
            features.append(radius)
            features.extend(center[:2])  # First two components of center

        elif curve_type.startswith('ELLIPSE'):
            axis = params.get('axis', [0, 0, 1])
            major_radius = params.get('major_radius', 0)
            minor_radius = params.get('minor_radius', 0)
            center = params.get('center', [0, 0, 0])
            features.extend(axis)  # axis_x, axis_y, axis_z
            features.append(major_radius)
            features.append(minor_radius)
            features.extend(center[:1])  # First component of center

        elif curve_type == 'SPLINE':
            # For spline curves
            sample_points = params.get('sample_points', [])
            degree = params.get('degree', 0)
            features.append(degree)
            # Use first sample point if available
            if sample_points and len(sample_points) > 0:
                features.extend(sample_points[0][:2])  # First two components of first sample
            else:
                features.extend([0, 0])
            features.extend([0, 0, 0])  # Padding

        else:  # PROCESSING_ERROR, UNKNOWN_CURVE, etc.
            features.extend([0, 0, 0, 0, 0, 0, 0])  # Placeholder

        # Common features for all edge types
        features.append(edge_data.get('length', 0))
        start_point = edge_data.get('start_point', [0, 0, 0])
        end_point = edge_data.get('end_point', [0, 0, 0])
        features.extend(start_point)  # start_x, start_y, start_z
        features.extend(end_point)    # end_x, end_y, end_z

        direction = edge_data.get('direction', [1, 0, 0])
        features.extend(direction)  # direction_x, direction_y, direction_z

        # Pad or truncate to fixed dimension
        if len(features) < self.edge_feature_dim:
            features.extend([0] * (self.edge_feature_dim - len(features)))
        else:
            features = features[:self.edge_feature_dim]

        return features

    def build_vertex_node_features(self, vertex_data, vertex_idx, entities,     topology_graph, valency=0, normal=None):
        features = []
    
        # 1. Type flags (3)
        features.extend([0, 0, 1])  # [IS_FACE, IS_EDGE, IS_VERTEX]
    
        # 2. Position (3)
        features.extend(vertex_data['point'])  # x, y, z
    
        # 3. Valency (1)
        features.append(float(valency))
    
        # 4. Normal vector (3)
        if normal is None:
            normal = [0, 0, 1]
        features.extend(normal)  # nx, ny, nz
    
        # 5. Vertex classification (2)
        is_boundary = 1.0 if valency < 3 else 0.0
        features.append(is_boundary)
        is_interior = 1.0 if valency >= 3 else 0.0
        features.append(is_interior)
    
        # 6. Density estimates (2) - enhanced
        density_estimate = 1.0 / (valency + 1.0)
        features.append(density_estimate)
    
        # NEW: Valency-weighted density (gives more weight to higher valency)
        weighted_density = valency / (valency + 5.0)  # Normalized between 0-1
        features.append(weighted_density)
    
        # 7. Curvature estimates (3) - enhanced with surface type context
        # Base curvature from valency
        if valency <= 2:
            base_curvature = 1.0  # High curvature (edges/corners)
        elif valency <= 4:
            base_curvature = 0.5  # Medium curvature
        else:
            base_curvature = 0.1  # Low curvature (flat regions)
        features.append(base_curvature)
    
        # NEW: Surface type diversity around vertex
        # Get all faces adjacent to this vertex
        adjacent_face_types = []
        if 'vertex_face_adjacency' in topology_graph:
            for v_idx, f_idx in zip(topology_graph['vertex_face_adjacency'][0], 
                                    topology_graph['vertex_face_adjacency'][1]):
                if v_idx == vertex_idx and f_idx < len(entities.get('faces', [])):
                    face_type = entities['faces'][f_idx].get('surface_type', 'UNKNOWN')
                    # Simplify to base type
                    if face_type.startswith('PLANE'):
                        adjacent_face_types.append('PLANE')
                    elif face_type.startswith('CYLINDER'):
                        adjacent_face_types.append('CYLINDER')
                    elif face_type.startswith('CONE'):
                        adjacent_face_types.append('CONE')
                    elif face_type.startswith('SPHERE'):
                        adjacent_face_types.append('SPHERE')
                    else:
                        adjacent_face_types.append('COMPLEX')
    
        # Diversity: ratio of unique surface types to total faces
        if adjacent_face_types:
            unique_types = len(set(adjacent_face_types))
            total_faces = len(adjacent_face_types)
            diversity = unique_types / total_faces if total_faces > 0 else 0
        else:
            diversity = 0
        features.append(diversity)  # High diversity means vertex connects different surface types
    
        # NEW: Edge type diversity around vertex
        adjacent_edge_types = []
        if 'edge_vertex_adjacency' in topology_graph:
            for e_idx, v_idx in zip(topology_graph['edge_vertex_adjacency'][0], 
                                    topology_graph['edge_vertex_adjacency'][1]):
                if v_idx == vertex_idx and e_idx < len(entities['edges']):
                    edge_type = entities['edges'][e_idx].get('curve_type', 'UNKNOWN_CURVE')
                    # Simplify
                    if edge_type.startswith('LINE'):
                        adjacent_edge_types.append('LINE')
                    elif edge_type.startswith('CIRCLE'):
                        adjacent_edge_types.append('CIRCLE')
                    elif edge_type.startswith('ELLIPSE'):
                        adjacent_edge_types.append('ELLIPSE')
                    else:
                        adjacent_edge_types.append('COMPLEX')
    
        if adjacent_edge_types:
            edge_diversity = len(set(adjacent_edge_types)) / len(adjacent_edge_types)
        else:
            edge_diversity = 0
        features.append(edge_diversity)
    
        # NEW: Is the vertex on a seam? (where different surface types meet)
        is_seam = 1.0 if diversity > 0.5 else 0.0  # Arbitrary threshold
        features.append(is_seam)
    
        # Count dimensions:
        # Type flags: 3
        # Position: 3
        # Valency: 1
        # Normal: 3
        # Boundary/interior: 2
        # Density estimates: 2
        # Curvature: 3
        # Diversity features: 3
        # Total: 3+3+1+3+2+2+3+3 = 20 features
    
        # Pad to target dimension (16 or 32)
        if len(features) < self.vertex_feature_dim:
            features.extend([0] * (self.vertex_feature_dim - len(features)))
        else:
            features = features[:self.vertex_feature_dim]
    
        return features

    def build_hybrid_graph_nodes(self, entities, topology_graph):
        """Build all node features for hybrid graph - FIXED DIMENSIONS"""
        face_nodes = []
        edge_nodes = []
        vertex_nodes = []
        node_type_mapping = []
        node_entity_mapping = []

        # Use the largest dimension (32) for all nodes and pad the others
        target_dim = self.face_feature_dim  # 32

        # Build face nodes (already 32 dimensions)
        for face in entities['faces']:
            face_features = self.build_face_node_features(face)
            face_nodes.append(face_features)
            node_type_mapping.append('face')
            node_entity_mapping.append(('face', face['id']))

        # Build edge nodes (pad from 24 to 32)
        for edge in entities['edges']:
            edge_features = self.build_edge_node_features(edge)
            # Pad edge features from 24 to 32 dimensions
            if len(edge_features) < target_dim:
                edge_features.extend([0] * (target_dim - len(edge_features)))
            edge_nodes.append(edge_features)
            node_type_mapping.append('edge')
            node_entity_mapping.append(('edge', edge['id']))

        # Compute vertex properties with UNBIASED averaging
        vertex_valency = [0] * len(entities['vertices'])
        vertex_normals = [[0, 0, 0] for _ in range(len(entities['vertices']))]  # Start with zero vectors
        vertex_face_counts = [0] * len(entities['vertices'])  # Track number of faces per vertex

        # Compute valency from edge_vertex_adjacency
        if 'edge_vertex_adjacency' in topology_graph:
            for edge_idx, vertex_idx in zip(topology_graph['edge_vertex_adjacency'][0], 
                                            topology_graph['edge_vertex_adjacency'][1]):
                if vertex_idx < len(vertex_valency):
                    vertex_valency[vertex_idx] += 1

        # Compute vertex normals from adjacent faces - UNBIASED VERSION
        # In build_hybrid_graph_nodes, enhance the normal calculation:
        if 'vertex_face_adjacency' in topology_graph:
            for vertex_idx, face_idx in zip(topology_graph['vertex_face_adjacency'][0], 
                                    topology_graph['vertex_face_adjacency'][1]):
                if vertex_idx < len(vertex_normals) and face_idx < len(entities['faces']):
                    face = entities['faces'][face_idx]
            
                    # Get appropriate direction based on surface type
                    if face['surface_type'].startswith('PLANE'):
                        face_normal = face.get('normal', [0, 0, 1])
                    elif face['surface_type'].startswith('CYLINDER'):
                        # For cylinders, use axis as normal proxy
                        face_normal = face.get('normal', face.get('parameters', {}).get('axis', [0, 0, 1]))
                    elif face['surface_type'].startswith('CONE'):
                        face_normal = face.get('normal', face.get('parameters', {}).get('axis', [0, 0, 1]))
                    else:
                        face_normal = face.get('normal', [0, 0, 1])
            
                    vertex_normals[vertex_idx][0] += face_normal[0]
                    vertex_normals[vertex_idx][1] += face_normal[1]
                    vertex_normals[vertex_idx][2] += face_normal[2]
                    vertex_face_counts[vertex_idx] += 1

        # Normalize vertex normals to unit length after accumulation
        for i in range(len(vertex_normals)):
            if vertex_face_counts[i] > 0:
                # Compute true average (divide by count) and normalize to unit length
                norm = np.linalg.norm(vertex_normals[i])
                if norm > 0:
                    vertex_normals[i] = [
                        vertex_normals[i][0] / norm,
                        vertex_normals[i][1] / norm,
                        vertex_normals[i][2] / norm
                    ]
                else:
                    vertex_normals[i] = [0, 0, 1]  # Fallback to default normal
            else:
                vertex_normals[i] = [0, 0, 1]  # Default normal for isolated vertices

        # Build vertex nodes (pad from 8 to 32)
        for i, vertex in enumerate(entities['vertices']):
            # Pass additional vertex properties with REAL COMPUTED VALUES
            vertex_features = self.build_vertex_node_features(
                vertex, 
                vertex_idx=i,
                entities=entities,
                topology_graph=topology_graph,
                valency=vertex_valency[i],
                normal=vertex_normals[i]
            )
            # Pad vertex features to 32 dimensions
            if len(vertex_features) < target_dim:
                vertex_features.extend([0] * (target_dim - len(vertex_features)))
            vertex_nodes.append(vertex_features)
            node_type_mapping.append('vertex')
            node_entity_mapping.append(('vertex', vertex['id']))

        # Combine all nodes (now all have 32 dimensions)
        all_nodes = face_nodes + edge_nodes + vertex_nodes
        num_faces = len(face_nodes)
        num_edges = len(edge_nodes)
        num_vertices = len(vertex_nodes)

        # Verify all nodes have the same dimension
        for i, node in enumerate(all_nodes):
            if len(node) != target_dim:
                print(f"ERROR: Node {i} has {len(node)} features, expected {target_dim}")
                raise ValueError(f"Node dimension mismatch: {len(node)} vs {target_dim}")

        return all_nodes, num_faces, num_edges, num_vertices, node_type_mapping, node_entity_mapping

    def calculate_face_face_relationship(self, face1, face2):
        """Calculate geometric relationship features between two faces"""
        features = []

        # 1. Angle between normals/axes (continuous feature)
        normal1 = face1.get('normal', [0, 0, 1])
        normal2 = face2.get('normal', [0, 0, 1])
        angle = self.calculate_angle_between_vectors(normal1, normal2)
        features.append(angle)

        # 2. Distance between centroids
        distance = self.calculate_distance(face1['centroid'], face2['centroid'])
        features.append(distance)

        # 3. Cosine of angle (alternative representation)
        cosine = np.cos(np.radians(angle))
        features.append(cosine)

        # 4. Area ratio (continuous)
        area_ratio = min(face1['area'], face2['area']) / max(face1['area'], face2['area']) if max(face1['area'], face2['area']) > 0 else 0.0
        features.append(area_ratio)

        # 5. Same surface type (now handles SIMPLE variants)
        if face1['surface_type'].startswith('PLANE') and        face2['surface_type'].startswith('PLANE'):
            same_type = 1.0
        elif face1['surface_type'].startswith('CYLINDER') and face2['surface_type'].startswith('CYLINDER'):
            same_type = 1.0
        elif face1['surface_type'].startswith('CONE') and       face2['surface_type'].startswith('CONE'):
            same_type = 1.0
        elif face1['surface_type'].startswith('SPHERE') and face2['surface_type'].startswith('SPHERE'):
            same_type = 1.0
        else:
            same_type = 1.0 if face1['surface_type'] == face2['surface_type'] else 0.0
        features.append(same_type)

        # 6. Dot product of normals
        dot_product = np.dot(normal1, normal2)
        features.append(dot_product)

        # 7. Centroid vector (relative position)
        centroid_vector = np.array(face2['centroid']) - np.array(face1['centroid'])
        features.extend(centroid_vector)

        # Pad to fixed dimension
        if len(features) < self.relationship_feature_dim:
            features.extend([0] * (self.relationship_feature_dim - len(features)))
        else:
            features = features[:self.relationship_feature_dim]

        return features

    def calculate_face_edge_relationship(self, face, edge):
        """Calculate geometric relationship features between face and edge"""
        features = []

        # 1. Angle between face normal and edge direction (continuous)
        face_normal = face.get('normal', [0, 0, 1])
        edge_direction = edge.get('direction', [1, 0, 0])
        angle = self.calculate_angle_between_vectors(face_normal, edge_direction)
        features.append(angle)

        # 2. Distance from edge midpoint to face centroid
        edge_midpoint = [
            (edge['start_point'][0] + edge['end_point'][0]) / 2,
            (edge['start_point'][1] + edge['end_point'][1]) / 2,
            (edge['start_point'][2] + edge['end_point'][2]) / 2
        ]
        distance = self.calculate_distance(edge_midpoint, face['centroid'])
        features.append(distance)

        # 3. Cosine of angle
        cosine = np.cos(np.radians(angle))
        features.append(cosine)

        # 4. Edge length to face area ratio
        length_area_ratio = edge['length'] / face['area'] if face['area'] > 0 else 0.0
        features.append(length_area_ratio)

        # 5. Dot product of face normal and edge direction
        dot_product = np.dot(face_normal, edge_direction)
        features.append(dot_product)

        # 6. Minimum distance from edge endpoints to face centroid
        dist_start = self.calculate_distance(edge['start_point'], face['centroid'])
        dist_end = self.calculate_distance(edge['end_point'], face['centroid'])
        min_endpoint_distance = min(dist_start, dist_end)
        features.append(min_endpoint_distance)

        # 7. Vector from face centroid to edge midpoint
        midpoint_vector = np.array(edge_midpoint) - np.array(face['centroid'])
        features.extend(midpoint_vector)  # Only take x,y for dimension control

        # Pad to fixed dimension
        if len(features) < self.relationship_feature_dim:
            features.extend([0] * (self.relationship_feature_dim - len(features)))
        else:
            features = features[:self.relationship_feature_dim]

        return features

    def calculate_edge_vertex_relationship(self, edge, vertex):
        """Calculate geometric relationship features between edge and vertex"""
        features = [0] * self.relationship_feature_dim

        # 1. Distance from vertex to edge midpoint
        edge_midpoint = [
            (edge['start_point'][0] + edge['end_point'][0]) / 2,
            (edge['start_point'][1] + edge['end_point'][1]) / 2,
            (edge['start_point'][2] + edge['end_point'][2]) / 2
        ]
        distance_to_midpoint = self.calculate_distance(vertex['point'], edge_midpoint)
        features[0] = distance_to_midpoint

        # 2. Distance from vertex to edge start point (continuous)
        distance_to_start = self.calculate_distance(vertex['point'], edge['start_point'])
        features[1] = distance_to_start

        # 3. Distance from vertex to edge end point (continuous)
        distance_to_end = self.calculate_distance(vertex['point'], edge['end_point'])
        features[2] = distance_to_end

        # 4. Ratio of distances (vertex position along edge)
        if distance_to_start + distance_to_end > 0:
            start_end_ratio = distance_to_start / (distance_to_start + distance_to_end)
            features[3] = start_end_ratio
        else:
            features[3] = 0.5

        # 5. Edge length for context
        features[4] = edge['length']

        # 6-7. Angles between vertex position and edge direction
        edge_dir = np.array(edge.get('direction', [1, 0, 0]))

        if np.linalg.norm(edge_dir) > 0:
            # Vector from vertex to start point
            vertex_to_start = np.array(edge['start_point']) - np.array(vertex['point'])
            if np.linalg.norm(vertex_to_start) > 0:
                vertex_to_start = vertex_to_start / np.linalg.norm(vertex_to_start)
                angle_to_start = self.calculate_angle_between_vectors(vertex_to_start, edge_dir)
                features[5] = angle_to_start

            # Vector from vertex to end point
            vertex_to_end = np.array(edge['end_point']) - np.array(vertex['point'])
            if np.linalg.norm(vertex_to_end) > 0:
                vertex_to_end = vertex_to_end / np.linalg.norm(vertex_to_end)
                angle_to_end = self.calculate_angle_between_vectors(vertex_to_end, edge_dir)
                features[6] = angle_to_end

        # 8. Dot products for alternative angle representation
        if np.linalg.norm(edge_dir) > 0:
            vertex_to_start = np.array(edge['start_point']) - np.array(vertex['point'])
            if np.linalg.norm(vertex_to_start) > 0:
                vertex_to_start = vertex_to_start / np.linalg.norm(vertex_to_start)
                dot_start = np.dot(vertex_to_start, edge_dir)
                features[7] = dot_start

            vertex_to_end = np.array(edge['end_point']) - np.array(vertex['point'])
            if np.linalg.norm(vertex_to_end) > 0:
                vertex_to_end = vertex_to_end / np.linalg.norm(vertex_to_end)
                dot_end = np.dot(vertex_to_end, edge_dir)
                features[8] = dot_end

        # 9. Vector from vertex to edge midpoint (relative position)
        midpoint_vector = np.array(edge_midpoint) - np.array(vertex['point'])
        features[9] = np.linalg.norm(midpoint_vector)  # Magnitude
        if len(midpoint_vector) >= 2:
            features[10] = midpoint_vector[0]  # x-component
            features[11] = midpoint_vector[1]  # y-component
            features[12] = midpoint_vector[2]  # z-component

        return features

    def calculate_vertex_face_relationship(self, vertex, face):
        """Calculate geometric relationship features between vertex and face"""
        features = [0] * self.relationship_feature_dim

        # 1. Distance from vertex to face centroid
        distance_to_centroid = self.calculate_distance(vertex['point'], face['centroid'])
        features[0] = distance_to_centroid

        # 2. Distance from vertex to face surface (for planes)
        if face['surface_type'] == 'PLANE':
            face_normal = np.array(face['parameters'].get('normal', [0, 0, 1]))
            face_point = np.array(face['parameters'].get('point', [0, 0, 0]))
            vertex_point = np.array(vertex['point'])

            # Distance from point to plane: |(point - plane_point) · normal|
            distance_to_plane = abs(np.dot(vertex_point - face_point, face_normal))
            features[1] = distance_to_plane
        else:
            # For curved surfaces, use centroid distance approximation
            features[1] = distance_to_centroid

        # 3. Angle between vertex position and face normal
        face_normal = np.array(face.get('normal', [0, 0, 1]))
        vertex_vector = np.array(vertex['point']) - np.array(face['centroid'])

        if np.linalg.norm(vertex_vector) > 0 and np.linalg.norm(face_normal) > 0:
            vertex_vector = vertex_vector / np.linalg.norm(vertex_vector)
            angle_to_normal = self.calculate_angle_between_vectors(vertex_vector, face_normal)
            features[2] = angle_to_normal

            # 4. Dot product alternative
            dot_product = np.dot(vertex_vector, face_normal)
            features[3] = dot_product

        # 5. Face area for context
        features[4] = face['area']

        # 6. Relative distance (normalized by face size)
        if face['area'] > 0:
            relative_distance = distance_to_centroid / (face['area'] ** 0.5)
            features[5] = relative_distance

        # 7. Vector from face centroid to vertex
        centroid_to_vertex = np.array(vertex['point']) - np.array(face['centroid'])
        features[6] = np.linalg.norm(centroid_to_vertex)  # Magnitude
        if len(centroid_to_vertex) >= 2:
            features[7] = centroid_to_vertex[0]  # x-component
            features[8] = centroid_to_vertex[1]  # y-component
            features[9] = centroid_to_vertex[2]  # z-component

        return features

    def calculate_edge_edge_relationship(self, edge1, edge2):
        """Calculate geometric relationship features between two edges"""
        features = []

        # 1. Angle between edge directions (continuous)
        direction1 = edge1.get('direction', [1, 0, 0])
        direction2 = edge2.get('direction', [1, 0, 0])
        angle = self.calculate_angle_between_vectors(direction1, direction2)
        features.append(angle)

        # 2. Distance between edge midpoints
        midpoint1 = [
            (edge1['start_point'][0] + edge1['end_point'][0]) / 2,
            (edge1['start_point'][1] + edge1['end_point'][1]) / 2,
            (edge1['start_point'][2] + edge1['end_point'][2]) / 2
        ]
        midpoint2 = [
            (edge2['start_point'][0] + edge2['end_point'][0]) / 2,
            (edge2['start_point'][1] + edge2['end_point'][1]) / 2,
            (edge2['start_point'][2] + edge2['end_point'][2]) / 2
        ]
        distance = self.calculate_distance(midpoint1, midpoint2)
        features.append(distance)

        # 3. Cosine of angle
        cosine = np.cos(np.radians(angle))
        features.append(cosine)

        # 4. Length ratio
        length_ratio = min(edge1['length'], edge2['length']) / max(edge1['length'], edge2['length']) if max(edge1['length'], edge2['length']) > 0 else 0.0
        features.append(length_ratio)

        # 5. Same curve type (now handles SIMPLE variants)
        if edge1['curve_type'].startswith('LINE') and edge2['curve_type'].startswith('LINE'):
            same_type = 1.0
        elif edge1['curve_type'].startswith('CIRCLE') and edge2['curve_type'].startswith('CIRCLE'):
            same_type = 1.0
        elif edge1['curve_type'].startswith('ELLIPSE') and edge2['curve_type'].startswith('ELLIPSE'):
            same_type = 1.0
        else:
            same_type = 1.0 if edge1['curve_type'] == edge2['curve_type'] else 0.0
        features.append(same_type)

        # 6. Dot product of directions
        dot_product = np.dot(direction1, direction2)
        features.append(dot_product)

        # 7. Minimum endpoint distance (continuous, not binary coincident)
        distances = [
            self.calculate_distance(edge1['start_point'], edge2['start_point']),
            self.calculate_distance(edge1['start_point'], edge2['end_point']),
            self.calculate_distance(edge1['end_point'], edge2['start_point']),
            self.calculate_distance(edge1['end_point'], edge2['end_point'])
        ]
        min_endpoint_distance = min(distances)
        features.append(min_endpoint_distance)

        # 8. Vector between midpoints
        midpoint_vector = np.array(midpoint2) - np.array(midpoint1)
        features.extend(midpoint_vector)  # Only take x,y for dimension control

        # Pad to fixed dimension
        if len(features) < self.relationship_feature_dim:
            features.extend([0] * (self.relationship_feature_dim - len(features)))
        else:
            features = features[:self.relationship_feature_dim]

        return features

    def build_hybrid_graph_edges(self, entities, topology_graph, num_faces, num_edges):
        """Build all edge relationships for hybrid graph"""
        edge_indices = []
        edge_features_list = []
        edge_type_mapping = []

        num_total_nodes = num_faces + num_edges + len(entities['vertices'])

        # 1. FACE-EDGE relationships
        for i in range(len(topology_graph['face_edge_adjacency'][0])):
            face_idx = topology_graph['face_edge_adjacency'][0][i]
            edge_idx = topology_graph['face_edge_adjacency'][1][i]

            # Skip if indices are out of bounds
            if face_idx >= len(entities['faces']) or edge_idx >= len(entities['edges']):
                continue

            face = entities['faces'][face_idx]
            edge = entities['edges'][edge_idx]

            relationship_features = self.calculate_face_edge_relationship(face, edge)

            # Add both directions for undirected graph
            edge_indices.append([face_idx, num_faces + edge_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('face_edge')

            edge_indices.append([num_faces + edge_idx, face_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('edge_face')

        # 2. EDGE-VERTEX relationships
        for i in range(len(topology_graph['edge_vertex_adjacency'][0])):
            edge_idx = topology_graph['edge_vertex_adjacency'][0][i]
            vertex_idx = topology_graph['edge_vertex_adjacency'][1][i]

            # Skip if indices are out of bounds
            if edge_idx >= len(entities['edges']) or vertex_idx >= len(entities['vertices']):
                continue

            edge = entities['edges'][edge_idx]
            vertex = entities['vertices'][vertex_idx]

            relationship_features = self.calculate_edge_vertex_relationship(edge, vertex)

            edge_indices.append([num_faces + edge_idx, num_faces + num_edges + vertex_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('edge_vertex')

            edge_indices.append([num_faces + num_edges + vertex_idx, num_faces + edge_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('vertex_edge')

        # 3. VERTEX-FACE relationships
        for i in range(len(topology_graph['vertex_face_adjacency'][0])):
            vertex_idx = topology_graph['vertex_face_adjacency'][0][i]
            face_idx = topology_graph['vertex_face_adjacency'][1][i]

            # Skip if indices are out of bounds
            if vertex_idx >= len(entities['vertices']) or face_idx >= len(entities['faces']):
                continue

            vertex = entities['vertices'][vertex_idx]
            face = entities['faces'][face_idx]

            relationship_features = self.calculate_vertex_face_relationship(vertex, face)

            edge_indices.append([num_faces + num_edges + vertex_idx, face_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('vertex_face')

            edge_indices.append([face_idx, num_faces + num_edges + vertex_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('face_vertex')

        # 4. EDGE-EDGE relationships
        for i in range(len(topology_graph['edge_edge_adjacency'][0])):
            edge1_idx = topology_graph['edge_edge_adjacency'][0][i]
            edge2_idx = topology_graph['edge_edge_adjacency'][1][i]

            # Skip if indices are out of bounds
            if edge1_idx >= len(entities['edges']) or edge2_idx >= len(entities['edges']):
                continue

            edge1 = entities['edges'][edge1_idx]
            edge2 = entities['edges'][edge2_idx]

            relationship_features = self.calculate_edge_edge_relationship(edge1, edge2)

            edge_indices.append([num_faces + edge1_idx, num_faces + edge2_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('edge_edge')

            edge_indices.append([num_faces + edge2_idx, num_faces + edge1_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('edge_edge')

        # 5. FACE-FACE relationships
        for i in range(len(topology_graph['face_face_adjacency'][0])):
            face1_idx = topology_graph['face_face_adjacency'][0][i]
            face2_idx = topology_graph['face_face_adjacency'][1][i]

            # Skip if indices are out of bounds
            if face1_idx >= len(entities['faces']) or face2_idx >= len(entities['faces']):
                continue

            face1 = entities['faces'][face1_idx]
            face2 = entities['faces'][face2_idx]

            relationship_features = self.calculate_face_face_relationship(face1, face2)

            edge_indices.append([face1_idx, face2_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('face_face')

            edge_indices.append([face2_idx, face1_idx])
            edge_features_list.append(relationship_features)
            edge_type_mapping.append('face_face')

        return edge_indices, edge_features_list, edge_type_mapping

    def build_hybrid_graph(self, phase1_data):
        """Build the complete hybrid graph with all entity types and relationships"""
        # Extract entities and topology from AP214 structure
        entities = self.extract_entities_from_ap214(phase1_data)
        topology_graph = self.extract_topology_from_ap214(phase1_data)
        
        print(f"DEBUG - Topology edges:")
        print(f"  face_face: {len(topology_graph['face_face_adjacency'][0])}")
        print(f"  face_edge: {len(topology_graph['face_edge_adjacency'][0])}") 
        print(f"  edge_vertex: {len(topology_graph['edge_vertex_adjacency'][0])}")
        print(f"  vertex_face: {len(topology_graph['vertex_face_adjacency'][0])}")
        print(f"  edge_edge: {len(topology_graph['edge_edge_adjacency'][0])}")
        
        should_skip, reason = self.should_skip_graph(entities, topology_graph)
        if should_skip:
            print(f"  ⚠ SKIPPING: {reason}")
            return self.create_minimal_graph(phase1_data)
        self.debug_all_feature_dimensions(entities)
        # Validate we have the basic entities
        if not entities.get('faces') and not entities.get('edges') and not entities.get('vertices'):
            raise ValueError("No geometric entities found in Phase 1 data")

        # Build all node features
        all_nodes, num_faces, num_edges, num_vertices, node_type_mapping, node_entity_mapping = self.build_hybrid_graph_nodes(entities, topology_graph)

        # Build all edge relationships
        edge_indices, edge_features_list, edge_type_mapping = self.build_hybrid_graph_edges(
            entities, topology_graph, num_faces, num_edges
        )

        # Convert to tensors
        if not edge_indices:  # Handle case with no edges
            edge_indices = [[0, 0]]
            edge_features_list = [[0] * self.relationship_feature_dim]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        x = torch.tensor(all_nodes, dtype=torch.float)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

        # Convert mappings to tensors for proper serialization
        node_entity_tensor = self._convert_mapping_to_tensor(node_entity_mapping)
        edge_type_tensor = self._convert_edge_types_to_tensor(edge_type_mapping)

        # Create enhanced PyG Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_faces=torch.tensor([num_faces], dtype=torch.long),
            num_edges=torch.tensor([num_edges], dtype=torch.long),
            num_vertices=torch.tensor([num_vertices], dtype=torch.long),
            # Store mappings as tensors for proper serialization
            node_entity_mapping=node_entity_tensor,
            edge_type_mapping=edge_type_tensor,
            # Store original lists as string representations for debugging
            node_type_mapping_str=json.dumps(node_type_mapping),
            edge_type_mapping_str=json.dumps(edge_type_mapping),
            source_file=phase1_data.get('source_file', 'unknown')
        )

        return graph_data

    def _convert_mapping_to_tensor(self, node_entity_mapping):
        """Convert node_entity_mapping to tensor format for proper serialization"""
        mapping_tensor = []

        for node_type, entity_id in node_entity_mapping:
            if node_type == 'face':
                type_encoding = [1, 0, 0]  # [IS_FACE, IS_EDGE, IS_VERTEX]
            elif node_type == 'edge':
                type_encoding = [0, 1, 0]
            elif node_type == 'vertex':
                type_encoding = [0, 0, 1]
            else:
                type_encoding = [0, 0, 0]

            # Combine type encoding with entity ID
            mapping_entry = type_encoding + [float(entity_id)]
            mapping_tensor.append(mapping_entry)

        return torch.tensor(mapping_tensor, dtype=torch.float)

    def _convert_edge_types_to_tensor(self, edge_type_mapping):
        """Convert edge_type_mapping to tensor format for proper serialization"""
        # Create one-hot encoding for edge types
        edge_type_categories = ['face_face', 'face_edge', 'edge_face', 'edge_vertex',
                               'vertex_edge', 'vertex_face', 'face_vertex', 'edge_edge']

        edge_type_tensor = []

        for edge_type in edge_type_mapping:
            encoding = [0] * len(edge_type_categories)
            if edge_type in edge_type_categories:
                encoding[edge_type_categories.index(edge_type)] = 1
            edge_type_tensor.append(encoding)

        return torch.tensor(edge_type_tensor, dtype=torch.float)

    def process_single_file(self, input_json_path, output_dir):
        """Process a single Phase 1 JSON file and save PyG graph"""
        try:
            print(f"Processing: {input_json_path}")

            # Load Phase 1 data
            phase1_data = self.load_phase1_data(input_json_path)

            # Build hybrid graph
            graph_data = self.build_hybrid_graph(phase1_data)

            # Save graph
            base_name = os.path.splitext(os.path.basename(input_json_path))[0].replace('_ap214_graph_data', '')
            output_path = os.path.join(output_dir, f"{base_name}.pt")
            torch.save(graph_data, output_path)

            print(f"Created hybrid graph: {output_path}")
            print(f"  - Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
            print(f"  - Faces: {graph_data.num_faces}, Edges: {graph_data.num_edges}, Vertices: {graph_data.num_vertices}")

            # Verify mappings were saved correctly
            if hasattr(graph_data, 'node_entity_mapping'):
                print(f"  - Node mappings: {graph_data.node_entity_mapping.shape}")
            if hasattr(graph_data, 'edge_type_mapping'):
                print(f"  - Edge types: {graph_data.edge_type_mapping.shape}")
            
            del graph_data
            del phase1_data
            gc.collect()
            
            return True

        except Exception as e:
            print(f"Error processing {input_json_path}: {e}")
            import traceback
            traceback.print_exc()
            if 'graph_data' in locals():
                del graph_data
            if 'phase1_data' in locals():
                del phase1_data
            gc.collect()
            return False

    def batch_process_phase1_files(self, input_dir, output_dir):
        """Process all Phase 1 JSON files in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Look for AP214 JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        print(f"Found {len(json_files)} Phase 1 AP214 graph files to process")

        success_count = 0
        failed_files = []

        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)
            success = self.process_single_file(json_path, output_dir)

            if success:
                success_count += 1
            else:
                failed_files.append(json_file)

        print(f"\nPhase 2 Processing Summary:")
        print(f"Successfully processed: {success_count}/{len(json_files)}")
        if failed_files:
            print(f"Failed files: {failed_files}")

def main():
    # Configuration
    PHASE1_OUTPUT_DIR = "/media/anubhab/External/Project/graph_data"  # Directory with Phase 1 JSON files
    PHASE2_OUTPUT_DIR = "/media/anubhab/External/Project/Pyg_graphs"  # Directory for PyG graph files

    # Initialize graph builder
    graph_builder = HybridGraphBuilder(
        angle_tolerance=1.0,    # degrees
        distance_tolerance=0.01 # normalized units
    )

    # Process all Phase 1 files
    graph_builder.batch_process_phase1_files(PHASE1_OUTPUT_DIR, PHASE2_OUTPUT_DIR)

if __name__ == "__main__":
    main()
