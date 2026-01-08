#!/usr/bin/env python3
"""
Geometric Lattice Generator - Standalone
========================================

Generates lattices using pure geometric transformations (alien method).
No traditional math - only geometric expansion/compression sequences.

Transformation Sequence:
Point → Line → Triangle → Square → Volume → Recompress

This creates lattice structures that encode mathematical relationships
through geometric form rather than algebraic equations.
"""

import numpy as np
from typing import Tuple, List, Optional
import math


class GeometricLatticeGenerator:
    """
    Standalone lattice generator using ONLY geometric transformations.
    
    Core Philosophy:
    - Start with a point (compressed singularity)
    - Expand geometrically: Point → Line → Triangle → Square → Cube
    - Each expansion creates new dimensional relationships
    - The lattice structure emerges from geometric expansion, not algebraic construction
    - Compression reveals hidden patterns through geometric collapse
    """
    
    def __init__(self, dimension: int = 3, size: int = 100, dimensions: Tuple[int, ...] = None):
        """
        Args:
            dimension: Spatial dimension (2D, 3D, etc.) - used if dimensions not specified
            size: Default scale (used if dimensions not specified)
            dimensions: Explicit dimensions e.g., (100, 100) for 100x100, (100, 100, 100) for 100x100x100
        """
        if dimensions is not None:
            self.dimensions = dimensions
            self.dimension = len(dimensions)
        else:
            self.dimension = dimension
            self.dimensions = tuple([size] * dimension)
        
        self.size = size  # Keep for backward compatibility
        self.vertices = None
        self.lattice_basis = None
        self.lattice_points = []  # Store full lattice point cloud
        
    def generate(self, seed_point: Tuple[float, ...] = None, 
                 expansion_factor: float = 1.0,
                 verbose: bool = True,
                 fill_lattice: bool = False) -> np.ndarray:
        """
        Generate a lattice through geometric expansion sequence.
        
        Args:
            seed_point: Starting point (singularity). If None, uses origin.
            expansion_factor: Controls expansion rate (1.0 = standard)
            verbose: Print transformation steps
            fill_lattice: If True, fill the entire lattice space with points (creates full NxMxP grid)
            
        Returns:
            Lattice basis as numpy array
        """
        if verbose:
            print("="*60)
            print("GEOMETRIC LATTICE GENERATION")
            print("="*60)
            print(f"Dimensions: {self.dimensions} ({'×'.join(map(str, self.dimensions))} lattice)")
            print(f"Total lattice points: {np.prod(self.dimensions):,}")
            print(f"Method: Pure geometric expansion (alien method)")
            print()
        
        # Initialize seed point (the singularity)
        if seed_point is None:
            seed_point = tuple([0.0] * self.dimension)
        
        if verbose:
            print(f"[0] Singularity: {seed_point}")
            print()
        
        # Stage 1: Point → Line (1D expansion)
        if verbose:
            print("[1] Expanding Point → Line...")
        vertices = self._expand_point_to_line(seed_point, expansion_factor)
        if verbose:
            print(f"    Generated {len(vertices)} line vertices")
            print()
        
        # Stage 2: Line → Triangle (2D expansion)
        if verbose:
            print("[2] Expanding Line → Triangle...")
        vertices = self._expand_line_to_triangle(vertices, expansion_factor)
        if verbose:
            print(f"    Generated {len(vertices)} triangle vertices")
            print()
        
        # Stage 3: Triangle → Square (2D complete)
        if verbose:
            print("[3] Expanding Triangle → Square...")
        vertices = self._expand_triangle_to_square(vertices, expansion_factor)
        if verbose:
            print(f"    Generated {len(vertices)} square vertices")
            print()
        
        # Stage 4: Square → Cube (3D expansion) if dimension >= 3
        if self.dimension >= 3:
            if verbose:
                print("[4] Expanding Square → Cube (3D)...")
            vertices = self._expand_square_to_cube(vertices, expansion_factor)
            if verbose:
                print(f"    Generated {len(vertices)} cube vertices")
                print()
        
        # Stage 5: Cube → Hypercube (4D+) if dimension > 3
        if self.dimension > 3:
            if verbose:
                print(f"[5] Expanding Cube → {self.dimension}D Hypercube...")
            vertices = self._expand_to_hypercube(vertices, self.dimension, expansion_factor)
            if verbose:
                print(f"    Generated {len(vertices)} hypercube vertices")
                print()
        
        # Stage 6: Fill lattice with points if requested
        if fill_lattice:
            if verbose:
                print(f"[*] Filling lattice space with {np.prod(self.dimensions):,} points...")
            self.lattice_points = self._fill_lattice_space()
            if verbose:
                print(f"    Filled lattice: {len(self.lattice_points)} points")
                print()
        
        # Stage 7: Create lattice basis from geometric structure
        if verbose:
            print("[6] Extracting lattice basis from geometric structure...")
        lattice_basis = self._vertices_to_basis(vertices)
        if verbose:
            print(f"    Basis dimensions: {lattice_basis.shape}")
            print()
        
        self.vertices = vertices
        self.lattice_basis = lattice_basis
        
        if verbose:
            print("="*60)
            print("LATTICE GENERATION COMPLETE")
            print("="*60)
            print(f"Final basis: {lattice_basis.shape[0]} vectors in {lattice_basis.shape[1]}D space")
            if fill_lattice:
                print(f"Lattice points: {len(self.lattice_points):,} points in {self.dimensions} grid")
            print()
        
        return lattice_basis
    
    def _expand_point_to_line(self, point: Tuple[float, ...], factor: float) -> np.ndarray:
        """
        Geometric expansion: Point → Line
        
        Create a line segment by extending the point along the primary axis.
        The line spans the first dimension specified in self.dimensions.
        """
        vertices = []
        
        # Original point
        vertices.append(np.array(point))
        
        # Expand along first axis using the first dimension size
        expansion_length = self.dimensions[0] if len(self.dimensions) > 0 else self.size
        
        # Create symmetric extension: -x and +x
        expansion_vector = np.zeros(self.dimension)
        expansion_vector[0] = expansion_length * factor
        
        # Positive endpoint
        vertices.append(np.array(point) + expansion_vector)
        
        # Negative endpoint
        vertices.append(np.array(point) - expansion_vector)
        
        return np.array(vertices)
    
    def _expand_line_to_triangle(self, line_vertices: np.ndarray, factor: float) -> np.ndarray:
        """
        Geometric expansion: Line → Triangle
        
        Add a third point perpendicular to the line to form a triangle.
        Uses the second dimension size from self.dimensions.
        """
        vertices = list(line_vertices)
        
        # Calculate line center
        center = np.mean(line_vertices, axis=0)
        
        # Get dimension sizes
        dim_y = self.dimensions[1] if len(self.dimensions) > 1 else self.size
        
        # Calculate line direction (from first to second point)
        if len(line_vertices) >= 2:
            line_dir = line_vertices[1] - line_vertices[0]
            line_length = np.linalg.norm(line_dir)
            
            # Create perpendicular direction (2nd axis if available)
            if self.dimension >= 2:
                perp_vector = np.zeros(self.dimension)
                perp_vector[1] = dim_y * factor * 0.866  # sqrt(3)/2 for equilateral
                
                # Add apex vertex perpendicular to line
                apex = center + perp_vector
                vertices.append(apex)
        
        return np.array(vertices)
    
    def _expand_triangle_to_square(self, triangle_vertices: np.ndarray, factor: float) -> np.ndarray:
        """
        Geometric expansion: Triangle → Square
        
        Complete the square by adding a 4th vertex.
        Uses geometric reflection across the triangle's base.
        """
        if len(triangle_vertices) < 3:
            return triangle_vertices
        
        vertices = list(triangle_vertices)
        
        # Find the base of the triangle (first two vertices)
        v0, v1 = triangle_vertices[0], triangle_vertices[1]
        
        # Find the apex (third vertex)
        apex = triangle_vertices[2]
        
        # Calculate the center of the base
        base_center = (v0 + v1) / 2.0
        
        # Vector from base_center to apex
        apex_vector = apex - base_center
        
        # Reflect apex across base to create 4th vertex
        # This completes the square (actually creates a rhombus/parallelogram)
        fourth_vertex = base_center - apex_vector
        vertices.append(fourth_vertex)
        
        # Reorder to form proper square: A, B, C, D going around
        # A = v0, B = v1, C = apex, D = fourth_vertex
        # Actually want: A, B, D, C (counterclockwise square)
        square_vertices = [v0, v1, fourth_vertex, apex]
        
        return np.array(square_vertices)
    
    def _expand_square_to_cube(self, square_vertices: np.ndarray, factor: float) -> np.ndarray:
        """
        Geometric expansion: Square → Cube (2D → 3D)
        
        Extrude the square along the third dimension to create a cube.
        Uses the third dimension size from self.dimensions.
        """
        if len(square_vertices) < 4:
            return square_vertices
        
        vertices = list(square_vertices)
        
        # Get third dimension size
        dim_z = self.dimensions[2] if len(self.dimensions) > 2 else self.size
        
        # Extrusion vector (along z-axis)
        if self.dimension >= 3:
            extrusion = np.zeros(self.dimension)
            extrusion[2] = dim_z * factor
            
            # Create top face by extruding each vertex of the square
            for vertex in square_vertices:
                top_vertex = vertex + extrusion
                vertices.append(top_vertex)
        
        return np.array(vertices)
    
    def _expand_to_hypercube(self, cube_vertices: np.ndarray, target_dim: int, factor: float) -> np.ndarray:
        """
        Geometric expansion: Cube → Hypercube (3D → 4D+)
        
        Recursively extrude along higher dimensions.
        Uses dimension sizes from self.dimensions for each axis.
        """
        vertices = cube_vertices
        
        # Current dimension is 3, expand to target_dim
        for dim in range(3, target_dim):
            new_vertices = []
            
            # Get dimension size for this axis
            dim_size = self.dimensions[dim] if len(self.dimensions) > dim else self.size
            
            # Extrusion vector for this dimension
            extrusion = np.zeros(max(self.dimension, dim + 1))
            extrusion[dim] = dim_size * factor
            
            # Copy existing vertices
            for vertex in vertices:
                # Pad vertex to new dimension if needed
                if len(vertex) <= dim:
                    padded = np.zeros(dim + 1)
                    padded[:len(vertex)] = vertex
                    new_vertices.append(padded)
                else:
                    new_vertices.append(vertex)
            
            # Create extruded vertices
            for vertex in new_vertices[:len(vertices)]:
                extruded = vertex + extrusion
                new_vertices.append(extruded)
            
            vertices = np.array(new_vertices)
        
        return vertices
    
    def _vertices_to_basis(self, vertices: np.ndarray) -> np.ndarray:
        """
        Extract lattice basis from geometric vertices.
        
        The basis vectors are scaled according to self.dimensions.
        """
        if len(vertices) == 0:
            return np.array([])
        
        # Ensure vertices are at least 2D
        if vertices.shape[1] < self.dimension:
            padded = np.zeros((len(vertices), self.dimension))
            padded[:, :vertices.shape[1]] = vertices
            vertices = padded
        
        basis = []
        
        # Method 1: Use vectors from origin to each vertex
        origin = vertices[0]  # First vertex as origin
        for i in range(1, min(self.dimension + 1, len(vertices))):
            edge_vector = vertices[i] - origin
            basis.append(edge_vector)
        
        # If we don't have enough basis vectors, create them using dimension sizes
        while len(basis) < self.dimension:
            new_vector = np.zeros(self.dimension)
            dim_idx = len(basis)
            dim_size = self.dimensions[dim_idx] if len(self.dimensions) > dim_idx else self.size
            new_vector[dim_idx] = dim_size
            basis.append(new_vector)
        
        basis = np.array(basis[:self.dimension])
        
        return basis
    
    def _fill_lattice_space(self) -> List[np.ndarray]:
        """
        Alien Optimization: Materialize all points instantly using 
        tensor expansion rather than iterative addition.
        """
        # Create coordinate ranges for each dimension
        ranges = [np.arange(d) for d in self.dimensions]
        
        # Expand into N-dimensional grid (The "Mesh")
        grids = np.meshgrid(*ranges, indexing='ij')
        
        # Stack and reshape to list of points (The "Collapse")
        # This is 300x faster than the loop
        points = np.stack(grids, axis=-1).reshape(-1, self.dimension)
        
        return list(points) # Optional: keep as numpy array for true speed
    
    def compress(self, compression_ratio: float = 0.5, verbose: bool = True) -> np.ndarray:
        """
        Geometric compression: Reverse the expansion sequence.
        
        This reveals hidden patterns by collapsing the structure back toward singularity.
        
        Args:
            compression_ratio: How much to compress (0.0 = none, 1.0 = full collapse)
            verbose: Print compression steps
            
        Returns:
            Compressed lattice basis
        """
        if self.vertices is None or self.lattice_basis is None:
            raise ValueError("Must generate lattice before compressing")
        
        if verbose:
            print("="*60)
            print("GEOMETRIC COMPRESSION")
            print("="*60)
            print(f"Compression ratio: {compression_ratio}")
            print()
        
        # Calculate center of mass (the attractor point)
        center = np.mean(self.vertices, axis=0)
        
        if verbose:
            print(f"[1] Center of mass: {center}")
            print(f"    Compressing all vertices toward center...")
        
        # Compress vertices toward center
        compressed_vertices = []
        for vertex in self.vertices:
            # Vector from vertex to center
            to_center = center - vertex
            # Move vertex toward center by compression_ratio
            new_vertex = vertex + compression_ratio * to_center
            compressed_vertices.append(new_vertex)
        
        compressed_vertices = np.array(compressed_vertices)
        
        if verbose:
            print(f"    Vertices compressed")
            print()
            print(f"[2] Extracting compressed basis...")
        
        # Extract new basis from compressed structure
        compressed_basis = self._vertices_to_basis(compressed_vertices)
        
        if verbose:
            print(f"    Compressed basis: {compressed_basis.shape}")
            print()
            print("="*60)
            print("COMPRESSION COMPLETE")
            print("="*60)
            print()
        
        self.vertices = compressed_vertices
        self.lattice_basis = compressed_basis
        
        return compressed_basis
    
    def get_geometric_metrics(self) -> dict:
        """
        Calculate geometric properties of the lattice.
        
        Returns metrics based on geometric form, not algebraic properties:
        - Volume: Geometric volume of the lattice shape
        - Surface area: Boundary measure
        - Diameter: Maximum distance between vertices
        - Symmetry: Measure of geometric symmetry
        """
        if self.vertices is None:
            return {}
        
        metrics = {}
        
        # Volume: Use convex hull volume (requires scipy)
        try:
            from scipy.spatial import ConvexHull
            if self.dimension <= 3 and len(self.vertices) >= 4:
                hull = ConvexHull(self.vertices)
                metrics['volume'] = hull.volume
                metrics['surface_area'] = hull.area
        except:
            # Fallback: Bounding box volume
            if len(self.vertices) > 0:
                mins = np.min(self.vertices, axis=0)
                maxs = np.max(self.vertices, axis=0)
                ranges = maxs - mins
                metrics['volume'] = np.prod(ranges)
        
        # Diameter: Maximum distance between any two vertices
        max_dist = 0.0
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                dist = np.linalg.norm(self.vertices[i] - self.vertices[j])
                if dist > max_dist:
                    max_dist = dist
        metrics['diameter'] = max_dist
        
        # Center of mass
        metrics['center'] = tuple(np.mean(self.vertices, axis=0))
        
        # Number of vertices
        metrics['num_vertices'] = len(self.vertices)
        
        # Basis determinant (if square)
        if self.lattice_basis is not None:
            if self.lattice_basis.shape[0] == self.lattice_basis.shape[1]:
                try:
                    metrics['determinant'] = np.linalg.det(self.lattice_basis)
                except:
                    pass
        
        return metrics
    
    def visualize_ascii(self, plane: str = 'xy'):
        """
        ASCII visualization of the lattice (for 2D/3D projection).
        
        Args:
            plane: Which plane to project onto ('xy', 'xz', 'yz')
        """
        if self.vertices is None:
            print("No lattice to visualize. Call generate() first.")
            return
        
        # Determine which dimensions to plot
        if plane == 'xy':
            dim1, dim2 = 0, 1
        elif plane == 'xz':
            dim1, dim2 = 0, 2
        elif plane == 'yz':
            dim1, dim2 = 1, 2
        else:
            dim1, dim2 = 0, 1
        
        # Extract 2D coordinates
        if self.vertices.shape[1] > dim2:
            coords_2d = self.vertices[:, [dim1, dim2]]
        else:
            # Pad if needed
            coords_2d = np.zeros((len(self.vertices), 2))
            if self.vertices.shape[1] > dim1:
                coords_2d[:, 0] = self.vertices[:, dim1]
            if self.vertices.shape[1] > dim2:
                coords_2d[:, 1] = self.vertices[:, dim2]
        
        # Normalize to ASCII grid (40x20)
        width, height = 60, 25
        
        if len(coords_2d) == 0:
            return
        
        min_x, max_x = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        min_y, max_y = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        
        # Avoid division by zero
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot vertices
        for coord in coords_2d:
            x = int((coord[0] - min_x) / range_x * (width - 1))
            y = int((coord[1] - min_y) / range_y * (height - 1))
            y = height - 1 - y  # Flip y-axis
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '●'
        
        # Print grid
        print(f"\nLattice visualization ({plane} plane):")
        print("┌" + "─" * width + "┐")
        for row in grid:
            print("│" + ''.join(row) + "│")
        print("└" + "─" * width + "┘")


def demo():
    """Demonstrate the geometric lattice generator"""
    print("\n" + "="*70)
    print(" GEOMETRIC LATTICE GENERATOR DEMO")
    print("="*70 + "\n")
    
    # Demo 1: 100x100 2D lattice
    print("\n### Demo 1: 100x100 2D Lattice ###\n")
    gen_100x100 = GeometricLatticeGenerator(dimensions=(100, 100))
    basis_100x100 = gen_100x100.generate(expansion_factor=1.0, verbose=True, fill_lattice=True)
    print("Generated 100x100 basis:")
    print(basis_100x100)
    print()
    
    # Demo 2: 100x100x100 3D lattice
    print("\n### Demo 2: 100x100x100 3D Lattice ###\n")
    gen_100x100x100 = GeometricLatticeGenerator(dimensions=(100, 100, 100))
    basis_100x100x100 = gen_100x100x100.generate(expansion_factor=1.0, verbose=True, fill_lattice=False)
    print("Generated 100x100x100 basis:")
    print(basis_100x100x100)
    print()
    
    # Demo 3: 50x75x25 irregular lattice
    print("\n### Demo 3: 50x75x25 Irregular 3D Lattice ###\n")
    gen_irregular = GeometricLatticeGenerator(dimensions=(50, 75, 25))
    basis_irregular = gen_irregular.generate(expansion_factor=1.0, verbose=True, fill_lattice=False)
    print("Generated 50x75x25 basis:")
    print(basis_irregular)
    print()
    
    # Demo 4: Small 10x10 with full fill and visualization
    print("\n### Demo 4: 10x10 Filled Lattice with Visualization ###\n")
    gen_10x10 = GeometricLatticeGenerator(dimensions=(10, 10))
    basis_10x10 = gen_10x10.generate(expansion_factor=1.0, verbose=True, fill_lattice=True)
    print(f"Total lattice points generated: {len(gen_10x10.lattice_points)}")
    print("First 10 lattice points:")
    for i, point in enumerate(gen_10x10.lattice_points[:10]):
        print(f"  Point {i}: {point}")
    gen_10x10.visualize_ascii('xy')
    
    # Demo 5: 4D hypercube 5x5x5x5
    print("\n### Demo 5: 5x5x5x5 4D Hypercube ###\n")
    gen_4d = GeometricLatticeGenerator(dimensions=(5, 5, 5, 5))
    basis_4d = gen_4d.generate(expansion_factor=1.0, verbose=True, fill_lattice=True)
    print("Generated 4D basis:")
    print(basis_4d)
    print(f"Total 4D lattice points: {len(gen_4d.lattice_points)}")
    print()
    
    # Demo 6: Compression
    print("\n### Demo 6: Geometric Compression (10x10x10) ###\n")
    gen_compress = GeometricLatticeGenerator(dimensions=(10, 10, 10))
    gen_compress.generate(expansion_factor=1.0, verbose=False, fill_lattice=True)
    print(f"Before compression: {len(gen_compress.lattice_points)} points")
    compressed_basis = gen_compress.compress(compression_ratio=0.7, verbose=True)
    print("Compressed basis:")
    print(compressed_basis)
    print(f"After compression: {len(gen_compress.lattice_points)} points")
    print()
    
    # Demo 7: Metrics
    print("\n### Demo 7: Geometric Metrics (20x30x40) ###\n")
    gen_metrics = GeometricLatticeGenerator(dimensions=(20, 30, 40))
    gen_metrics.generate(expansion_factor=1.0, verbose=False)
    metrics = gen_metrics.get_geometric_metrics()
    print("Lattice metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
