import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import colorsys

@dataclass
class Vertex:
    """Represents a vertex in 3D space."""
    x: float
    y: float
    z: float
    
    def normalize(self) -> 'Vertex':
        """Normalize the vertex coordinates to lie on a unit sphere."""
        length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vertex(self.x/length, self.y/length, self.z/length)
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))

@dataclass
class Face:
    """Represents a face defined by vertex indices."""
    vertices: List[int]

class IcosphereGenerator:
    """Generates an icosahedral sphere with subdivision."""
    
    def __init__(self, subdivision_level: int = 1):
        """
        Initialize the icosphere generator.
        
        Args:
            subdivision_level: Number of times to subdivide the base icosahedron
        """
        self.subdivision_level = subdivision_level
        self.vertices: List[Vertex] = []
        self.faces: List[Face] = []
        self._vertex_cache: Dict[Tuple[int, int], int] = {}
        self._generate_base_icosahedron()
        
    def _generate_base_icosahedron(self) -> None:
        """Generate the base icosahedron vertices and faces."""
        # Golden ratio for icosahedron construction
        t = (1.0 + np.sqrt(5.0)) / 2.0
        
        # Add vertices
        self.vertices = [
            Vertex(-1, t, 0), Vertex(1, t, 0), Vertex(-1, -t, 0), Vertex(1, -t, 0),
            Vertex(0, -1, t), Vertex(0, 1, t), Vertex(0, -1, -t), Vertex(0, 1, -t),
            Vertex(t, 0, -1), Vertex(t, 0, 1), Vertex(-t, 0, -1), Vertex(-t, 0, 1)
        ]
        
        # Normalize vertices to put them on unit sphere
        self.vertices = [v.normalize() for v in self.vertices]
        
        # Define faces
        self.faces = [
            Face([0, 11, 5]), Face([0, 5, 1]), Face([0, 1, 7]), Face([0, 7, 10]), Face([0, 10, 11]),
            Face([1, 5, 9]), Face([5, 11, 4]), Face([11, 10, 2]), Face([10, 7, 6]), Face([7, 1, 8]),
            Face([3, 9, 4]), Face([3, 4, 2]), Face([3, 2, 6]), Face([3, 6, 8]), Face([3, 8, 9]),
            Face([4, 9, 5]), Face([2, 4, 11]), Face([6, 2, 10]), Face([8, 6, 7]), Face([9, 8, 1])
        ]
    
    def _get_middle_point(self, p1_idx: int, p2_idx: int) -> int:
        """
        Find or create a vertex between two existing vertices.
        
        Args:
            p1_idx: Index of first vertex
            p2_idx: Index of second vertex
            
        Returns:
            Index of the middle vertex
        """
        key = tuple(sorted([p1_idx, p2_idx]))
        if key in self._vertex_cache:
            return self._vertex_cache[key]
            
        p1, p2 = self.vertices[p1_idx], self.vertices[p2_idx]
        middle = Vertex(
            (p1.x + p2.x) / 2.0,
            (p1.y + p2.y) / 2.0,
            (p1.z + p2.z) / 2.0
        ).normalize()
        
        self.vertices.append(middle)
        idx = len(self.vertices) - 1
        self._vertex_cache[key] = idx
        return idx
    
    def subdivide(self) -> None:
        """Perform one level of subdivision on all faces."""
        faces = self.faces[:]
        self.faces = []
        
        for face in faces:
            v1, v2, v3 = face.vertices
            
            # Get midpoints
            a = self._get_middle_point(v1, v2)
            b = self._get_middle_point(v2, v3)
            c = self._get_middle_point(v3, v1)
            
            # Create four new faces
            self.faces.append(Face([v1, a, c]))
            self.faces.append(Face([v2, b, a]))
            self.faces.append(Face([v3, c, b]))
            self.faces.append(Face([a, b, c]))
    
    def generate(self) -> Tuple[List[Vertex], List[Face]]:
        """
        Generate the subdivided icosphere.
        
        Returns:
            Tuple of (vertices, faces)
        """
        for _ in range(self.subdivision_level):
            self.subdivide()
        return self.vertices, self.faces

    def generate_goldberg(self) -> Tuple[List[Vertex], List[List[int]]]:
        """
        Generate the Goldberg polyhedron (hexagons and pentagons) from the icosahedron
        using the dual polyhedron approach.
        
        Returns:
            Tuple of (vertices, faces) where faces are lists of vertex indices forming polygons
        """
        # First generate the subdivided icosphere
        vertices, triangle_faces = self.generate()
        
        # Create a mapping of vertex to all faces that contain it
        vertex_to_faces: Dict[int, Set[int]] = defaultdict(set)
        for face_idx, face in enumerate(triangle_faces):
            for vertex_idx in face.vertices:
                vertex_to_faces[vertex_idx].add(face_idx)
        
        # Create face centers - these will be our new vertices
        face_centers: List[Vertex] = []
        for face in triangle_faces:
            v1, v2, v3 = [vertices[i] for i in face.vertices]
            center = Vertex(
                (v1.x + v2.x + v3.x) / 3.0,
                (v1.y + v2.y + v3.y) / 3.0,
                (v1.z + v2.z + v3.z) / 3.0
            ).normalize()
            face_centers.append(center)
        
        # For each original vertex, create a face by connecting the centers
        # of all triangles that share this vertex
        new_faces = []
        for vertex_idx in range(len(vertices)):
            if not vertex_to_faces[vertex_idx]:
                continue
            
            # Get all faces that share this vertex
            adjacent_faces = list(vertex_to_faces[vertex_idx])
            
            # Create a face by connecting the centers of these faces
            # We need to order these correctly to form the polygon
            current_face = adjacent_faces[0]
            ordered_centers = [current_face]
            used_faces = {current_face}
            
            while len(ordered_centers) < len(adjacent_faces):
                # Find the next face that shares an edge with the current face
                current_vertices = set(triangle_faces[current_face].vertices)
                
                for next_face in adjacent_faces:
                    if next_face in used_faces:
                        continue
                    
                    next_vertices = set(triangle_faces[next_face].vertices)
                    # If these faces share an edge (2 vertices), and one is our center vertex
                    if len(current_vertices & next_vertices) == 2 and vertex_idx in next_vertices:
                        ordered_centers.append(next_face)
                        used_faces.add(next_face)
                        current_face = next_face
                        break
            
            new_faces.append(ordered_centers)
        
        return face_centers, new_faces

def add_noise_to_vertex(vertex: Vertex, noise_level: float = 0.05) -> Vertex:
    """
    Add random noise to a vertex while keeping it roughly on the sphere surface.
    
    Args:
        vertex: The vertex to add noise to
        noise_level: Amount of noise to add (0-1)
        
    Returns:
        A new vertex with added noise
    """
    noise = np.random.normal(0, noise_level, 3)
    new_vertex = Vertex(
        vertex.x + noise[0],
        vertex.y + noise[1],
        vertex.z + noise[2]
    )
    return new_vertex.normalize()

def generate_organic_color(base_hue: float, saturation_range: Tuple[float, float] = (0.4, 0.6),
                         value_range: Tuple[float, float] = (0.6, 0.8)) -> Tuple[float, float, float]:
    """
    Generate an organic-looking color by varying saturation and value.
    
    Args:
        base_hue: Base hue value (0-1)
        saturation_range: Range for saturation variation
        value_range: Range for value variation
        
    Returns:
        RGB color tuple
    """
    saturation = np.random.uniform(*saturation_range)
    value = np.random.uniform(*value_range)
    return colorsys.hsv_to_rgb(base_hue, saturation, value)

def plot_icosphere(subdivision_level: int = 1) -> None:
    """
    Plot the icosphere using matplotlib.
    
    Args:
        subdivision_level: Number of subdivisions to perform
    """
    generator = IcosphereGenerator(subdivision_level)
    vertices, faces = generator.generate()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert vertices to numpy arrays for easier plotting
    verts = np.array([(v.x, v.y, v.z) for v in vertices])
    
    # Create list of triangles for Poly3DCollection
    triangles = []
    for face in faces:
        triangle = verts[face.vertices]
        triangles.append(triangle)
    
    # Create the 3D polygon collection
    poly3d = Poly3DCollection(triangles, alpha=0.6)
    poly3d.set_facecolor('skyblue')
    poly3d.set_edgecolor('black')
    ax.add_collection3d(poly3d)
    
    # Set equal aspect ratio and remove axes for better visualization
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    
    # Set the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Set reasonable axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    plt.title(f'Icosphere (subdivision level {subdivision_level})')
    plt.show()

def plot_goldberg(subdivision_level: int = 1, noise_level: float = 0.02) -> None:
    """
    Plot the Goldberg polyhedron using matplotlib with interactive zoom and organic appearance.
    
    Args:
        subdivision_level: Number of subdivisions to perform
        noise_level: Amount of noise to add to vertices (0-1)
    """
    generator = IcosphereGenerator(subdivision_level)
    vertices, faces = generator.generate_goldberg()
    
    # Count pentagons and hexagons
    num_pentagons = sum(1 for face in faces if len(face) == 5)
    num_hexagons = sum(1 for face in faces if len(face) == 6)
    total_faces = num_pentagons + num_hexagons
    
    # Create figure with mouse button and scroll wheel bindings
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add noise to vertices and convert to numpy arrays
    noisy_vertices = [add_noise_to_vertex(v, noise_level) for v in vertices]
    verts = np.array([(v.x, v.y, v.z) for v in noisy_vertices])
    
    # Create list of polygons for Poly3DCollection
    polygons = []
    for face in faces:
        polygon = verts[face]
        polygon = np.vstack([polygon, polygon[0]])  # Close the polygon
        polygons.append(polygon)
    
    # Create the 3D polygon collection
    poly3d = Poly3DCollection(polygons, alpha=1.0)
    
    # Generate base colors for faces
    base_colors = []
    for face in faces:
        if len(face) == 5:
            base_colors.append(generate_organic_color(0.05))  # Pentagons in reddish-brown
        else:
            base_colors.append(generate_organic_color(0.35))  # Hexagons in green
    
    poly3d.set_facecolor(base_colors)
    poly3d.set_edgecolor((0, 0, 0, 0.3))
    ax.add_collection3d(poly3d)
    
    # Create info text box in bottom right corner
    info_text = ax.text2D(0.98, 0.02, '', transform=ax.transAxes,
                         ha='right', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Store references to important objects
    ax._polygons = polygons
    ax._faces = faces
    ax._base_colors = base_colors
    ax._selected_face = None
    ax._poly3d = poly3d
    ax._info_text = info_text
    
    def update_info_text(face_idx: Optional[int] = None) -> None:
        """Update the information text box."""
        if face_idx is None:
            info_text.set_text('')
        else:
            face = faces[face_idx]
            face_type = 'Pentagon' if len(face) == 5 else 'Hexagon'
            center = np.mean(verts[face], axis=0)
            info_text.set_text(
                f'Face ID: {face_idx}\n'
                f'Type: {face_type}\n'
                f'Vertices: {len(face)}\n'
                f'Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})'
            )
    
    def on_click(event):
        """Handle mouse click events for face selection."""
        if event.inaxes != ax or event.button != 1:  # Left click only
            return
        
        # Get mouse click coordinates
        mouse_x, mouse_y = event.xdata, event.ydata
        
        # Find closest face by checking centers
        min_dist = float('inf')
        closest_face_idx = None
        
        for idx, polygon in enumerate(polygons):
            center = np.mean(polygon[:-1], axis=0)  # Exclude the last point (duplicate for closing)
            
            # Project 3D point to 2D
            x2d, y2d, _ = proj3d.proj_transform(center[0], center[1], center[2], ax.get_proj())
            
            # Calculate distance to mouse click
            dist = np.sqrt((x2d - mouse_x)**2 + (y2d - mouse_y)**2)
            
            # Only consider faces that are "in front" based on z-coordinate after projection
            _, _, z = proj3d.proj_transform(center[0], center[1], center[2], ax.get_proj())
            if z > 0.5:  # Adjust this threshold if needed
                continue
                
            if dist < min_dist and dist < 0.1:  # Only select if click is close enough
                min_dist = dist
                closest_face_idx = idx
        
        # Update selection
        if closest_face_idx is not None:
            # Reset previous selection
            colors = base_colors.copy()
            if ax._selected_face is not None:
                colors[ax._selected_face] = base_colors[ax._selected_face]
            
            # Highlight new selection
            if ax._selected_face != closest_face_idx:
                ax._selected_face = closest_face_idx
                # Brighten the selected face
                selected_color = np.array(colors[closest_face_idx])
                selected_color = np.clip(selected_color * 1.3, 0, 1)  # Brighten by 30%
                colors[closest_face_idx] = selected_color
            else:
                ax._selected_face = None
            
            # Update visualization
            poly3d.set_facecolor(colors)
            update_info_text(ax._selected_face)
            fig.canvas.draw_idle()
        else:
            # If clicked empty space, clear selection
            if ax._selected_face is not None:
                ax._selected_face = None
                poly3d.set_facecolor(base_colors)
                update_info_text(None)
                fig.canvas.draw_idle()
    
    def on_scroll(event):
        """Handle scroll events for zooming."""
        if event.inaxes == ax:
            # Get the current scale
            scale = np.mean([abs(lim) for lim in ax.get_xlim()])
            
            # Zoom factor (adjust this to change zoom sensitivity)
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            
            # Apply zoom
            new_scale = scale * zoom_factor
            
            # Limit maximum zoom out to 2x initial scale
            if new_scale > 2 * ax._initial_scale:
                new_scale = 2 * ax._initial_scale
            # Limit maximum zoom in to 0.2x initial scale
            elif new_scale < 0.2 * ax._initial_scale:
                new_scale = 0.2 * ax._initial_scale
            
            ax.set_xlim(-new_scale, new_scale)
            ax.set_ylim(-new_scale, new_scale)
            ax.set_zlim(-new_scale, new_scale)
            
            fig.canvas.draw_idle()
    
    # Set equal aspect ratio and remove axes for better visualization
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    
    # Set the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Set reasonable axis limits
    initial_scale = 1.5
    ax.set_xlim(-initial_scale, initial_scale)
    ax.set_ylim(-initial_scale, initial_scale)
    ax.set_zlim(-initial_scale, initial_scale)
    
    # Store initial view limits
    ax._initial_scale = initial_scale
    
    # Connect the event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.title(f'Goldberg Polyhedron (level {subdivision_level})\n'
             f'Pentagons: {num_pentagons}, Hexagons: {num_hexagons}, Total: {total_faces}\n'
             f'Click tiles to select, use scroll wheel to zoom')
    
    plt.show()

if __name__ == "__main__":
    subdivision_level = 3
    noise_factor = 0.05
    noise_level = (noise_factor) * (subdivision_level ** (-3/2))
    plot_goldberg(subdivision_level=subdivision_level, noise_level=noise_level)  # Start with level 1 for clearer visualization 