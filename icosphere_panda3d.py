import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter, GeomLines
from panda3d.core import GeomNode, NodePath
from panda3d.core import Point3, Vec4, DirectionalLight, AmbientLight
from panda3d.core import WindowProperties
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

class IcosphereViewer(ShowBase):
    """
    Panda3D viewer for the icosphere.
    """
    def __init__(self, subdivision_level: int = 1, water_level: float = 0.4, 
                 elevation_scale: float = 0.1, axial_tilt: float = 23.5):
        """
        Initialize the Panda3D viewer.
        
        Args:
            subdivision_level: Number of subdivisions for the icosphere
            water_level: Water level height (0-1), default 0.4
            elevation_scale: Scale factor for elevation height (default 0.1 = 10% of radius)
            axial_tilt: Planet's axial tilt in degrees (default 23.5 like Earth)
        """
        ShowBase.__init__(self)
        
        # Set up window properties
        props = WindowProperties()
        props.setSize(1024, 768)
        props.setTitle("Icosphere Viewer")
        self.win.requestProperties(props)
        
        # Set up camera
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(Point3(0, 0, 0))
        
        # Initialize camera parameters
        self.min_zoom = -2.0  # Closest distance
        self.max_zoom = -15.0  # Furthest distance
        self.zoom_speed = 0.5  # Zoom speed multiplier
        
        # Set water level and elevation scale
        self.water_level = max(0.0, min(1.0, water_level))  # Clamp between 0 and 1
        self.elevation_scale = max(0.0, elevation_scale)  # Ensure non-negative
        self.axial_tilt = axial_tilt  # Store axial tilt
        
        # Add lighting
        self._setup_lighting()
        
        # Create the icosphere
        self.create_icosphere(subdivision_level)
        
        # Create a pivot node for proper tilted rotation
        self.pivot = self.render.attachNewNode("pivot")
        self.icosphere.reparentTo(self.pivot)
        
        # Apply axial tilt
        self.pivot.setP(-self.axial_tilt)  # Negative for astronomical convention
        
        # Initialize rotation state
        self.is_rotating = True
        self.rotation_speed = 15  # degrees per second
        self.current_angle = 0
        self.last_time = self.taskMgr.globalClock.getFrameTime()
        self.pause_time = 0
        
        # Initialize borders state
        self.borders_visible = True
        
        # Add rotation task
        self.taskMgr.add(self.rotate_icosphere, "RotateIcosphere")
        
        # Set up keyboard input
        self.accept("space", self.toggle_rotation)
        self.accept("b", self.toggle_borders)
        
        # Set up mouse wheel zoom
        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)
        
    def _setup_lighting(self) -> None:
        """Set up the scene lighting to simulate sunlight."""
        # Create Ambient Light (dimmer for better directional light effect)
        amblight = AmbientLight('ambient')
        amblight.setColor(Vec4(0.15, 0.15, 0.15, 1))
        self.ambient = self.render.attachNewNode(amblight)
        self.render.setLight(self.ambient)
        
        # Create Directional Light (stronger and positioned like the Sun)
        dirlight = DirectionalLight('directional')
        dirlight.setColor(Vec4(1.0, 0.95, 0.9, 1))  # Slightly warm sunlight color
        self.dirlight = self.render.attachNewNode(dirlight)
        # Position light to show axial tilt effects (like equatorial sun)
        self.dirlight.setPos(0, -100, 0)  # Far away on equatorial plane
        self.dirlight.lookAt(Point3(0, 0, 0))
        self.render.setLight(self.dirlight)
        
    def create_icosphere(self, subdivision_level: int) -> None:
        """
        Create and display the icosphere.
        
        Args:
            subdivision_level: Number of subdivisions
        """
        generator = IcosphereGenerator(subdivision_level)
        vertices, faces = generator.generate()
        
        # Generate elevation data for each vertex using Perlin-like noise
        elevations = {}
        for i, vertex in enumerate(vertices):
            # Use position as seed for pseudo-random but smooth elevation
            base_elevation = (np.sin(vertex.x * 3.0) * np.cos(vertex.y * 2.0) + 
                            np.cos(vertex.z * 4.0) * np.sin(vertex.x * 2.0)) * 0.5
            # Add some smaller scale variation
            detail = (np.sin(vertex.x * 8.0) * np.cos(vertex.y * 7.0) + 
                     np.cos(vertex.z * 9.0) * np.sin(vertex.x * 6.0)) * 0.2
            elevations[i] = base_elevation + detail
        
        # Normalize elevations to 0-1 range
        min_elev = min(elevations.values())
        max_elev = max(elevations.values())
        elev_range = max_elev - min_elev
        for i in elevations:
            elevations[i] = (elevations[i] - min_elev) / elev_range
        
        # Create color gradient for elevation
        def elevation_to_color(elevation: float) -> Tuple[float, float, float, float]:
            """Convert elevation to color using Earth-like gradient."""
            if elevation < self.water_level:  # Deep ocean to shallow ocean
                t = elevation / self.water_level
                return (0.0, 0.0, 0.5 + 0.3 * t, 1.0)
            elif elevation < self.water_level + 0.05:  # Ocean to beach transition
                t = (elevation - self.water_level) / 0.05
                return (0.3 * t, 0.3 * t, 0.8 - 0.3 * t, 1.0)
            elif elevation < self.water_level + 0.25:  # Lowlands to highlands
                t = (elevation - (self.water_level + 0.05)) / 0.25
                return (0.3 + 0.2 * t, 0.3 + 0.1 * t, 0.1, 1.0)
            else:  # Mountains to peaks
                t = (elevation - (self.water_level + 0.25)) / (1.0 - (self.water_level + 0.25))
                return (0.5 + 0.5 * t, 0.4 + 0.6 * t, 0.1 + 0.9 * t, 1.0)
        
        # Create two vertex formats: one for faces and one for borders
        face_format = GeomVertexFormat.getV3n3c4()
        face_vdata = GeomVertexData('icosphere_faces', face_format, Geom.UHStatic)
        
        # Create writers for vertex positions, normals, and colors
        vertex = GeomVertexWriter(face_vdata, 'vertex')
        normal = GeomVertexWriter(face_vdata, 'normal')
        color = GeomVertexWriter(face_vdata, 'color')
        
        # Add vertices with elevation-based colors
        for i, v in enumerate(vertices):
            # Scale vertex based on elevation, but only for land
            elevation = elevations[i]
            # Only apply elevation scaling for land above water level
            scale = 1.0 + max(0, elevation - self.water_level) * self.elevation_scale
            vertex.addData3(v.x * scale, v.y * scale, v.z * scale)
            normal.addData3(v.x, v.y, v.z)
            r, g, b, a = elevation_to_color(elevation)
            color.addData4(r, g, b, a)
        
        # Create the triangles
        face_prim = GeomTriangles(Geom.UHStatic)
        
        for face in faces:
            face_prim.addVertices(face.vertices[0], face.vertices[1], face.vertices[2])
        
        # Create the face geometry
        face_geom = Geom(face_vdata)
        face_geom.addPrimitive(face_prim)
        
        # Create a separate geometry for black borders
        border_format = GeomVertexFormat.getV3c4()
        border_vdata = GeomVertexData('icosphere_borders', border_format, Geom.UHStatic)
        
        border_vertex = GeomVertexWriter(border_vdata, 'vertex')
        border_color = GeomVertexWriter(border_vdata, 'color')
        
        # Create lines for borders
        border_prim = GeomLines(Geom.UHStatic)
        
        # Keep track of edges we've already added
        edges_added = set()
        
        for face in faces:
            # For each face, create three edges
            for i in range(3):
                v1_idx = face.vertices[i]
                v2_idx = face.vertices[(i + 1) % 3]
                edge = tuple(sorted([v1_idx, v2_idx]))
                
                if edge not in edges_added:
                    edges_added.add(edge)
                    v1 = vertices[v1_idx]
                    v2 = vertices[v2_idx]
                    
                    # Scale vertices based on elevation, but only for land
                    scale1 = 1.0 + max(0, elevations[v1_idx] - self.water_level) * self.elevation_scale
                    scale2 = 1.0 + max(0, elevations[v2_idx] - self.water_level) * self.elevation_scale
                    
                    # Add slightly offset vertices for the border
                    border_scale = 1.001  # Minimal offset to prevent z-fighting
                    border_vertex.addData3(v1.x * scale1 * border_scale, 
                                         v1.y * scale1 * border_scale, 
                                         v1.z * scale1 * border_scale)
                    border_vertex.addData3(v2.x * scale2 * border_scale, 
                                         v2.y * scale2 * border_scale, 
                                         v2.z * scale2 * border_scale)
                    
                    # Add dark gray color for borders (less harsh than black)
                    border_color.addData4(0.2, 0.2, 0.2, 1.0)
                    border_color.addData4(0.2, 0.2, 0.2, 1.0)
                    
                    # Add the line
                    border_prim.addVertices(len(edges_added) * 2 - 2, len(edges_added) * 2 - 1)
        
        # Create the border geometry
        border_geom = Geom(border_vdata)
        border_geom.addPrimitive(border_prim)
        
        # Create nodes for both geometries
        face_node = GeomNode('icosphere_faces')
        face_node.addGeom(face_geom)
        
        border_node = GeomNode('icosphere_borders')
        border_node.addGeom(border_geom)
        
        # Create the nodepath and attach both geometries to render
        self.icosphere = self.render.attachNewNode(face_node)
        self.borders = self.render.attachNewNode(border_node)
        self.borders.reparentTo(self.icosphere)
        
    def toggle_rotation(self):
        """Toggle the rotation state when spacebar is pressed."""
        current_time = self.taskMgr.globalClock.getFrameTime()
        
        if self.is_rotating:
            # Pausing - store the time we paused
            self.pause_time = current_time
        else:
            # Unpausing - adjust last_time by the duration we were paused
            pause_duration = current_time - self.pause_time
            self.last_time += pause_duration
            
        self.is_rotating = not self.is_rotating

    def rotate_icosphere(self, task: Task) -> Task.cont:
        """
        Rotate the icosphere continuously around its tilted axis.
        
        Args:
            task: The task instance
            
        Returns:
            Task.cont to continue the task
        """
        if self.is_rotating:
            current_time = self.taskMgr.globalClock.getFrameTime()
            dt = current_time - self.last_time
            self.current_angle += self.rotation_speed * dt
            self.icosphere.setH(self.current_angle)  # Rotate around local axis
            self.last_time = current_time
        return Task.cont

    def zoom_in(self):
        """Zoom camera in towards the icosphere."""
        current_y = self.cam.getY()
        new_y = min(current_y + self.zoom_speed, self.min_zoom)
        self.cam.setY(new_y)
        
    def zoom_out(self):
        """Zoom camera out from the icosphere."""
        current_y = self.cam.getY()
        new_y = max(current_y - self.zoom_speed, self.max_zoom)
        self.cam.setY(new_y)

    def toggle_borders(self):
        """Toggle the visibility of borders."""
        self.borders_visible = not self.borders_visible
        if self.borders_visible:
            self.borders.show()
        else:
            self.borders.hide()

def main():
    """Main function to run the Panda3D viewer."""
    app = IcosphereViewer(
        subdivision_level=8,
        water_level=0.4,
        elevation_scale=0.05,
        axial_tilt=23.5  # Earth-like tilt in degrees
    )
    app.run()

if __name__ == "__main__":
    main() 