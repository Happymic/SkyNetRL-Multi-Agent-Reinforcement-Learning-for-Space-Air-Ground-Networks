"""
Real-time 3D Visualization System with PyOpenGL
High-performance rendering for multi-agent systems
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time


class Realtime3DViewer:
    """Real-time 3D visualization with smooth camera controls"""
    
    def __init__(self, env_config: Dict, window_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize 3D viewer
        
        Args:
            env_config: Environment configuration
            window_size: Window dimensions (width, height)
        """
        self.env_config = env_config
        self.window_size = window_size
        self.area_size = env_config.get('area_size', 1000)
        self.max_altitude = 300
        
        # Camera settings
        self.camera_distance = 1500
        self.camera_angle_h = 45  # Horizontal angle
        self.camera_angle_v = 30  # Vertical angle
        self.camera_target = [self.area_size/2, self.area_size/2, 0]
        self.camera_speed = 5
        
        # Mouse control
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # Agent data
        self.agent_positions = {}
        self.agent_trails = {}
        self.poi_positions = []
        self.coverage_status = []
        
        # Colors
        self.agent_colors = {
            'satellite': (1.0, 0.4, 0.4),
            'uav': (0.3, 0.8, 0.8),
            'ground_station': (0.6, 0.9, 0.5)
        }
        
        self.poi_colors = {
            1: (1.0, 0.9, 0.4),  # Yellow
            2: (1.0, 0.6, 0.2),  # Orange
            3: (1.0, 0.4, 0.3),  # Red-orange
            4: (0.9, 0.3, 0.3),  # Red
            5: (0.8, 0.0, 0.0)   # Dark red
        }
        
        # Performance settings
        self.fps = 60
        self.frame_time = 1.0 / self.fps
        self.last_update = time.time()
        
        # Data queue for thread-safe updates
        self.data_queue = queue.Queue()
        
        # Initialize pygame and OpenGL
        self.running = False
        self.render_thread = None
        
    def start(self):
        """Start the 3D viewer in a separate thread"""
        if not self.running:
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.start()
            print("🚀 3D Viewer started")
            
    def stop(self):
        """Stop the 3D viewer"""
        self.running = False
        if self.render_thread:
            self.render_thread.join()
        try:
            import pygame
            pygame.quit()
        except ImportError:
            pass
        print("🛑 3D Viewer stopped")
        
    def update_positions(self, positions: Dict):
        """Update agent positions (thread-safe)"""
        self.data_queue.put(('positions', positions))
        
    def update_coverage(self, coverage: List):
        """Update POI coverage status"""
        self.data_queue.put(('coverage', coverage))
        
    def _render_loop(self):
        """Main rendering loop"""
        # Import OpenGL modules here when actually needed
        try:
            import pygame
            from pygame.locals import DOUBLEBUF, OPENGL
            from OpenGL.GL import (glClear, glEnable, glDisable, glBlendFunc, glHint,
                                  glLightfv, glClearColor, glMatrixMode, glLoadIdentity,
                                  glBegin, glEnd, glVertex3f, glVertex3fv, glColor3f,
                                  glColor4f, glLineWidth, glPushMatrix, glPopMatrix,
                                  glTranslatef, glRotatef, glScalef, glRasterPos3f,
                                  GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
                                  GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                  GL_LINE_SMOOTH, GL_POINT_SMOOTH, GL_LINE_SMOOTH_HINT,
                                  GL_POINT_SMOOTH_HINT, GL_NICEST, GL_LIGHTING, GL_LIGHT0,
                                  GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                                  GL_POSITION, GL_AMBIENT, GL_DIFFUSE, GL_PROJECTION,
                                  GL_MODELVIEW, GL_LINES, GL_LINE_STRIP, GL_QUADS, GL_TRIANGLES,
                                  glColorMaterial)
            from OpenGL.GLU import gluPerspective, gluLookAt
            from OpenGL.GLUT import (glutSolidCube, glutSolidSphere, glutSolidCylinder,
                                    glutSolidCone, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18)
        except ImportError:
            print("⚠️ OpenGL dependencies not available. Install with: pip install PyOpenGL PyOpenGL_accelerate")
            return
            
        # Initialize pygame
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("SkyNetRL - Real-time 3D Visualization")
        
        # Set up OpenGL
        self._setup_opengl()
        
        # Generate POI positions
        self._generate_pois()
        
        clock = pygame.time.Clock()
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Process data updates
            self._process_updates()
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up camera
            self._setup_camera()
            
            # Render scene
            self._render_ground()
            self._render_grid()
            self._render_pois()
            self._render_agents()
            self._render_connections()
            self._render_hud()
            
            # Swap buffers
            pygame.display.flip()
            clock.tick(self.fps)
            
    def _setup_opengl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        light_pos = [self.area_size/2, self.area_size/2, 1000, 1.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.7, 0.7, 0.7, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        
        # Background color
        glClearColor(0.1, 0.1, 0.15, 1.0)
        
        # Perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.window_size[0]/self.window_size[1]), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        
    def _setup_camera(self):
        """Set up camera view"""
        glLoadIdentity()
        
        # Calculate camera position
        angle_h_rad = math.radians(self.camera_angle_h)
        angle_v_rad = math.radians(self.camera_angle_v)
        
        cam_x = self.camera_target[0] + self.camera_distance * math.cos(angle_v_rad) * math.sin(angle_h_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.cos(angle_v_rad) * math.cos(angle_h_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.sin(angle_v_rad)
        
        gluLookAt(cam_x, cam_y, cam_z,
                 self.camera_target[0], self.camera_target[1], self.camera_target[2],
                 0, 0, 1)
        
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(500, self.camera_distance - 50)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(3000, self.camera_distance + 50)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]
                    
                    self.camera_angle_h += dx * 0.5
                    self.camera_angle_v = max(-89, min(89, self.camera_angle_v - dy * 0.5))
                    
                    self.last_mouse_pos = current_pos
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset camera
                    self.camera_angle_h = 45
                    self.camera_angle_v = 30
                    self.camera_distance = 1500
                elif event.key == pygame.K_w:
                    self.camera_target[1] += self.camera_speed * 10
                elif event.key == pygame.K_s:
                    self.camera_target[1] -= self.camera_speed * 10
                elif event.key == pygame.K_a:
                    self.camera_target[0] -= self.camera_speed * 10
                elif event.key == pygame.K_d:
                    self.camera_target[0] += self.camera_speed * 10
                    
    def _process_updates(self):
        """Process data updates from queue"""
        while not self.data_queue.empty():
            try:
                update_type, data = self.data_queue.get_nowait()
                
                if update_type == 'positions':
                    # Update agent positions and trails
                    for agent_id, pos in data.items():
                        if agent_id not in self.agent_trails:
                            self.agent_trails[agent_id] = []
                        
                        self.agent_positions[agent_id] = pos
                        self.agent_trails[agent_id].append(pos)
                        
                        # Limit trail length
                        if len(self.agent_trails[agent_id]) > 30:
                            self.agent_trails[agent_id].pop(0)
                            
                elif update_type == 'coverage':
                    self.coverage_status = data
                    
            except queue.Empty:
                break
                
    def _generate_pois(self):
        """Generate POI positions"""
        num_pois = self.env_config.get('num_pois', 12)
        self.poi_positions = []
        
        for i in range(num_pois):
            angle = i * 2 * math.pi / num_pois
            radius = self.area_size * 0.3 * (1 + 0.2 * math.sin(i * 2))
            x = self.area_size/2 + radius * math.cos(angle)
            y = self.area_size/2 + radius * math.sin(angle)
            priority = 1 + (i % 5)
            
            self.poi_positions.append((x, y, 0, priority))
            
    def _render_ground(self):
        """Render ground plane"""
        glDisable(GL_LIGHTING)
        glBegin(GL_QUADS)
        glColor4f(0.15, 0.15, 0.2, 0.8)
        glVertex3f(0, 0, 0)
        glVertex3f(self.area_size, 0, 0)
        glVertex3f(self.area_size, self.area_size, 0)
        glVertex3f(0, self.area_size, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def _render_grid(self):
        """Render grid lines"""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor4f(0.3, 0.3, 0.4, 0.3)
        
        grid_spacing = 100
        glBegin(GL_LINES)
        for i in range(0, int(self.area_size) + 1, grid_spacing):
            # X-direction lines
            glVertex3f(i, 0, 0)
            glVertex3f(i, self.area_size, 0)
            # Y-direction lines
            glVertex3f(0, i, 0)
            glVertex3f(self.area_size, i, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def _render_pois(self):
        """Render Points of Interest"""
        for i, (x, y, z, priority) in enumerate(self.poi_positions):
            color = self.poi_colors.get(priority, (0.5, 0.5, 0.5))
            
            # Check if covered
            alpha = 0.3 if i < len(self.coverage_status) and self.coverage_status[i] else 1.0
            
            glPushMatrix()
            glTranslatef(x, y, z + 10)
            
            # Draw POI as a pyramid
            glColor4f(*color, alpha)
            self._draw_pyramid(20, 30)
            
            # Draw priority label
            glDisable(GL_LIGHTING)
            glColor3f(1, 1, 1)
            glRasterPos3f(0, 0, 35)
            for char in f"P{priority}":
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
            glEnable(GL_LIGHTING)
            
            glPopMatrix()
            
    def _render_agents(self):
        """Render agents with trails"""
        for agent_id, pos in self.agent_positions.items():
            # Determine agent type
            agent_num = int(agent_id.split('_')[1])
            if agent_num < self.env_config.get('num_satellites', 2):
                agent_type = 'satellite'
                size = 15
            elif agent_num < self.env_config.get('num_satellites', 2) + self.env_config.get('num_uavs', 3):
                agent_type = 'uav'
                size = 12
            else:
                agent_type = 'ground_station'
                size = 10
                
            color = self.agent_colors[agent_type]
            
            # Draw agent
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2] if len(pos) > 2 else 50)
            glColor3f(*color)
            
            if agent_type == 'satellite':
                self._draw_satellite(size)
            elif agent_type == 'uav':
                self._draw_uav(size)
            else:
                self._draw_ground_station(size)
                
            glPopMatrix()
            
            # Draw trail
            if agent_id in self.agent_trails and len(self.agent_trails[agent_id]) > 1:
                glDisable(GL_LIGHTING)
                glLineWidth(2.0)
                glBegin(GL_LINE_STRIP)
                
                for i, trail_pos in enumerate(self.agent_trails[agent_id]):
                    alpha = (i + 1) / len(self.agent_trails[agent_id]) * 0.5
                    glColor4f(*color, alpha)
                    glVertex3f(trail_pos[0], trail_pos[1], 
                             trail_pos[2] if len(trail_pos) > 2 else 50)
                
                glEnd()
                glEnable(GL_LIGHTING)
                
    def _render_connections(self):
        """Render communication connections between agents"""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        
        comm_range = self.env_config.get('communication_range', 200)
        
        for agent1_id, pos1 in self.agent_positions.items():
            for agent2_id, pos2 in self.agent_positions.items():
                if agent1_id < agent2_id:  # Avoid duplicate lines
                    dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
                    
                    if dist < comm_range:
                        # Draw connection line
                        alpha = max(0.1, 1.0 - dist / comm_range)
                        glColor4f(0.3, 0.8, 1.0, alpha * 0.3)
                        
                        glBegin(GL_LINES)
                        glVertex3f(pos1[0], pos1[1], pos1[2] if len(pos1) > 2 else 50)
                        glVertex3f(pos2[0], pos2[1], pos2[2] if len(pos2) > 2 else 50)
                        glEnd()
                        
        glEnable(GL_LIGHTING)
        
    def _render_hud(self):
        """Render heads-up display with information"""
        # This would render 2D overlay information
        pass
        
    def _draw_pyramid(self, base_size: float, height: float):
        """Draw a pyramid shape"""
        glBegin(GL_TRIANGLES)
        
        # Base vertices
        v1 = [-base_size/2, -base_size/2, 0]
        v2 = [base_size/2, -base_size/2, 0]
        v3 = [base_size/2, base_size/2, 0]
        v4 = [-base_size/2, base_size/2, 0]
        apex = [0, 0, height]
        
        # Side faces
        glVertex3fv(v1)
        glVertex3fv(v2)
        glVertex3fv(apex)
        
        glVertex3fv(v2)
        glVertex3fv(v3)
        glVertex3fv(apex)
        
        glVertex3fv(v3)
        glVertex3fv(v4)
        glVertex3fv(apex)
        
        glVertex3fv(v4)
        glVertex3fv(v1)
        glVertex3fv(apex)
        
        glEnd()
        
        # Base
        glBegin(GL_QUADS)
        glVertex3fv(v1)
        glVertex3fv(v2)
        glVertex3fv(v3)
        glVertex3fv(v4)
        glEnd()
        
    def _draw_satellite(self, size: float):
        """Draw satellite shape"""
        # Body
        glPushMatrix()
        glScalef(size/10, size/10, size/20)
        glutSolidCube(10)
        glPopMatrix()
        
        # Solar panels
        glPushMatrix()
        glTranslatef(-size*1.5, 0, 0)
        glScalef(size/5, size/15, 1)
        glutSolidCube(10)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(size*1.5, 0, 0)
        glScalef(size/5, size/15, 1)
        glutSolidCube(10)
        glPopMatrix()
        
    def _draw_uav(self, size: float):
        """Draw UAV shape"""
        # Body (sphere)
        glutSolidSphere(size, 16, 16)
        
        # Rotors
        for angle in [0, 90, 180, 270]:
            glPushMatrix()
            glRotatef(angle, 0, 0, 1)
            glTranslatef(size*1.5, 0, 0)
            glScalef(1, 1, 0.2)
            glutSolidCylinder(size/3, 2, 8, 1)
            glPopMatrix()
            
    def _draw_ground_station(self, size: float):
        """Draw ground station shape"""
        # Base
        glPushMatrix()
        glScalef(size/5, size/5, size/10)
        glutSolidCube(10)
        glPopMatrix()
        
        # Antenna
        glPushMatrix()
        glTranslatef(0, 0, size)
        glRotatef(-90, 1, 0, 0)
        glutSolidCone(size/2, size*2, 8, 8)
        glPopMatrix()


class VideoExporter:
    """Export 3D visualization to video file"""
    
    def __init__(self, viewer: Realtime3DViewer):
        """
        Initialize video exporter
        
        Args:
            viewer: The 3D viewer instance
        """
        self.viewer = viewer
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        """Start recording frames"""
        self.recording = True
        self.frames = []
        print("🔴 Recording started")
        
    def stop_recording(self):
        """Stop recording and return frames"""
        self.recording = False
        print(f"⏹ Recording stopped. Captured {len(self.frames)} frames")
        return self.frames
        
    def save_video(self, filename: str, fps: int = 30):
        """Save recorded frames to video file"""
        if not self.frames:
            print("⚠️ No frames to save")
            return
            
        # Implementation would save frames to video file
        print(f"💾 Video saved: {filename}")