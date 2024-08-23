import numpy as np
import cv2
import robotransforms.euclidean as t

class Camera:
    def __init__(self, K):
        self.K = K
        self.pose_lre = np.array([0, 0, 0, 0, 0, 0]).astype(float)
        self.H_world2camera = np.eye(4)

        self.lights = [
            (np.array([0.0, 1.0, 1.0]), np.array([50, 100, 50])),
            (np.array([-1.0, 1.0, 0.0]), np.array([50, 50, 100]))
        ]

        self.ambient_light = np.array([50., 50., 50.])

        self.lights = [ (l / np.linalg.norm(l), i.astype(np.float32)) for l, i in self.lights ]

    def set_world_position(self, vec):
        self.pose_lre[:3] = vec
        self.H_world2camera = t.lre2homo(self.pose_lre)

    def set_world_orientation(self, vec):
        self.pose_lre[3:] = vec
        self.H_world2camera = t.lre2homo(self.pose_lre)
        
    
    def render(self, scene):
        img = np.zeros((self.K[1][2]*2, self.K[0][2]*2, 3), dtype=np.uint8)

        vert_world = scene.world_vertices
        normals_world = scene.world_normals
        # vert_pixel_z = self.verticies_to_pixels(vert_world)
        vert_pixel_z = np.array([ self.vertex_to_pixel(v) for v in vert_world ])

        # Create a list of face indices sorted by depth
        face_depths = [-np.mean(vert_pixel_z[face][:, 2]) for face in scene.faces]
        sorted_faces = np.argsort(face_depths)
        
        
        for i in sorted_faces:

            face = scene.faces[i]
            normal = normals_world[i]

            pts = np.array([vert_pixel_z[i][:2] for i in face], np.int32)
            pts = pts.reshape((-1, 1, 2))

            brightness = self.ambient_light.copy()

            for light_vector, color in self.lights:
                brightness += color * np.dot(light_vector, normal)

            brightness = np.clip(brightness, 0, 255)


            cv2.fillPoly(img, [pts], color=brightness.astype(int).tolist())
            # cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255))
            
                

        return img
    
    def vertex_to_pixel(self, vertex):
        vertex_camera_space = self.H_world2camera @ np.array([*vertex, 1])
        z_depth = vertex_camera_space[2]

        # Perspective division -- divide by z
        vertex_camera_space = vertex_camera_space[:3] / vertex_camera_space[2]
        
        vertex_pixel_space = (self.K @ vertex_camera_space)[:2].astype(int)
        return np.array([vertex_pixel_space[0], vertex_pixel_space[1], z_depth])
    
    def verticies_to_pixels(self, vert_world):
        # Convert vertices to homogeneous coordinates by adding a 1 at the end of each vertex
        ones = np.ones((vert_world.shape[0], 1))
        vert_world_homogeneous = np.hstack((vert_world, ones))

        # Apply the world to camera transformation
        vertices_camera_space = (self.H_world2camera @ vert_world_homogeneous.T).T
        
        # Extract z_depth
        z_depth = vertices_camera_space[:, 2]

        # Perspective division -- divide by z
        vertices_camera_space /= vertices_camera_space[:, 2].reshape(-1, 1)

        # Apply the intrinsic matrix K to get pixel coordinates
        vertices_pixel_space = (self.K @ vertices_camera_space[:, :3].T).T
        
        # Convert to integer pixel coordinates
        vertices_pixel_space = vertices_pixel_space[:, :2].astype(int)
        
        # Combine pixel coordinates with z_depth
        vert_pixel_z = np.hstack((vertices_pixel_space, z_depth.reshape(-1, 1)))

        return vert_pixel_z