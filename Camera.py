import numpy as np
import cv2
import robotransforms.euclidean as t

class Camera:
    def __init__(self, K):
        self.K = K
        self.pose_lre = np.array([0, 0, 0, 0, 0, 0]).astype(float)
        self.H_world2camera = np.eye(4)

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
        vert_pixel_z = np.array([ self.vertex_to_pixel(v) for v in vert_world ])

        # Create a list of face indices sorted by depth
        face_depths = [-np.mean(vert_pixel_z[face][:, 2]) for face in scene.faces]
        sorted_faces = np.argsort(face_depths)
        
        for i in sorted_faces:

            face = scene.faces[i]
            normal = normals_world[i]

            pts = np.array([vert_pixel_z[i][:2] for i in face], np.int32)
            pts = pts.reshape((-1, 1, 2))

            vec_vert_to_camera = vert_world[face[0]] - self.pose_lre[:3]
            vec_vert_to_camera /= np.linalg.norm(vec_vert_to_camera)
            
            alignment = np.dot(normal, vec_vert_to_camera)

            if alignment < 0:
                continue

            brightness = 255 * alignment



            cv2.fillPoly(img, [pts], color=(brightness, brightness, brightness))
            # cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0))
            
                

        return img
    
    def vertex_to_pixel(self, vertex):
        vertex_camera_space = self.H_world2camera @ np.array([*vertex, 1])
        z_depth = vertex_camera_space[2]

        # Perspective division -- divide by z
        vertex_camera_space = vertex_camera_space[:3] / vertex_camera_space[2]
        
        vertex_pixel_space = (self.K @ vertex_camera_space)[:2].astype(int)
        return np.array([vertex_pixel_space[0], vertex_pixel_space[1], z_depth])
    
