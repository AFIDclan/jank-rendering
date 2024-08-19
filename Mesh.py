import numpy as np
import robotransforms.euclidean as t

class Mesh:
    def __init__(self):
        self._vertices = []
        self.faces = []
        self.pose_lre = np.array([0, 0, 0, 0, 0, 0]).astype(float)
        self.H_local2world = np.eye(4)

    def load(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    self._vertices.append(list(map(float, values[1:4])))
                elif values[0] == 'f':
                    self.faces.append(list(values[1:4]))

            if self.faces:
                if '/' in self.faces[0][0]:  # vertex/texture/normal format
                    self.faces = [[int(v.split('/')[0]) - 1 for v in face] for face in self.faces]
                else:
                    self.faces = [[int(v) - 1 for v in face] for face in self.faces]


        self._vertices = np.array(self._vertices)
        self.world_vertices = self._vertices


        # Compute normals
        self._normals = []
        for face in self.faces:
            v0 = self._vertices[face[0]]
            v1 = self._vertices[face[1]]
            v2 = self._vertices[face[2]]

            normal = np.cross(v2 - v0, v1 - v0)
            normal /= np.linalg.norm(normal)

            self._normals.append(normal)

        self._normals = np.array(self._normals)
        self.world_normals = self._normals

            

    def set_world_position(self, vec):
        self.pose_lre[:3] = vec
        self.H_local2world = np.linalg.inv(t.lre2homo(self.pose_lre))
        self._generate_world() # Update world vertices and normals

    
    def set_world_orientation(self, vec):
        self.pose_lre[3:] = vec
        self.H_local2world = np.linalg.inv(t.lre2homo(self.pose_lre))
        self._generate_world() # Update world vertices and normals

    def _generate_world(self):
        self.world_vertices = [(self.H_local2world @ np.array([*v, 1]))[:3] for v in self._vertices]
        self.world_normals = [self.H_local2world[:3, :3] @ n for n in self._normals]
