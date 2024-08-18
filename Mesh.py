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

    def set_world_position(self, vec):
        self.pose_lre[:3] = vec

        self.H_local2world = np.linalg.inv(t.lre2homo(self.pose_lre))

    
    def set_world_orientation(self, vec):
        self.pose_lre[3:] = vec

        self.H_local2world = np.linalg.inv(t.lre2homo(self.pose_lre))

    @property
    def vertices(self):
        vertices_world = [self.H_local2world @ np.array([*v, 1]) for v in self._vertices]
        return [v[:3] for v in vertices_world]