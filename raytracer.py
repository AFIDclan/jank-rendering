import numpy as np
from Mesh import Mesh
import cv2

class Ray:
    def __init__(self, origin, direction, pixel_origin):
        self.origin = origin
        self.direction = direction
        self.pixel_origin = pixel_origin


def ray_plane_intersection(ray_origin, ray_direction, plane_origin, plane_normal):

    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
    denom = np.dot(ray_direction, plane_normal)

    if abs(denom) < 1e-3:
        return None
    
    t =  np.dot(plane_origin - ray_origin, plane_normal) / denom

    return ray_origin + t * ray_direction




cube = Mesh()
cube.load('cube.obj')

cam_width = 300
cam_height = 160

K = np.array([
    [200, 0, cam_width//2],
    [0, 200, cam_height//2],
    [0, 0, 1]
])
# cam_width = 640
# cam_height = 360

# K = np.array([
#     [400, 0, cam_width//2],
#     [0, 400, cam_height//2],
#     [0, 0, 1]
# ])

ray_pixel_origins = []

for i in range(cam_width):
    for j in range(cam_height):
        ray_pixel_origins.append(np.array([i, j, 1]))
    

ray_pixel_origins = np.array(ray_pixel_origins)

ray_dirs = np.linalg.inv(K) @ ray_pixel_origins.T
ray_dirs = ray_dirs.T
ray_dirs = ray_dirs[:, :3]
ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1)[:, None]

rays = []


for i in range(len(ray_pixel_origins)):
    rays.append(Ray(np.array([0, 0, -7]), ray_dirs[i], ray_pixel_origins[i]))


img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

for i, ray in enumerate(rays):

    if (i+1) % 1000 == 0:
        print(f'{i+1}/{len(rays)}')

    for t, face in enumerate(cube.faces):
        v0 = cube.world_vertices[face[0]]
        v1 = cube.world_vertices[face[1]]
        v2 = cube.world_vertices[face[2]]

        # face_bbox = np.array([
        #     [min(v0[0], v1[0], v2[0]), max(v0[0], v1[0], v2[0])],
        #     [min(v0[1], v1[1], v2[1]), max(v0[1], v1[1], v2[1])],
        #     [min(v0[2], v1[2], v2[2]), max(v0[2], v1[2], v2[2])]
        # ])

        face_bbox = cube.world_face_bounding_boxes[t]

        intersection = ray_plane_intersection(ray.origin, ray.direction, v0, cube.world_normals[face[0]])

        if intersection is not None:
            if face_bbox[0][0] <= intersection[0] <= face_bbox[0][1] and \
                face_bbox[1][0] <= intersection[1] <= face_bbox[1][1] and \
                face_bbox[2][0] <= intersection[2] <= face_bbox[2][1]:

                img[int(ray.pixel_origin[1]), int(ray.pixel_origin[0])] = np.array([255, 255, 255])
                break

cv2.imshow('image', img)
cv2.waitKey(0)