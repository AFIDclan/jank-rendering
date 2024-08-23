import numpy as np
from Mesh import Mesh
import cv2

class Ray:
    def __init__(self, origin, direction, pixel_origin):
        self.origin = origin
        self.direction = direction
        self.pixel_origin = pixel_origin
        self.color = np.array([1.0, 1.0, 1.0])
        self.illumination = 0.2
        self.ended = False


def point_in_triangle_barycentric(p, a, b, c):
    # Compute vectors        
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)  # Changed to <= 1

def point_in_triangle(p, verts, normal):

    indicies = [[0, 1], [1, 2], [2, 0]]

    D0 = None

    for i in range(3):
        e0 = verts[indicies[i][0]]
        e1 = verts[indicies[i][1]]

        edge = e1 - e0
        vp = p - e0

        C = np.cross(edge, vp)
        D = np.dot(normal, C)

        if D0 is None:
            D0 = D
        else:
            if D0 * D < 0:
                return False
    return True

        
def reflect(v, n):
    return v - 2 * np.dot(v, n) * n



def ray_plane_intersection(ray_origin, ray_direction, plane_origin, plane_normal):

    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
    denom = np.dot(ray_direction, plane_normal)

    if abs(denom) < 1e-3:
        return None
    
    t =  np.dot(plane_origin - ray_origin, plane_normal) / denom

    if t < 0:
        return None  # Intersection point is behind the ray origin

    return ray_origin + t * ray_direction




cube = Mesh()
cube.load('cube.obj')
cube.set_world_position([0, 2.0, 4])
# cube.set_world_orientation([0, 45 * np.pi / 180, 45 * np.pi / 180])

light = Mesh()
light.load('cube.obj')
light.set_world_position([0, -3, 17])

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
    rays.append(Ray(np.array([0, 0, 0]), ray_dirs[i], ray_pixel_origins[i]))


img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)

for n in range(2):

    for i, ray in enumerate(rays):
        if ray.ended:
            continue

        if (i+1) % 1000 == 0:
            print(f'{i+1}/{len(rays)}')

        for t, face in enumerate(cube.faces):
            v0 = cube.world_vertices[face[0]]
            v1 = cube.world_vertices[face[1]]
            v2 = cube.world_vertices[face[2]]

            world_bounding_box = cube.world_bounding_box

            intersection = ray_plane_intersection(ray.origin, ray.direction, v0, cube.world_normals[t])

            if intersection is not None:
                if world_bounding_box[0][0] <= intersection[0] <= world_bounding_box[0][1] and \
                    world_bounding_box[1][0] <= intersection[1] <= world_bounding_box[1][1] and \
                    world_bounding_box[2][0] <= intersection[2] <= world_bounding_box[2][1]:
                    if point_in_triangle_barycentric(intersection, v0, v1, v2):
                        # img[int(ray.pixel_origin[1]), int(ray.pixel_origin[0])] = np.array([255, 255, 255])
                        ray.color *= np.array([1.0, 0.0, 0.0])
                        ray.origin = intersection
                        ray.direction = reflect(ray.direction, cube.world_normals[t])
                        # ray.direction += np.random.normal(0, 0.1, 3)
                        ray.direction /= np.linalg.norm(ray.direction)

                        # break

        for t, face in enumerate(light.faces):
            v0 = light.world_vertices[face[0]]
            v1 = light.world_vertices[face[1]]
            v2 = light.world_vertices[face[2]]

            world_bounding_box = light.world_bounding_box

            intersection = ray_plane_intersection(ray.origin, ray.direction, v0, light.world_normals[t])

            if intersection is not None:
                if world_bounding_box[0][0] <= intersection[0] <= world_bounding_box[0][1] and \
                    world_bounding_box[1][0] <= intersection[1] <= world_bounding_box[1][1] and \
                    world_bounding_box[2][0] <= intersection[2] <= world_bounding_box[2][1]:
                    if point_in_triangle_barycentric(intersection, v0, v1, v2):
                        ray.illumination = 1.0
                        ray.ended = True
                        # break


for i, ray in enumerate(rays):
    img[int(ray.pixel_origin[1]), int(ray.pixel_origin[0])] = (ray.illumination * 255 * ray.color).astype(int)

cv2.imshow('image', img)
cv2.waitKey(0)