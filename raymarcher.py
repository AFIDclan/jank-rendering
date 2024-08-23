import numpy as np
import cv2

# n_rays_x = 120
# n_rays_y = 64

# K = np.array([
#     [100, 0, n_rays_x//2],
#     [0, 100, n_rays_y//2],
#     [0, 0, 1]
# ])
n_rays_x = 640
n_rays_y = 360

K = np.array([
    [800, 0, n_rays_x//2],
    [0, 800, n_rays_y//2],
    [0, 0, 1]
])

img_out = np.zeros((n_rays_y, n_rays_x, 3), dtype=np.uint8)

sphere_center = np.array([0, 0, 6])
sphere_radius = 1.0


ray_dirs = []


for i in range(n_rays_x):
    for j in range(n_rays_y):

        ray = np.array([
            i,
            j,
            1
        ])

        ray_dirs.append(ray)
    
ray_dirs = np.array(ray_dirs)
ray_origins = ray_dirs[:, :2]

ray_dirs = np.linalg.inv(K) @ ray_dirs.T
ray_dirs = ray_dirs.T
ray_dirs = ray_dirs[:, :3]

ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1)[:, None]

ray_positons = np.zeros_like(ray_dirs)

rays = np.concatenate([ray_positons, ray_dirs, ray_origins], axis=1)

print(rays[0])

def check_ray_collision(ray):
    if np.linalg.norm(ray[:3] - sphere_center) < sphere_radius:

        center_to_ray = ray[:3] - sphere_center
        center_to_ray /= np.linalg.norm(center_to_ray)
        img_out[int(ray[7]), int(ray[6])] = [(128 * center_to_ray[0]) + 128, (128 * center_to_ray[1]) + 128, (128 * center_to_ray[2]) + 128]

        return True
    
    return False

np.set_printoptions(precision=3, suppress=True)

def step_rays(rays,ray_step_size=1e-1):
    ray_ind_to_remove = []
    print("Step rays: " + str(rays.shape[0]))
    for i in range(rays.shape[0]):
        # if (i==9):
        #     print(rays[i])
        ray = rays[i]
        ray[:3] += ray[3:6] * ray_step_size

        hit = check_ray_collision(ray)

        if hit:
            ray_ind_to_remove.append(i)

    rays = np.delete(rays, ray_ind_to_remove, axis=0)

    return rays

for i in range(100):
    print("Step", i)
    rays = step_rays(rays)

img_out = cv2.resize(img_out, (640, 360), interpolation=cv2.INTER_NEAREST)
cv2.imshow('image', img_out)
cv2.waitKey(0)