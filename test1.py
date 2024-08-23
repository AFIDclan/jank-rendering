import numpy as np
import cv2
from Mesh import Mesh
from Camera import Camera
from MouseDragTracker import MouseDragTracker

K = np.array([
    [800, 0, 640],
    [0, 800, 360],
    [0, 0, 1]
])

cube = Mesh()
# cube.load('teapot.obj')
# cube.load('cow.obj')
cube.load('cube.obj')

cube.set_world_orientation([0, np.pi, 0])

camera = Camera(K)

# camera.set_world_position([0, -4, -7])

tracker = MouseDragTracker()
cv2.namedWindow('image')
cv2.setMouseCallback('image', tracker.start_drag)

camera_orientation = [0, 0, 0]
camera_orientation_drag_start = [0, 0, 0]

was_dragging = False

while True:
    camera.set_world_orientation(camera_orientation)

    img = camera.render(cube)

    if tracker.dragging:

        if not was_dragging:
            camera_orientation_drag_start = camera_orientation.copy()

        was_dragging = True

        delta_x, delta_y = tracker.get_deltas()

        camera_orientation[1] = camera_orientation_drag_start[1] - delta_y / 150
        camera_orientation[2] = camera_orientation_drag_start[2] + delta_x / 150
    else:
        was_dragging = False

    cube.set_world_orientation([0, cube.pose_lre[4], cube.pose_lre[5]+0.01])

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
        break
