import cv2

class MouseDragTracker:
    def __init__(self):
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.delta_x = 0
        self.delta_y = 0

    def start_drag(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button pressed
            self.dragging = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.end_point = (x, y)
                self.delta_x = self.end_point[0] - self.start_point[0]
                self.delta_y = self.end_point[1] - self.start_point[1]

        elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
            self.dragging = False
            self.end_point = (x, y)
            self.delta_x = self.end_point[0] - self.start_point[0]
            self.delta_y = self.end_point[1] - self.start_point[1]

    def get_deltas(self):
        return self.delta_x, self.delta_y