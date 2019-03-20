class Info:
    def __init__(self):
        self.fps = None
        self.height = None
        self.width = None
        self.track_boxes = True

    def set_info(self, fps=None, height=None, width=None, track_boxes=True):
        self.fps = fps
        self.height = height
        self.width = width
        self.track_boxes = track_boxes

        print(f"fps: {self.fps}, height: {self.height}, width: {self.width}, track boxes: {self.track_boxes}")
