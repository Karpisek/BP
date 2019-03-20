class Info:
    def __init__(self):
        self.fps = None
        self.height = None
        self.width = None

    def set_info(self, fps=None, height=None, width=None):
        self.fps = fps
        self.height = height
        self.width = width

        print(f"fps: {self.fps}, height: {self.height}, width: {self.width}")
