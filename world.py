import glfw


class World:

    def __init__(self):
        self._sample_time = glfw.get_time()

    def sample_time(self):
        self._sample_time = glfw.get_time()
        return self._sample_time

    def time(self):
        return self._sample_time
