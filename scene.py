from renderable import Renderable


class Scene(Renderable):

    def __init__(self):
        self._models = []

    def close(self):
        for model in self._models:
            model.close()
        self._models.clear()

    def add_model(self, model):
        self._models.append(model)

    def render(self, projection_matrix, view_matrix, model_matrix):
        for model in self._models:
            model.render(projection_matrix, view_matrix, model_matrix)
