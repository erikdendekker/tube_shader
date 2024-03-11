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

    def render(self, m_projection, m_view, m_model):
        for model in self._models:
            model.render(m_projection, m_view, m_model)
