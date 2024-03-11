from renderable import Renderable


class Transformer(Renderable):

    def __init__(self, model, func):
        self._model = model
        self._func = func

    def close(self):
        self._model.close()
        self._model = None
        self._func = None

    def render(self, m_projection, m_view, m_model):
        m_func = self._func()
        self._model.render(m_projection, m_view, m_model @ m_func)
