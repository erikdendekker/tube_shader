#! /usr/bin/env python3
import numpy as np
import glfw
from OpenGL.GL import *

from matrices import translate, rotate, scale, perspective_projection
from floor import Floor
from scene import Scene
from transformer import Transformer
from sphere_imposter import SphereImpostor
from cylinder_imposter import CylinderImpostor
from world import World


def create_scene(world: World) -> Scene:
    scene = Scene()

    scene.add_model(
        Transformer(
            Floor(8.0, 8.0),
            lambda: translate((0, -1, 0))
        )
    )

    scene.add_model(
        Transformer(
            CylinderImpostor('./assets/moon.png'),
            lambda: translate((+0.25, 0.0, 0)) @
                    rotate((0, 1, 0), 0.0 * world.time()) @
                    rotate((1, 0, 0), 0.5 * world.time()) @
                    scale((0.2, 0.2, 4.0))
        )
    )

    scene.add_model(
        Transformer(
            SphereImpostor('./assets/earth.png'),
            lambda: translate((+0.0, 0.0, 0)) @
                    scale((1.0, 1.0, 1.0)) @
                    rotate((0, 1, 0), 1 * world.time())
        )
    )

    return scene


class Application:

    def __init__(self):
        self.render_distance = 12.0

    @staticmethod
    def create_glfw_window(version_major: int, version_minor: int):
        """ Create a window using GLFW """

        # Set up window creation hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, version_minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create the window
        window = glfw.create_window(1024, 768, 'DrawScene', None, None)
        if not window:
            raise RuntimeError('Unable to create window using GLFW')

        return window

    def run(self):
        if not glfw.init():
            raise RuntimeError('Unable to initialize GLFW')

        # Create a GLFW window and set it as the current OpenGL context
        window = Application.create_glfw_window(4, 1)

        glfw.set_framebuffer_size_callback(window, lambda *args: Application.framebuffer_size_callback(*args))
        glfw.set_key_callback(window, lambda *args: self.key_callback(*args))

        glfw.make_context_current(window)

        # Create World and Scene
        world = World()
        scene = create_scene(world)

        # Prepare loop
        glfw.swap_interval(1)
        glPointSize(1)

        glClearColor(0.12, 0.12, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        fov_degrees = 30.0
        near_plane = 0.5
        far_plane = 1000.0

        frame_nr = 0
        t_previous_wallclock = None
        while not glfw.window_should_close(window):

            t_wallclock = glfw.get_time()

            if t_previous_wallclock is not None:
                fps = 1.0 / (t_wallclock - t_previous_wallclock)
                glfw.set_window_title(window, f'frame: {frame_nr}, fps: {fps:.1f} Hz')
            t_previous_wallclock = t_wallclock

            # Sample time to ensure all queries to world.time() will be identical
            world.sample_time()

            # Make view matrix
            m_view = translate((0.0, 0, -self.render_distance)) @ rotate((0, 1, 0), world.time() * 0.0)

            # Make model matrix
            m_model = np.identity(4)

            # Make perspective projection matrix
            (framebuffer_width, framebuffer_height) = glfw.get_framebuffer_size(window)

            if framebuffer_width * framebuffer_height == 0:
                continue

            m_projection = perspective_projection(framebuffer_width, framebuffer_height, fov_degrees, near_plane, far_plane)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            scene.render(m_projection, m_view, m_model)

            glfw.swap_buffers(window)
            glfw.poll_events()
            frame_nr += 1

        scene.close()

        glfw.destroy_window(window)
        glfw.terminate()

    @staticmethod
    def framebuffer_size_callback(_window, width, height):
        print('Resizing framebuffer:', width, height)
        glViewport(0, 0, width, height)

    def key_callback(self, window, key: int, scancode: int, action: int, mods: int):

        if action in (glfw.PRESS, glfw.REPEAT):
            match key:
                case glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
