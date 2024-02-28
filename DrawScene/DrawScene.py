#! /usr/bin/env python3

import math

import glfw
from OpenGL.GL import *

from matrices import translate, rotate, scale, projection, scale_xyz

from renderables import RenderableSphere, RenderableFloor, RenderableScene, RenderableTransformer, RenderableCube, \
    RenderableDiamond, RenderableCylinder

from world import World


class Application:

    @staticmethod
    def create_glfw_window(version_major: int, version_minor: int):
        """Create a window using GLFW."""

        # Set up window creation hints.

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, version_minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create the window.

        window = glfw.create_window(640, 480, "DrawScene", None, None)
        if not window:
            raise RuntimeError("Unable to create window using GLFW.")

        return window

    @staticmethod
    def run():

        """Main entry point."""

        if not glfw.init():
            raise RuntimeError("Unable to initialize GLFW.")

        # Create a GLFW window and set it as the current OpenGL context.

        window = Application.create_glfw_window(4, 1)

        glfw.set_framebuffer_size_callback(window, lambda *args: Application.framebuffer_size_callback(*args))
        glfw.set_key_callback(window, lambda *args: Application.key_callback(*args))

        glfw.make_context_current(window)

        glfw.swap_interval(1)
        glPointSize(1)
        world = World()

        # Create the scene model.
        scene = RenderableScene()

        draw_floor = False
        if draw_floor:
            scene.add_model(RenderableFloor(8.0, 8.0))

        draw_earth = False
        if draw_earth:

            # add center earth with smaller earths around it.

            earth = RenderableSphere()

            scene.add_model(
                RenderableTransformer(
                    earth,
                     lambda: translate(0, 4.0, 0) @ scale(4.0) @ rotate(0, 1, 0, world.time())
                )
            )

            num_around = 0
            for ei_ in range(num_around):
                scene.add_model(
                    RenderableTransformer(
                        earth,
                        (lambda ei:
                         lambda: translate(6.5 * math.cos(ei / num_around * math.tau), 6.5 * math.sin(ei / num_around * math.tau), 0.0) @ scale(2.0) @ rotate(1, 0, 0, world.time())
                         )(ei_)
                    )
                )

        draw_cylinder = False
        if draw_cylinder:
            scene.add_model(
                RenderableTransformer(
                    RenderableCylinder(12, scale_xyz(0.1, 0.1, 20)),
                    lambda: translate(0, 4.0, 0) @ scale(4.0) @ rotate(1, 0, 0, 0.1 * world.time())
                )
            )

        draw_diamond = True
        if draw_diamond:
            scene.add_model(
                RenderableTransformer(
                    RenderableDiamond(),
                    lambda: translate(0, 0.0, 0) @ rotate(1, 0, 0, 0.1 * world.time())
                )
            )

        view_scene = RenderableTransformer(
            scene,
            lambda: translate(0.0, -2.5, -25.0) @ rotate(0, 1, 0, world.time() * 0)
        )

        # Prepare loop.

        frame_counter = 0
        t_prev = None

        glClearColor(0.12, 0.12, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)

        fov_degrees = 30.0
        near_plane = 0.5
        far_plane = 50.0

        while not glfw.window_should_close(window):

            t_now = glfw.get_time()
            if t_prev is not None:
                frame_duration = (t_now - t_prev)
                # print("@@ {:20.4f} ms".format(frame_duration * 1000.0))
            t_prev = t_now

            world.set_time(t_now)

            # Make perspective projection matrix.

            (framebuffer_width, framebuffer_height) = glfw.get_framebuffer_size(window)
            m_projection = projection(framebuffer_width, framebuffer_height, fov_degrees, near_plane, far_plane)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view_scene.render(m_projection)

            glfw.swap_buffers(window)
            glfw.poll_events()
            frame_counter += 1

        view_scene.close()

        glfw.destroy_window(window)
        glfw.terminate()

    @staticmethod
    def framebuffer_size_callback(_window, width, height):
        print("Resizing framebuffer:", width, height)
        glViewport(0, 0, width, height)

    @staticmethod
    def key_callback(window, key: int, scancode: int, action: int, mods: int):

        if action == glfw.PRESS:
            match key:
                case glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
