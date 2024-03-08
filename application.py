#! /usr/bin/env python3
import numpy as np

import glfw
from OpenGL.GL import *

from matrices import translate, rotate, scale, perspective_projection

from renderables import (RenderablePlanet, RenderableFloor, RenderableScene, RenderableModelTransformer, RenderableDiamond, RenderableSphereImpostor, RenderableCylinderImpostor)

from world import World


class Application:

    def __init__(self):
        self.diamond_model = None
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

        world = World()

        # Create the scene model
        scene = RenderableScene()

        draw_floor = True
        if draw_floor:
            scene.add_model(
                RenderableModelTransformer(
                    RenderableFloor(8.0, 8.0),
                    lambda: translate((0, -1, 0))
                )
            )

        draw_earth = False
        if draw_earth:

            # add center earth with smaller earths around it

            earth = RenderablePlanet('./assets/earth.png')

            scene.add_model(
                RenderableModelTransformer(
                    earth,
                     lambda: translate((0, 0.0, 0)) @
                             scale(4.0) @
                             rotate((0, 1, 0), world.time())
                )
            )

        draw_sphere_impostor = False
        if draw_sphere_impostor:

            sphere_imposter_constellation = RenderableScene()

            earth = RenderableSphereImpostor('./assets/earth.png')

            sphere_imposter_constellation.add_model(
                RenderableModelTransformer(
                    earth,
                    lambda: translate((+0.8, 0.0, 0)) @
                            scale((1.0, 1.0, 1.0)) @
                            rotate((0, 1, 0), 0 * world.time())
                )
            )

            moon = RenderableSphereImpostor('./assets/moon.png')

            sphere_imposter_constellation.add_model(
                RenderableModelTransformer(
                    moon,
                    lambda: translate((-0.8, 0.0, 0.3)) @
                            scale((1.0, 1.0, 1.0)) @
                            rotate((0, 1, 0), 0 * world.time())
                )
            )

            scene.add_model(
                RenderableModelTransformer(
                    sphere_imposter_constellation,
                    lambda: rotate((0, 1, 0), 1 * world.time())
                )
            )

        draw_cylinder_impostor = True
        if draw_cylinder_impostor:

            cylinder_impostor = RenderableCylinderImpostor('./assets/moon.png')

            scene.add_model(
                RenderableModelTransformer(
                    cylinder_impostor,
                    lambda: translate((+0.25, 0.0, 0)) @
                            rotate((0, 1, 0), 0.0 * world.time()) @
                            rotate((1, 0, 0), 0.5 * world.time()) @
                            scale((0.2, 0.2, 4.0))
                )
            )

            sphere_impostor = RenderableSphereImpostor('./assets/earth.png')

            scene.add_model(
                RenderableModelTransformer(
                    sphere_impostor,
                    lambda: translate((+0.0, 0.0, 0)) @
                            scale((1.0, 1.0, 1.0)) @
                            rotate((0, 1, 0), 1 * world.time())
                )
            )

        draw_diamond = False
        if draw_diamond:
            self.diamond_model = RenderableDiamond()

            scene.add_model(
                RenderableModelTransformer(
                    self.diamond_model,
                    lambda: translate((0, 0.0, 0)) @
                            rotate((1, 0, 0), 0.2 * world.time()) @
                            rotate((0, 0, 1), 0.3 * world.time()) @
                            rotate((0, 1, 0), 0.1 * world.time())
                )
            )

        # Prepare loop
        frame_counter = 0
        t_prev = None

        glfw.swap_interval(1)
        glPointSize(1)

        glClearColor(0.12, 0.12, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        fov_degrees = 30.0
        near_plane = 0.5
        far_plane = 1000.0

        while not glfw.window_should_close(window):

            t_now = world.sample_time()

            if t_prev is not None:
                frame_duration = (t_now - t_prev)
                #print("@@ {:20.4f} ms".format(frame_duration * 1000.0))
            t_prev = t_now

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
            frame_counter += 1

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
                case glfw.KEY_0:
                    if self.diamond_model is not None:
                        self.diamond_model.cut_mode = 0
                case glfw.KEY_1:
                    if self.diamond_model is not None:
                        self.diamond_model.cut_mode = 1 if self.diamond_model.cut_mode != 1 else 0
                case glfw.KEY_2:
                    if self.diamond_model is not None:
                        self.diamond_model.cut_mode = 2 if self.diamond_model.cut_mode != 2 else 0
                case glfw.KEY_3:
                    if self.diamond_model is not None:
                        self.diamond_model.cut_mode = 3 if self.diamond_model.cut_mode != 3 else 0
                case glfw.KEY_C:
                    if self.diamond_model is not None:
                        self.diamond_model.color_mode = (self.diamond_model.color_mode + 1) % 2
                case glfw.KEY_RIGHT_BRACKET:
                    if self.diamond_model is not None:
                        if (mods & glfw.MOD_ALT) != 0:
                            self.render_distance =  max(0.0, self.render_distance - 5.0)
                        elif (mods & glfw.MOD_SHIFT) != 0:
                            self.diamond_model.unit_cells_per_dimension = self.diamond_model.unit_cells_per_dimension + 2
                        else:
                            self.diamond_model.crystal_side_length = self.diamond_model.crystal_side_length + 1.0
                case glfw.KEY_LEFT_BRACKET:
                        if self.diamond_model is not None:
                            if (mods & glfw.MOD_ALT) != 0:
                                self.render_distance = self.render_distance + 5.0
                            elif (mods & glfw.MOD_SHIFT) != 0:
                                self.diamond_model.unit_cells_per_dimension = max(1, self.diamond_model.unit_cells_per_dimension - 2)
                            else:
                                self.diamond_model.crystal_side_length = max(0, self.diamond_model.crystal_side_length - 1.0)

            if self.diamond_model is not None:
                print('render distance {} diamond cut: {} diamond unit_cells_per_dimension: {} diamond crystal_side_length: {}'.format(
                    self.render_distance,
                    self.diamond_model.cut_mode,
                    self.diamond_model.unit_cells_per_dimension,
                    self.diamond_model.crystal_side_length
                ))


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
