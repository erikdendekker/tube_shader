""" OpenGL utility functions """
import os

from OpenGL.GL import *


def make_shader(filename: str, shader_type):
    """Read a shader source from disk and compile it"""
    try:
        with open(filename, "rb") as fi:
            shader_source = fi.read()
    except FileNotFoundError:
        print("Shader source not found: {!r} from {!r}".format(filename, os.getcwd()))
        return None

    shader = None
    try:
        shader = glCreateShader(shader_type)

        print("Compiling shader: {!r} ...".format(filename))

        glShaderSource(shader, shader_source)
        glCompileShader(shader)
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status != GL_TRUE:
            log = glGetShaderInfoLog(shader)
            print("Error while compiling shader:", repr(log))
            raise RuntimeError("Error while compiling shader")

    except BaseException:
        if shader is not None:
            glDeleteShader(shader)
        raise  # Re-raise exception

    return shader


def create_shader(shader_source: str, shader_type):
    """Compile shader"""

    shader = None
    try:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shader_source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            log = glGetShaderInfoLog(shader)
            print("Error while compiling shader:", repr(log))
            raise RuntimeError("Error while compiling shader")

    except BaseException:
        if shader is not None:
            glDeleteShader(shader)
        raise  # Re-raise exception

    return shader


def create_opengl_program(geometry_shader : str = None,
                          vertex_shader   : str = None,
                          fragment_shader : str = None) -> tuple:

    shader_types = [ (geometry_shader, GL_GEOMETRY_SHADER),
                     (vertex_shader,   GL_VERTEX_SHADER),
                     (fragment_shader, GL_FRAGMENT_SHADER) ]

    shaders = []
    shader_program = None

    try:
        shader_program = glCreateProgram()

        for (shader_source, shader_type) in shader_types:
            if shader_source is not None:
                shader = create_shader(shader_source, shader_type)
                if shader is not None:
                    shaders.append(shader)

        for shader in shaders:
            glAttachShader(shader_program, shader)

        print("Linking shader program with {} shaders ...".format(len(shaders)))
        glLinkProgram(shader_program)
        status = glGetProgramiv(shader_program,  GL_LINK_STATUS)
        if status != GL_TRUE:
            log = glGetProgramInfoLog(shader_program)
            print("Error while linking shader program:", repr(log))
            raise RuntimeError("Error while linking shader program")

        print("Shader program linked successfully")

    except BaseException:
        if shader_program is not None:
            glDeleteProgram(shader_program)
        for shader in shaders:
            glDeleteShader(shader)
        raise  # Re-raise exception

    return shaders, shader_program
