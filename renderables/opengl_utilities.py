"""OpenGL utility functions"""
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


def create_opengl_program(prefix: str) -> tuple:

    shader_type_definitions = [ ("v", GL_VERTEX_SHADER),
                                ("g", GL_GEOMETRY_SHADER),
                                ("f", GL_FRAGMENT_SHADER) ]

    shaders = []
    shader_program = None

    try:
        shader_program = glCreateProgram()

        for (letter, shader_type) in shader_type_definitions:
            shader = make_shader("{}_{}.glsl".format(prefix, letter), shader_type)
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
