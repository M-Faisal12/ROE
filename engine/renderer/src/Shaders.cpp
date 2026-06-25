// =============================================================================
// Shaders.cpp
//
//   Implementation of CreateShaderProgram() declared in Shaders.h.
// =============================================================================
#include "Shaders.h"
#include "Logger.h"

#include <string>

// =============================================================================
// CreateShaderProgram
// =============================================================================
GLuint CreateShaderProgram()
{
    Log("INIT", "Compiling vertex shader ...");

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint vertexCompileSuccess = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexCompileSuccess);
    if (!vertexCompileSuccess)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        Log("ERROR", std::string("Vertex shader compile failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Vertex shader compiled OK.");
    }

    Log("INIT", "Compiling fragment shader ...");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    GLint fragmentCompileSuccess = 0;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentCompileSuccess);
    if (!fragmentCompileSuccess)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        Log("ERROR", std::string("Fragment shader compile failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Fragment shader compiled OK.");
    }

    Log("INIT", "Linking shader program ...");

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint linkSuccess = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
    if (!linkSuccess)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        Log("ERROR", std::string("Shader program link failed:\n") + infoLog);
    }
    else
    {
        Log("INIT", "Shader program linked OK.  handle=" + std::to_string(program));
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}