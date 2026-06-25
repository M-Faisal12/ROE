// =============================================================================
// Shaders.h
//
//   Vertex/fragment shader source (as raw strings) + the helper that
//   compiles and links them into a GL shader program.
// =============================================================================
#pragma once

#include "../Externals/Glad/glad.h"

// =============================================================================
// Vertex Shader
// =============================================================================
inline const char *vertexShaderSource = R"(
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// =============================================================================
// Fragment Shader
// =============================================================================
inline const char *fragmentShaderSource = R"(
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D uDiffuseMap;

void main()
{
    vec3 lightDirection = normalize(vec3(-1.0, -1.0, -1.0));
    float diffuse = max(dot(normalize(Normal), -lightDirection), 0.0);
    float ambientStrength = 0.2;
    vec4 textureColor = texture(uDiffuseMap, TexCoord);
    vec3 finalColor   = textureColor.rgb * (ambientStrength + diffuse);
    FragColor = vec4(finalColor, textureColor.a);
}
)";

// Compiles vertexShaderSource + fragmentShaderSource above, links them into
// a program, logs compile/link errors via Logger.h, and returns the handle.
GLuint CreateShaderProgram();