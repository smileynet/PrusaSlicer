///|/ Copyright (c) 2026 Sam Biggins
///|/ PrusaSlicer is released under the terms of the AGPLv3 or higher
///|/
/// Headless EGL thumbnail renderer for PrusaSlicer CLI.
///
/// Uses EGL to create an offscreen OpenGL context without a display server,
/// renders the model with the same Gouraud lighting as the GUI, and returns
/// RGBA pixels in ThumbnailData format.

#include "CLIThumbnailRenderer.hpp"

#ifdef SLIC3R_CLI_THUMBNAILS

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include <boost/log/trivial.hpp>
#include <Eigen/Geometry>

#include "libslic3r/BoundingBox.hpp"

// GL function pointers (loaded at runtime to avoid link-time dependency on GL 2.0+)
static PFNGLCREATESHADERPROC        p_glCreateShader        = nullptr;
static PFNGLSHADERSOURCEPROC        p_glShaderSource        = nullptr;
static PFNGLCOMPILESHADERPROC       p_glCompileShader       = nullptr;
static PFNGLGETSHADERIVPROC         p_glGetShaderiv         = nullptr;
static PFNGLGETSHADERINFOLOGPROC    p_glGetShaderInfoLog    = nullptr;
static PFNGLCREATEPROGRAMPROC       p_glCreateProgram       = nullptr;
static PFNGLATTACHSHADERPROC        p_glAttachShader        = nullptr;
static PFNGLLINKPROGRAMPROC         p_glLinkProgram         = nullptr;
static PFNGLGETPROGRAMIVPROC        p_glGetProgramiv        = nullptr;
static PFNGLUSEPROGRAMPROC          p_glUseProgram          = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC  p_glGetUniformLocation  = nullptr;
static PFNGLUNIFORM4FPROC           p_glUniform4f           = nullptr;
static PFNGLUNIFORM1FPROC           p_glUniform1f           = nullptr;
static PFNGLUNIFORMMATRIX4FVPROC    p_glUniformMatrix4fv    = nullptr;
static PFNGLUNIFORMMATRIX3FVPROC    p_glUniformMatrix3fv    = nullptr;
static PFNGLGETATTRIBLOCATIONPROC   p_glGetAttribLocation   = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC  p_glEnableVertexAttribArray = nullptr;
static PFNGLDISABLEVERTEXATTRIBARRAYPROC p_glDisableVertexAttribArray = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC p_glVertexAttribPointer = nullptr;
static PFNGLGENFRAMEBUFFERSPROC     p_glGenFramebuffers     = nullptr;
static PFNGLBINDFRAMEBUFFERPROC     p_glBindFramebuffer     = nullptr;
static PFNGLFRAMEBUFFERTEXTURE2DPROC p_glFramebufferTexture2D = nullptr;
static PFNGLGENRENDERBUFFERSPROC    p_glGenRenderbuffers    = nullptr;
static PFNGLBINDRENDERBUFFERPROC    p_glBindRenderbuffer    = nullptr;
static PFNGLRENDERBUFFERSTORAGEPROC p_glRenderbufferStorage = nullptr;
static PFNGLFRAMEBUFFERRENDERBUFFERPROC p_glFramebufferRenderbuffer = nullptr;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC  p_glCheckFramebufferStatus  = nullptr;
static PFNGLDELETESHADERPROC        p_glDeleteShader        = nullptr;
static PFNGLDELETEPROGRAMPROC       p_glDeleteProgram       = nullptr;
static PFNGLDELETEFRAMEBUFFERSPROC  p_glDeleteFramebuffers  = nullptr;
static PFNGLDELETERENDERBUFFERSPROC p_glDeleteRenderbuffers = nullptr;
static PFNGLGENBUFFERSPROC          p_glGenBuffers          = nullptr;
static PFNGLBINDBUFFERPROC          p_glBindBuffer          = nullptr;
static PFNGLBUFFERDATAPROC          p_glBufferData          = nullptr;
static PFNGLDELETEBUFFERSPROC       p_glDeleteBuffers       = nullptr;
static PFNGLGENVERTEXARRAYSPROC     p_glGenVertexArrays     = nullptr;
static PFNGLBINDVERTEXARRAYPROC     p_glBindVertexArray     = nullptr;
static PFNGLDELETEVERTEXARRAYSPROC  p_glDeleteVertexArrays  = nullptr;

static bool load_gl_functions()
{
    #define LOAD(name) p_##name = (decltype(p_##name))eglGetProcAddress(#name); \
        if (!p_##name) { BOOST_LOG_TRIVIAL(error) << "CLI Thumbnail: Failed to load " #name; return false; }
    LOAD(glCreateShader)
    LOAD(glShaderSource)
    LOAD(glCompileShader)
    LOAD(glGetShaderiv)
    LOAD(glGetShaderInfoLog)
    LOAD(glCreateProgram)
    LOAD(glAttachShader)
    LOAD(glLinkProgram)
    LOAD(glGetProgramiv)
    LOAD(glUseProgram)
    LOAD(glGetUniformLocation)
    LOAD(glUniform4f)
    LOAD(glUniform1f)
    LOAD(glUniformMatrix4fv)
    LOAD(glUniformMatrix3fv)
    LOAD(glGetAttribLocation)
    LOAD(glEnableVertexAttribArray)
    LOAD(glDisableVertexAttribArray)
    LOAD(glVertexAttribPointer)
    LOAD(glGenFramebuffers)
    LOAD(glBindFramebuffer)
    LOAD(glFramebufferTexture2D)
    LOAD(glGenRenderbuffers)
    LOAD(glBindRenderbuffer)
    LOAD(glRenderbufferStorage)
    LOAD(glFramebufferRenderbuffer)
    LOAD(glCheckFramebufferStatus)
    LOAD(glDeleteShader)
    LOAD(glDeleteProgram)
    LOAD(glDeleteFramebuffers)
    LOAD(glDeleteRenderbuffers)
    LOAD(glGenBuffers)
    LOAD(glBindBuffer)
    LOAD(glBufferData)
    LOAD(glDeleteBuffers)
    LOAD(glGenVertexArrays)
    LOAD(glBindVertexArray)
    LOAD(glDeleteVertexArrays)
    #undef LOAD
    return true;
}

namespace Slic3r { namespace CLI {

bool ThumbnailRenderer::s_initialized = false;

// EGL state
static EGLDisplay s_display = EGL_NO_DISPLAY;
static EGLContext s_context = EGL_NO_CONTEXT;
static EGLSurface s_surface = EGL_NO_SURFACE;

// Embedded shaders -- same lighting as PrusaSlicer GUI's gouraud_light
static const char* VS_SOURCE = R"glsl(
#version 140
#define INTENSITY_CORRECTION 0.6
const vec3 LIGHT_TOP_DIR = vec3(-0.4574957, 0.4574957, 0.7624929);
#define LIGHT_TOP_DIFFUSE    (0.8 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SPECULAR   (0.125 * INTENSITY_CORRECTION)
#define LIGHT_TOP_SHININESS  20.0
const vec3 LIGHT_FRONT_DIR = vec3(0.6985074, 0.1397015, 0.6985074);
#define LIGHT_FRONT_DIFFUSE  (0.3 * INTENSITY_CORRECTION)
#define INTENSITY_AMBIENT    0.3
uniform mat4 view_model_matrix;
uniform mat4 projection_matrix;
uniform mat3 view_normal_matrix;
in vec3 v_position;
in vec3 v_normal;
out vec2 intensity;
void main() {
    vec3 normal = normalize(view_normal_matrix * v_normal);
    float NdotL = max(dot(normal, LIGHT_TOP_DIR), 0.0);
    intensity.x = INTENSITY_AMBIENT + NdotL * LIGHT_TOP_DIFFUSE;
    vec4 position = view_model_matrix * vec4(v_position, 1.0);
    intensity.y = LIGHT_TOP_SPECULAR * pow(max(dot(-normalize(position.xyz),
        reflect(-LIGHT_TOP_DIR, normal)), 0.0), LIGHT_TOP_SHININESS);
    NdotL = max(dot(normal, LIGHT_FRONT_DIR), 0.0);
    intensity.x += NdotL * LIGHT_FRONT_DIFFUSE;
    gl_Position = projection_matrix * position;
}
)glsl";

static const char* FS_SOURCE = R"glsl(
#version 140
uniform vec4 uniform_color;
uniform float emission_factor;
in vec2 intensity;
out vec4 out_color;
void main() {
    out_color = vec4(vec3(intensity.y) + uniform_color.rgb * (intensity.x + emission_factor), uniform_color.a);
}
)glsl";


bool ThumbnailRenderer::init()
{
    if (s_initialized) return true;

    // Get EGL display (platform-default, typically the GPU)
    s_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (s_display == EGL_NO_DISPLAY) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: No EGL display available";
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(s_display, &major, &minor)) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: eglInitialize failed";
        return false;
    }

    BOOST_LOG_TRIVIAL(info) << "CLI Thumbnail: EGL " << major << "." << minor;

    // Request OpenGL (not GLES) context
    if (!eglBindAPI(EGL_OPENGL_API)) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: eglBindAPI(EGL_OPENGL_API) failed";
        eglTerminate(s_display);
        s_display = EGL_NO_DISPLAY;
        return false;
    }

    // Choose config supporting pbuffer + OpenGL
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      8,
        EGL_DEPTH_SIZE,      24,
        EGL_NONE
    };
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(s_display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: No suitable EGL config";
        eglTerminate(s_display);
        s_display = EGL_NO_DISPLAY;
        return false;
    }

    // Create 1x1 pbuffer (actual rendering uses FBO)
    EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    s_surface = eglCreatePbufferSurface(s_display, config, pbuffer_attribs);
    if (s_surface == EGL_NO_SURFACE) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: eglCreatePbufferSurface failed";
        eglTerminate(s_display);
        s_display = EGL_NO_DISPLAY;
        return false;
    }

    // Create OpenGL context (request 3.1 core for #version 140 shaders)
    EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };
    s_context = eglCreateContext(s_display, config, EGL_NO_CONTEXT, context_attribs);
    if (s_context == EGL_NO_CONTEXT) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: eglCreateContext failed";
        eglDestroySurface(s_display, s_surface);
        eglTerminate(s_display);
        s_display = EGL_NO_DISPLAY;
        s_surface = EGL_NO_SURFACE;
        return false;
    }

    if (!eglMakeCurrent(s_display, s_surface, s_surface, s_context)) {
        BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: eglMakeCurrent failed";
        shutdown();
        return false;
    }

    // Load GL 2.0+ function pointers via EGL
    if (!load_gl_functions()) {
        shutdown();
        return false;
    }

    s_initialized = true;
    BOOST_LOG_TRIVIAL(info) << "CLI Thumbnail: EGL context ready (GL "
        << (const char*)glGetString(GL_VERSION) << ")";
    return true;
}

void ThumbnailRenderer::shutdown()
{
    if (s_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(s_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (s_context != EGL_NO_CONTEXT)
            eglDestroyContext(s_display, s_context);
        if (s_surface != EGL_NO_SURFACE)
            eglDestroySurface(s_display, s_surface);
        eglTerminate(s_display);
    }
    s_display = EGL_NO_DISPLAY;
    s_context = EGL_NO_CONTEXT;
    s_surface = EGL_NO_SURFACE;
    s_initialized = false;
}


// Compile a shader, return 0 on failure.
static GLuint compile_shader(GLenum type, const char* source)
{
    GLuint shader = p_glCreateShader(type);
    p_glShaderSource(shader, 1, &source, nullptr);
    p_glCompileShader(shader);

    GLint ok = 0;
    p_glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        p_glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        BOOST_LOG_TRIVIAL(error) << "CLI Thumbnail: Shader compile error: " << log;
        p_glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Interleaved vertex: position + normal, 24 bytes.
struct Vertex { float pos[3]; float normal[3]; };

// Append a mesh (with smooth normals) to the shared vertex/index buffers.
// Applies the given affine transform to positions and its inverse-transpose to normals.
static void append_mesh(
    const indexed_triangle_set& its,
    const Eigen::Matrix4d& xform_mat,
    std::vector<Vertex>& vertices,
    std::vector<unsigned int>& indices,
    BoundingBoxf3& bbox)
{
    using Eigen::Vector3f;
    using Eigen::Matrix3f;
    const Transform3d xform(xform_mat);
    const Matrix3f norm_xform = xform_mat.block<3,3>(0,0).inverse().transpose().cast<float>();

    // Pass 1: area-weighted per-vertex normals
    std::vector<Vector3f> vnormals(its.vertices.size(), Vector3f::Zero());
    for (const auto& face : its.indices) {
        const Vector3f& v0 = its.vertices[face[0]];
        const Vector3f& v1 = its.vertices[face[1]];
        const Vector3f& v2 = its.vertices[face[2]];
        Vector3f area_normal = (v1 - v0).cross(v2 - v0);
        vnormals[face[0]] += area_normal;
        vnormals[face[1]] += area_normal;
        vnormals[face[2]] += area_normal;
    }

    // Pass 2: transformed positions + normalized normals
    const unsigned int base_idx = (unsigned int)vertices.size();
    for (size_t i = 0; i < its.vertices.size(); ++i) {
        Vector3f pos = (xform * its.vertices[i].cast<double>()).cast<float>();
        Vector3f raw_n = norm_xform * vnormals[i];
        float len = raw_n.norm();
        Vector3f n = (len > 1e-10f) ? Vector3f(raw_n / len) : Vector3f::UnitZ();
        vertices.push_back({{pos.x(), pos.y(), pos.z()}, {n.x(), n.y(), n.z()}});
    }
    for (const auto& face : its.indices) {
        indices.push_back(base_idx + face[0]);
        indices.push_back(base_idx + face[1]);
        indices.push_back(base_idx + face[2]);
    }

    // Expand bounding box with transformed mesh bounds
    TriangleMesh tmesh(its);
    BoundingBoxf3 mesh_bb = tmesh.bounding_box();
    mesh_bb = mesh_bb.transformed(xform);
    bbox.merge(mesh_bb);
}


ThumbnailData ThumbnailRenderer::render(
    const Model& model,
    unsigned int width, unsigned int height,
    float color_r, float color_g, float color_b, float color_a,
    const std::string& bed_model_path,
    double bed_center_x, double bed_center_y)

{
    ThumbnailData data;
    if (!s_initialized) return data;

    // Collect all mesh geometry into shared vertex/index buffers.
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    BoundingBoxf3 bbox;

    // --- Bed plate (rendered first, behind model) ---
    unsigned int bed_index_count = 0;
    if (!bed_model_path.empty()) {
        TriangleMesh bed_mesh;
        if (bed_mesh.ReadSTLFile(bed_model_path.c_str())) {
            // Position bed at bed center, slightly below Z=0 to avoid z-fighting.
            Eigen::Matrix4d bed_xform = Eigen::Matrix4d::Identity();
            bed_xform(0, 3) = bed_center_x;
            bed_xform(1, 3) = bed_center_y;
            bed_xform(2, 3) = -0.03;
            append_mesh(bed_mesh.its, bed_xform, vertices, indices, bbox);
            bed_index_count = (unsigned int)indices.size();
        } else {
            BOOST_LOG_TRIVIAL(warning) << "CLI Thumbnail: Could not load bed model: " << bed_model_path;
        }
    }

    // --- Model geometry ---
    const unsigned int model_index_start = (unsigned int)indices.size();
    for (const ModelObject* obj : model.objects) {
        for (const ModelVolume* vol : obj->volumes) {
            if (!vol->is_model_part()) continue;
            const TriangleMesh& mesh = vol->mesh();
            for (const ModelInstance* inst : obj->instances)
                append_mesh(mesh.its, inst->get_matrix().matrix(), vertices, indices, bbox);
        }
    }
    const unsigned int model_index_count = (unsigned int)indices.size() - model_index_start;

    if (vertices.empty()) return data;
    // --- Compile shaders ---
    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS_SOURCE);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_SOURCE);
    if (!vs || !fs) { if (vs) p_glDeleteShader(vs); if (fs) p_glDeleteShader(fs); return data; }

    GLuint prog = p_glCreateProgram();
    p_glAttachShader(prog, vs);
    p_glAttachShader(prog, fs);
    p_glLinkProgram(prog);
    p_glDeleteShader(vs);
    p_glDeleteShader(fs);

    GLint link_ok = 0;
    p_glGetProgramiv(prog, GL_LINK_STATUS, &link_ok);
    if (!link_ok) { p_glDeleteProgram(prog); return data; }

    // --- Create FBO ---
    GLuint fbo = 0, color_tex = 0, depth_rb = 0;
    p_glGenFramebuffers(1, &fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &color_tex);
    glBindTexture(GL_TEXTURE_2D, color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);

    p_glGenRenderbuffers(1, &depth_rb);
    p_glBindRenderbuffer(GL_RENDERBUFFER, depth_rb);
    p_glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    p_glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);

    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        BOOST_LOG_TRIVIAL(error) << "CLI Thumbnail: FBO incomplete";
        p_glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &color_tex);
        p_glDeleteRenderbuffers(1, &depth_rb);
        p_glDeleteProgram(prog);
        return data;
    }

    // --- Camera: orthographic, zoom to bounding box ---
    // Same approach as GLCanvas3D::_render_thumbnail_internal:
    // camera.zoom_to_box(volumes_box) with ortho projection.
    Eigen::Vector3d center = bbox.center();
    Eigen::Vector3d size   = bbox.size();
    double max_dim = std::max({size.x(), size.y(), size.z()});
    if (max_dim < 1e-6) max_dim = 1.0;

    // Camera position: above-front-right, same as GUI default (45deg azimuth, 45deg zenith)
    Eigen::Vector3d eye = center + Eigen::Vector3d(max_dim * 0.8, -max_dim * 0.9, max_dim * 1.1);
    Eigen::Vector3d target = center;
    Eigen::Vector3d up_hint(0, 0, 1);

    // View matrix: exact same construction as PrusaSlicer Camera::look_at()
    const Eigen::Vector3d unit_z = (eye - target).normalized();
    const Eigen::Vector3d unit_x = up_hint.cross(unit_z).normalized();
    const Eigen::Vector3d unit_y = unit_z.cross(unit_x).normalized();
    const double dist = (eye - target).norm();
    const Eigen::Vector3d position = target + dist * unit_z;

    Eigen::Matrix4d view = Eigen::Matrix4d::Identity();
    view(0, 0) = unit_x.x(); view(0, 1) = unit_x.y(); view(0, 2) = unit_x.z();
    view(0, 3) = -unit_x.dot(position);
    view(1, 0) = unit_y.x(); view(1, 1) = unit_y.y(); view(1, 2) = unit_y.z();
    view(1, 3) = -unit_y.dot(position);
    view(2, 0) = unit_z.x(); view(2, 1) = unit_z.y(); view(2, 2) = unit_z.z();
    view(2, 3) = -unit_z.dot(position);

    // Compute projected bounding box in view space to determine ortho bounds.
    Eigen::Vector3d corners[8] = {
        {bbox.min.x(), bbox.min.y(), bbox.min.z()},
        {bbox.max.x(), bbox.min.y(), bbox.min.z()},
        {bbox.min.x(), bbox.max.y(), bbox.min.z()},
        {bbox.max.x(), bbox.max.y(), bbox.min.z()},
        {bbox.min.x(), bbox.min.y(), bbox.max.z()},
        {bbox.max.x(), bbox.min.y(), bbox.max.z()},
        {bbox.min.x(), bbox.max.y(), bbox.max.z()},
        {bbox.max.x(), bbox.max.y(), bbox.max.z()},
    };

    // Project bbox corners onto camera XY plane (same as Camera::calc_zoom_to_bounding_box_factor)
    double min_x_cam = 1e9, max_x_cam = -1e9, min_y_cam = 1e9, max_y_cam = -1e9;
    double min_z_eye = 1e9, max_z_eye = -1e9;

    for (int i = 0; i < 8; ++i) {
        const Eigen::Vector3d pos = corners[i] - center;
        const Eigen::Vector3d proj_on_plane = pos - pos.dot(-unit_z) * (-unit_z);
        const double x_on_plane = proj_on_plane.dot(unit_x);
        const double y_on_plane = proj_on_plane.dot(unit_y);
        min_x_cam = std::min(min_x_cam, x_on_plane);
        max_x_cam = std::max(max_x_cam, x_on_plane);
        min_y_cam = std::min(min_y_cam, y_on_plane);
        max_y_cam = std::max(max_y_cam, y_on_plane);
        // Also compute eye-space Z for near/far
        Eigen::Vector4d pe = view * Eigen::Vector4d(corners[i].x(), corners[i].y(), corners[i].z(), 1.0);
        min_z_eye = std::min(min_z_eye, pe.z());
        max_z_eye = std::max(max_z_eye, pe.z());
    }

    // Add 2.5% margin (matching Camera::DefaultZoomToBoxMarginFactor)
    double dx = (max_x_cam - min_x_cam) * 1.025;
    double dy = (max_y_cam - min_y_cam) * 1.025;

    // Compute ortho half-extents: same as Camera::apply_projection
    // zoom = min(viewport_w / dx, viewport_h / dy)
    double zoom = std::min((double)width / dx, (double)height / dy);
    double half_w = 0.5 * (double)width / zoom;
    double half_h = 0.5 * (double)height / zoom;

    // Near/far from eye-space Z (with margin)
    double near_z = -max_z_eye + 10.0;
    double far_z  = -min_z_eye + 10.0;
    if (near_z < 100.0) near_z = 100.0;
    if (far_z - near_z < 50.0) far_z = near_z + 50.0;

    // Orthographic projection (same as PrusaSlicer Camera::apply_projection)
    Eigen::Matrix4d proj = Eigen::Matrix4d::Zero();
    proj(0, 0) =  2.0 / (2.0 * half_w);
    proj(0, 3) =  0.0;  // symmetric around center
    proj(1, 1) =  2.0 / (2.0 * half_h);
    proj(1, 3) =  0.0;
    proj(2, 2) = -2.0 / (far_z - near_z);
    proj(2, 3) = -(far_z + near_z) / (far_z - near_z);
    proj(3, 3) =  1.0;

    // Normal matrix = inverse transpose of upper-left 3x3 of view
    Eigen::Matrix3d normal_matrix = view.block<3,3>(0,0).inverse().transpose();

    // Convert to float for GL
    Eigen::Matrix4f view_f = view.cast<float>();
    Eigen::Matrix4f proj_f = proj.cast<float>();
    Eigen::Matrix3f norm_f = normal_matrix.cast<float>();

    // --- Render ---
    glViewport(0, 0, width, height);
    glClearColor(0.93f, 0.93f, 0.95f, 1.0f);  // Light grey background
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);  // Mesh normals may not be consistently oriented

    p_glUseProgram(prog);

    GLint loc_vm = p_glGetUniformLocation(prog, "view_model_matrix");
    GLint loc_pj = p_glGetUniformLocation(prog, "projection_matrix");
    GLint loc_nm = p_glGetUniformLocation(prog, "view_normal_matrix");
    GLint loc_color = p_glGetUniformLocation(prog, "uniform_color");
    GLint loc_emission = p_glGetUniformLocation(prog, "emission_factor");

    // GL_FALSE: Eigen stores matrices column-major, same as OpenGL expects.
    // No transpose needed. (GL_TRUE was a latent bug masked by near-origin coords.)
    p_glUniformMatrix4fv(loc_vm, 1, GL_FALSE, view_f.data());
    p_glUniformMatrix4fv(loc_pj, 1, GL_FALSE, proj_f.data());
    p_glUniformMatrix3fv(loc_nm, 1, GL_FALSE, norm_f.data());
    p_glUniform1f(loc_emission, 0.0f);

    GLint loc_pos = p_glGetAttribLocation(prog, "v_position");
    GLint loc_nor = p_glGetAttribLocation(prog, "v_normal");

    // Upload all geometry (bed + model) to GPU via VAO + VBO + EBO.
    GLuint vao = 0, vbo = 0, ebo = 0;
    p_glGenVertexArrays(1, &vao);
    p_glBindVertexArray(vao);

    p_glGenBuffers(1, &vbo);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER,
        vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    p_glGenBuffers(1, &ebo);
    p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    p_glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    p_glEnableVertexAttribArray(loc_pos);
    p_glVertexAttribPointer(loc_pos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        (const void*)offsetof(Vertex, pos));
    p_glEnableVertexAttribArray(loc_nor);
    p_glVertexAttribPointer(loc_nor, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        (const void*)offsetof(Vertex, normal));

    // Draw bed plate first (dark grey, behind model).
    if (bed_index_count > 0) {
        p_glUniform4f(loc_color, 0.25f, 0.25f, 0.25f, 1.0f);
        glDrawElements(GL_TRIANGLES, (GLsizei)bed_index_count, GL_UNSIGNED_INT, nullptr);
    }

    // Draw model (user-specified color, default orange).
    if (model_index_count > 0) {
        p_glUniform4f(loc_color, color_r, color_g, color_b, color_a);
        glDrawElements(GL_TRIANGLES, (GLsizei)model_index_count, GL_UNSIGNED_INT,
            (const void*)(model_index_start * sizeof(unsigned int)));
    }

    p_glDisableVertexAttribArray(loc_pos);
    p_glDisableVertexAttribArray(loc_nor);
    p_glBindVertexArray(0);
    p_glDeleteBuffers(1, &ebo);
    p_glDeleteBuffers(1, &vbo);
    p_glDeleteVertexArrays(1, &vao);

    // --- Read pixels ---
    data.set(width, height);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data.pixels.data());

    // Flip vertically (OpenGL origin at bottom-left, thumbnails at top-left)
    const int row_bytes = width * 4;
    std::vector<unsigned char> temp_row(row_bytes);
    for (unsigned int y = 0; y < height / 2; ++y) {
        unsigned char* top    = data.pixels.data() + y * row_bytes;
        unsigned char* bottom = data.pixels.data() + (height - 1 - y) * row_bytes;
        std::memcpy(temp_row.data(), top, row_bytes);
        std::memcpy(top, bottom, row_bytes);
        std::memcpy(bottom, temp_row.data(), row_bytes);
    }

    // --- Cleanup ---
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    p_glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_tex);
    p_glDeleteRenderbuffers(1, &depth_rb);
    p_glDeleteProgram(prog);
    glDisable(GL_DEPTH_TEST);

    return data;
}

}} // namespace Slic3r::CLI

#else // !SLIC3R_CLI_THUMBNAILS

// Stub when EGL is not available
namespace Slic3r { namespace CLI {
bool ThumbnailRenderer::s_initialized = false;
bool ThumbnailRenderer::init()     { return false; }
void ThumbnailRenderer::shutdown() {}
ThumbnailData ThumbnailRenderer::render(const Model&, unsigned int, unsigned int,
    float, float, float, float) { return ThumbnailData(); }
}} // namespace Slic3r::CLI

#endif // SLIC3R_CLI_THUMBNAILS