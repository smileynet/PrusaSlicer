# Session Handoff: PrusaSlicer CLI Thumbnail Renderer

## Prompt for Next Session

```
I'm working on a PR to PrusaSlicer that adds headless EGL thumbnail rendering
for CLI mode. The code is at ~/code/PrusaSlicer on branch feature/cli-thumbnails
(fork: smileynet/PrusaSlicer). Issue: github.com/prusa3d/PrusaSlicer/issues/7878

The renderer works -- it produces thumbnails via EGL on NVIDIA GPU. But it needs
three fixes before the PR is ready:

1. SMOOTH NORMALS: Replace per-face flat normals with area-weighted per-vertex
   smooth normals in CLIThumbnailRenderer.cpp. The current code (lines ~315-336)
   duplicates vertices per face with flat normals. Replace with: accumulate
   unnormalized cross products per vertex, normalize, use indexed geometry.
   Do NOT use igl (GUI dependency). ~20 lines of new code.

2. INSTANCE TRANSFORMS: The render() method collects raw mesh vertices without
   applying ModelInstance transforms. Multi-object plates will render at origin.
   Apply inst->get_matrix() to vertices and normals before adding to buffer.

3. TEST AND VERIFY: Build on monolith (sam@monolith.lan), run the CLI, extract
   the thumbnail PNG, and visually verify the result looks like a properly lit
   3D model (not a flat triangle). Build command:
   
   scp ~/code/PrusaSlicer/src/CLI/CLIThumbnailRenderer.cpp sam@monolith.lan:/tmp/PrusaSlicer/src/CLI/CLIThumbnailRenderer.cpp
   ssh sam@monolith.lan "cd /tmp/PrusaSlicer/build_release && cmake --build . --target PrusaSlicer -j\$(nproc)"
   
   Test command:
   ssh sam@monolith.lan "./tmp/PrusaSlicer/build_release/src/prusa-slicer --export-gcode --load /tmp/test_profile.ini --thumbnails '300x200' --thumbnails-format PNG --output /tmp/test.gcode /tmp/test_model.stl"
   
   Extract thumbnail:
   ssh sam@monolith.lan "python3 -c \"import base64,re; c=open('/tmp/test.gcode').read(); m=re.search(r'; thumbnail begin 300x200 \d+\n((?:; [^\n]+\n)+); thumbnail end',c); open('/tmp/thumb.png','wb').write(base64.b64decode(m.group(1).replace('; ','').replace('\n',''))) if m else print('no thumb')\""
   scp sam@monolith.lan:/tmp/thumb.png /tmp/thumb.png

After all three fixes are verified visually, reopen PR #15327:
   cd ~/code/PrusaSlicer && gh pr reopen 15327 --repo prusa3d/PrusaSlicer

Key files:
- src/CLI/CLIThumbnailRenderer.cpp (the renderer, ~580 lines)
- src/CLI/CLIThumbnailRenderer.hpp (header, ~55 lines)  
- src/CLI/ProcessActions.cpp (integration point, ~20 lines changed)
- src/CMakeLists.txt (build config, ~14 lines added)

Key context:
- Matrices use GL_TRUE for upload (our row-by-row construction needs transpose)
- PrusaSlicer GUI uses GL_FALSE because its shader system wraps the upload differently
- The gouraud_light shader is embedded as string constants matching resources/shaders/140/
- EGL context is a 1x1 pbuffer; actual rendering uses FBO at requested resolution
- GL functions loaded at runtime via eglGetProcAddress (no link-time GL 2.0+ dep)
- Build deps already at /tmp/PrusaSlicer/deps/build/destdir/usr/local on monolith
- Test STL at /tmp/test_model.stl, test profile at /tmp/test_profile.ini on monolith
```

## File Locations

| File | Machine | Path |
|------|---------|------|
| Source code | workstation | `~/code/PrusaSlicer/src/CLI/CLIThumbnailRenderer.cpp` |
| Header | workstation | `~/code/PrusaSlicer/src/CLI/CLIThumbnailRenderer.hpp` |
| ProcessActions | workstation | `~/code/PrusaSlicer/src/CLI/ProcessActions.cpp` |
| CMakeLists | workstation | `~/code/PrusaSlicer/src/CMakeLists.txt` |
| Build dir | monolith | `/tmp/PrusaSlicer/build_release/` |
| Deps | monolith | `/tmp/PrusaSlicer/deps/build/destdir/usr/local/` |
| Test STL | monolith | `/tmp/test_model.stl` |
| Test profile | monolith | `/tmp/test_profile.ini` |
| Binary | monolith | `/tmp/PrusaSlicer/build_release/src/prusa-slicer` |
