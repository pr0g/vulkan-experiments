# Vulkan-Experiments

## Overview

This repo contains a simple implementation of a Vulkan renderer created following the excellent [Vulkan Tutorial](https://vulkan-tutorial.com/) website.

This implementation uses SDL instead of GLFW and follows a more C-like API design.

It's very much a WIP and is not ready for use other than as a learning tool.

![fruit](vulkan-fruit.png)

### Earlier work

- [Updating Models (Apples, Pears, Bananas)](Uhttps://twitter.com/tom_h_h/status/980474346434162688?s=20) (Command Buffer generation)
- [Varying Models (Apples, Pears, Bananas)](https://twitter.com/tom_h_h/status/957656446258307073?s=20)
- [Instanced Models (Apples)](https://twitter.com/tom_h_h/status/955035033923932161?s=20)
- [Multiple Models (Bananas)](https://twitter.com/tom_h_h/status/952264079737085952?s=20)
- [Model (Banana)](https://twitter.com/tom_h_h/status/947870573123817472?s=20)
- [Depth](https://twitter.com/tom_h_h/status/938162519117631490?s=20)
- [Textured Quad](https://twitter.com/tom_h_h/status/934558254847406080?s=20)
- [Quad](https://twitter.com/tom_h_h/status/931890595962015744?s=20)
- [Triangle](https://twitter.com/tom_h_h/status/927241120006004736?s=20)

## Dependencies

Currently tested on macOS.

Vulkan is installed on the system and is found with `find_package(Vulkan)` in the root `CMakeLists.txt` file. All other dependencies are found through the use of CMake `ExternalProject_Add` and `FetchContent`.

Vulkan setup instructions: https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html

```bash
VULKAN_SDK="<path/to>/vulkan-sdk/macOS"
export VULKAN_SDK
PATH=$PATH:$VULKAN_SDK/bin
export PATH
DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$VULKAN_SDK/lib
export DYLD_LIBRARY_PATH
VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export VK_LAYER_PATH
VK_ICD_FILENAMES=$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json
export VK_ICD_FILENAMES
```
