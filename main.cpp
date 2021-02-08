#include "as-vulkan.hpp"
#include "as-vulkan-sdl.hpp"

#include "SDL.h"
#include "SDL_syswm.h"

// hack
const size_t g_fruitCount = 20;
const size_t g_fruitTypeCount = 3;

size_t g_meshHandles[g_fruitTypeCount]{};
size_t g_uniformHandles[g_fruitTypeCount]{};
size_t g_meshCount[g_fruitTypeCount]{};

size_t g_meshInstanceIndex[g_fruitTypeCount][g_fruitCount]{};
size_t g_meshInstanceHandle[g_fruitTypeCount][g_fruitCount]{};

size_t g_meshCounter[g_fruitTypeCount]{ g_fruitCount, g_fruitCount, g_fruitCount };
float g_meshMultiplier[g_fruitTypeCount]{ 0.0f, 0.0f, 0.0f };

int main(int argc, char** argv) {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  const int width = 1024;
  const int height = 768;
  const float aspect = float(width) / float(height);
  SDL_Window* window = SDL_CreateWindow(
    argv[0], SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height,
    SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN);

  if (window == nullptr) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    return 1;
  }

  SDL_SysWMinfo wmi;
  SDL_VERSION(&wmi.version);
  if (SDL_GetWindowWMInfo(window, &wmi) == 0) {
    return 1;
  }

  struct App
  {
    AsVulkan* asVulkan = nullptr;
  };

  App app;

  as_vulkan_create(&app.asVulkan);
  as_vulkan_create_instance(app.asVulkan);
#ifndef NDEBUG
  as_vulkan_debug(app.asVulkan);
#endif // _DEBUG
  as_vulkan_create_surface(app.asVulkan, window);
  as_vulkan_pick_physical_device(app.asVulkan);
  as_vulkan_create_logical_device(app.asVulkan);
  as_vulkan_create_swap_chain(app.asVulkan);
  as_vulkan_create_image_views(app.asVulkan);
  as_vulkan_create_render_pass(app.asVulkan);
  as_vulkan_create_descriptor_set_layout(app.asVulkan);
  as_vulkan_create_graphics_pipeline(app.asVulkan);
  as_vulkan_create_rendering_resources(app.asVulkan);
  as_vulkan_create_depth_resources(app.asVulkan);
  as_vulkan_create_image_sampler(app.asVulkan);
  as_vulkan_create_descriptor_pool(app.asVulkan);

  for (bool quit = false; !quit;) {
    SDL_Event current_event;
    while (SDL_PollEvent(&current_event) != 0) {
      if (current_event.type == SDL_QUIT) {
        quit = true;
        break;
      }
    }
  }

  as_vulkan_cleanup(app.asVulkan);
  as_vulkan_destroy(&app.asVulkan);

  return 0;
}
