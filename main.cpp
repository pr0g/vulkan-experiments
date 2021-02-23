#include "as-vulkan.hpp"

#include "as/as-math-ops.hpp"
#include "as-camera-input/as-camera-input.hpp"

#include "SDL.h"
#include "SDL_syswm.h"

#include <chrono>

namespace asc
{
  Handedness handedness()
  {
    return Handedness::Right;
  }
}

asci::MouseButton mouseFromSdl(const SDL_MouseButtonEvent* event)
{
  switch (event->button) {
    case SDL_BUTTON_LEFT:
      return asci::MouseButton::Left;
    case SDL_BUTTON_RIGHT:
      return asci::MouseButton::Right;
    case SDL_BUTTON_MIDDLE:
      return asci::MouseButton::Middle;
    default:
      return asci::MouseButton::Nil;
  }
}

asci::KeyboardButton keyboardFromSdl(const int key)
{
  switch (key) {
    case SDL_SCANCODE_W:
      return asci::KeyboardButton::W;
    case SDL_SCANCODE_S:
      return asci::KeyboardButton::S;
    case SDL_SCANCODE_A:
      return asci::KeyboardButton::A;
    case SDL_SCANCODE_D:
      return asci::KeyboardButton::D;
    case SDL_SCANCODE_Q:
      return asci::KeyboardButton::Q;
    case SDL_SCANCODE_E:
      return asci::KeyboardButton::E;
    case SDL_SCANCODE_LALT:
      return asci::KeyboardButton::LAlt;
    case SDL_SCANCODE_LSHIFT:
      return asci::KeyboardButton::LShift;
    case SDL_SCANCODE_LCTRL:
      return asci::KeyboardButton::Ctrl;
    default:
      return asci::KeyboardButton::Nil;
  }
}

asci::InputEvent sdlToInput(const SDL_Event* event)
{
  switch (event->type) {
    case SDL_MOUSEMOTION: {
      const auto* mouse_motion_event = (SDL_MouseMotionEvent*)event;
      return asci::CursorMotionEvent{
        {mouse_motion_event->x, mouse_motion_event->y}};
    }
    case SDL_MOUSEWHEEL: {
      const auto* mouse_wheel_event = (SDL_MouseWheelEvent*)event;
      return asci::ScrollEvent{mouse_wheel_event->y};
    }
    case SDL_MOUSEBUTTONDOWN: {
      const auto* mouse_event = (SDL_MouseButtonEvent*)event;
      return asci::MouseButtonEvent{
        mouseFromSdl(mouse_event), asci::ButtonAction::Down};
    }
    case SDL_MOUSEBUTTONUP: {
      const auto* mouse_event = (SDL_MouseButtonEvent*)event;
      return asci::MouseButtonEvent{
        mouseFromSdl(mouse_event), asci::ButtonAction::Up};
    }
    case SDL_KEYDOWN: {
      const auto* keyboard_event = (SDL_KeyboardEvent*)event;
      return asci::KeyboardButtonEvent{
        keyboardFromSdl(keyboard_event->keysym.scancode),
        asci::ButtonAction::Down, event->key.repeat != 0u};
    }
    case SDL_KEYUP: {
      const auto* keyboard_event = (SDL_KeyboardEvent*)event;
      return asci::KeyboardButtonEvent{
        keyboardFromSdl(keyboard_event->keysym.scancode),
        asci::ButtonAction::Up, event->key.repeat != 0u};
    }
    default:
      return std::monostate{};
  }
}

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

  AsVulkan* asVulkan = nullptr;
  as_vulkan_create(&asVulkan);
  const char** instance_extensions;
	uint32_t instance_extensions_count = 0;
  sdl_vulkan_instance_extensions(
    window, instance_extensions, instance_extensions_count);
  as_vulkan_create_instance(
    asVulkan, instance_extensions, instance_extensions_count);
  delete[] instance_extensions;
#ifndef NDEBUG
  as_vulkan_debug(asVulkan);
#endif // _DEBUG
  as_sdl_vulkan_create_surface(asVulkan, window);
  as_vulkan_pick_physical_device(asVulkan);
  as_vulkan_create_logical_device(asVulkan);
  as_vulkan_create_swap_chain(asVulkan);
  as_vulkan_create_image_views(asVulkan);
  as_vulkan_create_render_pass(asVulkan);
  as_vulkan_create_descriptor_set_layout(asVulkan);
  as_vulkan_create_graphics_pipeline(asVulkan);
  as_vulkan_create_rendering_resources(asVulkan);
  as_vulkan_create_depth_resources(asVulkan);
  as_vulkan_create_image_sampler(asVulkan);
  as_vulkan_create_descriptor_pool(asVulkan);

  AsMesh* viking_mesh;
  as_create_mesh(&viking_mesh);
  as_load_mesh(viking_mesh, "assets/models/viking_room.obj");

  size_t viking_texture_handle = as_vulkan_allocate_image(asVulkan);
  as_vulkan_create_as_image(
    asVulkan, as_vulkan_image(asVulkan, viking_texture_handle),
    "assets/models/viking_room.png");

  size_t viking_mesh_handle = as_vulkan_allocate_mesh(asVulkan);
  as_vulkan_create_vertex_buffer(
    asVulkan, as_vulkan_mesh(asVulkan, viking_mesh_handle), viking_mesh);
  as_vulkan_create_index_buffer(
    asVulkan, as_vulkan_mesh(asVulkan, viking_mesh_handle), viking_mesh);

  size_t viking_uniform_handle = as_vulkan_allocate_uniform(asVulkan);
  as_vulkan_create_uniform_buffer(
    asVulkan, as_vulkan_uniform(asVulkan, viking_uniform_handle), 1);
  as_vulkan_create_descriptor_set(
    asVulkan, as_vulkan_uniform(asVulkan, viking_uniform_handle),
    as_vulkan_image(asVulkan, viking_texture_handle));

  size_t viking_mesh_instance_handle = as_vulkan_allocate_mesh_instance(asVulkan);
  size_t viking_mesh_instance_index = as_uniform_add_mesh_instance(
    as_vulkan_uniform(asVulkan, viking_uniform_handle), viking_mesh_instance_handle);

  using fp_seconds =
    std::chrono::duration<float, std::chrono::seconds::period>;

  asci::SmoothProps smooth_props{};
  asc::Camera camera{};
  // initial camera position and orientation
  camera.look_at = as::vec3(2.0f, 2.0f, 2.0f);
  camera.pitch = as::radians(32.0f);
  camera.yaw = as::radians(-45.0f);

  asc::Camera target_camera = camera;

  auto first_person_rotate_camera =
    asci::RotateCameraInput{asci::MouseButton::Right};
  auto first_person_pan_camera = asci::PanCameraInput{asci::lookPan};
  auto first_person_translate_camera =
    asci::TranslateCameraInput{asci::lookTranslation};
  auto first_person_wheel_camera = asci::ScrollTranslationCameraInput{};

  asci::Cameras cameras;
  cameras.addCamera(&first_person_rotate_camera);
  cameras.addCamera(&first_person_pan_camera);
  cameras.addCamera(&first_person_translate_camera);
  cameras.addCamera(&first_person_wheel_camera);

  asci::CameraSystem camera_system;
  camera_system.cameras_ = cameras;

  auto previousTime = std::chrono::steady_clock::now();
  for (bool quit = false; !quit;) {
    auto currentTime = std::chrono::steady_clock::now();
    auto deltaTime = fp_seconds(currentTime - previousTime);
    previousTime = currentTime;

    SDL_Event current_event;
    while (SDL_PollEvent(&current_event) != 0) {
      camera_system.handleEvents(sdlToInput(&current_event));
      if (current_event.type == SDL_QUIT) {
        quit = true;
        break;
      }
    }

    target_camera = camera_system.stepCamera(target_camera, deltaTime.count());
    camera = asci::smoothCamera(
      camera, target_camera, asci::SmoothProps{}, deltaTime.count());

    AsMeshInstance* viking_mesh_instance =
      as_vulkan_mesh_instance(asVulkan, viking_mesh_instance_handle);

    as_mesh_instance_mesh(viking_mesh_instance, viking_mesh_handle);
    as_mesh_instance_uniform(viking_mesh_instance, viking_uniform_handle);
    as_mesh_instance_index(viking_mesh_instance, viking_mesh_instance_index);
    as_mesh_instance_transform(viking_mesh_instance, as::mat4::identity());

    as_vulkan_update_uniform_buffer(
      asVulkan, as::mat4_from_affine(camera.view()), deltaTime.count());

    as_vulkan_draw_frame(asVulkan);
  }

  as_vulkan_cleanup(asVulkan);
  as_vulkan_destroy(&asVulkan);

  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
