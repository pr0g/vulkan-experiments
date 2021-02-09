#include "as-vulkan.hpp"
#include "as-vulkan-sdl.hpp"

#include "as/as-math-ops.hpp"
#include "as-camera-input/as-camera-input.hpp"

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
      return asci::MouseMotionEvent{
        {mouse_motion_event->x, mouse_motion_event->y}};
    }
    case SDL_MOUSEWHEEL: {
      const auto* mouse_wheel_event = (SDL_MouseWheelEvent*)event;
      return asci::MouseWheelEvent{mouse_wheel_event->y};
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

  struct App
  {
    AsVulkan* asVulkan = nullptr;
  };

  App app;

  as_vulkan_create(&app.asVulkan);
  as_vulkan_create_instance(app.asVulkan);
#ifndef NDEBUG
  // as_vulkan_debug(app.asVulkan); // update me
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

  ////////////////////////////////////////////////////////////////////

  AsMesh* appleMesh;
  as_create_mesh(&appleMesh);
  as_load_mesh(appleMesh, "assets/models/apple.obj");

  size_t appleTextureHandle = as_vulkan_allocate_image(app.asVulkan);
  as_vulkan_create_as_image(
    app.asVulkan, as_vulkan_image(app.asVulkan, appleTextureHandle),
    "assets/models/appleD.jpg");

  size_t appleMeshHandle = as_vulkan_allocate_mesh(app.asVulkan);
  as_vulkan_create_vertex_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, appleMeshHandle), appleMesh);
  as_vulkan_create_index_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, appleMeshHandle), appleMesh);

  const size_t appleCount = g_fruitCount;
  size_t appleUniformHandle = as_vulkan_allocate_uniform(app.asVulkan);
  as_vulkan_create_uniform_buffer(
    app.asVulkan, as_vulkan_uniform(app.asVulkan, appleUniformHandle),
    appleCount);
  as_vulkan_create_descriptor_set(
    app.asVulkan, as_vulkan_uniform(app.asVulkan, appleUniformHandle),
    as_vulkan_image(app.asVulkan, appleTextureHandle));

  /////////////////////////////////////////////////

  AsMesh* bananaMesh;
  as_create_mesh(&bananaMesh);
  as_load_mesh(bananaMesh, "assets/models/banana.obj");

  size_t bananaTextureHandle = as_vulkan_allocate_image(app.asVulkan);
  as_vulkan_create_as_image(
    app.asVulkan, as_vulkan_image(
      app.asVulkan, bananaTextureHandle), "assets/models/banana.jpg");

  size_t bananaMeshHandle = as_vulkan_allocate_mesh(app.asVulkan);
  as_vulkan_create_vertex_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, bananaMeshHandle), bananaMesh);
  as_vulkan_create_index_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, bananaMeshHandle), bananaMesh);

  const size_t bananaCount = g_fruitCount;
  size_t bananaUniformHandle = as_vulkan_allocate_uniform(app.asVulkan);
  as_vulkan_create_uniform_buffer(app.asVulkan, as_vulkan_uniform(app.asVulkan, bananaUniformHandle), bananaCount);
  as_vulkan_create_descriptor_set(app.asVulkan, as_vulkan_uniform(app.asVulkan, bananaUniformHandle), as_vulkan_image(app.asVulkan, bananaTextureHandle));

  /////////////////////////////////////////////////

  AsMesh* pearMesh;
  as_create_mesh(&pearMesh);
  as_load_mesh(pearMesh, "assets/models/pear.obj");

  size_t pearTextureHandle = as_vulkan_allocate_image(app.asVulkan);
  as_vulkan_create_as_image(
    app.asVulkan, as_vulkan_image(app.asVulkan, pearTextureHandle),
    "assets/models/pear.jpg");

  size_t pearMeshHandle = as_vulkan_allocate_mesh(app.asVulkan);
  as_vulkan_create_vertex_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, pearMeshHandle), pearMesh);
  as_vulkan_create_index_buffer(
    app.asVulkan, as_vulkan_mesh(app.asVulkan, pearMeshHandle), pearMesh);

  const size_t pearCount = g_fruitCount;
  size_t pearUniformHandle = as_vulkan_allocate_uniform(app.asVulkan);
  as_vulkan_create_uniform_buffer(
    app.asVulkan, as_vulkan_uniform(app.asVulkan, pearUniformHandle), pearCount);
  as_vulkan_create_descriptor_set(
    app.asVulkan, as_vulkan_uniform(app.asVulkan, pearUniformHandle),
    as_vulkan_image(app.asVulkan, pearTextureHandle));

  auto layoutFruit = [&app](
    float distance, size_t count, size_t offset,
    size_t meshHandle, size_t uniformHandle, const as::mat4& rot)
  {
    size_t counter = 2;
    for (size_t i = offset; i <= count; ++i)
    {
      if (counter != 2)
      {
          counter++;
          continue;
      }

      size_t meshInstanceHandle = as_vulkan_allocate_mesh_instance(app.asVulkan);
      AsMeshInstance* meshInstance =
        as_vulkan_mesh_instance(app.asVulkan, meshInstanceHandle);

      as_mesh_instance_mesh(meshInstance, meshHandle);
      as_mesh_instance_uniform(meshInstance, uniformHandle);
      as_mesh_instance_index(meshInstance,
        as_uniform_add_mesh_instance(
          as_vulkan_uniform(app.asVulkan, uniformHandle), meshInstanceHandle));

      float percent = (i / (float)count);
      float horizontalOffset = percent * distance;

      as_mesh_instance_transform(
        meshInstance, as::mat4_from_vec3(
          as::vec3{ (-distance * 0.5f) + horizontalOffset, 0.0f, 0.0f }));
      as_mesh_instance_percent(
        meshInstance, sinf(percent * as::radians(360.0f)) * 2.0f);
      as_mesh_instance_rot(meshInstance, rot);

      counter = 0;
    }
  };

  g_meshHandles[0] = appleMeshHandle;
  g_uniformHandles[0] = appleUniformHandle;
  g_meshHandles[1] = bananaMeshHandle;
  g_uniformHandles[1] = bananaUniformHandle;
  g_meshHandles[2] = pearMeshHandle;
  g_uniformHandles[2] = pearUniformHandle;

  using fp_seconds =
    std::chrono::duration<float, std::chrono::seconds::period>;

  auto previousTime = std::chrono::high_resolution_clock::now();

  float spawnDelay = 0.03f;
  float timer = spawnDelay;

  asci::SmoothProps smooth_props{};
  asc::Camera camera{};
  // initial camera position and orientation
  camera.look_at = as::vec3(0.0f, 0.0f, 15.0f);

  asc::Camera target_camera = camera;

  auto first_person_rotate_camera =
    asci::RotateCameraInput{asci::MouseButton::Right};
  auto first_person_pan_camera = asci::PanCameraInput{asci::lookPan};
  auto first_person_translate_camera =
    asci::TranslateCameraInput{asci::lookTranslation};
  auto first_person_wheel_camera = asci::WheelTranslationCameraInput{};

  asci::Cameras cameras;
  cameras.addCamera(&first_person_rotate_camera);
  cameras.addCamera(&first_person_pan_camera);
  cameras.addCamera(&first_person_translate_camera);
  cameras.addCamera(&first_person_wheel_camera);

  asci::CameraSystem camera_system;
  camera_system.cameras_ = cameras;

  for (bool quit = false; !quit;) {
    auto currentTime = std::chrono::high_resolution_clock::now();
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

    timer += deltaTime.count();
    if (timer >= spawnDelay)
    {
        static size_t meshIndex = 0;

        size_t meshInstanceHandle;
        size_t meshInstanceIndex;
        // actually allocate the mesh/uniform buffer
        if (g_meshCount[meshIndex] < g_fruitCount)
        {
          meshInstanceHandle = as_vulkan_allocate_mesh_instance(app.asVulkan);
          meshInstanceIndex = as_uniform_add_mesh_instance(as_vulkan_uniform(app.asVulkan, g_uniformHandles[meshIndex]), meshInstanceHandle);

          g_meshInstanceHandle[meshIndex][g_meshCount[meshIndex]] = meshInstanceHandle;
          g_meshInstanceIndex[meshIndex][g_meshCount[meshIndex]] = meshInstanceIndex;

          g_meshCount[meshIndex]++;
        }
        else
        {
          // if we've run out, reuse an existing one
          if (g_meshCounter[meshIndex] == g_fruitCount)
          {
              g_meshMultiplier[meshIndex] += 1.0f;
              g_meshCounter[meshIndex] -= g_fruitCount;
          }

          meshInstanceHandle = g_meshInstanceHandle[meshIndex][g_meshCounter[meshIndex]];
          meshInstanceIndex = g_meshInstanceIndex[meshIndex][g_meshCounter[meshIndex]];

          g_meshCounter[meshIndex] = g_meshCounter[meshIndex] + 1;
        }

        AsMeshInstance* meshInstance = as_vulkan_mesh_instance(app.asVulkan, meshInstanceHandle);

        as_mesh_instance_mesh(meshInstance, g_meshHandles[meshIndex]);
        as_mesh_instance_uniform(meshInstance, g_uniformHandles[meshIndex]);
        as_mesh_instance_index(meshInstance, meshInstanceIndex);

        as_mesh_instance_transform(
            meshInstance, as::mat4_from_vec3(
              as::vec3(-60.0f + (meshInstanceIndex) + (g_meshMultiplier[meshIndex] * 20.0f), -3.0f + meshIndex * 3.0f, 0.0f)));
        as_mesh_instance_percent(meshInstance, 0.0f);
        as_mesh_instance_rot(meshInstance, as::mat4::identity());
        as_mesh_instance_time(meshInstance, std::chrono::high_resolution_clock::now());

        meshIndex = (meshIndex + 1) % g_fruitTypeCount;

        timer -= spawnDelay;
    }

    as_vulkan_update_uniform_buffer(app.asVulkan, as::mat4_from_affine(camera.view()));

    as_vulkan_draw_frame(app.asVulkan);
  }

  as_vulkan_cleanup(app.asVulkan);
  as_vulkan_destroy(&app.asVulkan);

  return 0;
}
