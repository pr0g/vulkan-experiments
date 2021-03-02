#pragma once

#include <stdint.h>

#include "as/as-mat4.hpp"
#include "thh_handles/thh_handles.hpp"

struct AsVulkan;
struct AsVulkanMesh;
struct AsVulkanImage;
struct AsVulkanUniform;

struct AsMesh;
struct AsMeshInstance;

void as_vulkan_create(AsVulkan** asVulkan);
void as_vulkan_destroy(AsVulkan** asVulkan);

void as_vulkan_create_instance(
  AsVulkan* asVulkan, const char* required_instance_extensions[],
  int64_t required_instance_extension_count);

#ifdef AS_VULKAN_SDL
struct SDL_Window;
void as_sdl_vulkan_create_surface(AsVulkan* asVulkan, SDL_Window* window);
void sdl_vulkan_instance_extensions(
  SDL_Window* window, const char**& required_instance_extensions,
  uint32_t& required_instance_extension_count);
#endif // AS_VULKAN_SDL

void as_vulkan_pick_physical_device(AsVulkan* asVulkan);
void as_vulkan_create_logical_device(AsVulkan* asVulkan);
void as_vulkan_create_swap_chain(AsVulkan* asVulkan);
void as_vulkan_recreate_swap_chain(AsVulkan* asVulkan);
void as_vulkan_create_image_views(AsVulkan* asVulkan);
void as_vulkan_create_render_pass(AsVulkan* asVulkan);
void as_vulkan_create_descriptor_set_layout(AsVulkan* asVulkan);
void as_vulkan_create_graphics_pipeline(AsVulkan* asVulkan);
void as_vulkan_create_rendering_resources(AsVulkan* asVulkan);
void as_vulkan_create_color_resources(AsVulkan* asVulkan);
void as_vulkan_create_depth_resources(AsVulkan* asVulkan);
void as_vulkan_create_vertex_buffer(AsVulkan* asVulkan, AsVulkanMesh* vulkanMesh, const AsMesh* mesh);
void as_vulkan_create_index_buffer(AsVulkan* asVulkan, AsVulkanMesh* vulkanMesh, const AsMesh* mesh);
void as_vulkan_create_uniform_buffer(AsVulkan* asVulkan, AsVulkanUniform* asUniform, size_t uniformBufferObjectCount);
void as_vulkan_create_descriptor_pool(AsVulkan* asVulkan);
void as_vulkan_create_descriptor_set(AsVulkan* asVulkan, AsVulkanUniform* asUniform, AsVulkanImage* asImage);
void as_vulkan_create_image_sampler(AsVulkan* asVulkan);
void as_vulkan_update_uniform_buffer(AsVulkan* asVulkan, const as::mat4& view, float time);
void as_vulkan_draw_frame(AsVulkan* asVulkan);
void as_vulkan_debug(AsVulkan* asVulkan);
void as_vulkan_cleanup(AsVulkan* asVulkan);

void as_vulkan_create_as_image(AsVulkan* asVulkan, AsVulkanImage* asImage, const char* path);

AsVulkanMesh* as_vulkan_mesh(AsVulkan* asVulkan, thh::handle_t handle);
AsVulkanImage* as_vulkan_image(AsVulkan* asVulkan, size_t handle);
AsVulkanUniform* as_vulkan_uniform(AsVulkan* asVulkan, size_t handle);
AsMeshInstance* as_vulkan_mesh_instance(AsVulkan* asVulkan, thh::handle_t handle);

thh::handle_t as_vulkan_allocate_mesh(AsVulkan* asVulkan);
thh::handle_t as_vulkan_allocate_mesh_instance(AsVulkan* asVulkan);
size_t as_vulkan_allocate_image(AsVulkan* asVulkan);
size_t as_vulkan_allocate_uniform(AsVulkan* asVulkan);

void as_create_mesh(AsMesh** asMesh);
void as_load_mesh(AsMesh* mesh, const char* path);

void as_mesh_instance_mesh(AsMeshInstance* meshInstance, thh::handle_t meshHandle);
void as_mesh_instance_uniform(AsMeshInstance* meshInstance, size_t uniformHandle);
void as_mesh_instance_index(AsMeshInstance* meshInstance, size_t uniformIndex);

void as_mesh_instance_transform(AsMeshInstance* meshInstance, const as::mat4& transform);
void as_mesh_instance_rot(AsMeshInstance* meshInstance, const as::mat4& rot);
void as_mesh_instance_percent(AsMeshInstance* meshInstance, float offset);
void as_mesh_instance_time(AsMeshInstance* meshInstance, float time);

size_t as_uniform_add_mesh_instance(AsVulkanUniform* asUniform, thh::handle_t handle);
