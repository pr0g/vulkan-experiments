#include "as-vulkan.hpp"
#ifdef AS_VULKAN_SDL
#include "SDL_vulkan.h"
#endif // AS_VULKAN_SDL

#define VK_USE_PLATFORM_MACOS_MVK
#include <vulkan/vulkan.h>

#include "as/as-math-ops.hpp"
#include "as/as-view.hpp"
#include "file-ops.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#define VERTEX_BUFFER_BIND_ID 0

static const std::vector<const char*> s_validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static const std::vector<const char*> s_deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  "VK_KHR_portability_subset"
};

static const bool s_enableValidationLayers =
#ifdef NDEBUG
    false;
#else
    true;
#endif

struct AsVertex
{
    as::vec3 position;
    as::vec3 color;
    as::vec2 uv;
};

struct AsMesh
{
    std::vector<AsVertex> vertices;
    std::vector<uint32_t> indices;
};

struct AsVulkanUniform
{
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    std::vector<thh::handle_t> meshInstanceHandles;
    VkDescriptorSet descriptorSet;
};

struct AsVulkanMesh
{
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    size_t indexCount;
};

struct AsVulkanImage
{
    VkImage image;
    VkImageView imageView;
    VkDeviceMemory imageMemory;
    uint32_t mipLevels;
};

struct AsVulkanQueue
{
    VkQueue queue;
    uint32_t familyIndex;
};

struct AsMeshInstance
{
    as::mat4 transform;
    as::mat4 rot;
    float percent;
    float time;

    thh::handle_t meshHandle; // actual mesh the instance is referring to
    thh::handle_t uniformHandle; // uniform buffer being used for this mesh instance
    size_t uniformIndex; // which instance in the uniform block is this one (id)
};

struct AsVulkanAlignment
{
    VkDeviceSize minUniformBufferAlignment;
};

struct AsRenderResource
{
    VkFramebuffer framebuffer{ VK_NULL_HANDLE };
    VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
    VkSemaphore imageAvailable{ VK_NULL_HANDLE };
    VkSemaphore renderFinished{ VK_NULL_HANDLE };
    VkFence commandBufferFinished{ VK_NULL_HANDLE };
};

struct AsVulkan
{
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkInstance instance;
    AsVulkanQueue graphicsQueue;
    AsVulkanQueue presentQueue;
    VkSurfaceKHR surface;

    // swap chain
    VkSwapchainKHR swapChain;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    //

    enum { RenderingResourcesCount = 3 };
    AsRenderResource renderingResources[RenderingResourcesCount];
    size_t resourceIndex = 0;

    VkRenderPass renderPass;
    VkPipeline graphicsPipeline;
    VkPipelineLayout pipelineLayout;
    VkCommandPool commandPool;

    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    AsVulkanImage depth;
    AsVulkanImage color;
    VkSampler imageSampler;

    // app data
    thh::container_t<AsVulkanMesh> meshes;
    thh::container_t<AsVulkanImage> images;
    thh::container_t<AsVulkanUniform> uniforms;
    thh::container_t<AsMeshInstance> meshInstances;

    AsVulkanAlignment alignment;
    VkDebugUtilsMessengerEXT debugMessenger;
};

struct UniformBufferObject
{
    as::mat4 mvp;
};

template<typename T>
VkDeviceSize as_vulkan_uniform_alignment(AsVulkanAlignment alignment)
{
    size_t sizeBytes = sizeof(T);
    if (sizeBytes <= alignment.minUniformBufferAlignment)
    {
        return alignment.minUniformBufferAlignment;
    }
    else
    {
        VkDeviceSize alignOffset = (sizeBytes / alignment.minUniformBufferAlignment) +
            (((sizeBytes % alignment.minUniformBufferAlignment) > 0) ? 1 : 0);

        return alignment.minUniformBufferAlignment * alignOffset;
    }
}

void as_vulkan_create_descriptor_set_layout(AsVulkan* asVulkan)
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding bindings[] =
        { uboLayoutBinding, samplerLayoutBinding };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(asVulkan->device, &layoutInfo, nullptr, &asVulkan->descriptorSetLayout) != VK_SUCCESS)
    {
        std::cerr << "failed to create descriptor set layout\n";
        std::exit(EXIT_FAILURE);
    }
}

VkVertexInputBindingDescription as_vulkan_get_vertex_binding_description()
{
    VkVertexInputBindingDescription bindingDescription{};
    // index of a binding with which vertex data will be associated
    bindingDescription.binding = VERTEX_BUFFER_BIND_ID;
    // the distance in bytes between two consecutive elements
    bindingDescription.stride = sizeof(AsVertex);
    // defines how data should be consumed, per vertex or per instance
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> as_vulkan_get_attribute_descriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

    // index of an attribute, the same as defined by the location layout
    // specifier in a shader source code
    attributeDescriptions[0].location = 0;
    // the number of the slot from which data should be read, the same binding
    // as in a VkVertexInputBindingDescription structure and vkCmdBindVertexBuffers()
    attributeDescriptions[0].binding = VERTEX_BUFFER_BIND_ID;
    // data type and number of components per attribute
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    // beginning of data for a given attribute
    attributeDescriptions[0].offset = offsetof(AsVertex, position);

    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].binding = VERTEX_BUFFER_BIND_ID;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(AsVertex, color);

    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(AsVertex, uv);

    return attributeDescriptions;
}

// triangle data
const std::vector<AsVertex> triangle_vertices = {
    { { 0.0f, -0.5f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.0f }, { 0.0f, 1.0f, 1.0f }, { 0.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.0f }, { 1.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } }
};

const std::vector<uint16_t> triangle_indices = {
    0, 1, 2
};

// rectangle data
const std::vector<AsVertex> rectangle_vertices = {
    // top square
    { { -0.5f, 0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
    // bottom square
    { { -0.5f, 0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } }
};

const std::vector<uint16_t> rectangle_indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

// current vertices/indices
//const std::vector<AsVertex>& vertices = rectangle_vertices;
//const std::vector<uint16_t>& indices = rectangle_indices;

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

VkFormat as_vulkan_find_supported_format(
    AsVulkan* asVulkan, const std::vector<VkFormat>& candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(asVulkan->physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    std::cerr << "failed to find supported format\n";
    std::exit(EXIT_FAILURE);

    return VK_FORMAT_UNDEFINED;
}

bool as_vulkan_has_stencil_component(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat as_vulkan_find_depth_format(AsVulkan* asVulkan)
{
    return as_vulkan_find_supported_format(
        asVulkan,
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void as_vulkan_cleanup_rendering_resources(AsVulkan* asVulkan)
{
    for (AsRenderResource& renderingResource : asVulkan->renderingResources)
    {
        if (renderingResource.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(asVulkan->device, renderingResource.framebuffer, nullptr);
        }

        if (renderingResource.commandBuffer != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(asVulkan->device, asVulkan->commandPool, 1, &renderingResource.commandBuffer);
        }

        if (renderingResource.imageAvailable != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(asVulkan->device, renderingResource.imageAvailable, nullptr);
        }

        if (renderingResource.renderFinished != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(asVulkan->device, renderingResource.renderFinished, nullptr);
        }

        if (renderingResource.commandBufferFinished != VK_NULL_HANDLE)
        {
            vkDestroyFence(asVulkan->device, renderingResource.commandBufferFinished, nullptr);
        }
    }
}

void as_vulkan_cleanup_swap_chain(AsVulkan* asVulkan)
{
    vkDestroyImageView(asVulkan->device, asVulkan->color.imageView, nullptr);
    vkDestroyImage(asVulkan->device, asVulkan->color.image, nullptr);
    vkFreeMemory(asVulkan->device, asVulkan->color.imageMemory, nullptr);

    vkDestroyImageView(asVulkan->device, asVulkan->depth.imageView, nullptr);
    vkDestroyImage(asVulkan->device, asVulkan->depth.image, nullptr);
    vkFreeMemory(asVulkan->device, asVulkan->depth.imageMemory, nullptr);

    vkDestroyPipeline(asVulkan->device, asVulkan->graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(asVulkan->device, asVulkan->pipelineLayout, nullptr);
    vkDestroyRenderPass(asVulkan->device, asVulkan->renderPass, nullptr);

    for (auto& imageView : asVulkan->swapChainImageViews)
    {
        vkDestroyImageView(asVulkan->device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(asVulkan->device, asVulkan->swapChain, nullptr);
}

SwapChainSupportDetails as_vulkan_query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

bool as_vulkan_check_validation_layer_support()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (auto* layerName : s_validationLayers)
    {
        auto it = std::find_if(availableLayers.begin(), availableLayers.end(), [layerName](VkLayerProperties availableLayerName)
        {
            return strcmp(layerName, availableLayerName.layerName) == 0;
        });

        if (static_cast<size_t>(it - availableLayers.begin()) == availableLayers.size())
        {
            return false;
        }
    }

    return true;
}

bool as_vulkan_check_device_extension_support(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    for (const std::string& deviceExtension : s_deviceExtensions)
    {
        auto it = std::find_if(availableExtensions.begin(), availableExtensions.end(),
            [&deviceExtension](const VkExtensionProperties& extensionProperty)
        {
            return strcmp(extensionProperty.extensionName, deviceExtension.c_str()) == 0;
        });

        if (it == availableExtensions.end())
        {
            return false;
        }
    }

    return true;
}

VkSurfaceFormatKHR as_vulkan_choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
    {
        return { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }

    auto it = std::find_if(availableFormats.begin(), availableFormats.end(), [](const auto& availableFormat)
    {
        return availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });

    if (it != availableFormats.end())
    {
        return *it;
    }
    else
    {
        if (!availableFormats.empty())
        {
            return availableFormats[0];
        }

        std::cerr << "no viable VkSurfaceFormatKHR available\n";
        return { VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }
}

VkPresentModeKHR as_vulkan_choose_swap_present_mode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    auto it = std::find_if(availablePresentModes.begin(), availablePresentModes.end(), [](const auto& availablePresentMode)
    {
        return availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR;
    });

    if (it != availablePresentModes.end())
    {
        return *it;
    }

    it = std::find_if(availablePresentModes.begin(), availablePresentModes.end(), [](const auto& availablePresentMode)
    {
        return availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR;
    });

    if (it != availablePresentModes.end())
    {
        return *it;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL as_vulkan_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* pUserData)
{
    std::cerr << "validation layer: " << callbackData->pMessage << '\n';;
    return VK_FALSE;
}

void as_vulkan_create(AsVulkan** asVulkan)
{
    *asVulkan = new AsVulkan;
}

void as_vulkan_destroy(AsVulkan** asVulkan)
{
    delete *asVulkan;
    *asVulkan = nullptr;
}

AsVulkanMesh* as_vulkan_mesh(AsVulkan* asVulkan, thh::handle_t handle)
{
    return asVulkan->meshes.resolve(handle);
}

AsVulkanImage* as_vulkan_image(AsVulkan* asVulkan, thh::handle_t handle)
{
    return asVulkan->images.resolve(handle);
}

AsVulkanUniform* as_vulkan_uniform(AsVulkan* asVulkan, thh::handle_t handle)
{
    return asVulkan->uniforms.resolve(handle);
}

thh::handle_t as_vulkan_allocate_mesh_instance(AsVulkan* asVulkan)
{
    return asVulkan->meshInstances.add();
}

AsMeshInstance* as_vulkan_mesh_instance(AsVulkan* asVulkan, thh::handle_t handle)
{
    return asVulkan->meshInstances.resolve(handle);
}

void as_create_mesh(AsMesh** asMesh)
{
    *asMesh = new AsMesh;
}

thh::handle_t as_vulkan_allocate_image(AsVulkan* asVulkan)
{
    return asVulkan->images.add();
}

thh::handle_t as_vulkan_allocate_mesh(AsVulkan* asVulkan)
{
    return asVulkan->meshes.add();
}

thh::handle_t as_vulkan_allocate_uniform(AsVulkan* asVulkan)
{
    return asVulkan->uniforms.add();
}

void as_mesh_instance_transform(AsMeshInstance* asMesh, const as::mat4& transform)
{
    asMesh->transform = transform;
}

void as_mesh_instance_rot(AsMeshInstance* meshInstance, const as::mat4& rot)
{
    meshInstance->rot = rot;
}

void as_mesh_instance_percent(AsMeshInstance* meshInstance, float percent)
{
    meshInstance->percent = percent;
}

void as_mesh_instance_time(AsMeshInstance* meshInstance, float time)
{
    meshInstance->time = time;
}

void as_mesh_instance_mesh(AsMeshInstance* meshInstance, thh::handle_t meshHandle)
{
    meshInstance->meshHandle = meshHandle;
}

void as_mesh_instance_uniform(AsMeshInstance* meshInstance, thh::handle_t uniformHandle)
{
    meshInstance->uniformHandle = uniformHandle;
}

void as_mesh_instance_index(AsMeshInstance* meshInstance, size_t uniformIndex)
{
    meshInstance->uniformIndex = uniformIndex;
}

size_t as_uniform_add_mesh_instance(
    AsVulkanUniform* asUniform, thh::handle_t meshHandle)
{
    asUniform->meshInstanceHandles.push_back(meshHandle);
    return asUniform->meshInstanceHandles.size() - 1;
}

VkResult as_vulkan_create_debug_utils_messenger_ext(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* createInfo,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* debugMessenger)
{
    if (auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT"))
    {
        return func(instance, createInfo, allocator, debugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

VkResult as_vulkan_destroy_debug_utils_messenger_ext(
    VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* allocator)
{
    if (auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT"))
    {
        func(instance, debugMessenger, allocator);
        return VK_SUCCESS;
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void populate_debug_messenger_create_info(
    VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = as_vulkan_debug_callback;
    createInfo.pUserData = nullptr;
}

void as_vulkan_debug(AsVulkan* asVulkan)
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populate_debug_messenger_create_info(createInfo);

    if (as_vulkan_create_debug_utils_messenger_ext(
        asVulkan->instance, &createInfo, nullptr, &asVulkan->debugMessenger) != VK_SUCCESS)
    {
        std::cerr << "failed to setup debug callback\n";
        std::exit(EXIT_FAILURE);
    }
}

static VkSampleCountFlagBits find_max_sample_count(
    const VkPhysicalDeviceProperties& physicalDeviceProperties)
{
    const VkSampleCountFlags counts =
        physicalDeviceProperties.limits.framebufferColorSampleCounts
      & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

bool as_vulkan_check_physical_device_properties(
    VkPhysicalDevice device, VkSurfaceKHR surface,
    uint32_t& selectedGraphicsQueueFamilyIndex,
    uint32_t& selectedPresentQueueFamilyIndex,
    VkSampleCountFlagBits& msaaSamples,
    AsVulkanAlignment& alignment)
{
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    msaaSamples = find_max_sample_count(deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    alignment.minUniformBufferAlignment = deviceProperties.limits.minUniformBufferOffsetAlignment;

    if (!deviceFeatures.samplerAnisotropy)
    {
        return false;
    }

    if (deviceProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
        std::cerr << "Vulkan physical device is not a discrete GPU\n";
        return false;
    }

    if (!as_vulkan_check_device_extension_support(device))
    {
        std::cerr << "Vulkan physical device does not support required extensions\n";
        return false;
    }

    SwapChainSupportDetails swapChainSupport = as_vulkan_query_swap_chain_support(device, surface);
    if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty())
    {
        std::cerr << "Vulkan physical device required swap chain charateristics are not supported\n";
        return false;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    if (queueFamilyCount == 0)
    {
        std::cerr << "Vulkan physical device queue family is empty\n";
        return false;
    }

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilyProperties.data());

    std::vector<VkBool32> queuePresentSupport(queueFamilyCount);

    uint32_t graphicsQueueFamilyIndex = std::numeric_limits<uint32_t>::max();
    uint32_t presentQueueFamilyIndex = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i < queueFamilyCount; ++i)
    {
        vkGetPhysicalDeviceSurfaceSupportKHR(device, static_cast<uint32_t>(i), surface, &queuePresentSupport[i]);

        if ((queueFamilyProperties[i].queueCount > 0) &&
            (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            // select first queue that supports graphics
            if (graphicsQueueFamilyIndex == std::numeric_limits<uint32_t>::max())
            {
                graphicsQueueFamilyIndex = static_cast<uint32_t>(i);
            }

            if (queuePresentSupport[i])
            {
                selectedGraphicsQueueFamilyIndex = static_cast<uint32_t>(i);
                selectedPresentQueueFamilyIndex = static_cast<uint32_t>(i);
                return true;
            }
        }
    }

    for (size_t i = 0; i < queueFamilyCount; ++i)
    {
        if (queuePresentSupport[i])
        {
            presentQueueFamilyIndex = static_cast<uint32_t>(i);
            break;
        }
    }

    if (graphicsQueueFamilyIndex == std::numeric_limits<uint32_t>::max() ||
        presentQueueFamilyIndex == std::numeric_limits<uint32_t>::max())
    {
        std::cerr << "Vulkan physical device graphics or present family are not supported\n";
        return false;
    }

    selectedGraphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
    selectedPresentQueueFamilyIndex = presentQueueFamilyIndex;

    return true;
}

void as_vulkan_pick_physical_device(AsVulkan* asVulkan)
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(asVulkan->instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        std::cerr << "failed to find GPU(s) with Vulkan support\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(asVulkan->instance, &deviceCount, devices.data());

    uint32_t selectedGraphicsQueueFamilyIndex = std::numeric_limits<uint32_t>::max();
    uint32_t selectedPresentQueueFamilyIndex = std::numeric_limits<uint32_t>::max();
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    const auto& it = std::find_if(devices.begin(), devices.end(),
        [&asVulkan,
        &selectedGraphicsQueueFamilyIndex,
        &selectedPresentQueueFamilyIndex,
        &msaaSamples](const auto& device) {
        return as_vulkan_check_physical_device_properties(
            device, asVulkan->surface,
            selectedGraphicsQueueFamilyIndex,
            selectedPresentQueueFamilyIndex,
            msaaSamples,
            asVulkan->alignment);
    });

    if (it != devices.end())
    {
        asVulkan->physicalDevice = *it;
        asVulkan->graphicsQueue.familyIndex = selectedGraphicsQueueFamilyIndex;
        asVulkan->presentQueue.familyIndex = selectedPresentQueueFamilyIndex;
        asVulkan->msaaSamples = msaaSamples;
    }
    else
    {
        std::cerr << "failed to find a suitable GPU\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_logical_device(AsVulkan* asVulkan)
{
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<float> queuePriorities = { 1.0f };

    queueCreateInfos.push_back({
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        asVulkan->graphicsQueue.familyIndex,
        static_cast<uint32_t>(queuePriorities.size()),
        queuePriorities.data()
    });

    if (asVulkan->graphicsQueue.familyIndex != asVulkan->presentQueue.familyIndex)
    {
        queueCreateInfos.push_back({
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            asVulkan->presentQueue.familyIndex,
            static_cast<uint32_t>(queuePriorities.size()),
            queuePriorities.data()
        });
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;

    VkDeviceCreateInfo createInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
        0,
        static_cast<uint32_t>(queueCreateInfos.size()),
        queueCreateInfos.data(),
        0,
        nullptr,
        static_cast<uint32_t>(s_deviceExtensions.size()),
        s_deviceExtensions.data(),
        &deviceFeatures
    };

    if (s_enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(s_validationLayers.size());
        createInfo.ppEnabledLayerNames = s_validationLayers.data();
    }

    if (vkCreateDevice(asVulkan->physicalDevice, &createInfo, nullptr, &asVulkan->device) != VK_SUCCESS)
    {
        std::cerr << "failed to create logical device\n";
        std::exit(EXIT_FAILURE);
    }

    vkGetDeviceQueue(asVulkan->device, asVulkan->graphicsQueue.familyIndex, /*queueIndex=*/0, &asVulkan->graphicsQueue.queue);
    vkGetDeviceQueue(asVulkan->device, asVulkan->presentQueue.familyIndex, /*queueIndex=*/0, &asVulkan->presentQueue.queue);
}

void as_vulkan_create_swap_chain(AsVulkan* asVulkan)
{
    SwapChainSupportDetails swapChainSupport = as_vulkan_query_swap_chain_support(asVulkan->physicalDevice, asVulkan->surface);
    VkSurfaceFormatKHR surfaceFormat = as_vulkan_choose_swap_surface_format(swapChainSupport.formats);
    VkPresentModeKHR presentMode = as_vulkan_choose_swap_present_mode(swapChainSupport.presentModes);
    VkExtent2D extent = swapChainSupport.capabilities.currentExtent;

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = asVulkan->surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(asVulkan->device, &createInfo, nullptr, &asVulkan->swapChain) != VK_SUCCESS)
    {
        std::cerr << "failed to create vk swap chain\n";
        std::exit(EXIT_FAILURE);
    }

    vkGetSwapchainImagesKHR(asVulkan->device, asVulkan->swapChain, &imageCount, nullptr);
    asVulkan->swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(asVulkan->device, asVulkan->swapChain, &imageCount, asVulkan->swapChainImages.data());

    asVulkan->swapChainImageFormat = surfaceFormat.format;
    asVulkan->swapChainExtent = extent;
}

void as_vulkan_recreate_swap_chain(AsVulkan* asVulkan)
{
    vkDeviceWaitIdle(asVulkan->device);

    as_vulkan_cleanup_swap_chain(asVulkan);

    as_vulkan_create_swap_chain(asVulkan);
    as_vulkan_create_image_views(asVulkan);
    as_vulkan_create_render_pass(asVulkan);
    as_vulkan_create_graphics_pipeline(asVulkan);
    as_vulkan_create_color_resources(asVulkan);
    as_vulkan_create_depth_resources(asVulkan);
}

VkImageView as_vulkan_create_image_view(
    AsVulkan* asVulkan, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags,
    const uint32_t mipLevels)
{
    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = aspectFlags;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = mipLevels;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(asVulkan->device, &imageViewCreateInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        std::cerr << "failed to create image view\n";
        std::exit(EXIT_FAILURE);
    }

    return imageView;
}

void as_vulkan_create_image_views(AsVulkan* asVulkan)
{
    asVulkan->swapChainImageViews.resize(asVulkan->swapChainImages.size());
    for (size_t i = 0; i < asVulkan->swapChainImages.size(); ++i)
    {
        asVulkan->swapChainImageViews[i] = as_vulkan_create_image_view(
            asVulkan, asVulkan->swapChainImages[i],
            asVulkan->swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
}

VkShaderModule as_vulkan_create_shader_module(AsVulkan* asVulkan, const std::string& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(asVulkan->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        std::cerr << "failed to create shader module\n";
        std::exit(EXIT_FAILURE);
    }

    return shaderModule;
}

void as_vulkan_create_graphics_pipeline(AsVulkan* asVulkan)
{
    std::string vert_shader;
    if (!fileops::readFile("shaders/vert.spv", vert_shader))
    {
        std::cerr << "failed to read vert shader\n";
        std::exit(EXIT_FAILURE);
    }

    VkShaderModule vertShaderModule =
        as_vulkan_create_shader_module(asVulkan, vert_shader);

    std::string frag_shader;
    if (!fileops::readFile("shaders/frag.spv", frag_shader))
    {
        std::cerr << "failed to read frag shader\n";
        std::exit(EXIT_FAILURE);
    }

    VkShaderModule fragShaderModule =
        as_vulkan_create_shader_module(asVulkan, frag_shader);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageInfo, fragShaderStageInfo
    };

    auto bindingDescription = as_vulkan_get_vertex_binding_description();
    auto attributeDescriptions = as_vulkan_get_attribute_descriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    // number of elements in the pVertexBindingDescriptions array
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    // array describing all bindings defined for a given pipeline
    // (buffers from which values of all attributes are read)
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    // number of elements in the pVertexAttributeDescriptions array
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    // array with elements specifying all vertex attributes
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // topology used for drawing vertices (like triangle fan, strip, list)
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // parameter defining whether we want to restart assembling a primitive
    // by using a special value of vertex index
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(asVulkan->swapChainExtent.width);
    viewport.height = static_cast<float>(asVulkan->swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = asVulkan->swapChainExtent;

    // note - can use dynamic state instead - TODO
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    // number of viewports
    viewportState.viewportCount = 1;
    // pointer to a structure defining static viewport parameters
    viewportState.pViewports = &viewport;
    // number of scissor rectangles (must have the same value as viewportCount parameter)
    viewportState.scissorCount = 1;
    // pointer to an array of 2D rectangles defining static scissor test parameters for each viewport
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizerState{};
    rasterizerState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerState.depthClampEnable = VK_FALSE;
    rasterizerState.rasterizerDiscardEnable = VK_FALSE;
    rasterizerState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerState.lineWidth = 1.0f;
    rasterizerState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizerState.depthBiasEnable = VK_FALSE;
    rasterizerState.depthBiasConstantFactor = 0.0f;
    rasterizerState.depthBiasClamp = 0.0f;
    rasterizerState.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.minSampleShading = 0.2f; // 1.0f
    multisampling.rasterizationSamples = asVulkan->msaaSamples;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // depth/stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};
    depthStencil.back = {};

    // no blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    // number of elements in pDynamicStates array
    dynamicState.dynamicStateCount = std::size(dynamicStates);
    // array containing enums, specifying which parts of a pipeline should be
    // marked as dynamic. each element of this array is of type VkDynamicState
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &asVulkan->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(asVulkan->device, &pipelineLayoutInfo, nullptr, &asVulkan->pipelineLayout) != VK_SUCCESS)
    {
        std::cerr << "failed to create pipeline layout\n";
        std::exit(EXIT_FAILURE);
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = std::size(shaderStages);
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizerState;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = asVulkan->pipelineLayout;
    pipelineInfo.renderPass = asVulkan->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(asVulkan->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &asVulkan->graphicsPipeline) != VK_SUCCESS)
    {
        std::cerr << "failed to create graphics pipeline\n";
        std::exit(EXIT_FAILURE);
    }

    vkDestroyShaderModule(asVulkan->device, vertShaderModule, nullptr);
    vkDestroyShaderModule(asVulkan->device, fragShaderModule, nullptr);
}

void as_vulkan_create_render_pass(AsVulkan* asVulkan)
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = asVulkan->swapChainImageFormat;
    colorAttachment.samples = asVulkan->msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = as_vulkan_find_depth_format(asVulkan);
    depthAttachment.samples = asVulkan->msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = asVulkan->swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkAttachmentDescription attachments[] = {
        colorAttachment, depthAttachment, colorAttachmentResolve };

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = std::size(attachments);
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    std::vector<VkSubpassDependency> subpassDependencies;

    {
        VkSubpassDependency subpassDependency{};
        // index of first (previous) subpass or VK_SUBPASS_EXTERNAL to indicate
        // dependency between subpass and operations outside render pass
        subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        // index of second (next/later) subpass or VK_SUBPASS_EXTERNAL
        subpassDependency.dstSubpass = 0;
        // pipeline stage during which a given attachment was used before (in a src subpass)
        subpassDependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        // pipeline stage during which a given attachment will be used later(in a dst subpass)
        subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // types of memory operations that occurred in a src subpass or before a render pass
        subpassDependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        // types of memory operations that occurred in a dst subpass or after a render pass
        subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        // flag describing the type (region) of dependency
        subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        subpassDependencies.push_back(subpassDependency);
    }

    {
        VkSubpassDependency subpassDependency{};
        subpassDependency.srcSubpass = 0;
        subpassDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
        subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependency.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        subpassDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        subpassDependency.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        subpassDependencies.push_back(subpassDependency);
    }

    renderPassInfo.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
    renderPassInfo.pDependencies = subpassDependencies.data();

    if (vkCreateRenderPass(asVulkan->device, &renderPassInfo, nullptr, &asVulkan->renderPass) != VK_SUCCESS)
    {
        std::cerr << "failed to create render pass\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_framebuffer(AsVulkan* asVulkan, VkFramebuffer& framebuffer, VkImageView imageView)
{
    if (framebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(asVulkan->device, framebuffer, nullptr);
        framebuffer = VK_NULL_HANDLE;
    }

    VkImageView attachments[] = {
        asVulkan->color.imageView,
        asVulkan->depth.imageView,
        imageView
    };

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = asVulkan->renderPass;
    framebufferInfo.attachmentCount = std::size(attachments);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = asVulkan->swapChainExtent.width;
    framebufferInfo.height = asVulkan->swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(asVulkan->device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS)
    {
        std::cerr << "failed to create framebuffer\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_command_pool(AsVulkan* asVulkan)
{
    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags =
        // command buffers allocated from this pool may be reset individually (*2)
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
        // command buffers allocated form this pool may be living for a short amount of time
        // they will often be recorded and reset (re-recorded)
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolInfo.queueFamilyIndex = asVulkan->graphicsQueue.familyIndex;

    if (vkCreateCommandPool(asVulkan->device, &commandPoolInfo, nullptr, &asVulkan->commandPool) != VK_SUCCESS)
    {
        std::cerr << "failed to create command pool\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_allocate_command_buffers(AsVulkan* asVulkan, VkCommandPool pool, VkCommandBuffer* commandBuffers, uint32_t count)
{
    VkCommandBufferAllocateInfo commandBufferAllocInfo{};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandPool = pool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandBufferCount = count;

    if (vkAllocateCommandBuffers(asVulkan->device, &commandBufferAllocInfo, commandBuffers) != VK_SUCCESS)
    {
        std::cerr << "failed to allocate command buffers";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_command_buffers(AsVulkan* asVulkan)
{
    as_vulkan_create_command_pool(asVulkan);
    for (AsRenderResource& renderingResource : asVulkan->renderingResources)
    {
        as_vulkan_allocate_command_buffers(asVulkan, asVulkan->commandPool, &renderingResource.commandBuffer, 1);
    }
}

void as_vulkan_create_semaphores(AsVulkan* asVulkan)
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (AsRenderResource& renderingResource : asVulkan->renderingResources)
    {
        if (vkCreateSemaphore(asVulkan->device, &semaphoreInfo, nullptr, &renderingResource.imageAvailable) != VK_SUCCESS ||
            vkCreateSemaphore(asVulkan->device, &semaphoreInfo, nullptr, &renderingResource.renderFinished) != VK_SUCCESS)
        {
            std::cerr << "failed to create semaphores\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

void as_vulkan_create_fences(AsVulkan* asVulkan)
{
    VkFenceCreateInfo fenceInfo{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        // this parameter allows for creating a fence that is already signaled
        VK_FENCE_CREATE_SIGNALED_BIT
    };

    for (AsRenderResource& renderingResource : asVulkan->renderingResources)
    {
        if (vkCreateFence(asVulkan->device, &fenceInfo, nullptr, &renderingResource.commandBufferFinished) != VK_SUCCESS)
        {
            std::cerr << "failed to create fence\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

void as_vulkan_create_rendering_resources(AsVulkan* asVulkan)
{
    as_vulkan_create_command_buffers(asVulkan);
    as_vulkan_create_semaphores(asVulkan);
    as_vulkan_create_fences(asVulkan);
}

uint32_t as_vulkan_find_memory_type(AsVulkan* asVulkan, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(asVulkan->physicalDevice, &memoryProperties);

    for (size_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((memoryTypeBits & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) // check all properties match
        {
            return static_cast<uint32_t>(i);
        }
    }

    std::cerr << "failed to find memory type\n";
    std::exit(EXIT_FAILURE);

    return std::numeric_limits<uint32_t>::max();
}

VkCommandBuffer as_vulkan_begin_single_time_commands(AsVulkan* asVulkan)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = asVulkan->commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(asVulkan->device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void as_vulkan_end_single_time_commands(AsVulkan* asVulkan, VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(asVulkan->graphicsQueue.queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(asVulkan->graphicsQueue.queue);
    vkDeviceWaitIdle(asVulkan->device);

    vkFreeCommandBuffers(asVulkan->device, asVulkan->commandPool, 1, &commandBuffer);
}

void as_vulkan_transition_image_layout(
    AsVulkan* asVulkan, VkImage image, VkFormat format,
    VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
{
    VkCommandBuffer commandBuffer = as_vulkan_begin_single_time_commands(asVulkan);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (as_vulkan_has_stencil_component(format))
        {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VkPipelineStageFlags srcStage{};
    VkPipelineStageFlags dstStage{};

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else
    {
        std::cerr << "unsupported layout transition\n";
        std::exit(EXIT_FAILURE);
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    as_vulkan_end_single_time_commands(asVulkan, commandBuffer);
}

void as_vulkan_copy_buffer_to_image(AsVulkan* asVulkan, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = as_vulkan_begin_single_time_commands(asVulkan);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    as_vulkan_end_single_time_commands(asVulkan, commandBuffer);
}

void as_vulkan_copy_buffer(AsVulkan* asVulkan, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = as_vulkan_begin_single_time_commands(asVulkan);

    VkBufferCopy copyRegion{};
    // size of the data (in bytes) we want to copy
    copyRegion.size = size;
    // offset in bytes in a source buffer from which we want to copy data
    copyRegion.srcOffset = 0;
    // offset in bytes in a destination buffer into which we want to copy data
    copyRegion.dstOffset = 0;

    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    VkBufferMemoryBarrier bufferMemoryBarrier{};
    bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferMemoryBarrier.pNext = nullptr;
    // types of memory operations that were performed on this buffer before the barrier
    bufferMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    // memory operations that will be performed on a given buffer after the barrier
    bufferMemoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    // index of a queue family that accessed the buffer before
    bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // queue family that will access the buffer from now on
    bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // handle to the buffer for which we set up a barrier
    bufferMemoryBarrier.buffer = dstBuffer;
    // memory offset from the start of the buffer (from the memory’s base offset bound to the buffer)
    bufferMemoryBarrier.offset = 0;
    // size of the buffer’s memory area for which we want to setup a barrier
    bufferMemoryBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr,
        1, &bufferMemoryBarrier, 0, nullptr);

    as_vulkan_end_single_time_commands(asVulkan, commandBuffer);
}

void as_vulkan_create_buffer(
    AsVulkan* asVulkan, VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // size, in bytes, of a buffer
    bufferInfo.size = size;
    // this parameter defines how we intend to use this buffer in future
    bufferInfo.usage = usage;
    // defines whether a given buffer can be accessed by multiple queues at the same time
    // (concurrent sharing mode) or by just a single queue (exclusive sharing mode) (*1)
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // number of queue indices in pQueueFamilyIndices array (only when
    // concurrent sharing mode is specified)
    bufferInfo.queueFamilyIndexCount = 0;
    // array with indices of all queues that will reference buffer (only when
    // concurrent sharing mode is specified)
    bufferInfo.pQueueFamilyIndices = nullptr;

    if (vkCreateBuffer(asVulkan->device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        std::cerr << "failed to create vertex buffer\n";
        std::exit(EXIT_FAILURE);
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(asVulkan->device, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // minimum required memory size that should be allocated
    allocInfo.allocationSize = memoryRequirements.size;
    // index of a memory type we want to use for a created memory object
    // it is the index of one of the bits that are set (has value of one) in
    // buffer’s memory requirement
    allocInfo.memoryTypeIndex = as_vulkan_find_memory_type(
        asVulkan, memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(asVulkan->device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        std::cerr << "failed to allocate vertex buffer memory\n";
        std::exit(EXIT_FAILURE);
    }

    vkBindBufferMemory(asVulkan->device, buffer, bufferMemory, 0);
}

void as_vulkan_create_index_buffer(AsVulkan* asVulkan, AsVulkanMesh* asMesh, const AsMesh* mesh)
{
    VkDeviceSize bufferSize = sizeof(mesh->indices[0]) * mesh->indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    as_vulkan_create_buffer(
        asVulkan, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(asVulkan->device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh->indices.data(), static_cast<size_t>(bufferSize));

    VkMappedMemoryRange flushRange{};
    flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    flushRange.pNext = nullptr;
    flushRange.memory = stagingBufferMemory;
    flushRange.offset = 0;
    flushRange.size = bufferSize;

    vkFlushMappedMemoryRanges(asVulkan->device, 1, &flushRange);

    vkUnmapMemory(asVulkan->device, stagingBufferMemory);

    as_vulkan_create_buffer(
        asVulkan, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        asMesh->indexBuffer, asMesh->indexBufferMemory);

    as_vulkan_copy_buffer(asVulkan, stagingBuffer, asMesh->indexBuffer, bufferSize);

    asMesh->indexCount = mesh->indices.size();

    vkDestroyBuffer(asVulkan->device, stagingBuffer, nullptr);
    vkFreeMemory(asVulkan->device, stagingBufferMemory, nullptr);
}

void as_vulkan_create_vertex_buffer(AsVulkan* asVulkan, AsVulkanMesh* asMesh, const AsMesh* mesh)
{
    VkDeviceSize bufferSize = sizeof(mesh->vertices[0]) * mesh->vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    as_vulkan_create_buffer(
        asVulkan, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(asVulkan->device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh->vertices.data(), static_cast<size_t>(bufferSize));

    VkMappedMemoryRange flushRange{};
    flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    flushRange.pNext = nullptr;
    flushRange.memory = stagingBufferMemory;
    flushRange.offset = 0;
    flushRange.size = bufferSize;

    vkFlushMappedMemoryRanges(asVulkan->device, 1, &flushRange);

    vkUnmapMemory(asVulkan->device, stagingBufferMemory);

    as_vulkan_create_buffer(
        asVulkan, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        asMesh->vertexBuffer, asMesh->vertexBufferMemory);

    as_vulkan_copy_buffer(asVulkan, stagingBuffer, asMesh->vertexBuffer, bufferSize);

    vkDestroyBuffer(asVulkan->device, stagingBuffer, nullptr);
    vkFreeMemory(asVulkan->device, stagingBufferMemory, nullptr);
}

void as_vulkan_create_uniform_buffer(AsVulkan* asVulkan, AsVulkanUniform* asUniform, size_t uniformBufferObjectCount)
{
    VkDeviceSize bufferSize =
        as_vulkan_uniform_alignment<UniformBufferObject>(asVulkan->alignment) *
        uniformBufferObjectCount;

    as_vulkan_create_buffer(
        asVulkan,
        bufferSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        asUniform->uniformBuffer,
        asUniform->uniformBufferMemory);
}

void as_vulkan_create_descriptor_pool(AsVulkan* asVulkan)
{
    VkDescriptorPoolSize poolSizes[] = { {}, {} };
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    poolSizes[0].descriptorCount = 3;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 3;
    poolInfo.flags = 0;

    if (vkCreateDescriptorPool(asVulkan->device, &poolInfo, nullptr, &asVulkan->descriptorPool) != VK_SUCCESS)
    {
        std::cerr << "failed to create descriptor pool\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_descriptor_set(AsVulkan* asVulkan, AsVulkanUniform* asUniform, AsVulkanImage* asImage)
{
    VkDescriptorSetLayout layouts[] = { asVulkan->descriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = asVulkan->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(asVulkan->device, &allocInfo, &asUniform->descriptorSet) != VK_SUCCESS)
    {
        std::cerr << "failed to allocate descriptor set\n";
        std::exit(EXIT_FAILURE);
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = asUniform->uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = as_vulkan_uniform_alignment<UniformBufferObject>(asVulkan->alignment);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = asImage->imageView;
    imageInfo.sampler = asVulkan->imageSampler;

    VkWriteDescriptorSet descriptorWrites[] = { {}, {} };
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = asUniform->descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;
    descriptorWrites[0].pImageInfo = nullptr;
    descriptorWrites[0].pTexelBufferView = nullptr;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = asUniform->descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;
    descriptorWrites[1].pBufferInfo = nullptr;
    descriptorWrites[1].pTexelBufferView = nullptr;

    vkUpdateDescriptorSets(asVulkan->device, 2, descriptorWrites, 0, nullptr);
}

void as_vulkan_prepare_frame(
    AsVulkan* asVulkan, VkCommandBuffer commandBuffer, VkImage image, VkImageView imageView, VkFramebuffer& framebuffer)
{
    as_vulkan_create_framebuffer(asVulkan, framebuffer, imageView);

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = nullptr;

    vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

    VkImageSubresourceRange imageSubresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, // VkImageAspectFlags - aspectMask
        0, // uint32_t - baseMipLevel
        1, // uint32_t - levelCount
        0, // uint32_t - baseArrayLayer
        1 // uint32_t - layerCount
    };

    if (asVulkan->presentQueue.queue != asVulkan->graphicsQueue.queue)
    {
        VkImageMemoryBarrier barrierFromPresentToDraw = {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // VkStructureType - sType
            nullptr, // const void - *pNext
            VK_ACCESS_MEMORY_READ_BIT, // VkAccessFlags - srcAccessMask
            VK_ACCESS_MEMORY_READ_BIT, // VkAccessFlags - dstAccessMask
            VK_IMAGE_LAYOUT_UNDEFINED, // VkImageLayout - oldLayout
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // VkImageLayout - newLayout
            asVulkan->presentQueue.familyIndex, // uint32_t - srcQueueFamilyIndex
            asVulkan->graphicsQueue.familyIndex, // uint32_t - dstQueueFamilyIndex
            image, // VkImage - image
            imageSubresourceRange // VkImageSubresourceRange - subresourceRange
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierFromPresentToDraw);
    }

    VkClearValue clearValues[2]{};
    clearValues[0].color = { { 0.392f, 0.584f, 0.929f, 1.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = asVulkan->renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = asVulkan->swapChainExtent;
    renderPassInfo.clearValueCount = std::size(clearValues);
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, asVulkan->graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(asVulkan->swapChainExtent.width);
    viewport.height = static_cast<float>(asVulkan->swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = asVulkan->swapChainExtent;

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    VkDeviceSize uniformAlignment = as_vulkan_uniform_alignment<UniformBufferObject>(asVulkan->alignment);
    asVulkan->meshInstances.enumerate(
        [asVulkan, uniformAlignment, commandBuffer](const auto& meshInstance) {
        uint32_t dynamicOffset = static_cast<uint32_t>(meshInstance.uniformIndex * uniformAlignment);

        const AsVulkanMesh& mesh = *asVulkan->meshes.resolve(meshInstance.meshHandle);
        const AsVulkanUniform& uniform = *asVulkan->uniforms.resolve(meshInstance.uniformHandle);

        VkBuffer vertexBuffers[] = { mesh.vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            asVulkan->pipelineLayout, 0, 1, &uniform.descriptorSet, 1,
            &dynamicOffset);

        vkCmdDrawIndexed(
            commandBuffer,
            static_cast<uint32_t>(mesh.indexCount),
            1, 0, 0, 0);
    });

    vkCmdEndRenderPass(commandBuffer);

    if (asVulkan->presentQueue.queue != asVulkan->graphicsQueue.queue)
    {
        VkImageMemoryBarrier barrierFromPresentToDraw = {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,  // VkStructureType - sType
            nullptr,  // const void - *pNext
            VK_ACCESS_MEMORY_READ_BIT,  // VkAccessFlags - srcAccessMask
            VK_ACCESS_MEMORY_READ_BIT,  // VkAccessFlags - dstAccessMask
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,  // VkImageLayout - oldLayout
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,  // VkImageLayout - newLayout
            asVulkan->graphicsQueue.familyIndex,  // uint32_t - srcQueueFamilyIndex
            asVulkan->presentQueue.familyIndex,  // uint32_t - dstQueueFamilyIndex
            image,  // VkImage - image
            imageSubresourceRange // VkImageSubresourceRange - subresourceRange
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierFromPresentToDraw);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        std::cerr << "failed to record command buffer\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_update_uniform_buffer(
    AsVulkan* asVulkan, const as::mat4& view, float deltaTime)
{
    asVulkan->uniforms.enumerate([asVulkan, &view, deltaTime](auto& uniform) {
        if (uniform.meshInstanceHandles.empty())
        {
            return;
        }

        VkDeviceSize uniformAlignment = as_vulkan_uniform_alignment<UniformBufferObject>(asVulkan->alignment);

        char* uniformData;
        vkMapMemory(
            asVulkan->device, uniform.uniformBufferMemory, 0,
            uniformAlignment * uniform.meshInstanceHandles.size(),
            0, reinterpret_cast<void**>(&uniformData));

        as::mat4 proj = as::perspective_vulkan_rh(
            as::radians(45.0f),
            asVulkan->swapChainExtent.width / static_cast<float>(asVulkan->swapChainExtent.height),
            0.1f, 100.0f);

        for (size_t i = 0; i < uniform.meshInstanceHandles.size(); ++i)
        {
            const as::mat4 model =
              asVulkan->meshInstances.resolve(uniform.meshInstanceHandles[i])->transform;

            UniformBufferObject ubo;
            ubo.mvp = proj * view * model;

            memcpy(&uniformData[i * uniformAlignment], &ubo, sizeof(UniformBufferObject));
        }

        VkMappedMemoryRange flushRange = {
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            nullptr,
            // memory - handle of a mapped and modified memory object
            uniform.uniformBufferMemory,
            // offset - offset (from the beginning of a given memory
            // object’s storage) at which a given range starts
            0,
            // size, in bytes, of an affected region. if the whole memory,
            // from an offset to the end, was modified, we can use the special
            // value of VK_WHOLE_SIZE
            VK_WHOLE_SIZE
        };

        vkFlushMappedMemoryRanges(asVulkan->device, 1, &flushRange);

        vkUnmapMemory(
            asVulkan->device, uniform.uniformBufferMemory);
    });
}

void as_vulkan_draw_frame(AsVulkan* asVulkan)
{
    AsRenderResource& renderResource = asVulkan->renderingResources[asVulkan->resourceIndex];
    asVulkan->resourceIndex = (asVulkan->resourceIndex + 1) % AsVulkan::RenderingResourcesCount;

    if (vkWaitForFences(asVulkan->device, 1, &renderResource.commandBufferFinished, VK_FALSE, 1000000000) != VK_SUCCESS)
    {
        std::cerr << "waiting for fence took too long\n";
        std::exit(EXIT_FAILURE);
    }

    vkResetFences(asVulkan->device, 1, &renderResource.commandBufferFinished);

    uint32_t imageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(
        asVulkan->device,
        asVulkan->swapChain,
        std::numeric_limits<uint64_t>::max(),
        renderResource.imageAvailable,
        VK_NULL_HANDLE, &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        as_vulkan_recreate_swap_chain(asVulkan);
        return;
    }
    else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
    {
        std::cerr << "failed to acquire swap chain image\n";
        std::exit(EXIT_FAILURE);
    }

    as_vulkan_prepare_frame(
        asVulkan, renderResource.commandBuffer, asVulkan->swapChainImages[imageIndex],
        asVulkan->swapChainImageViews[imageIndex], renderResource.framebuffer);

    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore waitSemaphores[] = { renderResource.imageAvailable };
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &renderResource.commandBuffer;
    VkSemaphore signalSemaphores[] = { renderResource.renderFinished };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(asVulkan->graphicsQueue.queue, 1, &submitInfo, renderResource.commandBufferFinished) != VK_SUCCESS)
    {
        std::cerr << "failed to submit draw command buffer\n";
        std::exit(EXIT_FAILURE);
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = { asVulkan->swapChain };
    presentInfo.swapchainCount = std::size(swapChains);
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    VkResult presentResult = vkQueuePresentKHR(asVulkan->presentQueue.queue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        as_vulkan_recreate_swap_chain(asVulkan);
    }
    else if (presentResult != VK_SUCCESS)
    {
        std::cerr << "failed to present swap chain image\n";
        std::exit(EXIT_FAILURE);
    }
}

static void check_extensions_are_available(
    const char* required_instance_extensions[],
    int64_t required_instance_extension_count,
    const std::vector<VkExtensionProperties>& available_instance_extensions)
{
    // build vector of available extension names
    std::vector<const char*> available_extension_names;
    std::transform(
        available_instance_extensions.begin(),
        available_instance_extensions.end(),
        std::back_inserter(available_extension_names),
        [](const auto& extensionProperties){
            return extensionProperties.extensionName;
        });

    const auto string_sort_compare_less = [](const char* lhs, const char* rhs) {
        return std::strcmp(lhs, rhs) < 0;
    };

    const auto string_sort_compare_equal = [](const char* lhs, const char* rhs) {
        return std::strcmp(lhs, rhs) == 0;
    };

    // sort required and available extension lists
    std::sort(
        required_instance_extensions,
        required_instance_extensions + required_instance_extension_count,
        string_sort_compare_less);
    std::sort(
        available_extension_names.begin(), available_extension_names.end(),
        string_sort_compare_less);

    std::cout << "Available extensions: \n";
    for (const auto& extension : available_extension_names) {
        std::cout << extension << '\n';
    }

    std::cout << "Instance extensions: \n";
    for (int64_t index = 0; index < required_instance_extension_count; ++index) {
        std::cout << required_instance_extensions[index] << '\n';
    }

    // ensure required extensions can be found in the available list
    if (const auto found = std::search(
            available_extension_names.begin(), available_extension_names.end(),
            required_instance_extensions,
            required_instance_extensions + required_instance_extension_count,
            string_sort_compare_equal);
        found == available_extension_names.end())
    {
        std::cerr << "not all required instance extensions found "
                     "in supported extension list\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_instance(
    AsVulkan* asVulkan, const char* required_instance_extensions[],
    int64_t required_instance_extension_count)
{
    if (s_enableValidationLayers && !as_vulkan_check_validation_layer_support())
    {
        std::cerr << "validation layer requested, but not available\n";
        std::exit(EXIT_FAILURE);
    }

    uint32_t availableExtensionCount = 0;
    vkEnumerateInstanceExtensionProperties(
        nullptr, &availableExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
    vkEnumerateInstanceExtensionProperties(
        nullptr, &availableExtensionCount, availableExtensions.data());

    check_extensions_are_available(
        required_instance_extensions, required_instance_extension_count,
        availableExtensions);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "as-vulkan-app";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> enabledExtensions(
        required_instance_extensions,
        required_instance_extensions + required_instance_extension_count);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (s_enableValidationLayers)
    {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        createInfo.enabledLayerCount = static_cast<uint32_t>(s_validationLayers.size());
        createInfo.ppEnabledLayerNames = s_validationLayers.data();

        populate_debug_messenger_create_info(debugCreateInfo);
        createInfo.pNext = &debugCreateInfo;
    }

    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());

    if (vkCreateInstance(&createInfo, nullptr, &asVulkan->instance) != VK_SUCCESS)
    {
        std::cerr << "failed to create vk instance\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_cleanup(AsVulkan* asVulkan)
{
    vkDeviceWaitIdle(asVulkan->device);

    as_vulkan_cleanup_rendering_resources(asVulkan);

    vkDestroyCommandPool(asVulkan->device, asVulkan->commandPool, nullptr);

    as_vulkan_cleanup_swap_chain(asVulkan);

    vkDestroySampler(asVulkan->device, asVulkan->imageSampler, nullptr);

    asVulkan->images.enumerate([asVulkan](auto& image) {
        vkDestroyImageView(asVulkan->device, image.imageView, nullptr);
        vkDestroyImage(asVulkan->device, image.image, nullptr);
        vkFreeMemory(asVulkan->device, image.imageMemory, nullptr);
    });

    vkDestroyDescriptorPool(asVulkan->device, asVulkan->descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(asVulkan->device, asVulkan->descriptorSetLayout, nullptr);

    asVulkan->uniforms.enumerate([asVulkan](auto& uniform) {
        vkDestroyBuffer(asVulkan->device, uniform.uniformBuffer, nullptr);
        vkFreeMemory(asVulkan->device, uniform.uniformBufferMemory, nullptr);
    });

    asVulkan->meshes.enumerate([asVulkan](AsVulkanMesh& mesh) {
        vkDestroyBuffer(asVulkan->device, mesh.vertexBuffer, nullptr);
        vkFreeMemory(asVulkan->device, mesh.vertexBufferMemory, nullptr);
        vkDestroyBuffer(asVulkan->device, mesh.indexBuffer, nullptr);
        vkFreeMemory(asVulkan->device, mesh.indexBufferMemory, nullptr);
    });

    vkDestroyDevice(asVulkan->device, nullptr);
    as_vulkan_destroy_debug_utils_messenger_ext(
        asVulkan->instance, asVulkan->debugMessenger, nullptr);
    vkDestroySurfaceKHR(asVulkan->instance, asVulkan->surface, nullptr);
    vkDestroyInstance(asVulkan->instance, nullptr);
}

#ifdef AS_VULKAN_SDL
void as_sdl_vulkan_create_surface(AsVulkan* asVulkan, SDL_Window* window)
{
    if (!SDL_Vulkan_CreateSurface(
        window, asVulkan->instance, &asVulkan->surface))
    {
        std::cerr << "failed to create window surface\n";
        std::exit(EXIT_FAILURE);
    }
}

void sdl_vulkan_instance_extensions(
    SDL_Window* window, const char**& required_instance_extensions,
    uint32_t& required_instance_extension_count)
{
    if (!SDL_Vulkan_GetInstanceExtensions(
        window, &required_instance_extension_count, nullptr))
    {
        std::cerr << "failed to get instance extensions\n";
        std::exit(EXIT_FAILURE);
    }

    required_instance_extensions =
        new const char*[required_instance_extension_count];

    if (!SDL_Vulkan_GetInstanceExtensions(
        window, &required_instance_extension_count,
        required_instance_extensions))
    {
        std::cerr << "failed to get instance extensions\n";
        std::exit(EXIT_FAILURE);
    }
}
#endif // AS_VULKAN_SDL

void as_vulkan_create_image(
    AsVulkan* asVulkan, uint32_t width, uint32_t height, uint32_t mipLevels,
    VkSampleCountFlagBits sampleCount, VkFormat format, VkImageTiling tiling,
    VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
    VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = sampleCount;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    if (vkCreateImage(asVulkan->device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        std::cerr << "failed to create image\n";
        std::exit(EXIT_FAILURE);
    }

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(asVulkan->device, image, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = as_vulkan_find_memory_type(
        asVulkan, memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(asVulkan->device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        std::cerr << "failed to allocate image memory\n";
        std::exit(EXIT_FAILURE);
    }

    vkBindImageMemory(asVulkan->device, image, imageMemory, 0);
}

void as_vulkan_generate_mipmaps(
    AsVulkan* asVulkan, const VkImage image, const VkFormat imageFormat,
    const int32_t width, const int32_t height, const uint32_t mipLevels)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(
        asVulkan->physicalDevice, imageFormat, &formatProperties);

    if ((formatProperties.optimalTilingFeatures
            & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0)
    {
        std::cerr << "texture image format does not support linear blitting\n";
        std::exit(EXIT_FAILURE);
    }

    VkCommandBuffer commandBuffer = as_vulkan_begin_single_time_commands(asVulkan);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = width;
    int32_t mipHeight = height;

    for (uint64_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &barrier);
        
        VkImageBlit blit{};
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1; // source mip level
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(
            commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &barrier);

        if (mipWidth > 1) {
            mipWidth /= 2;
        }
        if (mipHeight > 1) {
            mipHeight /= 2;
        }
    }

    // handle last mip level
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, 
        &barrier);

    as_vulkan_end_single_time_commands(asVulkan, commandBuffer);
}

void as_vulkan_create_as_image(
    AsVulkan* asVulkan, AsVulkanImage* asImage, const char* path)
{
    int width, height, channels;
    stbi_uc* pixels = stbi_load(
        path, &width, &height, &channels, STBI_rgb_alpha);

    if (pixels == nullptr)
    {
        std::cerr << "failed to load texture image\n";
        std::exit(EXIT_FAILURE);
    }

    const uint32_t mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width * height * 4);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    as_vulkan_create_buffer(
        asVulkan, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(asVulkan->device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(asVulkan->device, stagingBufferMemory);

    stbi_image_free(pixels);

    as_vulkan_create_image(
        asVulkan, width, height, mipLevels, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asImage->image, asImage->imageMemory);

    as_vulkan_transition_image_layout(
        asVulkan, asImage->image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipLevels);

    as_vulkan_copy_buffer_to_image(
        asVulkan, stagingBuffer, asImage->image,
        static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    vkDestroyBuffer(asVulkan->device, stagingBuffer, nullptr);
    vkFreeMemory(asVulkan->device, stagingBufferMemory, nullptr);

    as_vulkan_generate_mipmaps(
        asVulkan, asImage->image, VK_FORMAT_R8G8B8A8_SRGB, width, height,
        mipLevels);

    asImage->imageView = as_vulkan_create_image_view(
        asVulkan, asImage->image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void as_vulkan_create_image_sampler(AsVulkan* asVulkan)
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    if (vkCreateSampler(asVulkan->device, &samplerInfo, nullptr, &asVulkan->imageSampler) != VK_SUCCESS)
    {
        std::cerr << "failed to create texture sampler\n";
        std::exit(EXIT_FAILURE);
    }
}

void as_vulkan_create_color_resources(AsVulkan* asVulkan)
{
    const VkFormat colorFormat = asVulkan->swapChainImageFormat;
    as_vulkan_create_image(
        asVulkan, asVulkan->swapChainExtent.width,
        asVulkan->swapChainExtent.height, 1, asVulkan->msaaSamples,
        colorFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asVulkan->color.image,
        asVulkan->color.imageMemory);
    
    asVulkan->color.imageView = as_vulkan_create_image_view(
        asVulkan, asVulkan->color.image, colorFormat,
        VK_IMAGE_ASPECT_COLOR_BIT, 1);
}

void as_vulkan_create_depth_resources(AsVulkan* asVulkan)
{
    VkFormat depthFormat = as_vulkan_find_depth_format(asVulkan);
    as_vulkan_create_image(
        asVulkan, asVulkan->swapChainExtent.width,
        asVulkan->swapChainExtent.height, 1, asVulkan->msaaSamples, depthFormat,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, asVulkan->depth.image,
        asVulkan->depth.imageMemory);

    asVulkan->depth.imageView = as_vulkan_create_image_view(
        asVulkan, asVulkan->depth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    as_vulkan_transition_image_layout(
        asVulkan, asVulkan->depth.image, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

void as_vulkan_create_mesh_data_vector(
    AsMesh* asMesh, const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes)
{
    // struct to hold indices into vertex and uv data
    struct Tri
    {
        // index lookups into loaded mesh vertex and uv data
        int vert_indices[3];
        int uv_indices[2];
        // index into populated mesh data (to eliminate duplicates)
        uint32_t vertex_index;
    };

    // less than operator for upper_bound and equal_range
    // (could be implemented as free function - operator <)
    auto triLessThan = [](const Tri& lhs, const Tri& rhs)
    {
        return (lhs.vert_indices[0] <  rhs.vert_indices[0]) ||
            (lhs.vert_indices[0] == rhs.vert_indices[0] && lhs.vert_indices[1] <  rhs.vert_indices[1]) ||
            (lhs.vert_indices[0] == rhs.vert_indices[0] && lhs.vert_indices[1] == rhs.vert_indices[1] && lhs.vert_indices[2] <  rhs.vert_indices[2]) ||
            (lhs.vert_indices[0] == rhs.vert_indices[0] && lhs.vert_indices[1] == rhs.vert_indices[1] && lhs.vert_indices[2] == rhs.vert_indices[2] && lhs.uv_indices[0] <  rhs.uv_indices[0]) ||
            (lhs.vert_indices[0] == rhs.vert_indices[0] && lhs.vert_indices[1] == rhs.vert_indices[1] && lhs.vert_indices[2] == rhs.vert_indices[2] && lhs.uv_indices[0] == rhs.uv_indices[0] && lhs.uv_indices[1] < rhs.uv_indices[1]);
    };

    for (const auto& shape : shapes)
    {
        // vector of unique vertex and uv look ups
        std::vector<Tri> tris;
        for (const auto& index : shape.mesh.indices)
        {
            // create temp tri, may be added if values do not
            // already exist in tris vector
            Tri tri = {
                {
                    3 * index.vertex_index,
                    3 * index.vertex_index + 1,
                    3 * index.vertex_index + 2
                },
                {
                    2 * index.texcoord_index,
                    2 * index.texcoord_index + 1
                },
                0
            };

            // find if the element exists. if range is 0 (first and second are the same)
            // element does not already exist, insert into sorted range, otherwise record
            // existing lookup into indices
            auto it_range = std::equal_range(tris.begin(), tris.end(), tri, triLessThan);
            if (it_range.first != it_range.second)
            {
                asMesh->indices.push_back(it_range.first->vertex_index);
            }
            else
            {
                // create new unique vertex data (combination of vertex and uvs)
                asMesh->vertices.push_back({
                    {
                        attrib.vertices[tri.vert_indices[0]],
                        attrib.vertices[tri.vert_indices[1]],
                        attrib.vertices[tri.vert_indices[2]]
                    },
                    {
                        1.0f, 1.0f, 1.0f
                    },
                    {
                        attrib.texcoords[tri.uv_indices[0]],
                        1.0f - attrib.texcoords[tri.uv_indices[1]]
                    }
                });

                // record index of newly added vertex data
                tri.vertex_index = static_cast<uint32_t>(asMesh->vertices.size() - 1);
                asMesh->indices.push_back(tri.vertex_index);

                // update unique triangle data (insert into sorted vector)
                tris.insert(std::upper_bound(tris.begin(), tris.end(), tri, triLessThan), tri);
            }
        }
    }
}

bool operator==(const AsVertex& lhs, const AsVertex& rhs)
{
    return lhs.position.x == rhs.position.x &&
        lhs.position.y == rhs.position.y &&
        lhs.position.z == rhs.position.z &&
        lhs.color.x == rhs.color.x &&
        lhs.color.y == rhs.color.y &&
        lhs.color.z == rhs.color.z &&
        lhs.uv.x == rhs.uv.x &&
        lhs.uv.y == rhs.uv.y;
}

// glm hash combine implementation (glm/glm/gtx/hash.inl)
void hash_combine(size_t &seed, size_t hash)
{
    hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash;
}

namespace std
{
    // hash implementations for various types
    template<>
    struct hash<as::vec3>
    {
        size_t operator()(const as::vec3& v) const
        {
            size_t seed = 0;
            std::hash<float> hasher;
            hash_combine(seed, hasher(v.x));
            hash_combine(seed, hasher(v.y));
            hash_combine(seed, hasher(v.z));
            return seed;
        }
    };

    template<>
    struct hash<as::vec2>
    {
        size_t operator()(const as::vec2& v) const
        {
            size_t seed = 0;
            std::hash<float> hasher;
            hash_combine(seed, hasher(v.x));
            hash_combine(seed, hasher(v.y));
            return seed;
        }
    };

    template<>
    struct hash<AsVertex>
    {
        size_t operator()(AsVertex const& vertex) const
        {
            return ((hash<as::vec3>()(vertex.position) ^
                    (hash<as::vec3>()(vertex.color) << 1)) >> 1) ^
                     (hash<as::vec2>()(vertex.uv) << 1);
        }
    };
}

void as_vulkan_create_mesh_data_unordered_map(
    AsMesh* mesh, const tinyobj::attrib_t& attrib, const std::vector<tinyobj::shape_t>& shapes)
{
    for (const auto& shape : shapes)
    {
        std::unordered_map<AsVertex, uint32_t> uniqueVertices{};
        for (const auto& index : shape.mesh.indices)
        {
            AsVertex vertex = {
                {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                },
                {
                    1.0f, 1.0f, 1.0f
                },
                {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                }
            };

            auto it = uniqueVertices.find(vertex);
            if (it != uniqueVertices.end())
            {
                mesh->indices.push_back(it->second);
            }
            else
            {
                auto pair = uniqueVertices.insert({
                    vertex, static_cast<uint32_t>(mesh->vertices.size())
                });

                mesh->vertices.push_back(vertex);
                mesh->indices.push_back(pair.first->second);
            }
        }
    }
}

void as_load_mesh(AsMesh* mesh, const char* meshPath)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    std::string warn;

    if (!tinyobj::LoadObj(
        &attrib, &shapes, &materials, &warn, &err, meshPath))
    {
        std::cerr << err.c_str();
        std::exit(EXIT_FAILURE);
    }

    as_vulkan_create_mesh_data_unordered_map(mesh, attrib, shapes);
}

// INFO
//
// #1
// in general, semaphores are used to synchronize queues (GPU) and fences
// are used to synchronize application (CPU)

// (*1)
// if a concurrent sharing mode is specified, we must provide indices of all
// queues that will have access to a buffer. If we want to define an exclusive
// sharing mode, we can still reference this buffer in different queues, but
// only in one at a time. If we want to use a buffer in a different queue (submit
// commands that reference this buffer to another queue), we need to specify buffer
// memory barrier that transitions buffer’s ownership from one queue to another
//
// (*2)
// normally, without this flag, we can’t rerecord the same command buffer multiple
// times. It must be reset first. And, what’s more, command buffers created from
// one pool may be reset only all at once. Specifying this flag allows us to reset
// command buffers individually, and (even better) it is done implicitly by calling
// the vkBeginCommandBuffer() function
