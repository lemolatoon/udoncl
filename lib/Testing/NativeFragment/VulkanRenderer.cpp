#include "Renderer.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
namespace {
void check(VkResult value) {
  if (value != VK_SUCCESS)
    throw std::runtime_error("Vulkan operation failed: " +
                             std::to_string(value));
}
template <typename T> T info(VkStructureType type) {
  T value{};
  value.sType = type;
  return value;
}
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};
struct State {
  VkInstance instance = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory imageMemory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule vertex = VK_NULL_HANDLE, fragment = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  Buffer uniform, readback, upload;
  std::vector<VkImage> inputs;
  std::vector<VkDeviceMemory> inputMemories;
  std::vector<VkImageView> inputViews;
  ~State() {
    if (device) {
      vkDeviceWaitIdle(device);
      vkDestroyFence(device, fence, nullptr);
      vkDestroyCommandPool(device, commandPool, nullptr);
      vkDestroyPipeline(device, pipeline, nullptr);
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
      vkDestroyFramebuffer(device, framebuffer, nullptr);
      vkDestroyRenderPass(device, renderPass, nullptr);
      vkDestroyImageView(device, view, nullptr);
      vkDestroyImage(device, image, nullptr);
      vkFreeMemory(device, imageMemory, nullptr);
      for (auto view : inputViews)
        vkDestroyImageView(device, view, nullptr);
      for (auto image : inputs)
        vkDestroyImage(device, image, nullptr);
      for (auto memory : inputMemories)
        vkFreeMemory(device, memory, nullptr);
      for (auto resource : {uniform, readback, upload}) {
        vkDestroyBuffer(device, resource.buffer, nullptr);
        vkFreeMemory(device, resource.memory, nullptr);
      }
      vkDestroyShaderModule(device, vertex, nullptr);
      vkDestroyShaderModule(device, fragment, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }
};
class Renderer final : public FragmentRenderer {
  State state;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t family = 0;
  VkDescriptorSet descriptor = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  std::string name;
  uint32_t width = 1, height = 1, inputCount = 0;
  bool gather = false;
  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
      if ((bits & (1u << i)) &&
          (properties.memoryTypes[i].propertyFlags & flags) == flags)
        return i;
    throw std::runtime_error("required Vulkan memory type unavailable");
  }
  void makeBuffer(Buffer &buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    auto description =
        info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    description.size = size;
    description.usage = usage;
    check(vkCreateBuffer(state.device, &description, nullptr, &buffer.buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(state.device, buffer.buffer, &requirements);
    auto allocation =
        info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(state.device, &allocation, nullptr, &buffer.memory));
    check(vkBindBufferMemory(state.device, buffer.buffer, buffer.memory, 0));
  }
  VkShaderModule loadShader(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
      throw std::runtime_error("cannot read SPIR-V shader");
    auto size = input.tellg();
    if (size <= 0 || size % 4 != 0)
      throw std::runtime_error("invalid SPIR-V size");
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(words.data()), size);
    if (!input)
      throw std::runtime_error("cannot read complete SPIR-V shader");
    auto description = info<VkShaderModuleCreateInfo>(
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    description.codeSize = static_cast<size_t>(size);
    description.pCode = words.data();
    VkShaderModule module;
    check(vkCreateShaderModule(state.device, &description, nullptr, &module));
    return module;
  }

public:
  Renderer(const std::string &fragment, const std::string &vertex,
           uint32_t x = 0, uint32_t y = 1, uint32_t inputsRequested = 1)
      : width(x ? x : 1), height(y), inputCount(x ? inputsRequested : 0),
        gather(x != 0) {
    if (!height || uint64_t(width) * height > 16384 || inputCount > 13 ||
        (gather && !inputCount))
      throw std::runtime_error("gather extent exceeds supported profile");
    auto application =
        info<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "udoncl native boundary test";
    application.apiVersion = VK_API_VERSION_1_0;
    auto instanceInfo =
        info<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    instanceInfo.pApplicationInfo = &application;
    check(vkCreateInstance(&instanceInfo, nullptr, &state.instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(state.instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(state.instance, &count, devices.data()));
    for (auto candidate : devices) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(candidate, &properties);
      if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        physical = candidate;
        name = properties.deviceName;
        break;
      }
    }
    if (!physical)
      throw std::runtime_error(
          "software Vulkan device unavailable; select Lavapipe ICD");
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    bool found = false;
    for (uint32_t i = 0; i < count; ++i)
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        family = i;
        found = true;
        break;
      }
    if (!found)
      throw std::runtime_error("Vulkan graphics queue unavailable");
    float priority = 1;
    auto queueInfo = info<VkDeviceQueueCreateInfo>(
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    auto deviceInfo =
        info<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    check(vkCreateDevice(physical, &deviceInfo, nullptr, &state.device));
    vkGetDeviceQueue(state.device, family, 0, &queue);
    makeBuffer(state.uniform, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    makeBuffer(state.readback, width * height * 4,
               VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (gather)
      makeBuffer(state.upload, width * height * 4 * inputCount,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto imageInfo =
        info<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    check(vkCreateImage(state.device, &imageInfo, nullptr, &state.image));
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(state.device, state.image, &requirements);
    auto allocation =
        info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType(requirements.memoryTypeBits, 0);
    check(vkAllocateMemory(state.device, &allocation, nullptr,
                           &state.imageMemory));
    check(vkBindImageMemory(state.device, state.image, state.imageMemory, 0));
    auto viewInfo =
        info<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewInfo.image = state.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(state.device, &viewInfo, nullptr, &state.view));
    if (gather) {
      imageInfo.usage =
          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      state.inputs.resize(inputCount);
      state.inputMemories.resize(inputCount);
      state.inputViews.resize(inputCount);
      for (uint32_t index = 0; index < inputCount; ++index) {
        check(vkCreateImage(state.device, &imageInfo, nullptr,
                            &state.inputs[index]));
        vkGetImageMemoryRequirements(state.device, state.inputs[index],
                                     &requirements);
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType(requirements.memoryTypeBits, 0);
        check(vkAllocateMemory(state.device, &allocation, nullptr,
                               &state.inputMemories[index]));
        check(vkBindImageMemory(state.device, state.inputs[index],
                                state.inputMemories[index], 0));
        viewInfo.image = state.inputs[index];
        check(vkCreateImageView(state.device, &viewInfo, nullptr,
                                &state.inputViews[index]));
      }
    }
    VkAttachmentDescription attachment{};
    attachment.format = imageInfo.format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference reference{0,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    auto renderInfo =
        info<VkRenderPassCreateInfo>(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO);
    renderInfo.attachmentCount = 1;
    renderInfo.pAttachments = &attachment;
    renderInfo.subpassCount = 1;
    renderInfo.pSubpasses = &subpass;
    renderInfo.dependencyCount = 1;
    renderInfo.pDependencies = &dependency;
    check(vkCreateRenderPass(state.device, &renderInfo, nullptr,
                             &state.renderPass));
    auto framebufferInfo = info<VkFramebufferCreateInfo>(
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO);
    framebufferInfo.renderPass = state.renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &state.view;
    framebufferInfo.width = width;
    framebufferInfo.height = height;
    framebufferInfo.layers = 1;
    check(vkCreateFramebuffer(state.device, &framebufferInfo, nullptr,
                              &state.framebuffer));
    VkDescriptorSetLayoutBinding binding{gather ? 13u : 0u,
                                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    auto descriptorInfo = info<VkDescriptorSetLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    std::vector<VkDescriptorSetLayoutBinding> bindings{binding};
    for (uint32_t index = 0; index < inputCount; ++index)
      bindings.push_back({index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                          VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    descriptorInfo.bindingCount = uint32_t(bindings.size());
    descriptorInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(state.device, &descriptorInfo, nullptr,
                                      &state.descriptorLayout));
    VkDescriptorPoolSize poolSizes[2]{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, inputCount}};
    auto poolInfo = info<VkDescriptorPoolCreateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = gather ? 2 : 1;
    poolInfo.pPoolSizes = poolSizes;
    check(vkCreateDescriptorPool(state.device, &poolInfo, nullptr,
                                 &state.descriptorPool));
    auto setInfo = info<VkDescriptorSetAllocateInfo>(
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    setInfo.descriptorPool = state.descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &state.descriptorLayout;
    check(vkAllocateDescriptorSets(state.device, &setInfo, &descriptor));
    VkDescriptorBufferInfo bufferInfo{state.uniform.buffer, 0, 64};
    auto write =
        info<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
    write.dstSet = descriptor;
    write.dstBinding = gather ? 13 : 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(state.device, 1, &write, 0, nullptr);
    if (gather) {
      for (uint32_t index = 0; index < inputCount; ++index) {
        VkDescriptorImageInfo sampled{};
        sampled.imageView = state.inputViews[index];
        sampled.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        write.dstBinding = index;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pBufferInfo = nullptr;
        write.pImageInfo = &sampled;
        vkUpdateDescriptorSets(state.device, 1, &write, 0, nullptr);
      }
    }
    auto layoutInfo = info<VkPipelineLayoutCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &state.descriptorLayout;
    check(vkCreatePipelineLayout(state.device, &layoutInfo, nullptr,
                                 &state.pipelineLayout));
    state.vertex = loadShader(vertex);
    state.fragment = loadShader(fragment);
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (auto &stage : stages)
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = state.vertex;
    stages[0].pName = "vert";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = state.fragment;
    stages[1].pName = "frag";
    auto vertexInput = info<VkPipelineVertexInputStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
    auto assembly = info<VkPipelineInputAssemblyStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0, 0, float(width), float(height), 0, 1};
    VkRect2D scissor{{0, 0}, {width, height}};
    auto viewportInfo = info<VkPipelineViewportStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
    viewportInfo.viewportCount = viewportInfo.scissorCount = 1;
    viewportInfo.pViewports = &viewport;
    viewportInfo.pScissors = &scissor;
    auto raster = info<VkPipelineRasterizationStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1;
    auto multisample = info<VkPipelineMultisampleStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    auto blending = info<VkPipelineColorBlendStateCreateInfo>(
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    auto pipelineInfo = info<VkGraphicsPipelineCreateInfo>(
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewportInfo;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.layout = state.pipelineLayout;
    pipelineInfo.renderPass = state.renderPass;
    check(vkCreateGraphicsPipelines(state.device, VK_NULL_HANDLE, 1,
                                    &pipelineInfo, nullptr, &state.pipeline));
    auto commandPoolInfo = info<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = family;
    check(vkCreateCommandPool(state.device, &commandPoolInfo, nullptr,
                              &state.commandPool));
    auto commandInfo = info<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    commandInfo.commandPool = state.commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(state.device, &commandInfo, &command));
    auto fenceInfo =
        info<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    check(vkCreateFence(state.device, &fenceInfo, nullptr, &state.fence));
  }
  std::string deviceName() const override {
    return "Vulkan " + name + " (software)";
  }
  std::array<uint8_t, 4> draw(float value) override {
    auto bytes = drawWords(
        {},
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&value), 4));
    std::array<uint8_t, 4> result;
    std::memcpy(result.data(), bytes.data(), 4);
    return result;
  }
  std::vector<uint8_t> drawWords(std::span<const uint8_t> input,
                                 std::span<const uint8_t> scalars) override {
    size_t inputBytes = size_t(width) * height * 4;
    if (scalars.size() > 64 ||
        (gather ? input.size() != inputBytes * inputCount : !input.empty()))
      throw std::runtime_error("invalid fragment input/payload size");
    void *mapped;
    check(vkMapMemory(state.device, state.uniform.memory, 0, 64, 0, &mapped));
    std::memset(mapped, 0, 64);
    if (!scalars.empty())
      std::memcpy(mapped, scalars.data(), scalars.size());
    vkUnmapMemory(state.device, state.uniform.memory);
    if (gather) {
      check(vkMapMemory(state.device, state.upload.memory, 0, input.size(), 0,
                        &mapped));
      std::memcpy(mapped, input.data(), input.size());
      vkUnmapMemory(state.device, state.upload.memory);
    }
    check(vkResetCommandBuffer(command, 0));
    auto begin = info<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin));
    if (gather) {
      for (uint32_t index = 0; index < inputCount; ++index) {
        auto barrier =
            info<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        barrier.image = state.inputs[index];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &barrier);
        VkBufferImageCopy upload{};
        upload.bufferOffset = inputBytes * index;
        upload.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        upload.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(
            command, state.upload.buffer, state.inputs[index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
      }
    }
    VkClearValue clear{};
    clear.color.float32[0] = clear.color.float32[2] = clear.color.float32[3] =
        1;
    auto render =
        info<VkRenderPassBeginInfo>(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO);
    render.renderPass = state.renderPass;
    render.framebuffer = state.framebuffer;
    render.renderArea.extent = {width, height};
    render.clearValueCount = 1;
    render.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &render, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            state.pipelineLayout, 0, 1, &descriptor, 0,
                            nullptr);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    auto imageBarrier =
        info<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = state.image;
    imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imageBarrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command, state.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           state.readback.buffer, 1, &copy);
    auto bufferBarrier =
        info<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = bufferBarrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = state.readback.buffer;
    bufferBarrier.size = width * height * 4;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &bufferBarrier, 0, nullptr);
    check(vkEndCommandBuffer(command));
    check(vkResetFences(state.device, 1, &state.fence));
    auto submit = info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, state.fence));
    check(vkWaitForFences(state.device, 1, &state.fence, VK_TRUE,
                          30000000000ull));
    check(vkMapMemory(state.device, state.readback.memory, 0,
                      width * height * 4, 0, &mapped));
    std::vector<uint8_t> result(width * height * 4);
    std::memcpy(result.data(), mapped, result.size());
    vkUnmapMemory(state.device, state.readback.memory);
    return result;
  }
};
} // namespace
std::unique_ptr<FragmentRenderer>
createFragmentRenderer(const std::string &shader, const std::string &vertex) {
  return std::make_unique<Renderer>(shader, vertex);
}

std::unique_ptr<FragmentRenderer>
createGatherRenderer(const std::string &shader, const std::string &vertex,
                     uint32_t width, uint32_t height, uint32_t inputCount) {
  if (!width || !height || uint64_t(width) * height < 4 || !inputCount)
    throw std::runtime_error("invalid gather extent");
  return std::make_unique<Renderer>(shader, vertex, width, height, inputCount);
}
