#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/device_memory.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

#define MNIST_INPUT_ELEMENTS 784u
#define MNIST_OUTPUT_ELEMENTS 10u

/* Unity imports the library under Assets/udoncl-generated/generic_onnx_mnist
 * and names its U# binding UdonclGenericOnnxMnistBinding. */
UDONCL_PROGRAM("generic_onnx_mnist");

/* The model input and output: U# writes the 28x28 image (CL_MEM_READ_ONLY)
 * with TryWriteImage before Run and reads the ten logits (CL_MEM_WRITE_ONLY)
 * with StartReadLogits after it. */
UDONCL_PUBLIC(float) cl_mem image, logits;

static cl_context context;
static cl_command_queue queue;
static udonclBufferHandle_t input_buffer;
static udonclBufferHandle_t output_buffer;
static udononnxTensor_t input_bindings[1];
static udononnxTensor_t output_bindings[1];
static udononnxLegacySession_t session;
static int session_created;
static int input_buffer_created;
static int output_buffer_created;

static udononnxStatus_t release_resources(void) {
  int failed = 0;
  if (session_created) {
    if (udononnxLegacySessionDispose(&session) != UDONONNX_STATUS_SUCCESS)
      failed = 1;
    session_created = 0;
  }
  if (input_buffer_created) {
    if (udonclDeviceBufferDestroy(&input_buffer) != UDONCL_STATUS_SUCCESS)
      failed = 1;
    input_buffer_created = 0;
  }
  if (output_buffer_created) {
    if (udonclDeviceBufferDestroy(&output_buffer) != UDONCL_STATUS_SUCCESS)
      failed = 1;
    output_buffer_created = 0;
  }
  if (queue) {
    if (clReleaseCommandQueue(queue) != CL_SUCCESS)
      failed = 1;
    queue = 0;
  }
  if (context) {
    if (clReleaseContext(context) != CL_SUCCESS)
      failed = 1;
    context = 0;
  }
  return failed ? UDONONNX_STATUS_OPENCL_ERROR : UDONONNX_STATUS_SUCCESS;
}

static udononnxStatus_t complete(cl_event event,
                                 udononnxStatus_t enqueue_status) {
  udononnxStatus_t status;
  cl_int wait_status;
  cl_int release_status;
  if (enqueue_status != UDONONNX_STATUS_SUCCESS)
    return enqueue_status;
  if (!event)
    return UDONONNX_STATUS_OPENCL_ERROR;
  wait_status = clWaitForEvents(1, &event);
  status = udononnxLegacySessionCompleteEvent(
      &session, event,
      wait_status == CL_SUCCESS ? UDONONNX_STATUS_SUCCESS
                                : UDONONNX_STATUS_EXECUTION_FAILURE);
  release_status = clReleaseEvent(event);
  return release_status == CL_SUCCESS ? status : UDONONNX_STATUS_OPENCL_ERROR;
}

/* Creates the Device buffer *memory of `elements` floats and its tensor. */
static udononnxStatus_t create_buffer(cl_mem_flags flags, size_t elements,
                                      cl_mem *memory,
                                      udonclBufferHandle_t *handle,
                                      udononnxTensor_t *tensor, uint32_t rank,
                                      const int64_t *dimensions) {
  const size_t bytes = elements * sizeof(float);
  udonclBufferView_t view;
  udononnxStatus_t status;
  cl_int error = CL_SUCCESS;
  *memory = clCreateBuffer(context, flags, bytes, 0, &error);
  if (!*memory || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (udonclDeviceBufferAdopt(context, *memory, bytes, handle) !=
      UDONCL_STATUS_SUCCESS) {
    clReleaseMemObject(*memory);
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  if (udonclDeviceBufferView(*handle, 0, elements, UDONCL_DATA_FLOAT32,
                             &view) != UDONCL_STATUS_SUCCESS) {
    udonclDeviceBufferDestroy(handle);
    return UDONONNX_STATUS_INVALID_VALUE;
  }
  status = udononnxCreateTensor(view, rank, dimensions, tensor);
  if (status != UDONONNX_STATUS_SUCCESS) {
    udonclDeviceBufferDestroy(handle);
    return status;
  }
  return status;
}

UDONCL_EXPORT udononnxStatus_t initialize(void) {
  const udononnxModel_t *model = 0;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_event completion = 0;
  cl_int error = CL_SUCCESS;
  udononnxTensor_t input_tensor;
  udononnxTensor_t output_tensor;
  udononnxStatus_t status;
  if (session_created)
    return UDONONNX_STATUS_BUSY;
  if (udononnxGetModel(0, &model) != UDONONNX_STATUS_SUCCESS ||
      clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  status = create_buffer(CL_MEM_READ_ONLY, MNIST_INPUT_ELEMENTS, &image,
                         &input_buffer, &input_tensor, model->inputs[0].rank,
                         model->inputs[0].dimensions);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  input_buffer_created = 1;
  status = create_buffer(CL_MEM_WRITE_ONLY, MNIST_OUTPUT_ELEMENTS, &logits,
                         &output_buffer, &output_tensor, model->outputs[0].rank,
                         model->outputs[0].dimensions);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  output_buffer_created = 1;
  status = udononnxLegacySessionCreate(&session, model, context, input_bindings,
                                       output_bindings);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  session_created = 1;
  status = udononnxLegacySessionEnqueueInitialize(&session, queue, 0, 0,
                                                  &completion);
  status = complete(completion, status);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxLegacySessionBindInput(&session, 0, input_tensor);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxLegacySessionBindOutput(&session, 0, output_tensor);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  return UDONONNX_STATUS_SUCCESS;
fail:
  release_resources();
  return status;
}

UDONCL_EXPORT udononnxStatus_t run(void) {
  cl_event completion = 0;
  udononnxStatus_t status;
  if (!session_created)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxLegacySessionEnqueueRun(&session, queue, 0, 0, &completion);
  return complete(completion, status);
}

UDONCL_EXPORT udononnxStatus_t shutdown(void) {
  if (!session_created)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return release_resources();
}
