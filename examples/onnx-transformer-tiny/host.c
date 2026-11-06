#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <udoncl/device_memory.h>
#include <udoncl/export.h>
#include <udononnxruntime.h>

#define TRANSFORMER_SEQUENCE 4u
#define TRANSFORMER_HIDDEN 8u
#define TRANSFORMER_INPUT_COUNT 3u
#define TRANSFORMER_OUTPUT_COUNT 3u
#define TRANSFORMER_Q4_SURFACE_ELEMENTS 131072u

static cl_context context;
static cl_command_queue queue;
static udonclBufferHandle_t buffers[6];
static int buffer_count;
static udononnxTensor_t input_bindings[TRANSFORMER_INPUT_COUNT];
static udononnxTensor_t output_bindings[TRANSFORMER_OUTPUT_COUNT];
static udononnxSession_t session;
static int session_created;

static udononnxStatus_t release_resources(void) {
  int failed = 0;
  int index;
  if (session_created) {
    if (udononnxSessionDispose(&session) != UDONONNX_STATUS_SUCCESS)
      failed = 1;
    session_created = 0;
  }
  for (index = buffer_count - 1; index >= 0; --index)
    if (udonclDeviceBufferDestroy(&buffers[index]) != UDONCL_STATUS_SUCCESS)
      failed = 1;
  buffer_count = 0;
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
  status = udononnxSessionCompleteEvent(
      &session, event,
      wait_status == CL_SUCCESS ? UDONONNX_STATUS_SUCCESS
                                : UDONONNX_STATUS_EXECUTION_FAILURE);
  release_status = clReleaseEvent(event);
  return release_status == CL_SUCCESS ? status : UDONONNX_STATUS_OPENCL_ERROR;
}

static udononnxStatus_t create_tensor(cl_mem_flags flags, size_t elements,
                                      size_t allocation_elements,
                                      udonclDataType_t data_type, uint32_t rank,
                                      const int64_t *dimensions,
                                      udononnxTensor_t *tensor) {
  size_t element_bytes =
      data_type == UDONCL_DATA_INT32 ? sizeof(int32_t) : sizeof(float);
  size_t bytes = allocation_elements * element_bytes;
  udonclBufferView_t view;
  udonclBufferHandle_t *handle;
  udononnxStatus_t status;
  cl_int error = CL_SUCCESS;
  cl_mem memory;
  if (buffer_count >= 6)
    return UDONONNX_STATUS_INVALID_VALUE;
  handle = &buffers[buffer_count];
  memory = clCreateBuffer(context, flags, bytes, 0, &error);
  if (!memory || error != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  if (udonclDeviceBufferAdopt(context, memory, bytes, handle) !=
      UDONCL_STATUS_SUCCESS) {
    clReleaseMemObject(memory);
    return UDONONNX_STATUS_OPENCL_ERROR;
  }
  if (udonclDeviceBufferView(*handle, 0, elements, data_type, &view) !=
      UDONCL_STATUS_SUCCESS) {
    udonclDeviceBufferDestroy(handle);
    return UDONONNX_STATUS_INVALID_VALUE;
  }
  status = udononnxCreateTensor(view, rank, dimensions, tensor);
  if (status != UDONONNX_STATUS_SUCCESS) {
    udonclDeviceBufferDestroy(handle);
    return status;
  }
  ++buffer_count;
  return UDONONNX_STATUS_SUCCESS;
}

UDONCL_EXPORT udononnxStatus_t transformer_initialize(void) {
  const udononnxModel_t *model = 0;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_event completion = 0;
  cl_int error = CL_SUCCESS;
  udononnxTensor_t input_tensors[TRANSFORMER_INPUT_COUNT];
  udononnxTensor_t output_tensors[TRANSFORMER_OUTPUT_COUNT];
  udononnxStatus_t status = UDONONNX_STATUS_OPENCL_ERROR;
  size_t hidden_elements = TRANSFORMER_SEQUENCE * TRANSFORMER_HIDDEN;
  size_t index;
  if (session_created || context)
    return UDONONNX_STATUS_BUSY;
  if (udononnxGetModel(0, &model) != UDONONNX_STATUS_SUCCESS ||
      clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0) != CL_SUCCESS)
    return UDONONNX_STATUS_OPENCL_ERROR;
  context = clCreateContext(0, 1, &device, 0, 0, &error);
  if (!context || error != CL_SUCCESS)
    goto fail;
  queue = clCreateCommandQueue(context, device, 0, &error);
  if (!queue || error != CL_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_READ_ONLY, TRANSFORMER_SEQUENCE,
                         TRANSFORMER_SEQUENCE, UDONCL_DATA_INT32,
                         model->inputs[0].rank, model->inputs[0].dimensions,
                         &input_tensors[0]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_READ_WRITE, hidden_elements, hidden_elements,
                         UDONCL_DATA_FLOAT32, model->inputs[1].rank,
                         model->inputs[1].dimensions, &input_tensors[1]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_READ_WRITE, hidden_elements, hidden_elements,
                         UDONCL_DATA_FLOAT32, model->inputs[2].rank,
                         model->inputs[2].dimensions, &input_tensors[2]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_WRITE_ONLY, hidden_elements,
                         TRANSFORMER_Q4_SURFACE_ELEMENTS, UDONCL_DATA_FLOAT32,
                         model->outputs[0].rank, model->outputs[0].dimensions,
                         &output_tensors[0]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_WRITE_ONLY, hidden_elements, hidden_elements,
                         UDONCL_DATA_FLOAT32, model->outputs[1].rank,
                         model->outputs[1].dimensions, &output_tensors[1]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = create_tensor(CL_MEM_WRITE_ONLY, hidden_elements, hidden_elements,
                         UDONCL_DATA_FLOAT32, model->outputs[2].rank,
                         model->outputs[2].dimensions, &output_tensors[2]);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  status = udononnxCreateSession(&session, model, context, input_bindings,
                                 output_bindings);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  session_created = 1;
  status = udononnxEnqueueInitialize(&session, queue, 0, 0, &completion);
  status = complete(completion, status);
  if (status != UDONONNX_STATUS_SUCCESS)
    goto fail;
  for (index = 0; index < TRANSFORMER_INPUT_COUNT; ++index) {
    status = udononnxSessionBindInput(&session, index, input_tensors[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      goto fail;
  }
  for (index = 0; index < TRANSFORMER_OUTPUT_COUNT; ++index) {
    status = udononnxSessionBindOutput(&session, index, output_tensors[index]);
    if (status != UDONONNX_STATUS_SUCCESS)
      goto fail;
  }
  return UDONONNX_STATUS_SUCCESS;
fail:
  release_resources();
  return status;
}

UDONCL_EXPORT udononnxStatus_t transformer_run(void) {
  cl_event completion = 0;
  udononnxStatus_t status;
  if (!session_created)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  status = udononnxEnqueueRun(&session, queue, 0, 0, &completion);
  return complete(completion, status);
}

UDONCL_EXPORT udononnxStatus_t transformer_dispose(void) {
  if (!session_created)
    return UDONONNX_STATUS_NOT_INITIALIZED;
  return release_resources();
}
