#include "cuda_wrapper.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace cudajs {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "           \
         << cudaGetErrorString(error);                                         \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

#define NVRTC_CHECK(call)                                                      \
  do {                                                                         \
    nvrtcResult result = call;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      std::stringstream ss;                                                    \
      ss << "NVRTC error at " << __FILE__ << ":" << __LINE__ << " - "          \
         << nvrtcGetErrorString(result);                                       \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char *errorStr;                                                    \
      cuGetErrorString(result, &errorStr);                                     \
      std::stringstream ss;                                                    \
      ss << "CUDA Driver error at " << __FILE__ << ":" << __LINE__ << " - "    \
         << errorStr;                                                          \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  } while (0)

// CudaContext implementation
CudaContext &CudaContext::getInstance() {
  static CudaContext instance;
  return instance;
}

void CudaContext::init() {
  if (!initialized_) {
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuInit(0));

    CUcontext context;
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, 0));
    CU_CHECK(cuCtxSetCurrent(context));

    initialized_ = true;
  }
}

int CudaContext::getDeviceCount() const {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

void CudaContext::setDevice(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  currentDevice_ = device;
}

int CudaContext::getCurrentDevice() const {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void CudaContext::synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

std::string CudaContext::getDeviceInfo(int device) const {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::stringstream ss;
  ss << "Device " << device << ": " << prop.name << "\n";
  ss << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
  ss << "  Total Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
  ss << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
  ss << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
  ss << "  Max Grid Size: [" << prop.maxGridSize[0] << ", "
     << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
  ss << "  Max Block Size: [" << prop.maxThreadsDim[0] << ", "
     << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]";

  return ss.str();
}

// GpuBuffer implementation
GpuBuffer::GpuBuffer(size_t size) : size_(size) {
  CUDA_CHECK(cudaMalloc(&d_ptr_, size));
}

GpuBuffer::~GpuBuffer() {
  if (d_ptr_) {
    cudaFree(d_ptr_);
  }
}

void GpuBuffer::upload(const void *hostData, size_t size) {
  if (size > size_) {
    throw std::runtime_error("Upload size exceeds buffer size");
  }
  CUDA_CHECK(cudaMemcpy(d_ptr_, hostData, size, cudaMemcpyHostToDevice));
}

void GpuBuffer::download(void *hostData, size_t size) const {
  if (size > size_) {
    throw std::runtime_error("Download size exceeds buffer size");
  }
  CUDA_CHECK(cudaMemcpy(hostData, d_ptr_, size, cudaMemcpyDeviceToHost));
}

void GpuBuffer::memset(int value) {
  CUDA_CHECK(cudaMemset(d_ptr_, value, size_));
}

// CudaKernel implementation
CudaKernel::CudaKernel(const std::string &source, const std::string &kernelName)
    : kernelName_(kernelName) {
  compileKernel(source);
}

CudaKernel::~CudaKernel() {
  if (module_) {
    cuModuleUnload(module_);
  }
}

void CudaKernel::compileKernel(const std::string &source) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), "kernel.cu", 0, nullptr,
                                 nullptr));

  // Usar compute capability más moderna y opciones optimizadas
  const char *opts[] = {"--gpu-architecture=compute_60", "--use_fast_math",
                        "--extra-device-vectorization"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 3, opts);

  size_t logSize;
  nvrtcGetProgramLogSize(prog, &logSize);
  if (logSize > 1) {
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(prog, log.data());
    if (compileResult != NVRTC_SUCCESS) {
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error(std::string("Kernel compilation failed: ") +
                               log.data());
    }
  }

  size_t ptxSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

  nvrtcDestroyProgram(&prog);

  CU_CHECK(cuModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(&kernel_, module_, kernelName_.c_str()));
}

void CudaKernel::launch(void **args, dim3 gridDim, dim3 blockDim,
                        size_t sharedMem, cudaStream_t stream) {
  CU_CHECK(cuLaunchKernel(kernel_, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
                          blockDim.y, blockDim.z, sharedMem, stream, args,
                          nullptr));

  CUDA_CHECK(cudaDeviceSynchronize());
}

// CudaStream implementation
CudaStream::CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

CudaStream::~CudaStream() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void CudaStream::synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

// CudaEvent implementation
CudaEvent::CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

CudaEvent::~CudaEvent() {
  if (event_) {
    cudaEventDestroy(event_);
  }
}

void CudaEvent::record(cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaEvent::synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

float CudaEvent::elapsedTime(const CudaEvent &start) const {
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
  return ms;
}
} // namespace cudajs