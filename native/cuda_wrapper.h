#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <string>
#include <memory>

namespace cudajs
{

    class CudaContext
    {
    public:
        static CudaContext &getInstance();
        void init();
        int getDeviceCount() const;
        void setDevice(int device);
        int getCurrentDevice() const;
        void synchronize();
        std::string getDeviceInfo(int device) const;

    private:
        CudaContext() = default;
        bool initialized_ = false;
        int currentDevice_ = 0;
    };

    class GpuBuffer
    {
    public:
        GpuBuffer(size_t size);
        ~GpuBuffer();

        void *getPointer() const { return d_ptr_; }
        size_t getSize() const { return size_; }

        void upload(const void *hostData, size_t size);
        void download(void *hostData, size_t size) const;
        void memset(int value);

    private:
        void *d_ptr_ = nullptr;
        size_t size_ = 0;
    };

    class CudaKernel
    {
    public:
        CudaKernel(const std::string &source, const std::string &kernelName);
        ~CudaKernel();

        void launch(void **args, dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);

    private:
        CUmodule module_ = nullptr;
        CUfunction kernel_ = nullptr;
        std::string kernelName_;

        void compileKernel(const std::string &source);
    };

    class CudaStream
    {
    public:
        CudaStream();
        ~CudaStream();

        cudaStream_t getStream() const { return stream_; }
        void synchronize();

    private:
        cudaStream_t stream_ = nullptr;
    };

    class CudaEvent
    {
    public:
        CudaEvent();
        ~CudaEvent();

        void record(cudaStream_t stream = 0);
        void synchronize();
        float elapsedTime(const CudaEvent &start) const;

    private:
        cudaEvent_t event_ = nullptr;
    };

} // namespace cudajs

#endif // CUDA_WRAPPER_H