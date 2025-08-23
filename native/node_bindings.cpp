#include "cuda_wrapper.h"
#include <memory>
#include <napi.h>
#include <unordered_map>

using namespace Napi;
using namespace cudajs;

// Global storage for objects
static std::unordered_map<uint32_t, std::shared_ptr<GpuBuffer>> buffers;
static std::unordered_map<uint32_t, std::shared_ptr<CudaKernel>> kernels;
static std::unordered_map<uint32_t, std::shared_ptr<CudaStream>> streams;
static std::unordered_map<uint32_t, std::shared_ptr<CudaEvent>> events;
static uint32_t nextId = 1;

// Helper function to get next ID
uint32_t getNextId() { return nextId++; }

// CUDA Context functions
Value CudaInit(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    CudaContext::getInstance().init();
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value GetDeviceCount(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    return Number::New(env, CudaContext::getInstance().getDeviceCount());
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value SetDevice(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Device ID expected");
  }

  try {
    CudaContext::getInstance().setDevice(info[0].As<Number>().Int32Value());
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value GetCurrentDevice(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    return Number::New(env, CudaContext::getInstance().getCurrentDevice());
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value Synchronize(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    CudaContext::getInstance().synchronize();
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value GetDeviceInfo(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Device ID expected");
  }

  try {
    std::string info_str = CudaContext::getInstance().getDeviceInfo(
        info[0].As<Number>().Int32Value());
    return String::New(env, info_str);
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

// GpuBuffer functions
Value CreateBuffer(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Buffer size expected");
  }

  try {
    size_t size = info[0].As<Number>().Int64Value();
    uint32_t id = getNextId();
    buffers[id] = std::make_shared<GpuBuffer>(size);

    Object result = Object::New(env);
    result.Set("id", id);
    result.Set("size", size);
    return result;
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value UploadBuffer(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsTypedArray()) {
    throw TypeError::New(env, "Buffer ID and TypedArray expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = buffers.find(id);
    if (it == buffers.end()) {
      throw Error::New(env, "Invalid buffer ID");
    }

    TypedArrayOf<float> array = info[1].As<TypedArrayOf<float>>();
    it->second->upload(array.Data(), array.ByteLength());
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value DownloadBuffer(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Buffer ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = buffers.find(id);
    if (it == buffers.end()) {
      throw Error::New(env, "Invalid buffer ID");
    }

    size_t size = it->second->getSize();
    Float32Array result = Float32Array::New(env, size / sizeof(float));
    it->second->download(result.Data(), size);
    return result;
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value FreeBuffer(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Buffer ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    buffers.erase(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value MemsetBuffer(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    throw TypeError::New(env, "Buffer ID and value expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    int value = info[1].As<Number>().Int32Value();

    auto it = buffers.find(id);
    if (it == buffers.end()) {
      throw Error::New(env, "Invalid buffer ID");
    }

    it->second->memset(value);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

// Kernel functions
Value CreateKernel(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsString() || !info[1].IsString()) {
    throw TypeError::New(env, "Kernel source and name expected");
  }

  try {
    std::string source = info[0].As<String>().Utf8Value();
    std::string name = info[1].As<String>().Utf8Value();

    uint32_t id = getNextId();
    kernels[id] = std::make_shared<CudaKernel>(source, name);

    Object result = Object::New(env);
    result.Set("id", id);
    result.Set("name", name);
    return result;
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value LaunchKernel(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsArray() ||
      !info[2].IsArray() || !info[3].IsArray()) {
    throw TypeError::New(env,
                         "Kernel ID, args, grid dims, and block dims expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = kernels.find(id);
    if (it == kernels.end()) {
      throw Error::New(env, "Invalid kernel ID");
    }

    Array args = info[1].As<Array>();
    Array gridDims = info[2].As<Array>();
    Array blockDims = info[3].As<Array>();

    // Prepare kernel arguments
    std::vector<void *> argPtrs;
    std::vector<void *> devicePointers;
    std::vector<std::unique_ptr<int>> intArgs;
    std::vector<std::unique_ptr<float>> floatArgs;

    for (uint32_t i = 0; i < args.Length(); i++) {
      Value arg = args[i];
      if (arg.IsObject()) {
        Object obj = arg.As<Object>();
        if (obj.Has("id") && obj.Has("size")) {
          uint32_t bufferId = obj.Get("id").As<Number>().Uint32Value();
          auto bufIt = buffers.find(bufferId);
          if (bufIt != buffers.end()) {
            void *devicePtr = bufIt->second->getPointer();
            devicePointers.push_back(devicePtr);
            argPtrs.push_back(&devicePointers.back());

            // Debug output
            std::cout << "Buffer arg " << i << ": id=" << bufferId
                      << ", devicePtr=" << devicePtr << std::endl;
          } else {
            throw Error::New(
                env,
                ("Invalid buffer ID: " + std::to_string(bufferId)).c_str());
          }
        } else {
          throw Error::New(
              env, "Object argument missing 'id' and 'size' properties");
        }
      } else if (arg.IsNumber()) {
        Number num = arg.As<Number>();
        if (num.Int32Value() == num.DoubleValue()) {
          // Integer
          intArgs.push_back(std::make_unique<int>(num.Int32Value()));
          argPtrs.push_back(intArgs.back().get());

          std::cout << "Int arg " << i << ": " << num.Int32Value() << std::endl;
        } else {
          // Float
          floatArgs.push_back(std::make_unique<float>(num.FloatValue()));
          argPtrs.push_back(floatArgs.back().get());

          std::cout << "Float arg " << i << ": " << num.FloatValue()
                    << std::endl;
        }
      } else {
        throw Error::New(
            env, ("Unsupported argument type at index " + std::to_string(i))
                     .c_str());
      }
    }

    // Get dimensions
    dim3 grid(gridDims.Get((uint32_t)0).As<Number>().Uint32Value(),
              gridDims.Length() > 1
                  ? gridDims.Get((uint32_t)1).As<Number>().Uint32Value()
                  : 1,
              gridDims.Length() > 2
                  ? gridDims.Get((uint32_t)2).As<Number>().Uint32Value()
                  : 1);

    dim3 block(blockDims.Get((uint32_t)0).As<Number>().Uint32Value(),
               blockDims.Length() > 1
                   ? blockDims.Get((uint32_t)1).As<Number>().Uint32Value()
                   : 1,
               blockDims.Length() > 2
                   ? blockDims.Get((uint32_t)2).As<Number>().Uint32Value()
                   : 1);

    // Launch kernel
    it->second->launch(argPtrs.data(), grid, block);

    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value FreeKernel(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Kernel ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    kernels.erase(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

// Stream functions
Value CreateStream(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    uint32_t id = getNextId();
    streams[id] = std::make_shared<CudaStream>();
    return Number::New(env, id);
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value SynchronizeStream(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Stream ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = streams.find(id);
    if (it == streams.end()) {
      throw Error::New(env, "Invalid stream ID");
    }

    it->second->synchronize();
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value FreeStream(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Stream ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    streams.erase(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

// Event functions
Value CreateEvent(const CallbackInfo &info) {
  Env env = info.Env();
  try {
    uint32_t id = getNextId();
    events[id] = std::make_shared<CudaEvent>();
    return Number::New(env, id);
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value RecordEvent(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Event ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = events.find(id);
    if (it == events.end()) {
      throw Error::New(env, "Invalid event ID");
    }

    cudaStream_t stream = 0;
    if (info.Length() > 1 && info[1].IsNumber()) {
      uint32_t streamId = info[1].As<Number>().Uint32Value();
      auto streamIt = streams.find(streamId);
      if (streamIt != streams.end()) {
        stream = streamIt->second->getStream();
      }
    }

    it->second->record(stream);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value SynchronizeEvent(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Event ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    auto it = events.find(id);
    if (it == events.end()) {
      throw Error::New(env, "Invalid event ID");
    }

    it->second->synchronize();
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value ElapsedTime(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    throw TypeError::New(env, "Start and end event IDs expected");
  }

  try {
    uint32_t startId = info[0].As<Number>().Uint32Value();
    uint32_t endId = info[1].As<Number>().Uint32Value();

    auto startIt = events.find(startId);
    auto endIt = events.find(endId);

    if (startIt == events.end() || endIt == events.end()) {
      throw Error::New(env, "Invalid event ID");
    }

    float ms = endIt->second->elapsedTime(*startIt->second);
    return Number::New(env, ms);
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Value FreeEvent(const CallbackInfo &info) {
  Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    throw TypeError::New(env, "Event ID expected");
  }

  try {
    uint32_t id = info[0].As<Number>().Uint32Value();
    events.erase(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    throw Error::New(env, e.what());
  }
}

Object Init(Env env, Object exports) {
  // CUDA Context
  exports.Set("cudaInit", Function::New(env, CudaInit));
  exports.Set("getDeviceCount", Function::New(env, GetDeviceCount));
  exports.Set("setDevice", Function::New(env, SetDevice));
  exports.Set("getCurrentDevice", Function::New(env, GetCurrentDevice));
  exports.Set("synchronize", Function::New(env, Synchronize));
  exports.Set("getDeviceInfo", Function::New(env, GetDeviceInfo));

  // Buffer operations
  exports.Set("createBuffer", Function::New(env, CreateBuffer));
  exports.Set("uploadBuffer", Function::New(env, UploadBuffer));
  exports.Set("downloadBuffer", Function::New(env, DownloadBuffer));
  exports.Set("freeBuffer", Function::New(env, FreeBuffer));
  exports.Set("memsetBuffer", Function::New(env, MemsetBuffer));

  // Kernel operations
  exports.Set("createKernel", Function::New(env, CreateKernel));
  exports.Set("launchKernel", Function::New(env, LaunchKernel));
  exports.Set("freeKernel", Function::New(env, FreeKernel));

  // Stream operations
  exports.Set("createStream", Function::New(env, CreateStream));
  exports.Set("synchronizeStream", Function::New(env, SynchronizeStream));
  exports.Set("freeStream", Function::New(env, FreeStream));

  // Event operations
  exports.Set("createEvent", Function::New(env, CreateEvent));
  exports.Set("recordEvent", Function::New(env, RecordEvent));
  exports.Set("synchronizeEvent", Function::New(env, SynchronizeEvent));
  exports.Set("elapsedTime", Function::New(env, ElapsedTime));
  exports.Set("freeEvent", Function::New(env, FreeEvent));

  return exports;
}

NODE_API_MODULE(cudajs, Init)