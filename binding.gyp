{
  "targets": [{
    "target_name": "cudajs",
    "sources": [
      "native/cuda_wrapper.cpp",
      "native/node_bindings.cpp"
    ],
    "include_dirs": [
      "node_modules/node-addon-api",
      "native",
      "/usr/include/node",
      "/usr/local/cuda/include"
    ],
    "library_dirs": [
      "/usr/local/cuda/lib64"
    ],
    "libraries": [
      "-lcudart"
    ],
    "cflags!": ["-fno-exceptions"],
    "cflags_cc!": ["-fno-exceptions"],
    "defines": [
      "NAPI_CPP_EXCEPTIONS",
      "NAPI_VERSION=6"
    ],
    "cflags_cc": ["-std=c++17"],
    "xcode_settings": {
      "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
      "CLANG_CXX_LIBRARY": "libc++",
      "MACOSX_DEPLOYMENT_TARGET": "10.15"
    }
  }]
}
