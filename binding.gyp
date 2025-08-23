{
  "targets": [
    {
      "target_name": "cudajs",
      "sources": [
        "native/cuda_wrapper.cpp",
        "native/node_bindings.cpp"
      ],
      "include_dirs": [
        "node_modules/node-addon-api",
        "native",
        "<!(node -p \"require('node-addon-api').include\")"
      ],
      "cflags_cc": ["-std=c++17"],
      "defines": [
        "NAPI_CPP_EXCEPTIONS",
        "NAPI_VERSION=6"
      ],
      "conditions": [
        # Linux
        [ "OS=='linux'", {
          "include_dirs": [
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
          "cflags_cc!": ["-fno-exceptions"]
        }],
        
        # macOS
        [ "OS=='mac'", {
          "include_dirs": [
            "/usr/local/cuda/include"
          ],
          "library_dirs": [
            "/usr/local/cuda/lib"
          ],
          "libraries": [
            "-lcudart"
          ],
          "xcode_settings": {
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.15"
          }
        }],
        
        # Windows
        [ "OS=='win'", {
          "include_dirs": [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include",
            "C:/Users/sam/AppData/Local/node-gyp/Cache/20.12.2/include/node"
          ],
          "library_dirs": [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"
          ],
          "libraries": [
            "cudart.lib",   # CUDA Runtime API
            "cuda.lib",     # CUDA Driver API (cuInit, cuLaunchKernel, etc.)
            "nvrtc.lib"     # NVRTC (nvrtcCreateProgram, nvrtcCompileProgram, etc.)
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/std:c++17"]
            }
          }
        }]
      ]
    }
  ]
}