# Cuda.JS

CUDA bindings for Node.js - bringing GPU computing to JavaScript.

[![npm version](https://badge.fury.io/js/cuda.js.svg)](https://www.npmjs.com/package/cuda.js)
[![Build Status](https://github.com/sammwyy/cuda.js/workflows/CI/badge.svg)](https://github.com/sammwyy/cuda.js/actions)

## 🚀 Features

- **PyCUDA-inspired API** - Familiar interface for CUDA developers
- **Runtime kernel compilation** - Compile CUDA C++ code at runtime using NVRTC  
- **High-level GPU arrays** - Easy data management with `GpuArray` class
- **Memory management** - Explicit control over GPU memory allocation
- **TypeScript support** - Full type definitions included
- **Cross-platform** - Works on Linux and Windows

## 📋 Requirements

- **CUDA Toolkit 11.0+** (12.0+ recommended)
- **Node.js 16+**
- **Python 3.7+** (for node-gyp)
- **Compatible C++ compiler**:
  - Linux: GCC 7+ or Clang 6+  
  - Windows: Visual Studio 2019+

## 🔧 Installation

### 1. Install CUDA Toolkit

Download and install from [NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit).

Make sure `nvcc` is in your PATH:
```bash
nvcc --version
```

### 2. Install Cuda.JS

```bash
npm install cuda.js
```

**Note**: First installation may take several minutes as it compiles native code.

## 🏃 Quick Start

```javascript
import { Cuda, GpuArray, Kernel } from 'cuda.js';

// Initialize CUDA
Cuda.init();
console.log(`Found ${Cuda.getDeviceCount()} CUDA devices`);
console.log(Cuda.getDeviceInfo(0));

// Create GPU arrays
const a = new GpuArray([1, 2, 3, 4, 5]);
const b = new GpuArray([5, 4, 3, 2, 1]);
const c = new GpuArray(5);

// Compile and run kernel
const kernel = new Kernel(`
extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}`, 'vector_add');

kernel.run([a, b, c, 5], [1, 1, 1], [256, 1, 1]);

// Get results
const result = c.download();
console.log('Result:', result); // [6, 6, 6, 6, 6]

// Cleanup
a.free();
b.free();
c.free();
kernel.free();
```

## 🔥 Examples

### Basic Example
```bash
npm run example:basic
```

## 🛠️ Development

### Building from Source

```bash
git clone https://github.com/sammwyy/cuda.js.git
cd cuda.js
npm install
npm run build
npm test
```

### Project Structure

```
cuda.js/
├── native/             # Native C bindings
├── src/                # JavaScript bindings
├── test/               # Test suite
└── lib/                # Compiled output
```

## 🧪 Testing

```bash
# Run all tests
npm test
```