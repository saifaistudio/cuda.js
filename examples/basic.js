const { Cuda, GpuArray, Kernel } = require('../lib');

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