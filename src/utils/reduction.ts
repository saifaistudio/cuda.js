import { GpuArray } from '../gpu-array';
import { Kernel } from '../kernel';

/**
 * Reduction operations on GPU arrays
 */
export const reduction = {
    sum: (array: GpuArray): number => {
        const kernel = new Kernel(`
extern "C" __global__ void sum_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}`, 'sum_reduction');

        const output = new GpuArray(1);
        output.zero();

        const blockSize = 256;
        const gridSize = Math.ceil(array.size / blockSize);
        const sharedMem = blockSize * 4; // sizeof(float) * blockSize

        kernel.run([array, output, array.size], gridSize, blockSize, sharedMem);

        const result = output.download()[0];
        output.free();
        kernel.free();

        return result;
    },

    max: (array: GpuArray): number => {
        const kernel = new Kernel(`
extern "C" __global__ void max_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(sdata[0]));
    }
}`, 'max_reduction');

        const output = new GpuArray(1);
        output.fill(-Infinity);

        const blockSize = 256;
        const gridSize = Math.ceil(array.size / blockSize);
        const sharedMem = blockSize * 4;

        kernel.run([array, output, array.size], gridSize, blockSize, sharedMem);

        const result = output.download()[0];
        output.free();
        kernel.free();

        return result;
    },

    min: (array: GpuArray): number => {
        const kernel = new Kernel(`
extern "C" __global__ void min_reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : INFINITY;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin((int*)output, __float_as_int(sdata[0]));
    }
}`, 'min_reduction');

        const output = new GpuArray(1);
        output.fill(Infinity);

        const blockSize = 256;
        const gridSize = Math.ceil(array.size / blockSize);
        const sharedMem = blockSize * 4;

        kernel.run([array, output, array.size], gridSize, blockSize, sharedMem);

        const result = output.download()[0];
        output.free();
        kernel.free();

        return result;
    },

    mean: (array: GpuArray): number => {
        return reduction.sum(array) / array.size;
    }
};