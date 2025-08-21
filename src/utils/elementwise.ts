import { GpuArray } from '../gpu-array';
import { Kernel } from '../kernel';

/**
 * Elementwise operations on GPU arrays
 */
export class ElementwiseKernel {
    private kernel: Kernel;
    private operation: string;

    constructor(operation: string, functionName: string = 'elementwise_op') {
        this.operation = operation;
        const source = `
extern "C" __global__ void ${functionName}(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ${operation};
    }
}`;
        this.kernel = new Kernel(source, functionName);
    }

    apply(a: GpuArray, b: GpuArray | number, out?: GpuArray): GpuArray {
        if (!out) {
            out = new GpuArray(a.size);
        }

        if (b instanceof GpuArray) {
            if (a.size !== b.size) {
                throw new Error('Array size mismatch');
            }
            const blockSize = 256;
            const gridSize = Math.ceil(a.size / blockSize);
            this.kernel.run([a, b, out, a.size], gridSize, blockSize);
        } else {
            // Handle scalar case
            const scalarArray = new GpuArray(a.size);
            scalarArray.fill(b);
            const blockSize = 256;
            const gridSize = Math.ceil(a.size / blockSize);
            this.kernel.run([a, scalarArray, out, a.size], gridSize, blockSize);
            scalarArray.free();
        }

        return out;
    }

    free(): void {
        this.kernel.free();
    }
}

// Pre-defined elementwise operations
export const elementwise = {
    add: (a: GpuArray, b: GpuArray | number, out?: GpuArray): GpuArray => {
        const kernel = new ElementwiseKernel('a[i] + b[i]', 'add_kernel');
        const result = kernel.apply(a, b, out);
        kernel.free();
        return result;
    },

    subtract: (a: GpuArray, b: GpuArray | number, out?: GpuArray): GpuArray => {
        const kernel = new ElementwiseKernel('a[i] - b[i]', 'sub_kernel');
        const result = kernel.apply(a, b, out);
        kernel.free();
        return result;
    },

    multiply: (a: GpuArray, b: GpuArray | number, out?: GpuArray): GpuArray => {
        const kernel = new ElementwiseKernel('a[i] * b[i]', 'mul_kernel');
        const result = kernel.apply(a, b, out);
        kernel.free();
        return result;
    },

    divide: (a: GpuArray, b: GpuArray | number, out?: GpuArray): GpuArray => {
        const kernel = new ElementwiseKernel('a[i] / b[i]', 'div_kernel');
        const result = kernel.apply(a, b, out);
        kernel.free();
        return result;
    },

    power: (a: GpuArray, b: number, out?: GpuArray): GpuArray => {
        const kernel = new ElementwiseKernel(`powf(a[i], ${b})`, 'pow_kernel');
        const result = kernel.apply(a, b, out);
        kernel.free();
        return result;
    },

    exp: (a: GpuArray, out?: GpuArray): GpuArray => {
        const kernel = new Kernel(`
extern "C" __global__ void exp_kernel(float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(a[i]);
    }
}`, 'exp_kernel');

        if (!out) {
            out = new GpuArray(a.size);
        }

        const blockSize = 256;
        const gridSize = Math.ceil(a.size / blockSize);
        kernel.run([a, out, a.size], gridSize, blockSize);
        kernel.free();

        return out;
    },

    log: (a: GpuArray, out?: GpuArray): GpuArray => {
        const kernel = new Kernel(`
extern "C" __global__ void log_kernel(float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = logf(a[i]);
    }
}`, 'log_kernel');

        if (!out) {
            out = new GpuArray(a.size);
        }

        const blockSize = 256;
        const gridSize = Math.ceil(a.size / blockSize);
        kernel.run([a, out, a.size], gridSize, blockSize);
        kernel.free();

        return out;
    },

    sqrt: (a: GpuArray, out?: GpuArray): GpuArray => {
        const kernel = new Kernel(`
extern "C" __global__ void sqrt_kernel(float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sqrtf(a[i]);
    }
}`, 'sqrt_kernel');

        if (!out) {
            out = new GpuArray(a.size);
        }

        const blockSize = 256;
        const gridSize = Math.ceil(a.size / blockSize);
        kernel.run([a, out, a.size], gridSize, blockSize);
        kernel.free();

        return out;
    }
};