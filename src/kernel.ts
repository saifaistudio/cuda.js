import bindings from './native';
import { KernelInfo, KernelArg, LaunchConfig } from './types';
import { GpuArray } from './gpu-array';
import { Cuda } from './cuda';

/**
 * CUDA kernel wrapper
 */
export class Kernel {
    private kernel: KernelInfo;
    private _source: string;

    /**
     * Create and compile a CUDA kernel
     * @param source - CUDA kernel source code
     * @param name - Kernel function name
     */
    constructor(source: string, name: string) {
        Cuda.init();
        this._source = source;
        this.kernel = bindings.createKernel(source, name);
    }

    /**
     * Get kernel source
     */
    get source(): string {
        return this._source;
    }

    /**
     * Get kernel name
     */
    get name(): string {
        return this.kernel.name;
    }

    /**
     * Launch the kernel
     * @param args - Kernel arguments (GpuArrays and scalars)
     * @param gridDim - Grid dimensions [x, y, z]
     * @param blockDim - Block dimensions [x, y, z]
     * @param sharedMem - Shared memory size in bytes (optional)
     */
    run(
        args: Array<GpuArray | number>,
        gridDim: number | number[],
        blockDim: number | number[],
        sharedMem: number = 0
    ): void {
        // Normalize dimensions
        const grid = typeof gridDim === 'number' ? [gridDim, 1, 1] :
            gridDim.length === 1 ? [...gridDim, 1, 1] :
                gridDim.length === 2 ? [...gridDim, 1] :
                    gridDim.slice(0, 3);

        const block = typeof blockDim === 'number' ? [blockDim, 1, 1] :
            blockDim.length === 1 ? [...blockDim, 1, 1] :
                blockDim.length === 2 ? [...blockDim, 1] :
                    blockDim.slice(0, 3);

        // Convert arguments
        const kernelArgs = args.map(arg => {
            if (arg instanceof GpuArray) {
                return (arg as any).buffer;
            } else {
                return arg;
            }
        });

        bindings.launchKernel(this.kernel.id, kernelArgs, grid, block);
    }

    /**
     * Launch kernel with config object
     */
    launch(args: Array<GpuArray | number>, config: LaunchConfig): void {
        this.run(args, config.gridDim, config.blockDim, config.sharedMem || 0);
    }

    /**
     * Calculate grid size for given problem size and block size
     */
    static gridSize(problemSize: number, blockSize: number): number {
        return Math.ceil(problemSize / blockSize);
    }

    /**
     * Get optimal block size for the kernel (heuristic)
     */
    static optimalBlockSize(problemSize: number): number {
        // Simple heuristic - can be improved with occupancy calculation
        if (problemSize < 32) return 32;
        if (problemSize < 128) return 128;
        if (problemSize < 256) return 256;
        return 512;
    }

    /**
     * Free kernel resources
     */
    free(): void {
        bindings.freeKernel(this.kernel.id);
    }
}