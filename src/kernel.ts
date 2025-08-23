import { Cuda } from "./cuda";
import { GpuArray } from "./gpu-array";
import bindings from "./native";
import { KernelInfo, LaunchConfig } from "./types";

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
    const grid =
      typeof gridDim === "number"
        ? [gridDim, 1, 1]
        : gridDim.length === 1
        ? [...gridDim, 1, 1]
        : gridDim.length === 2
        ? [...gridDim, 1]
        : gridDim.slice(0, 3);

    const block =
      typeof blockDim === "number"
        ? [blockDim, 1, 1]
        : blockDim.length === 1
        ? [...blockDim, 1, 1]
        : blockDim.length === 2
        ? [...blockDim, 1]
        : blockDim.slice(0, 3);

    // Validate block dimensions against device limits
    const deviceProps = Cuda.getDeviceProperties();
    if (
      block[0] > deviceProps.maxBlockSize[0] ||
      block[1] > deviceProps.maxBlockSize[1] ||
      block[2] > deviceProps.maxBlockSize[2]
    ) {
      throw new Error(
        `Block dimensions ${block} exceed device limits ${deviceProps.maxBlockSize}`
      );
    }

    const totalThreadsPerBlock = block[0] * block[1] * block[2];
    if (totalThreadsPerBlock > deviceProps.maxThreadsPerBlock) {
      throw new Error(
        `Total threads per block (${totalThreadsPerBlock}) exceeds device limit (${deviceProps.maxThreadsPerBlock})`
      );
    }

    // Convert arguments
    const kernelArgs = args.map((arg) => {
      if (arg instanceof GpuArray) {
        return (arg as any).buffer;
      } else if (typeof arg === "number") {
        return arg;
      } else {
        throw new Error(`Invalid kernel argument type: ${typeof arg}`);
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
    const warpSize = 32;
    let blockSize: number;

    if (problemSize < 32) {
      blockSize = 32;
    } else if (problemSize < 128) {
      blockSize = 128;
    } else if (problemSize < 256) {
      blockSize = 256;
    } else {
      blockSize = 512;
    }

    return Math.ceil(blockSize / warpSize) * warpSize;
  }

  /**
   * Auto-configure launch parameters for 1D problem
   */
  static autoConfig(problemSize: number): {
    gridDim: number[];
    blockDim: number[];
  } {
    const blockSize = this.optimalBlockSize(problemSize);
    const gridSize = this.gridSize(problemSize, blockSize);

    return {
      gridDim: [gridSize, 1, 1],
      blockDim: [blockSize, 1, 1],
    };
  }

  /**
   * Free kernel resources
   */
  free(): void {
    bindings.freeKernel(this.kernel.id);
  }
}
