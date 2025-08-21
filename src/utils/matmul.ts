import { GpuArray } from '../gpu-array';
import { Kernel } from '../kernel';

/**
 * Matrix multiplication on GPU
 */
export function matmul(a: GpuArray, b: GpuArray, m: number, n: number, k: number): GpuArray {
    const kernel = new Kernel(`
#define TILE_SIZE 16

extern "C" __global__ void matmul_kernel(
    float* A, float* B, float* C,
    int m, int n, int k
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < m && t * TILE_SIZE + tx < k) {
            As[ty][tx] = A[row * k + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < n && t * TILE_SIZE + ty < k) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}`, 'matmul_kernel');

    const c = new GpuArray(m * n);

    const blockSize = [16, 16, 1];
    const gridSize = [
        Math.ceil(n / 16),
        Math.ceil(m / 16),
        1
    ];

    kernel.run([a, b, c, m, n, k], gridSize, blockSize);
    kernel.free();

    return c;
}