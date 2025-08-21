export interface BufferInfo {
    id: number;
    size: number;
}

export interface KernelInfo {
    id: number;
    name: string;
}

export interface DeviceProperties {
    name: string;
    computeCapability: string;
    totalMemory: number;
    multiprocessors: number;
    maxThreadsPerBlock: number;
    maxGridSize: number[];
    maxBlockSize: number[];
}

export type KernelArg = BufferInfo | number;

export interface LaunchConfig {
    gridDim: number[];
    blockDim: number[];
    sharedMem?: number;
    stream?: number;
}