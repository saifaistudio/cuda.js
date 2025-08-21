import bindings from './native';
import { DeviceProperties } from './types';

/**
 * Main CUDA context manager
 */
export class Cuda {
    private static initialized = false;

    /**
     * Initialize CUDA runtime
     */
    static init(): void {
        if (!this.initialized) {
            bindings.cudaInit();
            this.initialized = true;
        }
    }

    /**
     * Get the number of available CUDA devices
     */
    static getDeviceCount(): number {
        this.ensureInitialized();
        return bindings.getDeviceCount();
    }

    /**
     * Set the current CUDA device
     */
    static setDevice(device: number): void {
        this.ensureInitialized();
        bindings.setDevice(device);
    }

    /**
     * Get the current CUDA device
     */
    static getCurrentDevice(): number {
        this.ensureInitialized();
        return bindings.getCurrentDevice();
    }

    /**
     * Synchronize all CUDA operations
     */
    static synchronize(): void {
        this.ensureInitialized();
        bindings.synchronize();
    }

    /**
     * Get device information as a string
     */
    static getDeviceInfo(device: number = 0): string {
        this.ensureInitialized();
        return bindings.getDeviceInfo(device);
    }

    /**
     * Parse device info string into structured data
     */
    static getDeviceProperties(device: number = 0): DeviceProperties {
        const info = this.getDeviceInfo(device);
        const lines = info.split('\n');

        const props: DeviceProperties = {
            name: '',
            computeCapability: '',
            totalMemory: 0,
            multiprocessors: 0,
            maxThreadsPerBlock: 0,
            maxGridSize: [],
            maxBlockSize: []
        };

        lines.forEach(line => {
            if (line.includes('Device')) {
                props.name = line.split(': ')[1];
            } else if (line.includes('Compute Capability')) {
                props.computeCapability = line.split(': ')[1];
            } else if (line.includes('Total Memory')) {
                props.totalMemory = parseInt(line.match(/\d+/)?.[0] || '0');
            } else if (line.includes('Multiprocessors')) {
                props.multiprocessors = parseInt(line.split(': ')[1]);
            } else if (line.includes('Max Threads per Block')) {
                props.maxThreadsPerBlock = parseInt(line.split(': ')[1]);
            } else if (line.includes('Max Grid Size')) {
                const matches = line.match(/\[(\d+), (\d+), (\d+)\]/);
                if (matches) {
                    props.maxGridSize = [parseInt(matches[1]), parseInt(matches[2]), parseInt(matches[3])];
                }
            } else if (line.includes('Max Block Size')) {
                const matches = line.match(/\[(\d+), (\d+), (\d+)\]/);
                if (matches) {
                    props.maxBlockSize = [parseInt(matches[1]), parseInt(matches[2]), parseInt(matches[3])];
                }
            }
        });

        return props;
    }

    private static ensureInitialized(): void {
        if (!this.initialized) {
            throw new Error('CUDA not initialized. Call Cuda.init() first.');
        }
    }
}