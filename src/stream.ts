import bindings from './native';
import { Cuda } from './cuda';

/**
 * CUDA stream wrapper for asynchronous operations
 */
export class Stream {
    private id: number;

    constructor() {
        Cuda.init();
        this.id = bindings.createStream();
    }

    /**
     * Synchronize the stream
     */
    synchronize(): void {
        bindings.synchronizeStream(this.id);
    }

    /**
     * Get stream ID for internal use
     */
    getId(): number {
        return this.id;
    }

    /**
     * Free stream resources
     */
    free(): void {
        bindings.freeStream(this.id);
    }
}