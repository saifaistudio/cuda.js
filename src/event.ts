import bindings from './native';
import { Cuda } from './cuda';
import { Stream } from './stream';

/**
 * CUDA event wrapper for timing and synchronization
 */
export class Event {
    private id: number;

    constructor() {
        Cuda.init();
        this.id = bindings.createEvent();
    }

    /**
     * Record the event
     */
    record(stream?: Stream): void {
        if (stream) {
            bindings.recordEvent(this.id, stream.getId());
        } else {
            bindings.recordEvent(this.id);
        }
    }

    /**
     * Synchronize the event
     */
    synchronize(): void {
        bindings.synchronizeEvent(this.id);
    }

    /**
     * Calculate elapsed time between two events in milliseconds
     */
    elapsedTime(start: Event): number {
        return bindings.elapsedTime(start.id, this.id);
    }

    /**
     * Free event resources
     */
    free(): void {
        bindings.freeEvent(this.id);
    }
}