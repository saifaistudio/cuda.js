import bindings from './native';
import { BufferInfo } from './types';
import { Cuda } from './cuda';

/**
 * GPU array wrapper similar to PyCuda's GPUArray
 */
export class GpuArray {
    private buffer: BufferInfo;
    private _shape: number[];
    private _dtype: string;
    private _size: number;

    /**
     * Create a new GPU array
     * @param data - Initial data or size
     * @param dtype - Data type (default: 'float32')
     */
    constructor(data?: number[] | Float32Array | number, dtype: string = 'float32') {
        Cuda.init();

        this._dtype = dtype;

        if (typeof data === 'number') {
            // Create empty array of given size
            this._size = data;
            this._shape = [data];
            const byteSize = this._size * this.getBytesPerElement();
            this.buffer = bindings.createBuffer(byteSize);
        } else if (data instanceof Float32Array) {
            // Create from Float32Array
            this._size = data.length;
            this._shape = [data.length];
            const byteSize = data.byteLength;
            this.buffer = bindings.createBuffer(byteSize);
            bindings.uploadBuffer(this.buffer.id, data);
        } else if (Array.isArray(data)) {
            // Create from array
            const typedArray = new Float32Array(data);
            this._size = typedArray.length;
            this._shape = [typedArray.length];
            const byteSize = typedArray.byteLength;
            this.buffer = bindings.createBuffer(byteSize);
            bindings.uploadBuffer(this.buffer.id, typedArray);
        } else {
            throw new Error('Invalid data type for GpuArray');
        }
    }

    /**
     * Get the shape of the array
     */
    get shape(): number[] {
        return [...this._shape];
    }

    /**
     * Get the data type
     */
    get dtype(): string {
        return this._dtype;
    }

    /**
     * Get the total number of elements
     */
    get size(): number {
        return this._size;
    }

    /**
     * Get bytes per element based on dtype
     */
    private getBytesPerElement(): number {
        switch (this._dtype) {
            case 'float32':
                return 4;
            case 'float64':
                return 8;
            case 'int32':
                return 4;
            case 'int8':
                return 1;
            default:
                return 4;
        }
    }

    /**
     * Upload data to GPU
     */
    upload(data: number[] | Float32Array): void {
        const typedArray = data instanceof Float32Array ? data : new Float32Array(data);
        if (typedArray.length !== this._size) {
            throw new Error(`Data size mismatch. Expected ${this._size}, got ${typedArray.length}`);
        }
        bindings.uploadBuffer(this.buffer.id, typedArray);
    }

    /**
     * Download data from GPU
     */
    download(): Float32Array {
        return bindings.downloadBuffer(this.buffer.id);
    }

    /**
     * Get data as regular array
     */
    toArray(): number[] {
        return Array.from(this.download());
    }

    /**
     * Fill array with zeros
     */
    zero(): void {
        bindings.memsetBuffer(this.buffer.id, 0);
    }

    /**
     * Fill array with a value
     */
    fill(value: number): void {
        if (value === 0) {
            this.zero();
        } else {
            // Upload filled array
            const data = new Float32Array(this._size).fill(value);
            this.upload(data);
        }
    }

    /**
     * Copy data from another GPU array
     */
    copyFrom(other: GpuArray): void {
        if (this._size !== other._size) {
            throw new Error('Size mismatch in copy operation');
        }
        const data = other.download();
        this.upload(data);
    }

    /**
     * Create a copy of this array
     */
    copy(): GpuArray {
        const newArray = new GpuArray(this._size, this._dtype);
        newArray.copyFrom(this);
        return newArray;
    }

    /**
     * Reshape the array (view only, doesn't change memory layout)
     */
    reshape(shape: number[]): GpuArray {
        const totalElements = shape.reduce((a, b) => a * b, 1);
        if (totalElements !== this._size) {
            throw new Error(`Cannot reshape array of size ${this._size} to shape [${shape.join(', ')}]`);
        }
        const reshaped = Object.create(this);
        reshaped._shape = shape;
        return reshaped;
    }

    /**
     * Free GPU memory
     */
    free(): void {
        bindings.freeBuffer(this.buffer.id);
    }

    /**
     * String representation
     */
    toString(): string {
        const data = this.download();
        const preview = Array.from(data.slice(0, Math.min(10, data.length)));
        const suffix = data.length > 10 ? ', ...' : '';
        return `GpuArray([${preview.join(', ')}${suffix}], shape=${JSON.stringify(this._shape)}, dtype='${this._dtype}')`;
    }
}