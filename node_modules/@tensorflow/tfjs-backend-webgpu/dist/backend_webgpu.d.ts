/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/backend_webgpu" />
/// <reference types="@webgpu/types/dist" />
import './flags_webgpu';
import { backend_util, BackendValues, DataStorage, DataType, GPUData, KernelBackend, Rank, RecursiveArray, Tensor, TensorBuffer, TensorInfo, TimingInfo, WebGPUData } from '@tensorflow/tfjs-core';
import { AdapterInfo } from './adapter_info';
import { BufferManager } from './buffer_manager';
import { TextureManager } from './texture_manager';
import * as webgpu_program from './webgpu_program';
export interface WebGPUMemoryInfo extends backend_util.MemoryInfo {
    numBytesInGPU: number;
    numBytesAllocatedInGPU: number;
    unreliable: boolean;
}
type TensorData = {
    values: BackendValues;
    dtype: DataType;
    shape: number[];
    refCount: number;
    resource?: GPUBuffer | GPUTexture | GPUExternalTexture;
    external?: boolean;
    complexTensorInfos?: {
        real: TensorInfo;
        imag: TensorInfo;
    };
};
interface DataId {
}
export type WebGPUKernelInfo = {
    name: string;
    query: Promise<number>;
};
export type TimerNode = RecursiveArray<WebGPUKernelInfo> | WebGPUKernelInfo;
export interface WebGPUTimingInfo extends TimingInfo {
    uploadWaitMs: number;
    downloadWaitMs: number;
}
type ProgramUniform = Array<{
    type: string;
    data: number[];
}>;
export declare class WebGPUBackend extends KernelBackend {
    bufferManager: BufferManager;
    adapterInfo: AdapterInfo;
    device: GPUDevice;
    queue: GPUQueue;
    tensorMap: DataStorage<TensorData>;
    textureManager: TextureManager;
    thresholdToIncreaseWorkgroups: number;
    private activeTimers;
    private commandEncoder;
    private computePassEncoder;
    private commandQueueOwnedIds;
    private dispatchCountInPass;
    private disposed;
    private downloadWaitMs;
    private dummyCanvas;
    private dummyContext;
    private tensorDataPendingDisposal;
    private static nextDataId;
    private pipelineCache;
    private programTimersStack;
    private queryResolveBuffer;
    private querySet;
    private querySetCount;
    private stagingPendingDisposal;
    private supportTimestampQuery;
    private uniformPendingDisposal;
    private uploadWaitMs;
    private hasReadSyncWarned;
    private hasTimestampQueryWarned;
    private nextDataId;
    constructor(device: GPUDevice, adapterInfo?: GPUAdapterInfo);
    floatPrecision(): 32;
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released or delayed in this backend, false if there are still
     * references.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId: DataId, force?: boolean): boolean;
    memory(): WebGPUMemoryInfo;
    private releaseResource;
    /** Return refCount of a `TensorData`. */
    refCount(dataId: DataId): number;
    /** Increase refCount of a `TensorData`. */
    incRef(dataId: DataId): void;
    /** Decrease refCount of a `TensorData`. */
    decRef(dataId: DataId): void;
    write(values: BackendValues, shape: number[], dtype: DataType): DataId;
    move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType, refCount: number): void;
    submitQueue(): void;
    ensureCommandEncoderReady(): void;
    endComputePassEncoder(): void;
    checkCompileCompletionAsync(): Promise<void>;
    getBufferData(buffer: GPUBuffer): Promise<ArrayBuffer>;
    private convertAndCacheOnCPU;
    readSync(dataId: object): BackendValues;
    read(dataId: object): Promise<BackendValues>;
    private copyBuffer;
    /**
     * Create a TF.js tensor out of an existing WebGPU buffer.
     */
    createTensorFromGPUData(webGPUData: WebGPUData, shape: number[], dtype: DataType): Tensor;
    /**
     * Read tensor to a new GPUBuffer.
     * @param dataId The source tensor.
     */
    readToGPU(dataId: DataId): GPUData;
    bufferSync<R extends Rank, D extends DataType>(t: TensorInfo): TensorBuffer<R, D>;
    time(f: () => void): Promise<WebGPUTimingInfo>;
    makeTensorInfo(shape: number[], dtype: DataType, values?: BackendValues | string[]): TensorInfo;
    private tensorToBinding;
    uploadToGPU(dataId: DataId): void;
    private makeUniforms;
    runWebGPUProgram(program: webgpu_program.WebGPUProgram, inputs: TensorInfo[], outputDtype: DataType, programDefinedUniform?: ProgramUniform, output?: TensorInfo): TensorInfo;
    private recordAndSubmit;
    getQueryTime(): Promise<number>;
    shouldExecuteOnCPU(inputs: TensorInfo[], sizeThreshold?: number): boolean;
    numDataIds(): number;
    dispose(): void;
}
export {};
