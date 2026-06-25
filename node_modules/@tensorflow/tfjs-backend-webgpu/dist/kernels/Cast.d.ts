/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/kernels/Cast" />
import { CastAttrs, CastInputs, KernelConfig, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '../backend_webgpu';
export declare function cast(args: {
    inputs: CastInputs;
    backend: WebGPUBackend;
    attrs: CastAttrs;
}): TensorInfo;
export declare const castConfig: KernelConfig;
