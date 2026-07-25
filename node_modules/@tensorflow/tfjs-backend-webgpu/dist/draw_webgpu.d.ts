/**
 * @license
 * Copyright 2023 Google LLC.
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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/draw_webgpu" />
/// <reference types="@webgpu/types/dist" />
import { DataType } from '@tensorflow/tfjs-core';
import { PixelsOpType, WebGPUProgram } from './webgpu_program';
export declare class DrawProgram implements WebGPUProgram {
    variableNames: string[];
    uniforms: string;
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number];
    type: DataType;
    textureFormat: GPUTextureFormat;
    pixelsOpType: PixelsOpType;
    size: boolean;
    constructor(outShape: number[], type: DataType, textureFormat: GPUTextureFormat);
    getUserCode(): string;
}
