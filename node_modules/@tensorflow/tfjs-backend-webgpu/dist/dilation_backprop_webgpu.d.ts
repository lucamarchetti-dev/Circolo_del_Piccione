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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/dilation_backprop_webgpu" />
import { backend_util, DataType } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from './webgpu_program';
export declare class Dilation2DBackpropInputProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms: string;
    workgroupSize: [number, number, number];
    atomic: boolean;
    type: DataType;
    constructor(convInfo: backend_util.Conv2DInfo, outputDtype: DataType);
    getUserCode(): string;
}
export declare class Dilation2DBackpropFilterProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms: string;
    workgroupSize: [number, number, number];
    atomic: boolean;
    type: DataType;
    constructor(convInfo: backend_util.Conv2DInfo, shape: number[], outputDtype: DataType);
    getUserCode(): string;
}
