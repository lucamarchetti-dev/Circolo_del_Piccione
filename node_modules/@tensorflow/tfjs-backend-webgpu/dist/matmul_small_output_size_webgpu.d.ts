/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/matmul_small_output_size_webgpu" />
import { backend_util, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from './webgpu_program';
export declare function makeMatMulSmallOutputSizeSource(workgroupSize: [number, number, number]): string;
export declare class MatMulSmallOutputSizeProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
        y: number[];
        z: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms: string;
    workgroupSize: [number, number, number];
    transposeA: boolean;
    transposeB: boolean;
    addBias: boolean;
    activation: backend_util.Activation;
    hasPreluActivationWeights: boolean;
    constructor(aShape: [number, number, number], bShape: [number, number, number], outputShape: [number, number, number], transposeA?: boolean, transposeB?: boolean, bias?: TensorInfo, activation?: backend_util.Activation, preluActivationWeights?: TensorInfo);
    getUserCode(): string;
}
