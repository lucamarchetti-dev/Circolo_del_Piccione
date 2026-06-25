/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/conv2d_naive_webgpu" />
import { backend_util } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from './webgpu_program';
export declare class Conv2DNaiveProgram implements WebGPUProgram {
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
    addBias: boolean;
    activation: backend_util.Activation;
    hasPreluActivationWeights: boolean;
    isChannelsLast: boolean;
    constructor(convInfo: backend_util.Conv2DInfo, addBias?: boolean, activation?: backend_util.Activation, hasPreluActivationWeights?: boolean);
    getUserCode(): string;
}
