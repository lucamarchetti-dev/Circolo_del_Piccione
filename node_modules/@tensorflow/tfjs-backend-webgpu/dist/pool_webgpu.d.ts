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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/pool_webgpu" />
import { backend_util } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from './webgpu_program';
export declare class Pool2DProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms: string;
    workgroupSize: [number, number, number];
    poolType: 'max' | 'avg';
    size: boolean;
    computePositions: boolean;
    flattenPositions: boolean;
    includeBatchIndex: boolean;
    constructor(convInfo: backend_util.Conv2DInfo, poolType: 'max' | 'avg', computePositions?: boolean, flattenPositions?: boolean, includeBatchIndex?: boolean);
    getUserCode(): string;
}
export declare class Pool3DProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms: string;
    workgroupSize: [number, number, number];
    poolType: 'max' | 'avg';
    size: boolean;
    computePositions: boolean;
    flattenPositions: boolean;
    includeBatchIndex: boolean;
    constructor(convInfo: backend_util.Conv3DInfo, poolType: 'max' | 'avg', computePositions?: boolean, flattenPositions?: boolean, includeBatchIndex?: boolean);
    getUserCode(): string;
}
