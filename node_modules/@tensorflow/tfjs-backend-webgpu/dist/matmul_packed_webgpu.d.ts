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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/matmul_packed_webgpu" />
import { backend_util, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from './webgpu_program';
export declare function matMulReadFnSource(transposeA: boolean, transposeB: boolean, fitAOuter?: boolean, fitBOuter?: boolean, fitInner?: boolean, component?: number): string;
export declare function matMulReadWriteFnSource(hasBias: boolean, activation: backend_util.Activation, transposeA: boolean, transposeB: boolean, fitAOuter?: boolean, fitBOuter?: boolean, fitInner?: boolean, component?: number): string;
export declare function makeMatMulPackedVec4Source(workPerThread: number[], workgroupSize: [number, number, number], transposeA?: boolean, tileInner?: number, splitK?: boolean, splitedDimInner?: number, broadcastBatch?: boolean): string;
export declare function makeMatMulPackedSource(workPerThread: number[], workgroupSize: [number, number, number], transposeA?: boolean, tileInner?: number, splitK?: boolean, splitedDimInner?: number, sequentialAccessByThreads?: boolean, broadcastBatch?: boolean): string;
export declare function makeVectorMatrixProductSource(workgroupSize: [number, number, number], transposeA?: boolean): string;
export declare class MatMulPackedProgram implements WebGPUProgram {
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
    elementsPerThread: [number, number, number];
    transposeA: boolean;
    transposeB: boolean;
    addBias: boolean;
    activation: backend_util.Activation;
    hasPreluActivationWeights: boolean;
    fitAOuter: boolean;
    fitBOuter: boolean;
    fitInner: boolean;
    tileInner: number;
    isVectorA: boolean;
    isVec4: boolean;
    outputComponent: number;
    private sequentialAccessByThreads;
    constructor(aShape: [number, number, number], outputShape: [number, number, number], transposeA?: boolean, transposeB?: boolean, bias?: TensorInfo, activation?: backend_util.Activation, preluActivationWeights?: TensorInfo, sequentialAccessByThreads?: boolean);
    getShapeFit(dimAOuter: number, dimBOuter: number, dimInner: number): boolean[];
    getUserCode(): string;
}
