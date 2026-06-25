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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/webgpu_program" />
/// <reference types="@webgpu/types/dist" />
import { DataType, Rank, TensorInfo } from '@tensorflow/tfjs-core';
export declare enum PixelsOpType {
    FROM_PIXELS = 0,
    DRAW = 1
}
export interface WebGPUProgram {
    atomic?: boolean;
    dispatch: [number, number, number];
    dispatchLayout: {
        x: number[];
        y?: number[];
        z?: number[];
    };
    outputComponent?: number;
    outputShape: number[];
    pixelsOpType?: PixelsOpType;
    shaderKey: string;
    size?: boolean;
    uniforms?: string;
    variableNames: string[];
    variableComponents?: number[];
    workgroupSize: [number, number, number];
    workPerThread?: number;
    pipeline?: GPUComputePipeline | Promise<GPUComputePipeline>;
    getUserCode: () => string;
}
export declare const compileProgram: (device: GPUDevice, program: WebGPUProgram, inputsData: InputInfo[], output: TensorInfo, parallelCompilation: boolean) => GPUComputePipeline | Promise<GPUComputePipeline>;
export declare const typeSnippet: (component: number, type?: string) => string;
export declare function getCoordsDataType(rank: number): string;
export declare function getCoordsXYZ(index: number): string;
export declare function getMainHeaderString(): string;
export declare function getMainHeaderString(index: string): string;
export declare function getStartHeaderString(useGlobalIndex: boolean, program: WebGPUProgram): string;
export declare function getWorkgroupSizeString(program: WebGPUProgram): string;
export declare function makeShaderKey<R extends Rank>(program: WebGPUProgram, inputsData: InputInfo[], output: TensorInfo): string;
type InputInfo = {
    dtype: DataType;
    shape: number[];
    name: string;
};
/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
export declare function getCoordsFromIndexSnippet(shape: number[], name?: string): string;
export declare function dataTypeToGPUType(type: DataType, component?: number): string;
export {};
