/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/webgpu_util" />
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
import { DataType, TensorInfo } from '@tensorflow/tfjs-core';
export declare function tilesFitEvenlyIntoShape(tileSize: number[], shape: number[]): boolean;
export declare function computeDispatch(layout: {
    x: number[];
    y?: number[];
    z?: number[];
}, outputShape: number[], workgroupSize?: [number, number, number], elementsPerThread?: [number, number, number]): [number, number, number];
export type WorkgroupInfo = {
    workgroupSize: [number, number, number];
    elementsPerThread: [number, number, number];
};
export declare function computeWorkgroupInfoForMatMul(dimAOuter: number, dimInner: number, dimBOuter: number, transposeA?: boolean): WorkgroupInfo;
export declare function computeWorkgroupSizeForConv2d(layout: {
    x: number[];
    y?: number[];
    z?: number[];
}, outputShape: number[], isVec4?: boolean): [number, number, number];
export declare function computeWorkPerThreadForConv2d(layout: {
    x: number[];
    y?: number[];
    z?: number[];
}, outputShape: number[], isVec4?: boolean): [number, number, number];
export declare function flatDispatchLayout(shape: number[]): {
    x: number[];
};
export declare function GPUBytesPerElement(dtype: DataType): number;
export declare function isWebGPUSupported(): boolean;
export declare function assertNotComplex(tensor: TensorInfo | TensorInfo[], opName: string): void;
export declare enum MatMulProgramType {
    MatMulReduceProgram = 0,
    MatMulSplitKProgram = 1,
    MatMulSmallOutputSizeProgram = 2,
    MatMulPackedProgram = 3,
    MatMulMax = 4
}
