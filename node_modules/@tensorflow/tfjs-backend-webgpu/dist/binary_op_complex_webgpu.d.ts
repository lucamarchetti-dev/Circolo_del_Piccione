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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/binary_op_complex_webgpu" />
import { BinaryOpType } from './binary_op_util';
import { WebGPUProgram } from './webgpu_program';
export declare class BinaryOpComplexProgram implements WebGPUProgram {
    variableNames: string[];
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number];
    op: BinaryOpType;
    size: boolean;
    constructor(op: BinaryOpType, aShape: number[], bShape: number[]);
    getUserCode(): string;
}
