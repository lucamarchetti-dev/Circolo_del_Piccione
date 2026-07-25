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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/unary_op_webgpu" />
import { UnaryOpType } from './unary_op_util';
import { WebGPUProgram } from './webgpu_program';
export declare class UnaryOpProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    variableNames: string[];
    workgroupSize: [number, number, number];
    op: UnaryOpType;
    uniforms?: string;
    size: boolean;
    constructor(outputShape: number[], op: UnaryOpType, uniforms?: string);
    getUserCode(): string;
}
