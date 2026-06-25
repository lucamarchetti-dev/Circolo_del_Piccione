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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/space_to_batchND_webgpu" />
import { WebGPUProgram } from './webgpu_program';
export declare class SpaceToBatchNDProgram implements WebGPUProgram {
    variableNames: string[];
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {
        x: number[];
    };
    dispatch: [number, number, number];
    uniforms: string;
    workgroupSize: [number, number, number];
    newDim: number[];
    xShape: number[];
    paddedXShape: number[];
    size: boolean;
    constructor(xShape: number[], paddedXShape: number[], paddings: Array<[number, number]>, reshapedPaddedXShape: number[], newDim: number[], paddedXShapeStridesShapeLength: number);
    getUserCode(): string;
}
