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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/kernel_utils/kernel_funcs_utils" />
import { DataType, KernelFunc } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { UnaryOpType } from '../unary_op_util';
import { SimpleBinaryKernelImplCPU, SimpleUnaryKernelImplCPU } from './shared';
type UnaryKernelFuncConfig = {
    opType: UnaryOpType;
    cpuKernelImpl?: SimpleUnaryKernelImplCPU;
    dtype?: DataType;
};
/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param opType Op type to create `UnaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export declare function unaryKernelFunc({ opType, cpuKernelImpl, dtype }: UnaryKernelFuncConfig): KernelFunc;
type BinaryKernelFuncConfig = {
    opType: BinaryOpType;
    cpuKernelImpl?: SimpleBinaryKernelImplCPU;
    supportsComplex?: boolean;
    dtype?: DataType;
};
/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opType Op type to create `BinaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
export declare function binaryKernelFunc({ opType, cpuKernelImpl, supportsComplex, dtype }: BinaryKernelFuncConfig): KernelFunc;
export {};
