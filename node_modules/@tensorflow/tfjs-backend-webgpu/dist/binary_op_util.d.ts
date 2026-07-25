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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/binary_op_util" />
export declare enum BinaryOpType {
    ADD = 0,
    ATAN2 = 1,
    COMPLEX_MULTIPLY_IMAG = 2,
    COMPLEX_MULTIPLY_REAL = 3,
    DIV = 4,
    ELU_DER = 5,
    EQUAL = 6,
    FLOOR_DIV = 7,
    GREATER = 8,
    GREATER_EQUAL = 9,
    LESS = 10,
    LESS_EQUAL = 11,
    LOGICAL_AND = 12,
    LOGICAL_OR = 13,
    MAX = 14,
    MIN = 15,
    MOD = 16,
    MUL = 17,
    NOT_EQUAL = 18,
    POW = 19,
    PRELU = 20,
    SQUARED_DIFFERENCE = 21,
    SUB = 22
}
export declare function getBinaryOpString(type: BinaryOpType, useVec4?: boolean): string;
