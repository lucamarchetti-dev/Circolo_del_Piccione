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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/unary_op_util" />
export declare enum UnaryOpType {
    ABS = 0,
    ACOS = 1,
    ACOSH = 2,
    ASIN = 3,
    ASINH = 4,
    ATAN = 5,
    ATANH = 6,
    CEIL = 7,
    COS = 8,
    COSH = 9,
    ELU = 10,
    ERF = 11,
    EXP = 12,
    EXPM1 = 13,
    FLOOR = 14,
    IS_FINITE = 15,
    IS_INF = 16,
    IS_NAN = 17,
    LINEAR = 18,
    LOG = 19,
    LOG1P = 20,
    LOGICAL_NOT = 21,
    NEG = 22,
    RELU = 23,
    RELU6 = 24,
    LEAKYRELU = 25,
    RECIPROCAL = 26,
    ROUND = 27,
    RSQRT = 28,
    SELU = 29,
    SIGMOID = 30,
    SIGN = 31,
    SIN = 32,
    SINH = 33,
    SOFTPLUS = 34,
    SQRT = 35,
    SQUARE = 36,
    STEP = 37,
    TAN = 38,
    TANH = 39,
    TO_INT = 40
}
export declare function getUnaryOpString(type: UnaryOpType, useVec4?: boolean): string;
