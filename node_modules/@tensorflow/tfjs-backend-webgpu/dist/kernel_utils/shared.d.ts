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
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/kernel_utils/shared" />
import * as shared from '@tensorflow/tfjs-backend-cpu/dist/shared';
import { SimpleBinaryKernelImpl } from '@tensorflow/tfjs-backend-cpu/dist/shared';
import { SimpleUnaryImpl } from '@tensorflow/tfjs-backend-cpu/dist/utils/unary_types';
export type SimpleBinaryKernelImplCPU = SimpleBinaryKernelImpl;
export type SimpleUnaryKernelImplCPU = SimpleUnaryImpl;
declare const addImplCPU: shared.SimpleBinaryKernelImpl, castImplCPU: typeof shared.castImpl, ceilImplCPU: SimpleUnaryImpl<number, number>, concatImplCPU: typeof shared.concatImpl, equalImplCPU: shared.SimpleBinaryKernelImpl, expImplCPU: SimpleUnaryImpl<number, number>, expm1ImplCPU: SimpleUnaryImpl<number, number>, floorImplCPU: SimpleUnaryImpl<number, number>, floorDivImplCPU: shared.SimpleBinaryKernelImpl, gatherNdImplCPU: typeof shared.gatherNdImpl, gatherV2ImplCPU: typeof shared.gatherV2Impl, greaterEqualImplCPU: shared.SimpleBinaryKernelImpl, greaterImplCPU: shared.SimpleBinaryKernelImpl, lessEqualImplCPU: shared.SimpleBinaryKernelImpl, lessImplCPU: shared.SimpleBinaryKernelImpl, logImplCPU: SimpleUnaryImpl<number, number>, maxImplCPU: typeof shared.maxImpl, maximumImplCPU: shared.SimpleBinaryKernelImpl, minimumImplCPU: shared.SimpleBinaryKernelImpl, multiplyImplCPU: shared.SimpleBinaryKernelImpl, negImplCPU: typeof shared.negImpl, notEqualImplCPU: shared.SimpleBinaryKernelImpl, prodImplCPU: typeof shared.prodImpl, rangeImplCPU: typeof shared.rangeImpl, rsqrtImplCPU: SimpleUnaryImpl<number, number>, scatterImplCPU: typeof shared.scatterImpl, simpleAbsImplCPU: typeof shared.simpleAbsImpl, sliceImplCPU: typeof shared.sliceImpl, stridedSliceImplCPU: typeof shared.stridedSliceImpl, stringNGramsImplCPU: typeof shared.stringNGramsImpl, subImplCPU: shared.SimpleBinaryKernelImpl, tileImplCPU: typeof shared.tileImpl, topKImplCPU: typeof shared.topKImpl, transposeImplCPU: typeof shared.transposeImpl, uniqueImplCPU: typeof shared.uniqueImpl;
export { addImplCPU, castImplCPU, ceilImplCPU, concatImplCPU, equalImplCPU, expImplCPU, expm1ImplCPU, floorImplCPU, floorDivImplCPU, gatherNdImplCPU, gatherV2ImplCPU, greaterEqualImplCPU, greaterImplCPU, lessEqualImplCPU, lessImplCPU, logImplCPU, maxImplCPU, maximumImplCPU, minimumImplCPU, multiplyImplCPU, prodImplCPU, negImplCPU, notEqualImplCPU, scatterImplCPU, simpleAbsImplCPU, sliceImplCPU, stridedSliceImplCPU, stringNGramsImplCPU, subImplCPU, rangeImplCPU, rsqrtImplCPU, tileImplCPU, topKImplCPU, transposeImplCPU, uniqueImplCPU, };
