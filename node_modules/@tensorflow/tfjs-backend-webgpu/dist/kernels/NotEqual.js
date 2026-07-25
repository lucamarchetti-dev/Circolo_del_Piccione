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
import { NotEqual } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { notEqualImplCPU as cpuNotEqual } from '../kernel_utils/shared';
export const notEqual = binaryKernelFunc({
    opType: BinaryOpType.NOT_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: cpuNotEqual
});
export const notEqualConfig = {
    kernelName: NotEqual,
    backendName: 'webgpu',
    kernelFunc: notEqual
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTm90RXF1YWwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL05vdEVxdWFsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBZSxRQUFRLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUU3RCxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDL0MsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sb0NBQW9DLENBQUM7QUFDcEUsT0FBTyxFQUFDLGVBQWUsSUFBSSxXQUFXLEVBQUMsTUFBTSx3QkFBd0IsQ0FBQztBQUV0RSxNQUFNLENBQUMsTUFBTSxRQUFRLEdBQUcsZ0JBQWdCLENBQUM7SUFDdkMsTUFBTSxFQUFFLFlBQVksQ0FBQyxTQUFTO0lBQzlCLEtBQUssRUFBRSxNQUFNO0lBQ2IsYUFBYSxFQUFFLFdBQVc7Q0FDM0IsQ0FBQyxDQUFDO0FBRUgsTUFBTSxDQUFDLE1BQU0sY0FBYyxHQUFpQjtJQUMxQyxVQUFVLEVBQUUsUUFBUTtJQUNwQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsUUFBUTtDQUNyQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgTm90RXF1YWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmluYXJ5T3BUeXBlfSBmcm9tICcuLi9iaW5hcnlfb3BfdXRpbCc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmN9IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuaW1wb3J0IHtub3RFcXVhbEltcGxDUFUgYXMgY3B1Tm90RXF1YWx9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuXG5leHBvcnQgY29uc3Qgbm90RXF1YWwgPSBiaW5hcnlLZXJuZWxGdW5jKHtcbiAgb3BUeXBlOiBCaW5hcnlPcFR5cGUuTk9UX0VRVUFMLFxuICBkdHlwZTogJ2Jvb2wnLFxuICBjcHVLZXJuZWxJbXBsOiBjcHVOb3RFcXVhbFxufSk7XG5cbmV4cG9ydCBjb25zdCBub3RFcXVhbENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBOb3RFcXVhbCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBub3RFcXVhbFxufTtcbiJdfQ==