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
import { Sub } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { subImplCPU as cpuSub } from '../kernel_utils/shared';
export const sub = binaryKernelFunc({ opType: BinaryOpType.SUB, cpuKernelImpl: cpuSub, supportsComplex: true });
export const subConfig = {
    kernelName: Sub,
    backendName: 'webgpu',
    kernelFunc: sub
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3ViLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9TdWIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXhELE9BQU8sRUFBQyxZQUFZLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUMvQyxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNwRSxPQUFPLEVBQUMsVUFBVSxJQUFJLE1BQU0sRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBRTVELE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxnQkFBZ0IsQ0FDL0IsRUFBQyxNQUFNLEVBQUUsWUFBWSxDQUFDLEdBQUcsRUFBRSxhQUFhLEVBQUUsTUFBTSxFQUFFLGVBQWUsRUFBRSxJQUFJLEVBQUMsQ0FBQyxDQUFDO0FBRTlFLE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBaUI7SUFDckMsVUFBVSxFQUFFLEdBQUc7SUFDZixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsR0FBRztDQUNoQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgU3VifSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0JpbmFyeU9wVHlwZX0gZnJvbSAnLi4vYmluYXJ5X29wX3V0aWwnO1xuaW1wb3J0IHtiaW5hcnlLZXJuZWxGdW5jfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMva2VybmVsX2Z1bmNzX3V0aWxzJztcbmltcG9ydCB7c3ViSW1wbENQVSBhcyBjcHVTdWJ9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuXG5leHBvcnQgY29uc3Qgc3ViID0gYmluYXJ5S2VybmVsRnVuYyhcbiAgICB7b3BUeXBlOiBCaW5hcnlPcFR5cGUuU1VCLCBjcHVLZXJuZWxJbXBsOiBjcHVTdWIsIHN1cHBvcnRzQ29tcGxleDogdHJ1ZX0pO1xuXG5leHBvcnQgY29uc3Qgc3ViQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFN1YixcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBzdWJcbn07XG4iXX0=