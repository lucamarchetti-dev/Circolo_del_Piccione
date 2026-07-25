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
import { Minimum } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { minimumImplCPU as cpuMinimum } from '../kernel_utils/shared';
export const minimum = binaryKernelFunc({
    opType: BinaryOpType.MIN,
    cpuKernelImpl: cpuMinimum,
});
export const minimumConfig = {
    kernelName: Minimum,
    backendName: 'webgpu',
    kernelFunc: minimum
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWluaW11bS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvTWluaW11bS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQWUsT0FBTyxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFNUQsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQy9DLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLG9DQUFvQyxDQUFDO0FBQ3BFLE9BQU8sRUFBQyxjQUFjLElBQUksVUFBVSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFcEUsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGdCQUFnQixDQUFDO0lBQ3RDLE1BQU0sRUFBRSxZQUFZLENBQUMsR0FBRztJQUN4QixhQUFhLEVBQUUsVUFBVTtDQUMxQixDQUFDLENBQUM7QUFFSCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQWlCO0lBQ3pDLFVBQVUsRUFBRSxPQUFPO0lBQ25CLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxPQUFPO0NBQ3BCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBNaW5pbXVtfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0JpbmFyeU9wVHlwZX0gZnJvbSAnLi4vYmluYXJ5X29wX3V0aWwnO1xuaW1wb3J0IHtiaW5hcnlLZXJuZWxGdW5jfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMva2VybmVsX2Z1bmNzX3V0aWxzJztcbmltcG9ydCB7bWluaW11bUltcGxDUFUgYXMgY3B1TWluaW11bX0gZnJvbSAnLi4va2VybmVsX3V0aWxzL3NoYXJlZCc7XG5cbmV4cG9ydCBjb25zdCBtaW5pbXVtID0gYmluYXJ5S2VybmVsRnVuYyh7XG4gIG9wVHlwZTogQmluYXJ5T3BUeXBlLk1JTixcbiAgY3B1S2VybmVsSW1wbDogY3B1TWluaW11bSxcbn0pO1xuXG5leHBvcnQgY29uc3QgbWluaW11bUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBNaW5pbXVtLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IG1pbmltdW1cbn07XG4iXX0=