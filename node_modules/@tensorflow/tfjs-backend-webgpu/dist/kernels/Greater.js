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
import { Greater } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { greaterImplCPU as cpuGreater } from '../kernel_utils/shared';
export const greater = binaryKernelFunc({
    opType: BinaryOpType.GREATER,
    cpuKernelImpl: cpuGreater,
    dtype: 'bool',
});
export const greaterConfig = {
    kernelName: Greater,
    backendName: 'webgpu',
    kernelFunc: greater
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiR3JlYXRlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvR3JlYXRlci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsT0FBTyxFQUFlLE1BQU0sdUJBQXVCLENBQUM7QUFFNUQsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQy9DLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLG9DQUFvQyxDQUFDO0FBQ3BFLE9BQU8sRUFBQyxjQUFjLElBQUksVUFBVSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFcEUsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGdCQUFnQixDQUFDO0lBQ3RDLE1BQU0sRUFBRSxZQUFZLENBQUMsT0FBTztJQUM1QixhQUFhLEVBQUUsVUFBVTtJQUN6QixLQUFLLEVBQUUsTUFBTTtDQUNkLENBQUMsQ0FBQztBQUVILE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBaUI7SUFDekMsVUFBVSxFQUFFLE9BQU87SUFDbkIsV0FBVyxFQUFFLFFBQVE7SUFDckIsVUFBVSxFQUFFLE9BQU87Q0FDcEIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtHcmVhdGVyLCBLZXJuZWxDb25maWd9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmluYXJ5T3BUeXBlfSBmcm9tICcuLi9iaW5hcnlfb3BfdXRpbCc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmN9IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuaW1wb3J0IHtncmVhdGVySW1wbENQVSBhcyBjcHVHcmVhdGVyfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvc2hhcmVkJztcblxuZXhwb3J0IGNvbnN0IGdyZWF0ZXIgPSBiaW5hcnlLZXJuZWxGdW5jKHtcbiAgb3BUeXBlOiBCaW5hcnlPcFR5cGUuR1JFQVRFUixcbiAgY3B1S2VybmVsSW1wbDogY3B1R3JlYXRlcixcbiAgZHR5cGU6ICdib29sJyxcbn0pO1xuXG5leHBvcnQgY29uc3QgZ3JlYXRlckNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBHcmVhdGVyLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGdyZWF0ZXJcbn07XG4iXX0=