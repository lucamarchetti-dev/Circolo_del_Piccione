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
import { LessEqual } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
import { lessEqualImplCPU as cpuLessEqual } from '../kernel_utils/shared';
export const lessEqual = binaryKernelFunc({
    opType: BinaryOpType.LESS_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: cpuLessEqual
});
export const lessEqualConfig = {
    kernelName: LessEqual,
    backendName: 'webgpu',
    kernelFunc: lessEqual
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTGVzc0VxdWFsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9MZXNzRXF1YWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLFNBQVMsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRTlELE9BQU8sRUFBQyxZQUFZLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUMvQyxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNwRSxPQUFPLEVBQUMsZ0JBQWdCLElBQUksWUFBWSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFeEUsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFHLGdCQUFnQixDQUFDO0lBQ3hDLE1BQU0sRUFBRSxZQUFZLENBQUMsVUFBVTtJQUMvQixLQUFLLEVBQUUsTUFBTTtJQUNiLGFBQWEsRUFBRSxZQUFZO0NBQzVCLENBQUMsQ0FBQztBQUVILE1BQU0sQ0FBQyxNQUFNLGVBQWUsR0FBaUI7SUFDM0MsVUFBVSxFQUFFLFNBQVM7SUFDckIsV0FBVyxFQUFFLFFBQVE7SUFDckIsVUFBVSxFQUFFLFNBQVM7Q0FDdEIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIExlc3NFcXVhbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCaW5hcnlPcFR5cGV9IGZyb20gJy4uL2JpbmFyeV9vcF91dGlsJztcbmltcG9ydCB7YmluYXJ5S2VybmVsRnVuY30gZnJvbSAnLi4va2VybmVsX3V0aWxzL2tlcm5lbF9mdW5jc191dGlscyc7XG5pbXBvcnQge2xlc3NFcXVhbEltcGxDUFUgYXMgY3B1TGVzc0VxdWFsfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvc2hhcmVkJztcblxuZXhwb3J0IGNvbnN0IGxlc3NFcXVhbCA9IGJpbmFyeUtlcm5lbEZ1bmMoe1xuICBvcFR5cGU6IEJpbmFyeU9wVHlwZS5MRVNTX0VRVUFMLFxuICBkdHlwZTogJ2Jvb2wnLFxuICBjcHVLZXJuZWxJbXBsOiBjcHVMZXNzRXF1YWxcbn0pO1xuXG5leHBvcnQgY29uc3QgbGVzc0VxdWFsQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IExlc3NFcXVhbCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBsZXNzRXF1YWxcbn07XG4iXX0=