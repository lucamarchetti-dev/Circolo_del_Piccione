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
import { RealDiv } from '@tensorflow/tfjs-core';
import { BinaryOpType } from '../binary_op_util';
import { binaryKernelFunc } from '../kernel_utils/kernel_funcs_utils';
export const realDiv = binaryKernelFunc({ opType: BinaryOpType.DIV });
export const realDivConfig = {
    kernelName: RealDiv,
    backendName: 'webgpu',
    kernelFunc: realDiv
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUmVhbERpdi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvUmVhbERpdi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLE9BQU8sRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXhFLE9BQU8sRUFBQyxZQUFZLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUMvQyxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUVwRSxNQUFNLENBQUMsTUFBTSxPQUFPLEdBQUcsZ0JBQWdCLENBQUMsRUFBQyxNQUFNLEVBQUUsWUFBWSxDQUFDLEdBQUcsRUFBQyxDQUFDLENBQUM7QUFFcEUsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFpQjtJQUN6QyxVQUFVLEVBQUUsT0FBTztJQUNuQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsT0FBZ0M7Q0FDN0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFJlYWxEaXZ9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmluYXJ5T3BUeXBlfSBmcm9tICcuLi9iaW5hcnlfb3BfdXRpbCc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmN9IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgcmVhbERpdiA9IGJpbmFyeUtlcm5lbEZ1bmMoe29wVHlwZTogQmluYXJ5T3BUeXBlLkRJVn0pO1xuXG5leHBvcnQgY29uc3QgcmVhbERpdkNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBSZWFsRGl2LFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IHJlYWxEaXYgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19