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
import { Imag } from '@tensorflow/tfjs-core';
import { identity } from './Identity';
export function imag(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputData = backend.tensorMap.get(input.dataId);
    return identity({ inputs: { x: inputData.complexTensorInfos.imag }, backend });
}
export const imagConfig = {
    kernelName: Imag,
    backendName: 'webgpu',
    kernelFunc: imag
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiSW1hZy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvSW1hZy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsSUFBSSxFQUFtRCxNQUFNLHVCQUF1QixDQUFDO0FBRzdGLE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFcEMsTUFBTSxVQUFVLElBQUksQ0FBQyxJQUFrRDtJQUVyRSxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBQyxHQUFHLElBQUksQ0FBQztJQUMvQixNQUFNLEVBQUMsS0FBSyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ3ZCLE1BQU0sU0FBUyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUV0RCxPQUFPLFFBQVEsQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztBQUM3RSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sVUFBVSxHQUFpQjtJQUN0QyxVQUFVLEVBQUUsSUFBSTtJQUNoQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsSUFBNkI7Q0FDMUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtJbWFnLCBJbWFnSW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtpZGVudGl0eX0gZnJvbSAnLi9JZGVudGl0eSc7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbWFnKGFyZ3M6IHtpbnB1dHM6IEltYWdJbnB1dHMsIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmR9KTpcbiAgICBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZH0gPSBhcmdzO1xuICBjb25zdCB7aW5wdXR9ID0gaW5wdXRzO1xuICBjb25zdCBpbnB1dERhdGEgPSBiYWNrZW5kLnRlbnNvck1hcC5nZXQoaW5wdXQuZGF0YUlkKTtcblxuICByZXR1cm4gaWRlbnRpdHkoe2lucHV0czoge3g6IGlucHV0RGF0YS5jb21wbGV4VGVuc29ySW5mb3MuaW1hZ30sIGJhY2tlbmR9KTtcbn1cblxuZXhwb3J0IGNvbnN0IGltYWdDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogSW1hZyxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBpbWFnIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==