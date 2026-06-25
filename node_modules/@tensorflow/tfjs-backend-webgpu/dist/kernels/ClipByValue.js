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
import { ClipByValue, util } from '@tensorflow/tfjs-core';
import { ClipVec4Program } from '../clip_vec4_webgpu';
import { ClipProgram } from '../clip_webgpu';
export function clipByValue(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { clipValueMin, clipValueMax } = attrs;
    let program;
    const uniformData = [
        { type: 'float32', data: [clipValueMin] },
        { type: 'float32', data: [clipValueMax] }
    ];
    if (util.sizeFromShape(x.shape) % 4 === 0) {
        program = new ClipVec4Program(x.shape);
    }
    else {
        program = new ClipProgram(x.shape);
    }
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
export const clipByValueConfig = {
    kernelName: ClipByValue,
    backendName: 'webgpu',
    kernelFunc: clipByValue
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ2xpcEJ5VmFsdWUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL0NsaXBCeVZhbHVlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxXQUFXLEVBQTZFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBSW5JLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUNwRCxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsTUFBTSxVQUFVLFdBQVcsQ0FBQyxJQUkzQjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25CLE1BQU0sRUFBQyxZQUFZLEVBQUUsWUFBWSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRTNDLElBQUksT0FBb0MsQ0FBQztJQUN6QyxNQUFNLFdBQVcsR0FBRztRQUNsQixFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsWUFBWSxDQUFDLEVBQUM7UUFDdkMsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLFlBQVksQ0FBQyxFQUFDO0tBQ3hDLENBQUM7SUFDRixJQUFJLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLEVBQUU7UUFDekMsT0FBTyxHQUFHLElBQUksZUFBZSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztLQUN4QztTQUFNO1FBQ0wsT0FBTyxHQUFHLElBQUksV0FBVyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztLQUNwQztJQUNELE9BQU8sT0FBTyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDdEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGlCQUFpQixHQUFpQjtJQUM3QyxVQUFVLEVBQUUsV0FBVztJQUN2QixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsV0FBb0M7Q0FDakQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtDbGlwQnlWYWx1ZSwgQ2xpcEJ5VmFsdWVBdHRycywgQ2xpcEJ5VmFsdWVJbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5cbmltcG9ydCB7Q2xpcFZlYzRQcm9ncmFtfSBmcm9tICcuLi9jbGlwX3ZlYzRfd2ViZ3B1JztcbmltcG9ydCB7Q2xpcFByb2dyYW19IGZyb20gJy4uL2NsaXBfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGNsaXBCeVZhbHVlKGFyZ3M6IHtcbiAgaW5wdXRzOiBDbGlwQnlWYWx1ZUlucHV0cyxcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZCxcbiAgYXR0cnM6IENsaXBCeVZhbHVlQXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7Y2xpcFZhbHVlTWluLCBjbGlwVmFsdWVNYXh9ID0gYXR0cnM7XG5cbiAgbGV0IHByb2dyYW06IENsaXBQcm9ncmFtfENsaXBWZWM0UHJvZ3JhbTtcbiAgY29uc3QgdW5pZm9ybURhdGEgPSBbXG4gICAge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2NsaXBWYWx1ZU1pbl19LFxuICAgIHt0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFtjbGlwVmFsdWVNYXhdfVxuICBdO1xuICBpZiAodXRpbC5zaXplRnJvbVNoYXBlKHguc2hhcGUpICUgNCA9PT0gMCkge1xuICAgIHByb2dyYW0gPSBuZXcgQ2xpcFZlYzRQcm9ncmFtKHguc2hhcGUpO1xuICB9IGVsc2Uge1xuICAgIHByb2dyYW0gPSBuZXcgQ2xpcFByb2dyYW0oeC5zaGFwZSk7XG4gIH1cbiAgcmV0dXJuIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbeF0sIHguZHR5cGUsIHVuaWZvcm1EYXRhKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNsaXBCeVZhbHVlQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IENsaXBCeVZhbHVlLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGNsaXBCeVZhbHVlIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==