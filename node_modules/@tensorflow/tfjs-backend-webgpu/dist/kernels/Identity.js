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
import { Identity } from '@tensorflow/tfjs-core';
export function identity(args) {
    const { inputs } = args;
    const { x } = inputs;
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: x.shape, dtype: x.dtype };
}
export const identityConfig = {
    kernelName: Identity,
    backendName: 'webgpu',
    kernelFunc: identity
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiSWRlbnRpdHkuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL0lkZW50aXR5LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxRQUFRLEVBQXVELE1BQU0sdUJBQXVCLENBQUM7QUFHckcsTUFBTSxVQUFVLFFBQVEsQ0FDcEIsSUFBc0Q7SUFDeEQsTUFBTSxFQUFDLE1BQU0sRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QixNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBRW5CLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUM5QixPQUFPLEVBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUMsQ0FBQztBQUM1RCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sY0FBYyxHQUFpQjtJQUMxQyxVQUFVLEVBQUUsUUFBUTtJQUNwQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsUUFBaUM7Q0FDOUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtJZGVudGl0eSwgSWRlbnRpdHlJbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuXG5leHBvcnQgZnVuY3Rpb24gaWRlbnRpdHkoXG4gICAgYXJnczoge2lucHV0czogSWRlbnRpdHlJbnB1dHMsIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmR9KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuXG4gIGFyZ3MuYmFja2VuZC5pbmNSZWYoeC5kYXRhSWQpO1xuICByZXR1cm4ge2RhdGFJZDogeC5kYXRhSWQsIHNoYXBlOiB4LnNoYXBlLCBkdHlwZTogeC5kdHlwZX07XG59XG5cbmV4cG9ydCBjb25zdCBpZGVudGl0eUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBJZGVudGl0eSxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBpZGVudGl0eSBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=