/**
 * @license
 * Copyright 2023 Google LLC.
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
import { backend_util, Conv3D, upcastType } from '@tensorflow/tfjs-core';
import { Conv3DNaiveProgram } from '../conv3d_naive_webgpu';
export function conv3D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations } = attrs;
    const convInfo = backend_util.computeConv3DInfo(x.shape, filter.shape, strides, dilations, pad);
    const padInfo = [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
        {
            type: 'int32',
            data: [convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth]
        },
        { type: 'int32', data: [...padInfo] }, {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth
            ]
        }
    ];
    const program = new Conv3DNaiveProgram(convInfo);
    const dtype = upcastType(x.dtype, filter.dtype);
    return backend.runWebGPUProgram(program, [x, filter], dtype, dimensions);
}
export const conv3DConfig = {
    kernelName: Conv3D,
    backendName: 'webgpu',
    kernelFunc: conv3D,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29udjNELmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9Db252M0QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxNQUFNLEVBQXVELFVBQVUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRzVILE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBRTFELE1BQU0sVUFBVSxNQUFNLENBQ2xCLElBQXdFO0lBQzFFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMzQixNQUFNLEVBQUMsT0FBTyxFQUFFLEdBQUcsRUFBRSxTQUFTLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFeEMsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLGlCQUFpQixDQUMzQyxDQUFDLENBQUMsS0FBaUQsRUFDbkQsTUFBTSxDQUFDLEtBQWlELEVBQUUsT0FBTyxFQUNqRSxTQUFTLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFFcEIsTUFBTSxPQUFPLEdBQ1QsQ0FBQyxRQUFRLENBQUMsT0FBTyxDQUFDLEtBQUssRUFBRSxRQUFRLENBQUMsT0FBTyxDQUFDLEdBQUcsRUFBRSxRQUFRLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzFFLE1BQU0sVUFBVSxHQUFHO1FBQ2pCO1lBQ0UsSUFBSSxFQUFFLE9BQU87WUFDYixJQUFJLEVBQUUsQ0FBQyxRQUFRLENBQUMsV0FBVyxFQUFFLFFBQVEsQ0FBQyxZQUFZLEVBQUUsUUFBUSxDQUFDLFdBQVcsQ0FBQztTQUMxRTtRQUNELEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxJQUFJLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxFQUFDLEVBQUU7WUFDbkMsSUFBSSxFQUFFLE9BQU87WUFDYixJQUFJLEVBQUUsQ0FBQyxRQUFRLENBQUMsV0FBVyxFQUFFLFFBQVEsQ0FBQyxZQUFZLEVBQUUsUUFBUSxDQUFDLFdBQVcsQ0FBQztTQUMxRTtRQUNEO1lBQ0UsSUFBSSxFQUFFLE9BQU87WUFDYixJQUFJLEVBQUU7Z0JBQ0osUUFBUSxDQUFDLGFBQWEsRUFBRSxRQUFRLENBQUMsY0FBYyxFQUFFLFFBQVEsQ0FBQyxhQUFhO2FBQ3hFO1NBQ0Y7S0FDRixDQUFDO0lBQ0YsTUFBTSxPQUFPLEdBQUcsSUFBSSxrQkFBa0IsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNqRCxNQUFNLEtBQUssR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDaEQsT0FBTyxPQUFPLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxFQUFFLEtBQUssRUFBRSxVQUFVLENBQUMsQ0FBQztBQUMzRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFpQjtJQUN4QyxVQUFVLEVBQUUsTUFBTTtJQUNsQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsTUFBMEI7Q0FDdkMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIENvbnYzRCwgQ29udjNEQXR0cnMsIENvbnYzRElucHV0cywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCB1cGNhc3RUeXBlfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7Q29udjNETmFpdmVQcm9ncmFtfSBmcm9tICcuLi9jb252M2RfbmFpdmVfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGNvbnYzRChcbiAgICBhcmdzOiB7aW5wdXRzOiBDb252M0RJbnB1dHMsIGF0dHJzOiBDb252M0RBdHRycywgYmFja2VuZDogV2ViR1BVQmFja2VuZH0pIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3gsIGZpbHRlcn0gPSBpbnB1dHM7XG4gIGNvbnN0IHtzdHJpZGVzLCBwYWQsIGRpbGF0aW9uc30gPSBhdHRycztcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlQ29udjNESW5mbyhcbiAgICAgIHguc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIGZpbHRlci5zaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBzdHJpZGVzLFxuICAgICAgZGlsYXRpb25zLCBwYWQpO1xuXG4gIGNvbnN0IHBhZEluZm8gPVxuICAgICAgW2NvbnZJbmZvLnBhZEluZm8uZnJvbnQsIGNvbnZJbmZvLnBhZEluZm8udG9wLCBjb252SW5mby5wYWRJbmZvLmxlZnRdO1xuICBjb25zdCBkaW1lbnNpb25zID0gW1xuICAgIHtcbiAgICAgIHR5cGU6ICdpbnQzMicsXG4gICAgICBkYXRhOiBbY29udkluZm8uZmlsdGVyRGVwdGgsIGNvbnZJbmZvLmZpbHRlckhlaWdodCwgY29udkluZm8uZmlsdGVyV2lkdGhdXG4gICAgfSxcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogWy4uLnBhZEluZm9dfSwge1xuICAgICAgdHlwZTogJ2ludDMyJyxcbiAgICAgIGRhdGE6IFtjb252SW5mby5zdHJpZGVEZXB0aCwgY29udkluZm8uc3RyaWRlSGVpZ2h0LCBjb252SW5mby5zdHJpZGVXaWR0aF1cbiAgICB9LFxuICAgIHtcbiAgICAgIHR5cGU6ICdpbnQzMicsXG4gICAgICBkYXRhOiBbXG4gICAgICAgIGNvbnZJbmZvLmRpbGF0aW9uRGVwdGgsIGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LCBjb252SW5mby5kaWxhdGlvbldpZHRoXG4gICAgICBdXG4gICAgfVxuICBdO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IENvbnYzRE5haXZlUHJvZ3JhbShjb252SW5mbyk7XG4gIGNvbnN0IGR0eXBlID0gdXBjYXN0VHlwZSh4LmR0eXBlLCBmaWx0ZXIuZHR5cGUpO1xuICByZXR1cm4gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKHByb2dyYW0sIFt4LCBmaWx0ZXJdLCBkdHlwZSwgZGltZW5zaW9ucyk7XG59XG5cbmV4cG9ydCBjb25zdCBjb252M0RDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQ29udjNELFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGNvbnYzRCBhcyB7fSBhcyBLZXJuZWxGdW5jLFxufTtcbiJdfQ==