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
import { Fill, util } from '@tensorflow/tfjs-core';
import { FillProgram } from '../fill_webgpu';
export function fill(args) {
    const { backend, attrs } = args;
    const { shape, value } = attrs;
    let { dtype } = attrs;
    dtype = dtype || util.inferDtype(value);
    if (dtype === 'string') {
        // String type should be handled in CPU memory.
        const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
        values.fill(value);
        return backend.makeTensorInfo(shape, dtype, values);
    }
    else {
        const program = new FillProgram(shape);
        const uniformData = [{ type: 'float32', data: [value] }];
        return backend.runWebGPUProgram(program, [], dtype, uniformData);
    }
}
export const fillConfig = {
    kernelName: Fill,
    backendName: 'webgpu',
    kernelFunc: fill
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRmlsbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvRmlsbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsSUFBSSxFQUFtRCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUdsRyxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsTUFBTSxVQUFVLElBQUksQ0FBQyxJQUFnRDtJQUVuRSxNQUFNLEVBQUMsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUM5QixNQUFNLEVBQUMsS0FBSyxFQUFFLEtBQUssRUFBQyxHQUFHLEtBQUssQ0FBQztJQUM3QixJQUFJLEVBQUMsS0FBSyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRXBCLEtBQUssR0FBRyxLQUFLLElBQUksSUFBSSxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUV4QyxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7UUFDdEIsK0NBQStDO1FBQy9DLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1FBQ3hFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBZSxDQUFDLENBQUM7UUFDN0IsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7S0FDckQ7U0FBTTtRQUNMLE1BQU0sT0FBTyxHQUFHLElBQUksV0FBVyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3ZDLE1BQU0sV0FBVyxHQUFHLENBQUMsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLEtBQWUsQ0FBQyxFQUFDLENBQUMsQ0FBQztRQUNqRSxPQUFPLE9BQU8sQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUUsRUFBRSxFQUFFLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztLQUNsRTtBQUNILENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxVQUFVLEdBQWlCO0lBQ3RDLFVBQVUsRUFBRSxJQUFJO0lBQ2hCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxJQUE2QjtDQUMxQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0ZpbGwsIEZpbGxBdHRycywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBUZW5zb3JJbmZvLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7RmlsbFByb2dyYW19IGZyb20gJy4uL2ZpbGxfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGZpbGwoYXJnczoge2JhY2tlbmQ6IFdlYkdQVUJhY2tlbmQsIGF0dHJzOiBGaWxsQXR0cnN9KTpcbiAgICBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2JhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtzaGFwZSwgdmFsdWV9ID0gYXR0cnM7XG4gIGxldCB7ZHR5cGV9ID0gYXR0cnM7XG5cbiAgZHR5cGUgPSBkdHlwZSB8fCB1dGlsLmluZmVyRHR5cGUodmFsdWUpO1xuXG4gIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAvLyBTdHJpbmcgdHlwZSBzaG91bGQgYmUgaGFuZGxlZCBpbiBDUFUgbWVtb3J5LlxuICAgIGNvbnN0IHZhbHVlcyA9IHV0aWwuZ2V0QXJyYXlGcm9tRFR5cGUoZHR5cGUsIHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSkpO1xuICAgIHZhbHVlcy5maWxsKHZhbHVlIGFzIHN0cmluZyk7XG4gICAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8oc2hhcGUsIGR0eXBlLCB2YWx1ZXMpO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgRmlsbFByb2dyYW0oc2hhcGUpO1xuICAgIGNvbnN0IHVuaWZvcm1EYXRhID0gW3t0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFt2YWx1ZSBhcyBudW1iZXJdfV07XG4gICAgcmV0dXJuIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbXSwgZHR5cGUsIHVuaWZvcm1EYXRhKTtcbiAgfVxufVxuXG5leHBvcnQgY29uc3QgZmlsbENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBGaWxsLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGZpbGwgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19