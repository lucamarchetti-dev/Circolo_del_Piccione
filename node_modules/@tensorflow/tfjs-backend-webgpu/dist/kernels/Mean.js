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
import { Mean } from '@tensorflow/tfjs-core';
import { reduce } from '../kernel_utils/reduce';
export function mean(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { keepDims, axis } = attrs;
    return reduce(x, axis, keepDims, 'mean', backend);
}
export const meanConfig = {
    kernelName: Mean,
    backendName: 'webgpu',
    kernelFunc: mean
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWVhbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvTWVhbi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLElBQUksRUFBb0MsTUFBTSx1QkFBdUIsQ0FBQztBQUd4RyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFFOUMsTUFBTSxVQUFVLElBQUksQ0FDaEIsSUFBb0U7SUFFdEUsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFL0IsT0FBTyxNQUFNLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0FBQ3BELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxVQUFVLEdBQWlCO0lBQ3RDLFVBQVUsRUFBRSxJQUFJO0lBQ2hCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxJQUE2QjtDQUMxQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgTWVhbiwgTWVhbkF0dHJzLCBNZWFuSW5wdXRzLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7cmVkdWNlfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvcmVkdWNlJztcblxuZXhwb3J0IGZ1bmN0aW9uIG1lYW4oXG4gICAgYXJnczoge2lucHV0czogTWVhbklucHV0cywgYXR0cnM6IE1lYW5BdHRycywgYmFja2VuZDogV2ViR1BVQmFja2VuZH0pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtrZWVwRGltcywgYXhpc30gPSBhdHRycztcblxuICByZXR1cm4gcmVkdWNlKHgsIGF4aXMsIGtlZXBEaW1zLCAnbWVhbicsIGJhY2tlbmQpO1xufVxuXG5leHBvcnQgY29uc3QgbWVhbkNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBNZWFuLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IG1lYW4gYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19