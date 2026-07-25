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
import { Reshape, util } from '@tensorflow/tfjs-core';
export function reshape(args) {
    const { inputs, attrs } = args;
    const { x } = inputs;
    const { shape } = attrs;
    const xSize = util.sizeFromShape(x.shape);
    const $shape = util.inferFromImplicitShape(shape, xSize);
    const $xSize = util.sizeFromShape($shape);
    util.assert(xSize === $xSize, () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
        `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
        `shape must have the same number of elements.`);
    // Backend needs to track refCount for the dataId for reshape op
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: $shape, dtype: x.dtype };
}
export const reshapeConfig = {
    kernelName: Reshape,
    backendName: 'webgpu',
    kernelFunc: reshape
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUmVzaGFwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvUmVzaGFwZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLE9BQU8sRUFBMkMsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFJdkgsTUFBTSxVQUFVLE9BQU8sQ0FDbkIsSUFBMEU7SUFFNUUsTUFBTSxFQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDN0IsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUNuQixNQUFNLEVBQUMsS0FBSyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRXRCLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzFDLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDekQsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUUxQyxJQUFJLENBQUMsTUFBTSxDQUNQLEtBQUssS0FBSyxNQUFNLEVBQ2hCLEdBQUcsRUFBRSxDQUFDLGtCQUFrQixNQUFNLFNBQVMsTUFBTSx3QkFBd0I7UUFDakUsVUFBVSxDQUFDLENBQUMsS0FBSyxTQUFTLEtBQUssbUNBQW1DO1FBQ2xFLDhDQUE4QyxDQUFDLENBQUM7SUFFeEQsZ0VBQWdFO0lBQ2hFLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUM5QixPQUFPLEVBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBQyxDQUFDO0FBQzNELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQWlCO0lBQ3pDLFVBQVUsRUFBRSxPQUFPO0lBQ25CLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxPQUFnQztDQUM3QyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgUmVzaGFwZSwgUmVzaGFwZUF0dHJzLCBSZXNoYXBlSW5wdXRzLCBUZW5zb3JJbmZvLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIHJlc2hhcGUoXG4gICAgYXJnczoge2lucHV0czogUmVzaGFwZUlucHV0cywgYmFja2VuZDogV2ViR1BVQmFja2VuZCwgYXR0cnM6IFJlc2hhcGVBdHRyc30pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtzaGFwZX0gPSBhdHRycztcblxuICBjb25zdCB4U2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZSh4LnNoYXBlKTtcbiAgY29uc3QgJHNoYXBlID0gdXRpbC5pbmZlckZyb21JbXBsaWNpdFNoYXBlKHNoYXBlLCB4U2l6ZSk7XG4gIGNvbnN0ICR4U2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZSgkc2hhcGUpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeFNpemUgPT09ICR4U2l6ZSxcbiAgICAgICgpID0+IGBUaGUgbmV3IHNoYXBlICgkeyRzaGFwZX0pIGhhcyAkeyR4U2l6ZX0gZWxlbWVudHMgYW5kIHRoZSBvbGQgYCArXG4gICAgICAgICAgYHNoYXBlICgke3guc2hhcGV9KSBoYXMgJHt4U2l6ZX0gZWxlbWVudHMuIFRoZSBuZXcgc2hhcGUgYW5kIG9sZCBgICtcbiAgICAgICAgICBgc2hhcGUgbXVzdCBoYXZlIHRoZSBzYW1lIG51bWJlciBvZiBlbGVtZW50cy5gKTtcblxuICAvLyBCYWNrZW5kIG5lZWRzIHRvIHRyYWNrIHJlZkNvdW50IGZvciB0aGUgZGF0YUlkIGZvciByZXNoYXBlIG9wXG4gIGFyZ3MuYmFja2VuZC5pbmNSZWYoeC5kYXRhSWQpO1xuICByZXR1cm4ge2RhdGFJZDogeC5kYXRhSWQsIHNoYXBlOiAkc2hhcGUsIGR0eXBlOiB4LmR0eXBlfTtcbn1cblxuZXhwb3J0IGNvbnN0IHJlc2hhcGVDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogUmVzaGFwZSxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiByZXNoYXBlIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==