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
import { AvgPool3DGrad, backend_util } from '@tensorflow/tfjs-core';
import { AvgPool3DBackpropProgram } from '../avg_pool_backprop_webgpu';
export function avgPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const program = new AvgPool3DBackpropProgram(convInfo);
    const avgMultiplier = 1 / (convInfo.filterDepth * convInfo.filterHeight * convInfo.filterWidth);
    const uniformData = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth - 1 - convInfo.padInfo.front,
                convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
                convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
            ]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        },
        { type: 'int32', data: [convInfo.outDepth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'float32', data: [avgMultiplier] }
    ];
    return backend.runWebGPUProgram(program, [dy], x.dtype, uniformData);
}
export const avgPool3DGradConfig = {
    kernelName: AvgPool3DGrad,
    backendName: 'webgpu',
    kernelFunc: avgPool3DGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXZnUG9vbDNER3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvQXZnUG9vbDNER3JhZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsYUFBYSxFQUEyQyxZQUFZLEVBQXVDLE1BQU0sdUJBQXVCLENBQUM7QUFFakosT0FBTyxFQUFDLHdCQUF3QixFQUFDLE1BQU0sNkJBQTZCLENBQUM7QUFHckUsTUFBTSxVQUFVLGFBQWEsQ0FBQyxJQUk3QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsRUFBRSxFQUFFLEtBQUssRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMzQixNQUFNLENBQUMsR0FBRyxLQUFLLENBQUM7SUFDaEIsTUFBTSxFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUUxRCxNQUFNLFFBQVEsR0FBRyxZQUFZLENBQUMsaUJBQWlCLENBQzNDLENBQUMsQ0FBQyxLQUFpRCxFQUFFLFVBQVUsRUFBRSxPQUFPLEVBQ3hFLENBQUMsQ0FBQyxlQUFlLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQzdDLE1BQU0sT0FBTyxHQUFHLElBQUksd0JBQXdCLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDdkQsTUFBTSxhQUFhLEdBQ2YsQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsWUFBWSxHQUFHLFFBQVEsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUM5RSxNQUFNLFdBQVcsR0FBRztRQUNsQjtZQUNFLElBQUksRUFBRSxPQUFPO1lBQ2IsSUFBSSxFQUFFLENBQUMsUUFBUSxDQUFDLFdBQVcsRUFBRSxRQUFRLENBQUMsWUFBWSxFQUFFLFFBQVEsQ0FBQyxXQUFXLENBQUM7U0FDMUU7UUFDRDtZQUNFLElBQUksRUFBRSxPQUFPO1lBQ2IsSUFBSSxFQUFFO2dCQUNKLFFBQVEsQ0FBQyxvQkFBb0IsR0FBRyxDQUFDLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxLQUFLO2dCQUMxRCxRQUFRLENBQUMscUJBQXFCLEdBQUcsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUMsR0FBRztnQkFDekQsUUFBUSxDQUFDLG9CQUFvQixHQUFHLENBQUMsR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDLElBQUk7YUFDMUQ7U0FDRjtRQUNEO1lBQ0UsSUFBSSxFQUFFLE9BQU87WUFDYixJQUFJLEVBQUU7Z0JBQ0osUUFBUSxDQUFDLG9CQUFvQixFQUFFLFFBQVEsQ0FBQyxxQkFBcUI7Z0JBQzdELFFBQVEsQ0FBQyxvQkFBb0I7YUFDOUI7U0FDRjtRQUNELEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxJQUFJLEVBQUUsQ0FBQyxRQUFRLENBQUMsUUFBUSxDQUFDLEVBQUM7UUFDMUMsRUFBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBRSxDQUFDLFFBQVEsQ0FBQyxTQUFTLENBQUMsRUFBQztRQUMzQyxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsUUFBUSxDQUFDLFFBQVEsQ0FBQyxFQUFDO1FBQzFDLEVBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxhQUFhLENBQUMsRUFBQztLQUN6QyxDQUFDO0lBQ0YsT0FBTyxPQUFPLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztBQUN2RSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQWlCO0lBQy9DLFVBQVUsRUFBRSxhQUFhO0lBQ3pCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxhQUFzQztDQUNuRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0F2Z1Bvb2wzREdyYWQsIEF2Z1Bvb2wzREdyYWRBdHRycywgQXZnUG9vbDNER3JhZElucHV0cywgYmFja2VuZF91dGlsLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QXZnUG9vbDNEQmFja3Byb3BQcm9ncmFtfSBmcm9tICcuLi9hdmdfcG9vbF9iYWNrcHJvcF93ZWJncHUnO1xuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5cbmV4cG9ydCBmdW5jdGlvbiBhdmdQb29sM0RHcmFkKGFyZ3M6IHtcbiAgaW5wdXRzOiBBdmdQb29sM0RHcmFkSW5wdXRzLFxuICBiYWNrZW5kOiBXZWJHUFVCYWNrZW5kLFxuICBhdHRyczogQXZnUG9vbDNER3JhZEF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtkeSwgaW5wdXR9ID0gaW5wdXRzO1xuICBjb25zdCB4ID0gaW5wdXQ7XG4gIGNvbnN0IHtmaWx0ZXJTaXplLCBzdHJpZGVzLCBwYWQsIGRpbVJvdW5kaW5nTW9kZX0gPSBhdHRycztcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlUG9vbDNESW5mbyhcbiAgICAgIHguc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmlsdGVyU2l6ZSwgc3RyaWRlcyxcbiAgICAgIDEgLyogZGlsYXRpb25zICovLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgQXZnUG9vbDNEQmFja3Byb3BQcm9ncmFtKGNvbnZJbmZvKTtcbiAgY29uc3QgYXZnTXVsdGlwbGllciA9XG4gICAgICAxIC8gKGNvbnZJbmZvLmZpbHRlckRlcHRoICogY29udkluZm8uZmlsdGVySGVpZ2h0ICogY29udkluZm8uZmlsdGVyV2lkdGgpO1xuICBjb25zdCB1bmlmb3JtRGF0YSA9IFtcbiAgICB7XG4gICAgICB0eXBlOiAnaW50MzInLFxuICAgICAgZGF0YTogW2NvbnZJbmZvLnN0cmlkZURlcHRoLCBjb252SW5mby5zdHJpZGVIZWlnaHQsIGNvbnZJbmZvLnN0cmlkZVdpZHRoXVxuICAgIH0sXG4gICAge1xuICAgICAgdHlwZTogJ2ludDMyJyxcbiAgICAgIGRhdGE6IFtcbiAgICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVyRGVwdGggLSAxIC0gY29udkluZm8ucGFkSW5mby5mcm9udCxcbiAgICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0IC0gMSAtIGNvbnZJbmZvLnBhZEluZm8udG9wLFxuICAgICAgICBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJXaWR0aCAtIDEgLSBjb252SW5mby5wYWRJbmZvLmxlZnRcbiAgICAgIF1cbiAgICB9LFxuICAgIHtcbiAgICAgIHR5cGU6ICdpbnQzMicsXG4gICAgICBkYXRhOiBbXG4gICAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckRlcHRoLCBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJIZWlnaHQsXG4gICAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlcldpZHRoXG4gICAgICBdXG4gICAgfSxcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2NvbnZJbmZvLm91dERlcHRoXX0sXG4gICAge3R5cGU6ICdpbnQzMicsIGRhdGE6IFtjb252SW5mby5vdXRIZWlnaHRdfSxcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2NvbnZJbmZvLm91dFdpZHRoXX0sXG4gICAge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2F2Z011bHRpcGxpZXJdfVxuICBdO1xuICByZXR1cm4gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKHByb2dyYW0sIFtkeV0sIHguZHR5cGUsIHVuaWZvcm1EYXRhKTtcbn1cblxuZXhwb3J0IGNvbnN0IGF2Z1Bvb2wzREdyYWRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQXZnUG9vbDNER3JhZCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBhdmdQb29sM0RHcmFkIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==