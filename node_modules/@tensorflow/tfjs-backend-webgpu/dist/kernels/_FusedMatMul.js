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
import { _FusedMatMul } from '@tensorflow/tfjs-core';
import { batchMatMulImpl } from './BatchMatMul_impl';
export function _fusedMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b, bias, preluActivationWeights } = inputs;
    const { transposeA, transposeB, activation, leakyreluAlpha } = attrs;
    return batchMatMulImpl({
        a,
        b,
        transposeA,
        transposeB,
        backend,
        bias,
        preluActivationWeights,
        leakyreluAlpha,
        activation
    });
}
export const _fusedMatMulConfig = {
    kernelName: _FusedMatMul,
    backendName: 'webgpu',
    kernelFunc: _fusedMatMul,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiX0Z1c2VkTWF0TXVsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9fRnVzZWRNYXRNdWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBa0UsTUFBTSx1QkFBdUIsQ0FBQztBQUdwSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsTUFBTSxVQUFVLFlBQVksQ0FBQyxJQUk1QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxJQUFJLEVBQUUsc0JBQXNCLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDcEQsTUFBTSxFQUFDLFVBQVUsRUFBRSxVQUFVLEVBQUUsVUFBVSxFQUFFLGNBQWMsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUVuRSxPQUFPLGVBQWUsQ0FBQztRQUNyQixDQUFDO1FBQ0QsQ0FBQztRQUNELFVBQVU7UUFDVixVQUFVO1FBQ1YsT0FBTztRQUNQLElBQUk7UUFDSixzQkFBc0I7UUFDdEIsY0FBYztRQUNkLFVBQVU7S0FDWCxDQUFDLENBQUM7QUFDTCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sa0JBQWtCLEdBQWlCO0lBQzlDLFVBQVUsRUFBRSxZQUFZO0lBQ3hCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxZQUFxQztDQUNsRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge19GdXNlZE1hdE11bCwgX0Z1c2VkTWF0TXVsQXR0cnMsIF9GdXNlZE1hdE11bElucHV0cywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7YmF0Y2hNYXRNdWxJbXBsfSBmcm9tICcuL0JhdGNoTWF0TXVsX2ltcGwnO1xuXG5leHBvcnQgZnVuY3Rpb24gX2Z1c2VkTWF0TXVsKGFyZ3M6IHtcbiAgaW5wdXRzOiBfRnVzZWRNYXRNdWxJbnB1dHMsXG4gIGF0dHJzOiBfRnVzZWRNYXRNdWxBdHRycyxcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZFxufSkge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7YSwgYiwgYmlhcywgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0c30gPSBpbnB1dHM7XG4gIGNvbnN0IHt0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCLCBhY3RpdmF0aW9uLCBsZWFreXJlbHVBbHBoYX0gPSBhdHRycztcblxuICByZXR1cm4gYmF0Y2hNYXRNdWxJbXBsKHtcbiAgICBhLFxuICAgIGIsXG4gICAgdHJhbnNwb3NlQSxcbiAgICB0cmFuc3Bvc2VCLFxuICAgIGJhY2tlbmQsXG4gICAgYmlhcyxcbiAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLFxuICAgIGxlYWt5cmVsdUFscGhhLFxuICAgIGFjdGl2YXRpb25cbiAgfSk7XG59XG5cbmV4cG9ydCBjb25zdCBfZnVzZWRNYXRNdWxDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogX0Z1c2VkTWF0TXVsLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IF9mdXNlZE1hdE11bCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmMsXG59O1xuIl19