/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import { backend_util } from '@tensorflow/tfjs-core';
import { CumProgram } from '../cum_webgpu';
import { identity } from './Identity';
import { transpose } from './Transpose';
export function cumImpl(op, x, backend, axis, exclusive, reverse) {
    const xRank = x.shape.length;
    const permutation = backend_util.getAxesPermutation([axis], xRank);
    let permutedX = x;
    if (permutation != null) {
        permutedX = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
    }
    const permutedAxis = backend_util.getInnerMostAxes(1, xRank)[0];
    if (permutedAxis !== xRank - 1) {
        throw new Error(`WebGPU cumprod shader expects an inner-most axis=${x.shape.length - 1} ` +
            `but got axis=${axis}`);
    }
    const size = permutedX.shape[permutedAxis];
    let result = identity({ inputs: { x: permutedX }, backend });
    // Use cum parallel algorithm, inspired by:
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    // Note: although the algorithm is called sum, it works for any associtative
    // operator with an identity.
    for (let i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
        const program = new CumProgram(op, permutedX.shape, false, reverse);
        const prevResult = result;
        const uniformData = [{ type: 'float32', data: [i] }];
        result =
            backend.runWebGPUProgram(program, [result], result.dtype, uniformData);
        backend.disposeData(prevResult.dataId);
    }
    // For exclusive cum, shift the end result in the direction of product or sum
    // and add 1 for product or 0 for sum to the front index.
    if (exclusive) {
        const program = new CumProgram(op, permutedX.shape, exclusive, reverse);
        const prevResult = result;
        const uniformData = [{ type: 'float32', data: [0] }];
        result =
            backend.runWebGPUProgram(program, [result], result.dtype, uniformData);
        backend.disposeData(prevResult.dataId);
    }
    if (permutation != null) {
        const reversePermutation = backend_util.getUndoAxesPermutation(permutation);
        const reverseTransposedResult = transpose({ inputs: { x: result }, backend, attrs: { perm: reversePermutation } });
        backend.disposeData(result.dataId);
        backend.disposeData(permutedX.dataId);
        return reverseTransposedResult;
    }
    return result;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ3VtX2ltcGwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL0N1bV9pbXBsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQWEsTUFBTSx1QkFBdUIsQ0FBQztBQUcvRCxPQUFPLEVBQVksVUFBVSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRXBELE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDcEMsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0QyxNQUFNLFVBQVUsT0FBTyxDQUNuQixFQUFhLEVBQUUsQ0FBYSxFQUFFLE9BQXNCLEVBQUUsSUFBWSxFQUNsRSxTQUFrQixFQUFFLE9BQWdCO0lBQ3RDLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBQzdCLE1BQU0sV0FBVyxHQUFHLFlBQVksQ0FBQyxrQkFBa0IsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ25FLElBQUksU0FBUyxHQUFHLENBQUMsQ0FBQztJQUNsQixJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7UUFDdkIsU0FBUyxHQUFHLFNBQVMsQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUUsV0FBVyxFQUFDLEVBQUMsQ0FBQyxDQUFDO0tBQzNFO0lBQ0QsTUFBTSxZQUFZLEdBQUcsWUFBWSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUVoRSxJQUFJLFlBQVksS0FBSyxLQUFLLEdBQUcsQ0FBQyxFQUFFO1FBQzlCLE1BQU0sSUFBSSxLQUFLLENBQ1gsb0RBQ0ksQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxHQUFHO1lBQ3pCLGdCQUFnQixJQUFJLEVBQUUsQ0FBQyxDQUFDO0tBQzdCO0lBQ0QsTUFBTSxJQUFJLEdBQUcsU0FBUyxDQUFDLEtBQUssQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUMzQyxJQUFJLE1BQU0sR0FBRyxRQUFRLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsU0FBUyxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztJQUN6RCwyQ0FBMkM7SUFDM0MsK0dBQStHO0lBQy9HLDRFQUE0RTtJQUM1RSw2QkFBNkI7SUFFN0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUN4RCxNQUFNLE9BQU8sR0FBRyxJQUFJLFVBQVUsQ0FBQyxFQUFFLEVBQUUsU0FBUyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDcEUsTUFBTSxVQUFVLEdBQUcsTUFBTSxDQUFDO1FBQzFCLE1BQU0sV0FBVyxHQUFHLENBQUMsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQztRQUNuRCxNQUFNO1lBQ0YsT0FBTyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxDQUFDLE1BQU0sQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7UUFDM0UsT0FBTyxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7S0FDeEM7SUFDRCw2RUFBNkU7SUFDN0UseURBQXlEO0lBQ3pELElBQUksU0FBUyxFQUFFO1FBQ2IsTUFBTSxPQUFPLEdBQUcsSUFBSSxVQUFVLENBQUMsRUFBRSxFQUFFLFNBQVMsQ0FBQyxLQUFLLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3hFLE1BQU0sVUFBVSxHQUFHLE1BQU0sQ0FBQztRQUMxQixNQUFNLFdBQVcsR0FBRyxDQUFDLEVBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDLENBQUM7UUFDbkQsTUFBTTtZQUNGLE9BQU8sQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxNQUFNLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxFQUFFLFdBQVcsQ0FBQyxDQUFDO1FBQzNFLE9BQU8sQ0FBQyxXQUFXLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0tBQ3hDO0lBRUQsSUFBSSxXQUFXLElBQUksSUFBSSxFQUFFO1FBQ3ZCLE1BQU0sa0JBQWtCLEdBQUcsWUFBWSxDQUFDLHNCQUFzQixDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzVFLE1BQU0sdUJBQXVCLEdBQUcsU0FBUyxDQUNyQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxNQUFNLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsSUFBSSxFQUFFLGtCQUFrQixFQUFDLEVBQUMsQ0FBQyxDQUFDO1FBRXZFLE9BQU8sQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ25DLE9BQU8sQ0FBQyxXQUFXLENBQUMsU0FBUyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRXRDLE9BQU8sdUJBQXVCLENBQUM7S0FDaEM7SUFFRCxPQUFPLE1BQU0sQ0FBQztBQUNoQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5pbXBvcnQge0N1bU9wVHlwZSwgQ3VtUHJvZ3JhbX0gZnJvbSAnLi4vY3VtX3dlYmdwdSc7XG5cbmltcG9ydCB7aWRlbnRpdHl9IGZyb20gJy4vSWRlbnRpdHknO1xuaW1wb3J0IHt0cmFuc3Bvc2V9IGZyb20gJy4vVHJhbnNwb3NlJztcblxuZXhwb3J0IGZ1bmN0aW9uIGN1bUltcGwoXG4gICAgb3A6IEN1bU9wVHlwZSwgeDogVGVuc29ySW5mbywgYmFja2VuZDogV2ViR1BVQmFja2VuZCwgYXhpczogbnVtYmVyLFxuICAgIGV4Y2x1c2l2ZTogYm9vbGVhbiwgcmV2ZXJzZTogYm9vbGVhbik6IFRlbnNvckluZm8ge1xuICBjb25zdCB4UmFuayA9IHguc2hhcGUubGVuZ3RoO1xuICBjb25zdCBwZXJtdXRhdGlvbiA9IGJhY2tlbmRfdXRpbC5nZXRBeGVzUGVybXV0YXRpb24oW2F4aXNdLCB4UmFuayk7XG4gIGxldCBwZXJtdXRlZFggPSB4O1xuICBpZiAocGVybXV0YXRpb24gIT0gbnVsbCkge1xuICAgIHBlcm11dGVkWCA9IHRyYW5zcG9zZSh7aW5wdXRzOiB7eH0sIGJhY2tlbmQsIGF0dHJzOiB7cGVybTogcGVybXV0YXRpb259fSk7XG4gIH1cbiAgY29uc3QgcGVybXV0ZWRBeGlzID0gYmFja2VuZF91dGlsLmdldElubmVyTW9zdEF4ZXMoMSwgeFJhbmspWzBdO1xuXG4gIGlmIChwZXJtdXRlZEF4aXMgIT09IHhSYW5rIC0gMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYFdlYkdQVSBjdW1wcm9kIHNoYWRlciBleHBlY3RzIGFuIGlubmVyLW1vc3QgYXhpcz0ke1xuICAgICAgICAgICAgeC5zaGFwZS5sZW5ndGggLSAxfSBgICtcbiAgICAgICAgYGJ1dCBnb3QgYXhpcz0ke2F4aXN9YCk7XG4gIH1cbiAgY29uc3Qgc2l6ZSA9IHBlcm11dGVkWC5zaGFwZVtwZXJtdXRlZEF4aXNdO1xuICBsZXQgcmVzdWx0ID0gaWRlbnRpdHkoe2lucHV0czoge3g6IHBlcm11dGVkWH0sIGJhY2tlbmR9KTtcbiAgLy8gVXNlIGN1bSBwYXJhbGxlbCBhbGdvcml0aG0sIGluc3BpcmVkIGJ5OlxuICAvLyBodHRwczovL2RldmVsb3Blci5udmlkaWEuY29tL2dwdWdlbXMvZ3B1Z2VtczMvcGFydC12aS1ncHUtY29tcHV0aW5nL2NoYXB0ZXItMzktcGFyYWxsZWwtcHJlZml4LXN1bS1zY2FuLWN1ZGFcbiAgLy8gTm90ZTogYWx0aG91Z2ggdGhlIGFsZ29yaXRobSBpcyBjYWxsZWQgc3VtLCBpdCB3b3JrcyBmb3IgYW55IGFzc29jaXRhdGl2ZVxuICAvLyBvcGVyYXRvciB3aXRoIGFuIGlkZW50aXR5LlxuXG4gIGZvciAobGV0IGkgPSAwOyBpIDw9IE1hdGguY2VpbChNYXRoLmxvZzIoc2l6ZSkpIC0gMTsgaSsrKSB7XG4gICAgY29uc3QgcHJvZ3JhbSA9IG5ldyBDdW1Qcm9ncmFtKG9wLCBwZXJtdXRlZFguc2hhcGUsIGZhbHNlLCByZXZlcnNlKTtcbiAgICBjb25zdCBwcmV2UmVzdWx0ID0gcmVzdWx0O1xuICAgIGNvbnN0IHVuaWZvcm1EYXRhID0gW3t0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFtpXX1dO1xuICAgIHJlc3VsdCA9XG4gICAgICAgIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbcmVzdWx0XSwgcmVzdWx0LmR0eXBlLCB1bmlmb3JtRGF0YSk7XG4gICAgYmFja2VuZC5kaXNwb3NlRGF0YShwcmV2UmVzdWx0LmRhdGFJZCk7XG4gIH1cbiAgLy8gRm9yIGV4Y2x1c2l2ZSBjdW0sIHNoaWZ0IHRoZSBlbmQgcmVzdWx0IGluIHRoZSBkaXJlY3Rpb24gb2YgcHJvZHVjdCBvciBzdW1cbiAgLy8gYW5kIGFkZCAxIGZvciBwcm9kdWN0IG9yIDAgZm9yIHN1bSB0byB0aGUgZnJvbnQgaW5kZXguXG4gIGlmIChleGNsdXNpdmUpIHtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IEN1bVByb2dyYW0ob3AsIHBlcm11dGVkWC5zaGFwZSwgZXhjbHVzaXZlLCByZXZlcnNlKTtcbiAgICBjb25zdCBwcmV2UmVzdWx0ID0gcmVzdWx0O1xuICAgIGNvbnN0IHVuaWZvcm1EYXRhID0gW3t0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFswXX1dO1xuICAgIHJlc3VsdCA9XG4gICAgICAgIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbcmVzdWx0XSwgcmVzdWx0LmR0eXBlLCB1bmlmb3JtRGF0YSk7XG4gICAgYmFja2VuZC5kaXNwb3NlRGF0YShwcmV2UmVzdWx0LmRhdGFJZCk7XG4gIH1cblxuICBpZiAocGVybXV0YXRpb24gIT0gbnVsbCkge1xuICAgIGNvbnN0IHJldmVyc2VQZXJtdXRhdGlvbiA9IGJhY2tlbmRfdXRpbC5nZXRVbmRvQXhlc1Blcm11dGF0aW9uKHBlcm11dGF0aW9uKTtcbiAgICBjb25zdCByZXZlcnNlVHJhbnNwb3NlZFJlc3VsdCA9IHRyYW5zcG9zZShcbiAgICAgICAge2lucHV0czoge3g6IHJlc3VsdH0sIGJhY2tlbmQsIGF0dHJzOiB7cGVybTogcmV2ZXJzZVBlcm11dGF0aW9ufX0pO1xuXG4gICAgYmFja2VuZC5kaXNwb3NlRGF0YShyZXN1bHQuZGF0YUlkKTtcbiAgICBiYWNrZW5kLmRpc3Bvc2VEYXRhKHBlcm11dGVkWC5kYXRhSWQpO1xuXG4gICAgcmV0dXJuIHJldmVyc2VUcmFuc3Bvc2VkUmVzdWx0O1xuICB9XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cbiJdfQ==