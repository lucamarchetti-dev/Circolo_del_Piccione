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
import { backend_util, Einsum, util } from '@tensorflow/tfjs-core';
import { multiplyKernelFunc } from './Multiply';
import { reshape } from './Reshape';
import { sum } from './Sum';
import { transpose } from './Transpose';
export function einsum(args) {
    const { inputs, backend, attrs } = args;
    const { equation } = attrs;
    const tensors = inputs;
    const { allDims, summedDims, idDims } = backend_util.decodeEinsumEquation(equation, tensors.length);
    backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
    const { path, steps } = backend_util.getEinsumComputePath(summedDims, idDims);
    const nSteps = steps.length;
    let out = null;
    let numDimsRemaining = allDims.length;
    const tensorsToDispose = [];
    for (let i = 0; i < nSteps; ++i) {
        for (const idTerm of steps[i]) {
            const { permutationIndices: perm, expandDims: dimsToExpand } = backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]);
            let x;
            if (backend_util.isIdentityPermutation(perm)) {
                x = tensors[idTerm];
            }
            else {
                x = transpose({ inputs: { x: tensors[idTerm] }, backend, attrs: { perm } });
                tensorsToDispose.push(x);
            }
            const targetShape = x.shape.slice();
            for (let k = 0; k < dimsToExpand.length; ++k) {
                targetShape.splice(dimsToExpand[k], 0, 1);
            }
            if (!util.arraysEqual(x.shape, targetShape)) {
                x = reshape({ inputs: { x }, backend, attrs: { shape: targetShape } });
                tensorsToDispose.push(x);
            }
            if (out === null) {
                out = x;
            }
            else {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                out =
                    multiplyKernelFunc({ inputs: { a: x, b: out }, backend });
                tensorsToDispose.push(out);
            }
        }
        if (i < nSteps - 1) {
            if (path[i] >= 0) {
                out = sum({
                    inputs: { x: out },
                    backend,
                    attrs: {
                        axis: path[i] - (allDims.length - numDimsRemaining),
                        keepDims: false
                    }
                });
                tensorsToDispose.push(out);
            }
            numDimsRemaining--;
        }
    }
    // Clean up intermediate tensors.
    for (const tensorInfo of tensorsToDispose) {
        if (tensorInfo === out) {
            continue;
        }
        backend.disposeData(tensorInfo.dataId);
    }
    return out;
}
export const einsumConfig = {
    kernelName: Einsum,
    backendName: 'webgpu',
    kernelFunc: einsum
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRWluc3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9FaW5zdW0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxNQUFNLEVBQTJFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBSTFJLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUM5QyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0QyxNQUFNLFVBQVUsTUFBTSxDQUNsQixJQUF3RTtJQUUxRSxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLFFBQVEsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUN6QixNQUFNLE9BQU8sR0FBRyxNQUFrQixDQUFDO0lBRW5DLE1BQU0sRUFBQyxPQUFPLEVBQUUsVUFBVSxFQUFFLE1BQU0sRUFBQyxHQUMvQixZQUFZLENBQUMsb0JBQW9CLENBQUMsUUFBUSxFQUFFLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUNoRSxZQUFZLENBQUMsbUJBQW1CLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDbEUsTUFBTSxFQUFDLElBQUksRUFBRSxLQUFLLEVBQUMsR0FBRyxZQUFZLENBQUMsb0JBQW9CLENBQUMsVUFBVSxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBRTVFLE1BQU0sTUFBTSxHQUFHLEtBQUssQ0FBQyxNQUFNLENBQUM7SUFDNUIsSUFBSSxHQUFHLEdBQW9CLElBQUksQ0FBQztJQUNoQyxJQUFJLGdCQUFnQixHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUM7SUFDdEMsTUFBTSxnQkFBZ0IsR0FBaUIsRUFBRSxDQUFDO0lBQzFDLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDL0IsS0FBSyxNQUFNLE1BQU0sSUFBSSxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUU7WUFDN0IsTUFBTSxFQUFDLGtCQUFrQixFQUFFLElBQUksRUFBRSxVQUFVLEVBQUUsWUFBWSxFQUFDLEdBQ3RELFlBQVksQ0FBQyxvQkFBb0IsQ0FBQyxnQkFBZ0IsRUFBRSxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztZQUN4RSxJQUFJLENBQWEsQ0FBQztZQUNsQixJQUFJLFlBQVksQ0FBQyxxQkFBcUIsQ0FBQyxJQUFJLENBQUMsRUFBRTtnQkFDNUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQzthQUNyQjtpQkFBTTtnQkFDTCxDQUFDLEdBQUcsU0FBUyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxNQUFNLENBQUMsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUMsRUFBQyxDQUFDLENBQUM7Z0JBQ3RFLGdCQUFnQixDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMxQjtZQUNELE1BQU0sV0FBVyxHQUFhLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLENBQUM7WUFDOUMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFlBQVksQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7Z0JBQzVDLFdBQVcsQ0FBQyxNQUFNLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQzthQUMzQztZQUVELElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLEVBQUU7Z0JBQzNDLENBQUMsR0FBRyxPQUFPLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLFdBQVcsRUFBQyxFQUFDLENBQUMsQ0FBQztnQkFDakUsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzFCO1lBQ0QsSUFBSSxHQUFHLEtBQUssSUFBSSxFQUFFO2dCQUNoQixHQUFHLEdBQUcsQ0FBQyxDQUFDO2FBQ1Q7aUJBQU07Z0JBQ0wsMERBQTBEO2dCQUMxRCxHQUFHO29CQUNDLGtCQUFrQixDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsR0FBRyxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQWUsQ0FBQztnQkFDeEUsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDO2FBQzVCO1NBQ0Y7UUFDRCxJQUFJLENBQUMsR0FBRyxNQUFNLEdBQUcsQ0FBQyxFQUFFO1lBQ2xCLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsRUFBRTtnQkFDaEIsR0FBRyxHQUFHLEdBQUcsQ0FBQztvQkFDUixNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsR0FBRyxFQUFDO29CQUNoQixPQUFPO29CQUNQLEtBQUssRUFBRTt3QkFDTCxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsT0FBTyxDQUFDLE1BQU0sR0FBRyxnQkFBZ0IsQ0FBQzt3QkFDbkQsUUFBUSxFQUFFLEtBQUs7cUJBQ2hCO2lCQUNGLENBQUMsQ0FBQztnQkFDSCxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7YUFDNUI7WUFDRCxnQkFBZ0IsRUFBRSxDQUFDO1NBQ3BCO0tBQ0Y7SUFFRCxpQ0FBaUM7SUFDakMsS0FBSyxNQUFNLFVBQVUsSUFBSSxnQkFBZ0IsRUFBRTtRQUN6QyxJQUFJLFVBQVUsS0FBSyxHQUFHLEVBQUU7WUFDdEIsU0FBUztTQUNWO1FBQ0QsT0FBTyxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7S0FDeEM7SUFFRCxPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxZQUFZLEdBQWlCO0lBQ3hDLFVBQVUsRUFBRSxNQUFNO0lBQ2xCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxNQUErQjtDQUM1QyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgRWluc3VtLCBFaW5zdW1BdHRycywgRWluc3VtSW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvciwgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5cbmltcG9ydCB7bXVsdGlwbHlLZXJuZWxGdW5jfSBmcm9tICcuL011bHRpcGx5JztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9SZXNoYXBlJztcbmltcG9ydCB7c3VtfSBmcm9tICcuL1N1bSc7XG5pbXBvcnQge3RyYW5zcG9zZX0gZnJvbSAnLi9UcmFuc3Bvc2UnO1xuXG5leHBvcnQgZnVuY3Rpb24gZWluc3VtKFxuICAgIGFyZ3M6IHtpbnB1dHM6IEVpbnN1bUlucHV0cywgYmFja2VuZDogV2ViR1BVQmFja2VuZCwgYXR0cnM6IEVpbnN1bUF0dHJzfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtlcXVhdGlvbn0gPSBhdHRycztcbiAgY29uc3QgdGVuc29ycyA9IGlucHV0cyBhcyBUZW5zb3JbXTtcblxuICBjb25zdCB7YWxsRGltcywgc3VtbWVkRGltcywgaWREaW1zfSA9XG4gICAgICBiYWNrZW5kX3V0aWwuZGVjb2RlRWluc3VtRXF1YXRpb24oZXF1YXRpb24sIHRlbnNvcnMubGVuZ3RoKTtcbiAgYmFja2VuZF91dGlsLmNoZWNrRWluc3VtRGltU2l6ZXMoYWxsRGltcy5sZW5ndGgsIGlkRGltcywgdGVuc29ycyk7XG4gIGNvbnN0IHtwYXRoLCBzdGVwc30gPSBiYWNrZW5kX3V0aWwuZ2V0RWluc3VtQ29tcHV0ZVBhdGgoc3VtbWVkRGltcywgaWREaW1zKTtcblxuICBjb25zdCBuU3RlcHMgPSBzdGVwcy5sZW5ndGg7XG4gIGxldCBvdXQ6IFRlbnNvckluZm98bnVsbCA9IG51bGw7XG4gIGxldCBudW1EaW1zUmVtYWluaW5nID0gYWxsRGltcy5sZW5ndGg7XG4gIGNvbnN0IHRlbnNvcnNUb0Rpc3Bvc2U6IFRlbnNvckluZm9bXSA9IFtdO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IG5TdGVwczsgKytpKSB7XG4gICAgZm9yIChjb25zdCBpZFRlcm0gb2Ygc3RlcHNbaV0pIHtcbiAgICAgIGNvbnN0IHtwZXJtdXRhdGlvbkluZGljZXM6IHBlcm0sIGV4cGFuZERpbXM6IGRpbXNUb0V4cGFuZH0gPVxuICAgICAgICAgIGJhY2tlbmRfdXRpbC5nZXRFaW5zdW1QZXJtdXRhdGlvbihudW1EaW1zUmVtYWluaW5nLCBpZERpbXNbaWRUZXJtXSk7XG4gICAgICBsZXQgeDogVGVuc29ySW5mbztcbiAgICAgIGlmIChiYWNrZW5kX3V0aWwuaXNJZGVudGl0eVBlcm11dGF0aW9uKHBlcm0pKSB7XG4gICAgICAgIHggPSB0ZW5zb3JzW2lkVGVybV07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB4ID0gdHJhbnNwb3NlKHtpbnB1dHM6IHt4OiB0ZW5zb3JzW2lkVGVybV19LCBiYWNrZW5kLCBhdHRyczoge3Blcm19fSk7XG4gICAgICAgIHRlbnNvcnNUb0Rpc3Bvc2UucHVzaCh4KTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHRhcmdldFNoYXBlOiBudW1iZXJbXSA9IHguc2hhcGUuc2xpY2UoKTtcbiAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgZGltc1RvRXhwYW5kLmxlbmd0aDsgKytrKSB7XG4gICAgICAgIHRhcmdldFNoYXBlLnNwbGljZShkaW1zVG9FeHBhbmRba10sIDAsIDEpO1xuICAgICAgfVxuXG4gICAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwoeC5zaGFwZSwgdGFyZ2V0U2hhcGUpKSB7XG4gICAgICAgIHggPSByZXNoYXBlKHtpbnB1dHM6IHt4fSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogdGFyZ2V0U2hhcGV9fSk7XG4gICAgICAgIHRlbnNvcnNUb0Rpc3Bvc2UucHVzaCh4KTtcbiAgICAgIH1cbiAgICAgIGlmIChvdXQgPT09IG51bGwpIHtcbiAgICAgICAgb3V0ID0geDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgICAgICAgb3V0ID1cbiAgICAgICAgICAgIG11bHRpcGx5S2VybmVsRnVuYyh7aW5wdXRzOiB7YTogeCwgYjogb3V0fSwgYmFja2VuZH0pIGFzIFRlbnNvckluZm87XG4gICAgICAgIHRlbnNvcnNUb0Rpc3Bvc2UucHVzaChvdXQpO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAoaSA8IG5TdGVwcyAtIDEpIHtcbiAgICAgIGlmIChwYXRoW2ldID49IDApIHtcbiAgICAgICAgb3V0ID0gc3VtKHtcbiAgICAgICAgICBpbnB1dHM6IHt4OiBvdXR9LFxuICAgICAgICAgIGJhY2tlbmQsXG4gICAgICAgICAgYXR0cnM6IHtcbiAgICAgICAgICAgIGF4aXM6IHBhdGhbaV0gLSAoYWxsRGltcy5sZW5ndGggLSBudW1EaW1zUmVtYWluaW5nKSxcbiAgICAgICAgICAgIGtlZXBEaW1zOiBmYWxzZVxuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIHRlbnNvcnNUb0Rpc3Bvc2UucHVzaChvdXQpO1xuICAgICAgfVxuICAgICAgbnVtRGltc1JlbWFpbmluZy0tO1xuICAgIH1cbiAgfVxuXG4gIC8vIENsZWFuIHVwIGludGVybWVkaWF0ZSB0ZW5zb3JzLlxuICBmb3IgKGNvbnN0IHRlbnNvckluZm8gb2YgdGVuc29yc1RvRGlzcG9zZSkge1xuICAgIGlmICh0ZW5zb3JJbmZvID09PSBvdXQpIHtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cbiAgICBiYWNrZW5kLmRpc3Bvc2VEYXRhKHRlbnNvckluZm8uZGF0YUlkKTtcbiAgfVxuXG4gIHJldHVybiBvdXQ7XG59XG5cbmV4cG9ydCBjb25zdCBlaW5zdW1Db25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogRWluc3VtLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGVpbnN1bSBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=