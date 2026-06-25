/**
 * @license
 * Copyright 2022 Google LLC.
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
import { LRN } from '@tensorflow/tfjs-core';
import { LRNProgram, LRNSharedProgram } from '../lrn_webgpu';
export function lrn(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    // When the adjacent channels is less than or equal to 16, which could cover
    // most cases, we use shared memory version to get better performance.
    // The theoretical adjacent channels may be very large, but the shared memory
    // size of hardware is limited, so we use the naive version when the adjacent
    // channels is large.
    let program;
    if (depthRadius > 16) {
        program = new LRNProgram(x.shape);
    }
    else {
        program = new LRNSharedProgram(x.shape, depthRadius);
    }
    const uniformData = [
        { type: 'int32', data: [depthRadius] }, { type: 'float32', data: [bias] },
        { type: 'float32', data: [alpha] }, { type: 'float32', data: [beta] }
    ];
    const res = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
    return res;
}
export const lrnConfig = {
    kernelName: LRN,
    backendName: 'webgpu',
    kernelFunc: lrn
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTFJOLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9MUk4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixHQUFHLEVBQWtDLE1BQU0sdUJBQXVCLENBQUM7QUFHckcsT0FBTyxFQUFDLFVBQVUsRUFBRSxnQkFBZ0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUUzRCxNQUFNLFVBQVUsR0FBRyxDQUNmLElBQWtFO0lBRXBFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25CLE1BQU0sRUFBQyxXQUFXLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxJQUFJLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFL0MsNEVBQTRFO0lBQzVFLHNFQUFzRTtJQUN0RSw2RUFBNkU7SUFDN0UsNkVBQTZFO0lBQzdFLHFCQUFxQjtJQUNyQixJQUFJLE9BQW9DLENBQUM7SUFDekMsSUFBSSxXQUFXLEdBQUcsRUFBRSxFQUFFO1FBQ3BCLE9BQU8sR0FBRyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7S0FDbkM7U0FBTTtRQUNMLE9BQU8sR0FBRyxJQUFJLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7S0FDdEQ7SUFDRCxNQUFNLFdBQVcsR0FBRztRQUNsQixFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsV0FBVyxDQUFDLEVBQUMsRUFBRSxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsSUFBSSxDQUFDLEVBQUM7UUFDckUsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLEtBQUssQ0FBQyxFQUFDLEVBQUUsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFDO0tBQ2xFLENBQUM7SUFDRixNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztJQUV6RSxPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQWlCO0lBQ3JDLFVBQVUsRUFBRSxHQUFHO0lBQ2YsV0FBVyxFQUFFLFFBQVE7SUFDckIsVUFBVSxFQUFFLEdBQTRCO0NBQ3pDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBMUk4sIExSTkF0dHJzLCBMUk5JbnB1dHMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtMUk5Qcm9ncmFtLCBMUk5TaGFyZWRQcm9ncmFtfSBmcm9tICcuLi9scm5fd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGxybihcbiAgICBhcmdzOiB7aW5wdXRzOiBMUk5JbnB1dHMsIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmQsIGF0dHJzOiBMUk5BdHRyc30pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtkZXB0aFJhZGl1cywgYmlhcywgYWxwaGEsIGJldGF9ID0gYXR0cnM7XG5cbiAgLy8gV2hlbiB0aGUgYWRqYWNlbnQgY2hhbm5lbHMgaXMgbGVzcyB0aGFuIG9yIGVxdWFsIHRvIDE2LCB3aGljaCBjb3VsZCBjb3ZlclxuICAvLyBtb3N0IGNhc2VzLCB3ZSB1c2Ugc2hhcmVkIG1lbW9yeSB2ZXJzaW9uIHRvIGdldCBiZXR0ZXIgcGVyZm9ybWFuY2UuXG4gIC8vIFRoZSB0aGVvcmV0aWNhbCBhZGphY2VudCBjaGFubmVscyBtYXkgYmUgdmVyeSBsYXJnZSwgYnV0IHRoZSBzaGFyZWQgbWVtb3J5XG4gIC8vIHNpemUgb2YgaGFyZHdhcmUgaXMgbGltaXRlZCwgc28gd2UgdXNlIHRoZSBuYWl2ZSB2ZXJzaW9uIHdoZW4gdGhlIGFkamFjZW50XG4gIC8vIGNoYW5uZWxzIGlzIGxhcmdlLlxuICBsZXQgcHJvZ3JhbTogTFJOUHJvZ3JhbXxMUk5TaGFyZWRQcm9ncmFtO1xuICBpZiAoZGVwdGhSYWRpdXMgPiAxNikge1xuICAgIHByb2dyYW0gPSBuZXcgTFJOUHJvZ3JhbSh4LnNoYXBlKTtcbiAgfSBlbHNlIHtcbiAgICBwcm9ncmFtID0gbmV3IExSTlNoYXJlZFByb2dyYW0oeC5zaGFwZSwgZGVwdGhSYWRpdXMpO1xuICB9XG4gIGNvbnN0IHVuaWZvcm1EYXRhID0gW1xuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbZGVwdGhSYWRpdXNdfSwge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2JpYXNdfSxcbiAgICB7dHlwZTogJ2Zsb2F0MzInLCBkYXRhOiBbYWxwaGFdfSwge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2JldGFdfVxuICBdO1xuICBjb25zdCByZXMgPSBiYWNrZW5kLnJ1bldlYkdQVVByb2dyYW0ocHJvZ3JhbSwgW3hdLCB4LmR0eXBlLCB1bmlmb3JtRGF0YSk7XG5cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGxybkNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBMUk4sXG4gIGJhY2tlbmROYW1lOiAnd2ViZ3B1JyxcbiAga2VybmVsRnVuYzogbHJuIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==