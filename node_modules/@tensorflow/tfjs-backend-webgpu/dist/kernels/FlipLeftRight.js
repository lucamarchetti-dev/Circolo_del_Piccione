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
import { FlipLeftRight } from '@tensorflow/tfjs-core';
import { FlipLeftRightProgram } from '../flip_left_right_webgpu';
export const flipLeftRightConfig = {
    kernelName: FlipLeftRight,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, backend }) => {
        const { image } = inputs;
        const webgpuBackend = backend;
        const program = new FlipLeftRightProgram(image.shape);
        const output = webgpuBackend.runWebGPUProgram(program, [image], image.dtype);
        return output;
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRmxpcExlZnRSaWdodC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvRmxpcExlZnRSaWdodC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsYUFBYSxFQUFzQixNQUFNLHVCQUF1QixDQUFDO0FBR3pFLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLDJCQUEyQixDQUFDO0FBRS9ELE1BQU0sQ0FBQyxNQUFNLG1CQUFtQixHQUFpQjtJQUM3QyxVQUFVLEVBQUUsYUFBYTtJQUN6QixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsRUFBRSxFQUFFO1FBQ2hDLE1BQU0sRUFBQyxLQUFLLEVBQUMsR0FBRyxNQUE2QixDQUFDO1FBQzlDLE1BQU0sYUFBYSxHQUFHLE9BQXdCLENBQUM7UUFFL0MsTUFBTSxPQUFPLEdBQUcsSUFBSSxvQkFBb0IsQ0FBRSxLQUFrQixDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sTUFBTSxHQUNSLGFBQWEsQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxLQUFLLENBQUMsRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDbEUsT0FBTyxNQUFNLENBQUM7SUFDbEIsQ0FBQztDQUNGLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBUZW5zb3I0RH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7RmxpcExlZnRSaWdodCwgRmxpcExlZnRSaWdodElucHV0c30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5pbXBvcnQge0ZsaXBMZWZ0UmlnaHRQcm9ncmFtfSBmcm9tICcuLi9mbGlwX2xlZnRfcmlnaHRfd2ViZ3B1JztcblxuZXhwb3J0IGNvbnN0IGZsaXBMZWZ0UmlnaHRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAgICBrZXJuZWxOYW1lOiBGbGlwTGVmdFJpZ2h0LFxuICAgIGJhY2tlbmROYW1lOiAnd2ViZ3B1JyxcbiAgICBrZXJuZWxGdW5jOiAoe2lucHV0cywgYmFja2VuZH0pID0+IHtcbiAgICAgIGNvbnN0IHtpbWFnZX0gPSBpbnB1dHMgYXMgRmxpcExlZnRSaWdodElucHV0cztcbiAgICAgIGNvbnN0IHdlYmdwdUJhY2tlbmQgPSBiYWNrZW5kIGFzIFdlYkdQVUJhY2tlbmQ7XG5cbiAgICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgRmxpcExlZnRSaWdodFByb2dyYW0oKGltYWdlIGFzIFRlbnNvcjREKS5zaGFwZSk7XG4gICAgICBjb25zdCBvdXRwdXQgPVxuICAgICAgICAgIHdlYmdwdUJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbaW1hZ2VdLCBpbWFnZS5kdHlwZSk7XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICB9XG59O1xuIl19