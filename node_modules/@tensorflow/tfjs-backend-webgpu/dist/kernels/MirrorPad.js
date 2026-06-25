/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { MirrorPad } from '@tensorflow/tfjs-core';
import { MirrorPadProgram } from '../mirror_pad_webgpu';
export const mirrorPadConfig = {
    kernelName: MirrorPad,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        const { paddings, mode } = attrs;
        const webGPUBackend = backend;
        const uniformData = paddings.map(p => {
            return { type: 'int32', data: [p[0], p[1]] };
        });
        const program = new MirrorPadProgram(x.shape, paddings, mode);
        const output = webGPUBackend.runWebGPUProgram(program, [x], x.dtype, uniformData);
        return output;
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWlycm9yUGFkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9NaXJyb3JQYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLFNBQVMsRUFBa0MsTUFBTSx1QkFBdUIsQ0FBQztBQUkvRixPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUV0RCxNQUFNLENBQUMsTUFBTSxlQUFlLEdBQWlCO0lBQzNDLFVBQVUsRUFBRSxTQUFTO0lBQ3JCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxDQUFDLEVBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUMsRUFBRSxFQUFFO1FBQ3ZDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUF5QixDQUFDO1FBQ3RDLE1BQU0sRUFBQyxRQUFRLEVBQUUsSUFBSSxFQUFDLEdBQUcsS0FBa0MsQ0FBQztRQUM1RCxNQUFNLGFBQWEsR0FBRyxPQUF3QixDQUFDO1FBRS9DLE1BQU0sV0FBVyxHQUFHLFFBQVEsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUU7WUFDbkMsT0FBTyxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUM7UUFDN0MsQ0FBQyxDQUFDLENBQUM7UUFDSCxNQUFNLE9BQU8sR0FBRyxJQUFJLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQzlELE1BQU0sTUFBTSxHQUNSLGFBQWEsQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLFdBQVcsQ0FBQyxDQUFDO1FBRXZFLE9BQU8sTUFBTSxDQUFDO0lBQ2hCLENBQUM7Q0FDRixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgTWlycm9yUGFkLCBNaXJyb3JQYWRBdHRycywgTWlycm9yUGFkSW5wdXRzfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcblxuaW1wb3J0IHtNaXJyb3JQYWRQcm9ncmFtfSBmcm9tICcuLi9taXJyb3JfcGFkX3dlYmdwdSc7XG5cbmV4cG9ydCBjb25zdCBtaXJyb3JQYWRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTWlycm9yUGFkLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6ICh7aW5wdXRzLCBhdHRycywgYmFja2VuZH0pID0+IHtcbiAgICBjb25zdCB7eH0gPSBpbnB1dHMgYXMgTWlycm9yUGFkSW5wdXRzO1xuICAgIGNvbnN0IHtwYWRkaW5ncywgbW9kZX0gPSBhdHRycyBhcyB1bmtub3duIGFzIE1pcnJvclBhZEF0dHJzO1xuICAgIGNvbnN0IHdlYkdQVUJhY2tlbmQgPSBiYWNrZW5kIGFzIFdlYkdQVUJhY2tlbmQ7XG5cbiAgICBjb25zdCB1bmlmb3JtRGF0YSA9IHBhZGRpbmdzLm1hcChwID0+IHtcbiAgICAgIHJldHVybiB7dHlwZTogJ2ludDMyJywgZGF0YTogW3BbMF0sIHBbMV1dfTtcbiAgICB9KTtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IE1pcnJvclBhZFByb2dyYW0oeC5zaGFwZSwgcGFkZGluZ3MsIG1vZGUpO1xuICAgIGNvbnN0IG91dHB1dCA9XG4gICAgICAgIHdlYkdQVUJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbeF0sIHguZHR5cGUsIHVuaWZvcm1EYXRhKTtcblxuICAgIHJldHVybiBvdXRwdXQ7XG4gIH1cbn07XG4iXX0=