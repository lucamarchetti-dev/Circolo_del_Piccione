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
import { Square } from '@tensorflow/tfjs-core';
import { UnaryOpProgram } from '../unary_op_webgpu';
import { UnaryOpType } from '../unary_op_util';
export const squareConfig = {
    kernelName: Square,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, backend }) => {
        const { x } = inputs;
        const webGPUBackend = backend;
        const program = new UnaryOpProgram(x.shape, UnaryOpType.SQUARE);
        return webGPUBackend.runWebGPUProgram(program, [x], x.dtype);
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3F1YXJlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9TcXVhcmUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLE1BQU0sRUFBZSxNQUFNLHVCQUF1QixDQUFDO0FBRXpFLE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUNsRCxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFFN0MsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFpQjtJQUN4QyxVQUFVLEVBQUUsTUFBTTtJQUNsQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsRUFBRSxFQUFFO1FBQ2hDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFzQixDQUFDO1FBQ25DLE1BQU0sYUFBYSxHQUFHLE9BQXdCLENBQUM7UUFDL0MsTUFBTSxPQUFPLEdBQUcsSUFBSSxjQUFjLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDaEUsT0FBTyxhQUFhLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQy9ELENBQUM7Q0FDRixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgU3F1YXJlLCBTcXVhcmVJbnB1dHN9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7VW5hcnlPcFByb2dyYW19IGZyb20gJy4uL3VuYXJ5X29wX3dlYmdwdSc7XG5pbXBvcnQge1VuYXJ5T3BUeXBlfSBmcm9tICcuLi91bmFyeV9vcF91dGlsJztcblxuZXhwb3J0IGNvbnN0IHNxdWFyZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBTcXVhcmUsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ3B1JyxcbiAga2VybmVsRnVuYzogKHtpbnB1dHMsIGJhY2tlbmR9KSA9PiB7XG4gICAgY29uc3Qge3h9ID0gaW5wdXRzIGFzIFNxdWFyZUlucHV0cztcbiAgICBjb25zdCB3ZWJHUFVCYWNrZW5kID0gYmFja2VuZCBhcyBXZWJHUFVCYWNrZW5kO1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oeC5zaGFwZSwgVW5hcnlPcFR5cGUuU1FVQVJFKTtcbiAgICByZXR1cm4gd2ViR1BVQmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKHByb2dyYW0sIFt4XSwgeC5kdHlwZSk7XG4gIH1cbn07XG4iXX0=