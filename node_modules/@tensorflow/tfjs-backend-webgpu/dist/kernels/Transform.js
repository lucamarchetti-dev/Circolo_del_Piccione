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
import { Transform } from '@tensorflow/tfjs-core';
import { TransformProgram } from '../transform_webgpu';
export function transform(args) {
    const { inputs, backend, attrs } = args;
    const { image, transforms } = inputs;
    const { interpolation, fillMode, fillValue, outputShape } = attrs;
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;
    const [outHeight, outWidth] = outputShape != null ? outputShape : [imageHeight, imageWidth];
    const outShape = [batch, outHeight, outWidth,
        numChannels];
    const program = new TransformProgram(outShape);
    const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
    let fillModeId;
    switch (fillMode) {
        case 'constant':
            fillModeId = 1;
            break;
        case 'reflect':
            fillModeId = 2;
            break;
        case 'wrap':
            fillModeId = 3;
            break;
        case 'nearest':
            fillModeId = 4;
            break;
        default:
            fillModeId = 1;
            break;
    }
    const uniformData = [
        { type: 'int32', data: [interpolationModeId] },
        { type: 'int32', data: [fillModeId] }, { type: 'float32', data: [fillValue] }
    ];
    return backend.runWebGPUProgram(program, [image, transforms], 'float32', uniformData);
}
export const transformConfig = {
    kernelName: Transform,
    backendName: 'webgpu',
    kernelFunc: transform
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVHJhbnNmb3JtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9UcmFuc2Zvcm0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUF1QyxTQUFTLEVBQWtDLE1BQU0sdUJBQXVCLENBQUM7QUFHdkgsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFFckQsTUFBTSxVQUFVLFNBQVMsQ0FBQyxJQUl6QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsS0FBSyxFQUFFLFVBQVUsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUNuQyxNQUFNLEVBQUMsYUFBYSxFQUFFLFFBQVEsRUFBRSxTQUFTLEVBQUUsV0FBVyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRWhFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxXQUFXLENBQUMsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDO0lBQ2xFLE1BQU0sQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLEdBQ3ZCLFdBQVcsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxXQUFXLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFDbEUsTUFBTSxRQUFRLEdBQ1YsQ0FBQyxLQUFLLEVBQUUsU0FBUyxFQUFFLFFBQVE7UUFDMUIsV0FBVyxDQUFxQyxDQUFDO0lBRXRELE1BQU0sT0FBTyxHQUFHLElBQUksZ0JBQWdCLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDL0MsTUFBTSxtQkFBbUIsR0FBRyxhQUFhLEtBQUssU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoRSxJQUFJLFVBQWtCLENBQUM7SUFDdkIsUUFBUSxRQUFRLEVBQUU7UUFDaEIsS0FBSyxVQUFVO1lBQ2IsVUFBVSxHQUFHLENBQUMsQ0FBQztZQUNmLE1BQU07UUFDUixLQUFLLFNBQVM7WUFDWixVQUFVLEdBQUcsQ0FBQyxDQUFDO1lBQ2YsTUFBTTtRQUNSLEtBQUssTUFBTTtZQUNULFVBQVUsR0FBRyxDQUFDLENBQUM7WUFDZixNQUFNO1FBQ1IsS0FBSyxTQUFTO1lBQ1osVUFBVSxHQUFHLENBQUMsQ0FBQztZQUNmLE1BQU07UUFDUjtZQUNFLFVBQVUsR0FBRyxDQUFDLENBQUM7WUFDZixNQUFNO0tBQ1Q7SUFDRCxNQUFNLFdBQVcsR0FBRztRQUNsQixFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsbUJBQW1CLENBQUMsRUFBQztRQUM1QyxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsVUFBVSxDQUFDLEVBQUMsRUFBRSxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsU0FBUyxDQUFDLEVBQUM7S0FDMUUsQ0FBQztJQUNGLE9BQU8sT0FBTyxDQUFDLGdCQUFnQixDQUMzQixPQUFPLEVBQUUsQ0FBQyxLQUFLLEVBQUUsVUFBVSxDQUFDLEVBQUUsU0FBUyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQzVELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxlQUFlLEdBQWlCO0lBQzNDLFVBQVUsRUFBRSxTQUFTO0lBQ3JCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxTQUFrQztDQUMvQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgVHJhbnNmb3JtLCBUcmFuc2Zvcm1BdHRycywgVHJhbnNmb3JtSW5wdXRzfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7VHJhbnNmb3JtUHJvZ3JhbX0gZnJvbSAnLi4vdHJhbnNmb3JtX3dlYmdwdSc7XG5cbmV4cG9ydCBmdW5jdGlvbiB0cmFuc2Zvcm0oYXJnczoge1xuICBpbnB1dHM6IFRyYW5zZm9ybUlucHV0cyxcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZCxcbiAgYXR0cnM6IFRyYW5zZm9ybUF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtpbWFnZSwgdHJhbnNmb3Jtc30gPSBpbnB1dHM7XG4gIGNvbnN0IHtpbnRlcnBvbGF0aW9uLCBmaWxsTW9kZSwgZmlsbFZhbHVlLCBvdXRwdXRTaGFwZX0gPSBhdHRycztcblxuICBjb25zdCBbYmF0Y2gsIGltYWdlSGVpZ2h0LCBpbWFnZVdpZHRoLCBudW1DaGFubmVsc10gPSBpbWFnZS5zaGFwZTtcbiAgY29uc3QgW291dEhlaWdodCwgb3V0V2lkdGhdID1cbiAgICAgIG91dHB1dFNoYXBlICE9IG51bGwgPyBvdXRwdXRTaGFwZSA6IFtpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aF07XG4gIGNvbnN0IG91dFNoYXBlID1cbiAgICAgIFtiYXRjaCwgb3V0SGVpZ2h0LCBvdXRXaWR0aCxcbiAgICAgICBudW1DaGFubmVsc10gYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG5cbiAgY29uc3QgcHJvZ3JhbSA9IG5ldyBUcmFuc2Zvcm1Qcm9ncmFtKG91dFNoYXBlKTtcbiAgY29uc3QgaW50ZXJwb2xhdGlvbk1vZGVJZCA9IGludGVycG9sYXRpb24gPT09ICduZWFyZXN0JyA/IDEgOiAyO1xuICBsZXQgZmlsbE1vZGVJZDogbnVtYmVyO1xuICBzd2l0Y2ggKGZpbGxNb2RlKSB7XG4gICAgY2FzZSAnY29uc3RhbnQnOlxuICAgICAgZmlsbE1vZGVJZCA9IDE7XG4gICAgICBicmVhaztcbiAgICBjYXNlICdyZWZsZWN0JzpcbiAgICAgIGZpbGxNb2RlSWQgPSAyO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSAnd3JhcCc6XG4gICAgICBmaWxsTW9kZUlkID0gMztcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgJ25lYXJlc3QnOlxuICAgICAgZmlsbE1vZGVJZCA9IDQ7XG4gICAgICBicmVhaztcbiAgICBkZWZhdWx0OlxuICAgICAgZmlsbE1vZGVJZCA9IDE7XG4gICAgICBicmVhaztcbiAgfVxuICBjb25zdCB1bmlmb3JtRGF0YSA9IFtcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2ludGVycG9sYXRpb25Nb2RlSWRdfSxcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2ZpbGxNb2RlSWRdfSwge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2ZpbGxWYWx1ZV19XG4gIF07XG4gIHJldHVybiBiYWNrZW5kLnJ1bldlYkdQVVByb2dyYW0oXG4gICAgICBwcm9ncmFtLCBbaW1hZ2UsIHRyYW5zZm9ybXNdLCAnZmxvYXQzMicsIHVuaWZvcm1EYXRhKTtcbn1cblxuZXhwb3J0IGNvbnN0IHRyYW5zZm9ybUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBUcmFuc2Zvcm0sXG4gIGJhY2tlbmROYW1lOiAnd2ViZ3B1JyxcbiAga2VybmVsRnVuYzogdHJhbnNmb3JtIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==