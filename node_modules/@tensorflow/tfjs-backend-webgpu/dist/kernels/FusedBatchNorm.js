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
import { FusedBatchNorm } from '@tensorflow/tfjs-core';
import { BatchNormProgram } from '../batchnorm_webgpu';
export const fusedBatchNormConfig = {
    kernelName: FusedBatchNorm,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x, scale, offset, mean, variance } = inputs;
        const { varianceEpsilon } = attrs;
        const webGPUBackend = backend;
        const batchNormInputs = [x, mean, variance];
        let offsetShape = null;
        if (offset != null) {
            offsetShape = offset.shape;
            batchNormInputs.push(offset);
        }
        let scaleShape = null;
        if (scale != null) {
            scaleShape = scale.shape;
            batchNormInputs.push(scale);
        }
        const program = new BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape);
        const uniformData = [{ type: 'float32', data: [varianceEpsilon] }];
        return webGPUBackend.runWebGPUProgram(program, batchNormInputs, x.dtype, uniformData);
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRnVzZWRCYXRjaE5vcm0uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL0Z1c2VkQmF0Y2hOb3JtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxjQUFjLEVBQWtFLE1BQU0sdUJBQXVCLENBQUM7QUFJdEgsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFFckQsTUFBTSxDQUFDLE1BQU0sb0JBQW9CLEdBQWlCO0lBQ2hELFVBQVUsRUFBRSxjQUFjO0lBQzFCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxDQUFDLEVBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUMsRUFBRSxFQUFFO1FBQ3ZDLE1BQU0sRUFBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsUUFBUSxFQUFDLEdBQUcsTUFBOEIsQ0FBQztRQUMxRSxNQUFNLEVBQUMsZUFBZSxFQUFDLEdBQUcsS0FBdUMsQ0FBQztRQUNsRSxNQUFNLGFBQWEsR0FBRyxPQUF3QixDQUFDO1FBQy9DLE1BQU0sZUFBZSxHQUFHLENBQUMsQ0FBVyxFQUFFLElBQWMsRUFBRSxRQUFrQixDQUFDLENBQUM7UUFDMUUsSUFBSSxXQUFXLEdBQUcsSUFBSSxDQUFDO1FBQ3ZCLElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtZQUNsQixXQUFXLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQztZQUMzQixlQUFlLENBQUMsSUFBSSxDQUFDLE1BQWdCLENBQUMsQ0FBQztTQUN4QztRQUNELElBQUksVUFBVSxHQUFHLElBQUksQ0FBQztRQUN0QixJQUFJLEtBQUssSUFBSSxJQUFJLEVBQUU7WUFDakIsVUFBVSxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUM7WUFDekIsZUFBZSxDQUFDLElBQUksQ0FBQyxLQUFlLENBQUMsQ0FBQztTQUN2QztRQUNELE1BQU0sT0FBTyxHQUFHLElBQUksZ0JBQWdCLENBQ2hDLENBQUMsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLEtBQUssRUFBRSxRQUFRLENBQUMsS0FBSyxFQUFFLFdBQVcsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsRSxNQUFNLFdBQVcsR0FBRyxDQUFDLEVBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxlQUFlLENBQUMsRUFBQyxDQUFDLENBQUM7UUFDakUsT0FBTyxhQUFhLENBQUMsZ0JBQWdCLENBQ2pDLE9BQU8sRUFBRSxlQUFlLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztJQUN0RCxDQUFDO0NBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtGdXNlZEJhdGNoTm9ybSwgRnVzZWRCYXRjaE5vcm1BdHRycywgRnVzZWRCYXRjaE5vcm1JbnB1dHMsIEtlcm5lbENvbmZpZywgVGVuc29yfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcblxuaW1wb3J0IHtCYXRjaE5vcm1Qcm9ncmFtfSBmcm9tICcuLi9iYXRjaG5vcm1fd2ViZ3B1JztcblxuZXhwb3J0IGNvbnN0IGZ1c2VkQmF0Y2hOb3JtQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IEZ1c2VkQmF0Y2hOb3JtLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6ICh7aW5wdXRzLCBhdHRycywgYmFja2VuZH0pID0+IHtcbiAgICBjb25zdCB7eCwgc2NhbGUsIG9mZnNldCwgbWVhbiwgdmFyaWFuY2V9ID0gaW5wdXRzIGFzIEZ1c2VkQmF0Y2hOb3JtSW5wdXRzO1xuICAgIGNvbnN0IHt2YXJpYW5jZUVwc2lsb259ID0gYXR0cnMgYXMgdW5rbm93biBhcyBGdXNlZEJhdGNoTm9ybUF0dHJzO1xuICAgIGNvbnN0IHdlYkdQVUJhY2tlbmQgPSBiYWNrZW5kIGFzIFdlYkdQVUJhY2tlbmQ7XG4gICAgY29uc3QgYmF0Y2hOb3JtSW5wdXRzID0gW3ggYXMgVGVuc29yLCBtZWFuIGFzIFRlbnNvciwgdmFyaWFuY2UgYXMgVGVuc29yXTtcbiAgICBsZXQgb2Zmc2V0U2hhcGUgPSBudWxsO1xuICAgIGlmIChvZmZzZXQgIT0gbnVsbCkge1xuICAgICAgb2Zmc2V0U2hhcGUgPSBvZmZzZXQuc2hhcGU7XG4gICAgICBiYXRjaE5vcm1JbnB1dHMucHVzaChvZmZzZXQgYXMgVGVuc29yKTtcbiAgICB9XG4gICAgbGV0IHNjYWxlU2hhcGUgPSBudWxsO1xuICAgIGlmIChzY2FsZSAhPSBudWxsKSB7XG4gICAgICBzY2FsZVNoYXBlID0gc2NhbGUuc2hhcGU7XG4gICAgICBiYXRjaE5vcm1JbnB1dHMucHVzaChzY2FsZSBhcyBUZW5zb3IpO1xuICAgIH1cbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IEJhdGNoTm9ybVByb2dyYW0oXG4gICAgICAgIHguc2hhcGUsIG1lYW4uc2hhcGUsIHZhcmlhbmNlLnNoYXBlLCBvZmZzZXRTaGFwZSwgc2NhbGVTaGFwZSk7XG4gICAgY29uc3QgdW5pZm9ybURhdGEgPSBbe3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW3ZhcmlhbmNlRXBzaWxvbl19XTtcbiAgICByZXR1cm4gd2ViR1BVQmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBiYXRjaE5vcm1JbnB1dHMsIHguZHR5cGUsIHVuaWZvcm1EYXRhKTtcbiAgfVxufTtcbiJdfQ==