/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
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
import './flags_webgpu';
import { env, registerBackend } from '@tensorflow/tfjs-core';
import { WebGPUBackend } from './backend_webgpu';
import { isWebGPUSupported } from './webgpu_util';
if (isWebGPUSupported()) {
    registerBackend('webgpu', async () => {
        const gpuDescriptor = {
            powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
                'low-power' :
                'high-performance'
        };
        const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
        const deviceDescriptor = {};
        const requiredFeatures = [];
        if (adapter.features.has('timestamp-query')) {
            requiredFeatures.push('timestamp-query');
        }
        if (adapter.features.has('bgra8unorm-storage')) {
            requiredFeatures.push(['bgra8unorm-storage']);
        }
        deviceDescriptor.requiredFeatures =
            requiredFeatures;
        const adapterLimits = adapter.limits;
        deviceDescriptor.requiredLimits = {
            'maxComputeWorkgroupStorageSize': adapterLimits.maxComputeWorkgroupStorageSize,
            'maxComputeWorkgroupsPerDimension': adapterLimits.maxComputeWorkgroupsPerDimension,
            'maxStorageBufferBindingSize': adapterLimits.maxStorageBufferBindingSize,
            'maxBufferSize': adapterLimits.maxBufferSize,
            'maxComputeWorkgroupSizeX': adapterLimits.maxComputeWorkgroupSizeX,
            'maxComputeInvocationsPerWorkgroup': adapterLimits.maxComputeInvocationsPerWorkgroup,
        };
        const device = await adapter.requestDevice(deviceDescriptor);
        const adapterInfo = 'info' in adapter
            ? adapter.info
            : 'requestAdapterInfo' in adapter
                // tslint:disable-next-line:no-any
                ? await adapter.requestAdapterInfo()
                : undefined;
        return new WebGPUBackend(device, adapterInfo);
    }, 3 /*priority*/);
}
// Export webgpu utilities
export * from './webgpu';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFzZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2Jhc2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxnQkFBZ0IsQ0FBQztBQUV4QixPQUFPLEVBQUMsR0FBRyxFQUFFLGVBQWUsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRTNELE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUMvQyxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFaEQsSUFBSSxpQkFBaUIsRUFBRSxFQUFFO0lBQ3ZCLGVBQWUsQ0FBQyxRQUFRLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDbkMsTUFBTSxhQUFhLEdBQTZCO1lBQzlDLGVBQWUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsMEJBQTBCLENBQUMsQ0FBQyxDQUFDO2dCQUNwRCxXQUFXLENBQUMsQ0FBQztnQkFDYixrQkFBa0I7U0FDdkIsQ0FBQztRQUVGLE1BQU0sT0FBTyxHQUFHLE1BQU0sU0FBUyxDQUFDLEdBQUcsQ0FBQyxjQUFjLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDbEUsTUFBTSxnQkFBZ0IsR0FBd0IsRUFBRSxDQUFDO1FBRWpELE1BQU0sZ0JBQWdCLEdBQUcsRUFBRSxDQUFDO1FBQzVCLElBQUksT0FBTyxDQUFDLFFBQVEsQ0FBQyxHQUFHLENBQUMsaUJBQWlCLENBQUMsRUFBRTtZQUMzQyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQztTQUMxQztRQUNELElBQUksT0FBTyxDQUFDLFFBQVEsQ0FBQyxHQUFHLENBQUMsb0JBQW9CLENBQUMsRUFBRTtZQUM5QyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDLENBQUM7U0FDL0M7UUFDRCxnQkFBZ0IsQ0FBQyxnQkFBZ0I7WUFDN0IsZ0JBQTRDLENBQUM7UUFFakQsTUFBTSxhQUFhLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQztRQUNyQyxnQkFBZ0IsQ0FBQyxjQUFjLEdBQUc7WUFDaEMsZ0NBQWdDLEVBQzVCLGFBQWEsQ0FBQyw4QkFBOEI7WUFDaEQsa0NBQWtDLEVBQzlCLGFBQWEsQ0FBQyxnQ0FBZ0M7WUFDbEQsNkJBQTZCLEVBQUUsYUFBYSxDQUFDLDJCQUEyQjtZQUN4RSxlQUFlLEVBQUUsYUFBYSxDQUFDLGFBQWE7WUFDNUMsMEJBQTBCLEVBQUUsYUFBYSxDQUFDLHdCQUF3QjtZQUNsRSxtQ0FBbUMsRUFDL0IsYUFBYSxDQUFDLGlDQUFpQztTQUNwRCxDQUFDO1FBRUYsTUFBTSxNQUFNLEdBQWMsTUFBTSxPQUFPLENBQUMsYUFBYSxDQUFDLGdCQUFnQixDQUFDLENBQUM7UUFDeEUsTUFBTSxXQUFXLEdBQ2YsTUFBTSxJQUFJLE9BQU87WUFDZixDQUFDLENBQUMsT0FBTyxDQUFDLElBQUk7WUFDZCxDQUFDLENBQUMsb0JBQW9CLElBQUksT0FBTztnQkFDL0Isa0NBQWtDO2dCQUNsQyxDQUFDLENBQUMsTUFBTyxPQUFlLENBQUMsa0JBQWtCLEVBQUU7Z0JBQzdDLENBQUMsQ0FBQyxTQUFTLENBQUM7UUFDbEIsT0FBTyxJQUFJLGFBQWEsQ0FBQyxNQUFNLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDaEQsQ0FBQyxFQUFFLENBQUMsQ0FBQyxZQUFZLENBQUMsQ0FBQztDQUNwQjtBQUVELDBCQUEwQjtBQUMxQixjQUFjLFVBQVUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICcuL2ZsYWdzX3dlYmdwdSc7XG5cbmltcG9ydCB7ZW52LCByZWdpc3RlckJhY2tlbmR9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi9iYWNrZW5kX3dlYmdwdSc7XG5pbXBvcnQge2lzV2ViR1BVU3VwcG9ydGVkfSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuaWYgKGlzV2ViR1BVU3VwcG9ydGVkKCkpIHtcbiAgcmVnaXN0ZXJCYWNrZW5kKCd3ZWJncHUnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgZ3B1RGVzY3JpcHRvcjogR1BVUmVxdWVzdEFkYXB0ZXJPcHRpb25zID0ge1xuICAgICAgcG93ZXJQcmVmZXJlbmNlOiBlbnYoKS5nZXQoJ1dFQkdQVV9VU0VfTE9XX1BPV0VSX0dQVScpID9cbiAgICAgICAgICAnbG93LXBvd2VyJyA6XG4gICAgICAgICAgJ2hpZ2gtcGVyZm9ybWFuY2UnXG4gICAgfTtcblxuICAgIGNvbnN0IGFkYXB0ZXIgPSBhd2FpdCBuYXZpZ2F0b3IuZ3B1LnJlcXVlc3RBZGFwdGVyKGdwdURlc2NyaXB0b3IpO1xuICAgIGNvbnN0IGRldmljZURlc2NyaXB0b3I6IEdQVURldmljZURlc2NyaXB0b3IgPSB7fTtcblxuICAgIGNvbnN0IHJlcXVpcmVkRmVhdHVyZXMgPSBbXTtcbiAgICBpZiAoYWRhcHRlci5mZWF0dXJlcy5oYXMoJ3RpbWVzdGFtcC1xdWVyeScpKSB7XG4gICAgICByZXF1aXJlZEZlYXR1cmVzLnB1c2goJ3RpbWVzdGFtcC1xdWVyeScpO1xuICAgIH1cbiAgICBpZiAoYWRhcHRlci5mZWF0dXJlcy5oYXMoJ2JncmE4dW5vcm0tc3RvcmFnZScpKSB7XG4gICAgICByZXF1aXJlZEZlYXR1cmVzLnB1c2goWydiZ3JhOHVub3JtLXN0b3JhZ2UnXSk7XG4gICAgfVxuICAgIGRldmljZURlc2NyaXB0b3IucmVxdWlyZWRGZWF0dXJlcyA9XG4gICAgICAgIHJlcXVpcmVkRmVhdHVyZXMgYXMgSXRlcmFibGU8R1BVRmVhdHVyZU5hbWU+O1xuXG4gICAgY29uc3QgYWRhcHRlckxpbWl0cyA9IGFkYXB0ZXIubGltaXRzO1xuICAgIGRldmljZURlc2NyaXB0b3IucmVxdWlyZWRMaW1pdHMgPSB7XG4gICAgICAnbWF4Q29tcHV0ZVdvcmtncm91cFN0b3JhZ2VTaXplJzpcbiAgICAgICAgICBhZGFwdGVyTGltaXRzLm1heENvbXB1dGVXb3JrZ3JvdXBTdG9yYWdlU2l6ZSxcbiAgICAgICdtYXhDb21wdXRlV29ya2dyb3Vwc1BlckRpbWVuc2lvbic6XG4gICAgICAgICAgYWRhcHRlckxpbWl0cy5tYXhDb21wdXRlV29ya2dyb3Vwc1BlckRpbWVuc2lvbixcbiAgICAgICdtYXhTdG9yYWdlQnVmZmVyQmluZGluZ1NpemUnOiBhZGFwdGVyTGltaXRzLm1heFN0b3JhZ2VCdWZmZXJCaW5kaW5nU2l6ZSxcbiAgICAgICdtYXhCdWZmZXJTaXplJzogYWRhcHRlckxpbWl0cy5tYXhCdWZmZXJTaXplLFxuICAgICAgJ21heENvbXB1dGVXb3JrZ3JvdXBTaXplWCc6IGFkYXB0ZXJMaW1pdHMubWF4Q29tcHV0ZVdvcmtncm91cFNpemVYLFxuICAgICAgJ21heENvbXB1dGVJbnZvY2F0aW9uc1Blcldvcmtncm91cCc6XG4gICAgICAgICAgYWRhcHRlckxpbWl0cy5tYXhDb21wdXRlSW52b2NhdGlvbnNQZXJXb3JrZ3JvdXAsXG4gICAgfTtcblxuICAgIGNvbnN0IGRldmljZTogR1BVRGV2aWNlID0gYXdhaXQgYWRhcHRlci5yZXF1ZXN0RGV2aWNlKGRldmljZURlc2NyaXB0b3IpO1xuICAgIGNvbnN0IGFkYXB0ZXJJbmZvID1cbiAgICAgICdpbmZvJyBpbiBhZGFwdGVyXG4gICAgICAgID8gYWRhcHRlci5pbmZvXG4gICAgICAgIDogJ3JlcXVlc3RBZGFwdGVySW5mbycgaW4gYWRhcHRlclxuICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICA/IGF3YWl0IChhZGFwdGVyIGFzIGFueSkucmVxdWVzdEFkYXB0ZXJJbmZvKClcbiAgICAgICAgICA6IHVuZGVmaW5lZDtcbiAgICByZXR1cm4gbmV3IFdlYkdQVUJhY2tlbmQoZGV2aWNlLCBhZGFwdGVySW5mbyk7XG4gIH0sIDMgLypwcmlvcml0eSovKTtcbn1cblxuLy8gRXhwb3J0IHdlYmdwdSB1dGlsaXRpZXNcbmV4cG9ydCAqIGZyb20gJy4vd2ViZ3B1JztcbiJdfQ==