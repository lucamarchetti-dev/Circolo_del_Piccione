/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ReduceProgram {
    constructor(reduceInfo, reduceType, maxComputeWorkgroupSizeX) {
        this.variableNames = ['x'];
        this.uniforms = 'reduceSize : i32,';
        this.size = true;
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        const [outputShape,] = backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        // If reduceSize |reduceInfo.inSize| is very large, the I/O accessing will
        // become the bottleneck. Increasing workgroupSize can reduce the times of
        // accessing global memory. The threshold value is just to make sure the
        // reduceSize is large enough for a bigger workgroupSize.
        if (reduceInfo.inSize >= 32768 && maxComputeWorkgroupSizeX >= 512) {
            this.workgroupSize = [512, 1, 1];
        }
        else if (reduceInfo.inSize >= 4096) {
            this.workgroupSize = [256, 1, 1];
        }
        else {
            this.workgroupSize = [64, 1, 1];
        }
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        // A work group only outputs a data, so we transfer [1, 1, 1] to compute
        // dispatch size.
        this.dispatch =
            computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);
        this.reduceType = reduceType;
        this.shaderKey = `reduce_${reduceType}`;
    }
    getUserCode() {
        let reduceOp = ``;
        let initValue = '0.0';
        const workgroupSizeX = this.workgroupSize[0];
        if (this.reduceType === 'min' || this.reduceType === 'max') {
            reduceOp = `
         if (isnan(candidate)) {
          bestValue = uniforms.NAN;
         } else if (!isnan(bestValue) && candidate ${this.reduceType === 'min' ? '<' : '>'} bestValue)
           {  bestValue = candidate; }`;
            initValue = 'f32(x[offset])';
        }
        else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
            reduceOp = ' bestValue = bestValue + candidate; ';
        }
        else if (this.reduceType === 'prod') {
            reduceOp = ' bestValue = bestValue * candidate; ';
            initValue = '1.0';
        }
        else if (this.reduceType === 'all') {
            reduceOp = ' bestValue = f32(bestValue >= 1.0 && candidate >= 1.0); ';
            initValue = '1.0';
        }
        else if (this.reduceType === 'any') {
            reduceOp = ' bestValue = f32(bestValue >= 1.0 || candidate >= 1.0); ';
            initValue = '0.0';
        }
        const outputSnippet = this.reduceType === 'mean' ?
            // tslint:disable-next-line:max-line-length
            `setOutputAtIndex(outputIndex, bestValue / f32(uniforms.reduceSize));` :
            `setOutputAtIndex(outputIndex, bestValue);`;
        const sharedMemorySnippet = `
         var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
       `;
        const userCode = `
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }

       ${sharedMemorySnippet}
       fn getOffset(outputIndex : i32) -> i32 {
         let outputCoords = getCoordsFromIndex(outputIndex);
         let offset = ${this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
          return offset;
       }
       ${main('index')} {
         let outputIndex = index / ${workgroupSizeX};
         let offset = getOffset(outputIndex);
         var bestValue = ${initValue};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(u32(Length), ${workgroupSizeX}u);
         for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
             k = k + ${workgroupSizeX}) {
           let candidate = f32(x[offset + k]);
           ${reduceOp}
         }
         xBestValues[localId.x] = bestValue;
         workgroupBarrier();

         var reduceSize = min(u32(Length), ${workgroupSizeX}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            ${reduceOp}
            xBestValues[localId.x] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (localId.x == 0u && outputIndex < uniforms.size) {
          ${outputSnippet}
        }
       }
     `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVkdWNlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3JlZHVjZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ25ELE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sYUFBYTtJQVl4QixZQUNJLFVBQW1DLEVBQ25DLFVBQXVELEVBQ3ZELHdCQUFnQztRQVRwQyxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUFHLG1CQUFtQixDQUFDO1FBRy9CLFNBQUksR0FBRyxJQUFJLENBQUM7UUFNVixJQUFJLENBQUMsVUFBVSxHQUFHLENBQUMsVUFBVSxDQUFDLFNBQVMsRUFBRSxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDNUQsTUFBTSxDQUFDLFdBQVcsRUFBRyxHQUNqQixZQUFZLENBQUMseUJBQXlCLENBQUMsSUFBSSxDQUFDLFVBQVUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDakUsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUMsTUFBTSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsV0FBVyxDQUFDO1FBQ2hFLDBFQUEwRTtRQUMxRSwwRUFBMEU7UUFDMUUsd0VBQXdFO1FBQ3hFLHlEQUF5RDtRQUN6RCxJQUFJLFVBQVUsQ0FBQyxNQUFNLElBQUksS0FBSyxJQUFJLHdCQUF3QixJQUFJLEdBQUcsRUFBRTtZQUNqRSxJQUFJLENBQUMsYUFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztTQUNsQzthQUFNLElBQUksVUFBVSxDQUFDLE1BQU0sSUFBSSxJQUFJLEVBQUU7WUFDcEMsSUFBSSxDQUFDLGFBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7U0FDbEM7YUFBTTtZQUNMLElBQUksQ0FBQyxhQUFhLEdBQUcsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1NBQ2pDO1FBQ0QsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0Qsd0VBQXdFO1FBQ3hFLGlCQUFpQjtRQUNqQixJQUFJLENBQUMsUUFBUTtZQUNULGVBQWUsQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdEUsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7UUFDN0IsSUFBSSxDQUFDLFNBQVMsR0FBRyxVQUFVLFVBQVUsRUFBRSxDQUFDO0lBQzFDLENBQUM7SUFFRCxXQUFXO1FBQ1QsSUFBSSxRQUFRLEdBQUcsRUFBRSxDQUFDO1FBQ2xCLElBQUksU0FBUyxHQUFHLEtBQUssQ0FBQztRQUN0QixNQUFNLGNBQWMsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzdDLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxLQUFLLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxLQUFLLEVBQUU7WUFDMUQsUUFBUSxHQUFHOzs7cURBSVAsSUFBSSxDQUFDLFVBQVUsS0FBSyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRzt1Q0FDUixDQUFDO1lBQ2xDLFNBQVMsR0FBRyxnQkFBZ0IsQ0FBQztTQUM5QjthQUFNLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxLQUFLLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxNQUFNLEVBQUU7WUFDbEUsUUFBUSxHQUFHLHNDQUFzQyxDQUFDO1NBQ25EO2FBQU0sSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLE1BQU0sRUFBRTtZQUNyQyxRQUFRLEdBQUcsc0NBQXNDLENBQUM7WUFDbEQsU0FBUyxHQUFHLEtBQUssQ0FBQztTQUNuQjthQUFNLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxLQUFLLEVBQUU7WUFDcEMsUUFBUSxHQUFHLDBEQUEwRCxDQUFDO1lBQ3RFLFNBQVMsR0FBRyxLQUFLLENBQUM7U0FDbkI7YUFBTSxJQUFJLElBQUksQ0FBQyxVQUFVLEtBQUssS0FBSyxFQUFFO1lBQ3BDLFFBQVEsR0FBRywwREFBMEQsQ0FBQztZQUN0RSxTQUFTLEdBQUcsS0FBSyxDQUFDO1NBQ25CO1FBRUQsTUFBTSxhQUFhLEdBQUcsSUFBSSxDQUFDLFVBQVUsS0FBSyxNQUFNLENBQUMsQ0FBQztZQUM5QywyQ0FBMkM7WUFDM0Msc0VBQXNFLENBQUMsQ0FBQztZQUN4RSwyQ0FBMkMsQ0FBQztRQUVoRCxNQUFNLG1CQUFtQixHQUFHO21EQUNtQixjQUFjO1FBQ3pELENBQUM7UUFFTCxNQUFNLFFBQVEsR0FBRzs7Ozs7U0FLWixtQkFBbUI7Ozt3QkFJcEIsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDM0IsY0FBYyxDQUFDLENBQUM7WUFDaEIsaUJBQWlCOzs7U0FHcEIsSUFBSSxDQUFDLE9BQU8sQ0FBQztxQ0FDZSxjQUFjOzsyQkFFeEIsU0FBUzs7cURBRWlCLGNBQWM7O3VCQUU1QyxjQUFjOzthQUV4QixRQUFROzs7Ozs2Q0FLd0IsY0FBYzs7Ozs7O2NBTTdDLFFBQVE7Ozs7Ozs7O1lBUVYsYUFBYTs7O01BR25CLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBSZWR1Y2VQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgdW5pZm9ybXMgPSAncmVkdWNlU2l6ZSA6IGkzMiwnO1xuICByZWR1Y2VUeXBlOiAnYWxsJ3wnYW55J3wnbWF4J3wnbWVhbid8J21pbid8J3Byb2QnfCdzdW0nO1xuICBpbnB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICByZWR1Y2VJbmZvOiBiYWNrZW5kX3V0aWwuUmVkdWNlSW5mbyxcbiAgICAgIHJlZHVjZVR5cGU6ICdhbGwnfCdhbnknfCdtYXgnfCdtZWFuJ3wnbWluJ3wncHJvZCd8J3N1bScsXG4gICAgICBtYXhDb21wdXRlV29ya2dyb3VwU2l6ZVg6IG51bWJlcikge1xuICAgIHRoaXMuaW5wdXRTaGFwZSA9IFtyZWR1Y2VJbmZvLmJhdGNoU2l6ZSwgcmVkdWNlSW5mby5pblNpemVdO1xuICAgIGNvbnN0IFtvdXRwdXRTaGFwZSwgXSA9XG4gICAgICAgIGJhY2tlbmRfdXRpbC5jb21wdXRlT3V0QW5kUmVkdWNlU2hhcGVzKHRoaXMuaW5wdXRTaGFwZSwgWzFdKTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGUubGVuZ3RoID09PSAwID8gWzFdIDogb3V0cHV0U2hhcGU7XG4gICAgLy8gSWYgcmVkdWNlU2l6ZSB8cmVkdWNlSW5mby5pblNpemV8IGlzIHZlcnkgbGFyZ2UsIHRoZSBJL08gYWNjZXNzaW5nIHdpbGxcbiAgICAvLyBiZWNvbWUgdGhlIGJvdHRsZW5lY2suIEluY3JlYXNpbmcgd29ya2dyb3VwU2l6ZSBjYW4gcmVkdWNlIHRoZSB0aW1lcyBvZlxuICAgIC8vIGFjY2Vzc2luZyBnbG9iYWwgbWVtb3J5LiBUaGUgdGhyZXNob2xkIHZhbHVlIGlzIGp1c3QgdG8gbWFrZSBzdXJlIHRoZVxuICAgIC8vIHJlZHVjZVNpemUgaXMgbGFyZ2UgZW5vdWdoIGZvciBhIGJpZ2dlciB3b3JrZ3JvdXBTaXplLlxuICAgIGlmIChyZWR1Y2VJbmZvLmluU2l6ZSA+PSAzMjc2OCAmJiBtYXhDb21wdXRlV29ya2dyb3VwU2l6ZVggPj0gNTEyKSB7XG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSBbNTEyLCAxLCAxXTtcbiAgICB9IGVsc2UgaWYgKHJlZHVjZUluZm8uaW5TaXplID49IDQwOTYpIHtcbiAgICAgIHRoaXMud29ya2dyb3VwU2l6ZSA9IFsyNTYsIDEsIDFdO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSBbNjQsIDEsIDFdO1xuICAgIH1cbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIC8vIEEgd29yayBncm91cCBvbmx5IG91dHB1dHMgYSBkYXRhLCBzbyB3ZSB0cmFuc2ZlciBbMSwgMSwgMV0gdG8gY29tcHV0ZVxuICAgIC8vIGRpc3BhdGNoIHNpemUuXG4gICAgdGhpcy5kaXNwYXRjaCA9XG4gICAgICAgIGNvbXB1dGVEaXNwYXRjaCh0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCBbMSwgMSwgMV0pO1xuXG4gICAgdGhpcy5yZWR1Y2VUeXBlID0gcmVkdWNlVHlwZTtcbiAgICB0aGlzLnNoYWRlcktleSA9IGByZWR1Y2VfJHtyZWR1Y2VUeXBlfWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGxldCByZWR1Y2VPcCA9IGBgO1xuICAgIGxldCBpbml0VmFsdWUgPSAnMC4wJztcbiAgICBjb25zdCB3b3JrZ3JvdXBTaXplWCA9IHRoaXMud29ya2dyb3VwU2l6ZVswXTtcbiAgICBpZiAodGhpcy5yZWR1Y2VUeXBlID09PSAnbWluJyB8fCB0aGlzLnJlZHVjZVR5cGUgPT09ICdtYXgnKSB7XG4gICAgICByZWR1Y2VPcCA9IGBcbiAgICAgICAgIGlmIChpc25hbihjYW5kaWRhdGUpKSB7XG4gICAgICAgICAgYmVzdFZhbHVlID0gdW5pZm9ybXMuTkFOO1xuICAgICAgICAgfSBlbHNlIGlmICghaXNuYW4oYmVzdFZhbHVlKSAmJiBjYW5kaWRhdGUgJHtcbiAgICAgICAgICB0aGlzLnJlZHVjZVR5cGUgPT09ICdtaW4nID8gJzwnIDogJz4nfSBiZXN0VmFsdWUpXG4gICAgICAgICAgIHsgIGJlc3RWYWx1ZSA9IGNhbmRpZGF0ZTsgfWA7XG4gICAgICBpbml0VmFsdWUgPSAnZjMyKHhbb2Zmc2V0XSknO1xuICAgIH0gZWxzZSBpZiAodGhpcy5yZWR1Y2VUeXBlID09PSAnc3VtJyB8fCB0aGlzLnJlZHVjZVR5cGUgPT09ICdtZWFuJykge1xuICAgICAgcmVkdWNlT3AgPSAnIGJlc3RWYWx1ZSA9IGJlc3RWYWx1ZSArIGNhbmRpZGF0ZTsgJztcbiAgICB9IGVsc2UgaWYgKHRoaXMucmVkdWNlVHlwZSA9PT0gJ3Byb2QnKSB7XG4gICAgICByZWR1Y2VPcCA9ICcgYmVzdFZhbHVlID0gYmVzdFZhbHVlICogY2FuZGlkYXRlOyAnO1xuICAgICAgaW5pdFZhbHVlID0gJzEuMCc7XG4gICAgfSBlbHNlIGlmICh0aGlzLnJlZHVjZVR5cGUgPT09ICdhbGwnKSB7XG4gICAgICByZWR1Y2VPcCA9ICcgYmVzdFZhbHVlID0gZjMyKGJlc3RWYWx1ZSA+PSAxLjAgJiYgY2FuZGlkYXRlID49IDEuMCk7ICc7XG4gICAgICBpbml0VmFsdWUgPSAnMS4wJztcbiAgICB9IGVsc2UgaWYgKHRoaXMucmVkdWNlVHlwZSA9PT0gJ2FueScpIHtcbiAgICAgIHJlZHVjZU9wID0gJyBiZXN0VmFsdWUgPSBmMzIoYmVzdFZhbHVlID49IDEuMCB8fCBjYW5kaWRhdGUgPj0gMS4wKTsgJztcbiAgICAgIGluaXRWYWx1ZSA9ICcwLjAnO1xuICAgIH1cblxuICAgIGNvbnN0IG91dHB1dFNuaXBwZXQgPSB0aGlzLnJlZHVjZVR5cGUgPT09ICdtZWFuJyA/XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTptYXgtbGluZS1sZW5ndGhcbiAgICAgICAgYHNldE91dHB1dEF0SW5kZXgob3V0cHV0SW5kZXgsIGJlc3RWYWx1ZSAvIGYzMih1bmlmb3Jtcy5yZWR1Y2VTaXplKSk7YCA6XG4gICAgICAgIGBzZXRPdXRwdXRBdEluZGV4KG91dHB1dEluZGV4LCBiZXN0VmFsdWUpO2A7XG5cbiAgICBjb25zdCBzaGFyZWRNZW1vcnlTbmlwcGV0ID0gYFxuICAgICAgICAgdmFyPHdvcmtncm91cD4geEJlc3RWYWx1ZXMgOiBhcnJheTxmMzIsICR7d29ya2dyb3VwU2l6ZVh9PjtcbiAgICAgICBgO1xuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAgZm4gRElWX0NFSUwoYSA6IHUzMiwgYiA6IHUzMikgLT4gdTMyIHtcbiAgICAgICAgcmV0dXJuICgoYSAtIDF1KSAvIGIgKyAxdSk7XG4gICAgICAgfVxuXG4gICAgICAgJHtzaGFyZWRNZW1vcnlTbmlwcGV0fVxuICAgICAgIGZuIGdldE9mZnNldChvdXRwdXRJbmRleCA6IGkzMikgLT4gaTMyIHtcbiAgICAgICAgIGxldCBvdXRwdXRDb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgob3V0cHV0SW5kZXgpO1xuICAgICAgICAgbGV0IG9mZnNldCA9ICR7XG4gICAgICAgIHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoID09PSAxID9cbiAgICAgICAgICAgICdvdXRwdXRDb29yZHMnIDpcbiAgICAgICAgICAgICdvdXRwdXRDb29yZHNbMF0nfSAqIHVuaWZvcm1zLnJlZHVjZVNpemU7XG4gICAgICAgICAgcmV0dXJuIG9mZnNldDtcbiAgICAgICB9XG4gICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgICBsZXQgb3V0cHV0SW5kZXggPSBpbmRleCAvICR7d29ya2dyb3VwU2l6ZVh9O1xuICAgICAgICAgbGV0IG9mZnNldCA9IGdldE9mZnNldChvdXRwdXRJbmRleCk7XG4gICAgICAgICB2YXIgYmVzdFZhbHVlID0gJHtpbml0VmFsdWV9O1xuICAgICAgICAgbGV0IExlbmd0aCA9IHVuaWZvcm1zLnJlZHVjZVNpemU7XG4gICAgICAgICBsZXQgV29ya1BlclRocmVhZCA9IERJVl9DRUlMKHUzMihMZW5ndGgpLCAke3dvcmtncm91cFNpemVYfXUpO1xuICAgICAgICAgZm9yICh2YXIgayA9IGkzMihsb2NhbElkLngpOyBrIDwgTGVuZ3RoICYmIG91dHB1dEluZGV4IDwgdW5pZm9ybXMuc2l6ZTtcbiAgICAgICAgICAgICBrID0gayArICR7d29ya2dyb3VwU2l6ZVh9KSB7XG4gICAgICAgICAgIGxldCBjYW5kaWRhdGUgPSBmMzIoeFtvZmZzZXQgKyBrXSk7XG4gICAgICAgICAgICR7cmVkdWNlT3B9XG4gICAgICAgICB9XG4gICAgICAgICB4QmVzdFZhbHVlc1tsb2NhbElkLnhdID0gYmVzdFZhbHVlO1xuICAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICAgICB2YXIgcmVkdWNlU2l6ZSA9IG1pbih1MzIoTGVuZ3RoKSwgJHt3b3JrZ3JvdXBTaXplWH11KTtcbiAgICAgICAgIGZvciAodmFyIGN1cnJlbnRTaXplID0gcmVkdWNlU2l6ZSAvIDJ1OyByZWR1Y2VTaXplID4gMXU7XG4gICAgICAgICAgICAgY3VycmVudFNpemUgPSByZWR1Y2VTaXplIC8gMnUpIHtcbiAgICAgICAgICAgbGV0IGludGVydmFsID0gRElWX0NFSUwocmVkdWNlU2l6ZSwgMnUpO1xuICAgICAgICAgICBpZiAobG9jYWxJZC54IDwgY3VycmVudFNpemUpIHtcbiAgICAgICAgICAgIGxldCBjYW5kaWRhdGUgPSB4QmVzdFZhbHVlc1tsb2NhbElkLnggKyBpbnRlcnZhbF07XG4gICAgICAgICAgICAke3JlZHVjZU9wfVxuICAgICAgICAgICAgeEJlc3RWYWx1ZXNbbG9jYWxJZC54XSA9IGJlc3RWYWx1ZTtcbiAgICAgICAgICAgfVxuICAgICAgICAgICByZWR1Y2VTaXplID0gaW50ZXJ2YWw7XG4gICAgICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcbiAgICAgICAgIH1cblxuICAgICAgICAgaWYgKGxvY2FsSWQueCA9PSAwdSAmJiBvdXRwdXRJbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICAke291dHB1dFNuaXBwZXR9XG4gICAgICAgIH1cbiAgICAgICB9XG4gICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=