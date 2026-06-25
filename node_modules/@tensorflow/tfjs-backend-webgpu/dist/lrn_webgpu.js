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
import { util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
const powOperatorSnippet = `
  var powValue = 0.0;
  let basis = uniforms.bias + uniforms.alpha * sum;
  if (uniforms.beta == 0.5) {
    powValue = inverseSqrt(basis);
  } else if (uniforms.beta == 1.0) {
    powValue = 1.0 / basis;
  } else {
    powValue = exp(log(basis) * (-uniforms.beta));
  }
`;
export class LRNProgram {
    constructor(xShape) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'lrn';
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];
        let d = coords[3];

        let x = getX(b, r, c, d);
        var sum = 0.0;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let idx = d + i;
          if (idx >= 0 && idx < uniforms.xShape[3]) {
            let z = getX(b, r, c, idx);
            sum = sum + z * z;
          }
        }
        ${powOperatorSnippet}

        setOutputAtIndex(index, x * powValue);
      }
    }
  `;
        return userCode;
    }
}
export class LRNSharedProgram {
    constructor(xShape, radius) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
        this.workgroupSize = [256, 1, 1];
        this.maxAllowRadius = 16;
        util.assert(radius <= this.maxAllowRadius, () => `Radius must be less than or equal to ${this.maxAllowRadius}, current radius is ${radius}`);
        this.outputShape = xShape;
        // The reason why not using this.workgroupSize[0] + 2 * maxAllowRadius here
        // is to make sure that there is only one time global memory load access for
        // each thread.
        this.elementsPerWorkgroup = this.workgroupSize[0] - 2 * this.maxAllowRadius;
        this.dispatchLayout = { x: [3], y: [2], z: [0, 1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, [
            this.elementsPerWorkgroup, this.workgroupSize[1], this.workgroupSize[2]
        ]);
        this.shaderKey = 'lrn_shared';
    }
    getUserCode() {
        const userCode = `
    var <workgroup>lrnSub: array<f32, ${this.workgroupSize[0]}>;
    const elementsPerWorkgroup = ${this.elementsPerWorkgroup};
    const maxAllowRadius = ${this.maxAllowRadius};

    ${main()} {
      let localDepth = i32(localId.x);
      let workgroupDepth = i32(workgroupId.x) * elementsPerWorkgroup;
      let xDepth = workgroupDepth + localDepth - maxAllowRadius;
      let b = i32(globalId.z) / uniforms.xShape[1];
      let r = i32(globalId.z) - b * uniforms.xShape[1];
      let c = i32(globalId.y);
      let d = workgroupDepth + localDepth;

      var x = 0.0;
      if (xDepth >= 0 && xDepth < uniforms.xShape[3]) {
        x = getX(b, r, c, xDepth);
      }
      lrnSub[localDepth] = x;
      workgroupBarrier();

      if (localDepth < elementsPerWorkgroup && d < uniforms.outShape[3]) {
        var sum = 0.0;
        let index = localDepth + maxAllowRadius;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let z = lrnSub[index + i];
          sum = sum + z * z;
        }
        ${powOperatorSnippet}

        setOutputAtCoords(b, r, c, d, lrnSub[index] * powValue);
      }
    } `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibHJuX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2xybl93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQzNDLE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLGtCQUFrQixHQUFHOzs7Ozs7Ozs7O0NBVTFCLENBQUM7QUFFRixNQUFNLE9BQU8sVUFBVTtJQVVyQixZQUFZLE1BQWdCO1FBVDVCLGdCQUFXLEdBQWEsRUFBRSxDQUFDO1FBSTNCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN0QixhQUFRLEdBQUcsb0RBQW9ELENBQUM7UUFDaEUsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQztRQUMxQixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsU0FBUyxHQUFHLEtBQUssQ0FBQztJQUN6QixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7VUFpQlQsa0JBQWtCOzs7OztHQUt6QixDQUFDO1FBQ0EsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGO0FBRUQsTUFBTSxPQUFPLGdCQUFnQjtJQVczQixZQUFZLE1BQWdCLEVBQUUsTUFBYztRQVY1QyxnQkFBVyxHQUFhLEVBQUUsQ0FBQztRQUkzQixrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUFHLG9EQUFvRCxDQUFDO1FBQ2hFLGtCQUFhLEdBQTZCLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUN0RCxtQkFBYyxHQUFHLEVBQUUsQ0FBQztRQUlsQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sSUFBSSxJQUFJLENBQUMsY0FBYyxFQUM3QixHQUFHLEVBQUUsQ0FBQyx3Q0FDRixJQUFJLENBQUMsY0FBYyx1QkFBdUIsTUFBTSxFQUFFLENBQUMsQ0FBQztRQUU1RCxJQUFJLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQztRQUMxQiwyRUFBMkU7UUFDM0UsNEVBQTRFO1FBQzVFLGVBQWU7UUFDZixJQUFJLENBQUMsb0JBQW9CLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQztRQUM1RSxJQUFJLENBQUMsY0FBYyxHQUFHLEVBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFDLENBQUM7UUFDbEQsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQUMsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFO1lBQ3JFLElBQUksQ0FBQyxvQkFBb0IsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDO1NBQ3hFLENBQUMsQ0FBQztRQUNILElBQUksQ0FBQyxTQUFTLEdBQUcsWUFBWSxDQUFDO0lBQ2hDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7d0NBQ21CLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDO21DQUMxQixJQUFJLENBQUMsb0JBQW9COzZCQUMvQixJQUFJLENBQUMsY0FBYzs7TUFFMUMsSUFBSSxFQUFFOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztVQXVCRixrQkFBa0I7Ozs7T0FJckIsQ0FBQztRQUNKLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHt1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmNvbnN0IHBvd09wZXJhdG9yU25pcHBldCA9IGBcbiAgdmFyIHBvd1ZhbHVlID0gMC4wO1xuICBsZXQgYmFzaXMgPSB1bmlmb3Jtcy5iaWFzICsgdW5pZm9ybXMuYWxwaGEgKiBzdW07XG4gIGlmICh1bmlmb3Jtcy5iZXRhID09IDAuNSkge1xuICAgIHBvd1ZhbHVlID0gaW52ZXJzZVNxcnQoYmFzaXMpO1xuICB9IGVsc2UgaWYgKHVuaWZvcm1zLmJldGEgPT0gMS4wKSB7XG4gICAgcG93VmFsdWUgPSAxLjAgLyBiYXNpcztcbiAgfSBlbHNlIHtcbiAgICBwb3dWYWx1ZSA9IGV4cChsb2coYmFzaXMpICogKC11bmlmb3Jtcy5iZXRhKSk7XG4gIH1cbmA7XG5cbmV4cG9ydCBjbGFzcyBMUk5Qcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgdW5pZm9ybXMgPSAncmFkaXVzIDogaTMyLCBiaWFzIDogZjMyLCBhbHBoYSA6IGYzMiwgYmV0YSA6IGYzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3Rvcih4U2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IHhTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnbHJuJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgbGV0IGIgPSBjb29yZHNbMF07XG4gICAgICAgIGxldCByID0gY29vcmRzWzFdO1xuICAgICAgICBsZXQgYyA9IGNvb3Jkc1syXTtcbiAgICAgICAgbGV0IGQgPSBjb29yZHNbM107XG5cbiAgICAgICAgbGV0IHggPSBnZXRYKGIsIHIsIGMsIGQpO1xuICAgICAgICB2YXIgc3VtID0gMC4wO1xuICAgICAgICBmb3IgKHZhciBpID0gLXVuaWZvcm1zLnJhZGl1czsgaSA8PSB1bmlmb3Jtcy5yYWRpdXM7IGkgPSBpICsgMSkge1xuICAgICAgICAgIGxldCBpZHggPSBkICsgaTtcbiAgICAgICAgICBpZiAoaWR4ID49IDAgJiYgaWR4IDwgdW5pZm9ybXMueFNoYXBlWzNdKSB7XG4gICAgICAgICAgICBsZXQgeiA9IGdldFgoYiwgciwgYywgaWR4KTtcbiAgICAgICAgICAgIHN1bSA9IHN1bSArIHogKiB6O1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICAke3Bvd09wZXJhdG9yU25pcHBldH1cblxuICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCB4ICogcG93VmFsdWUpO1xuICAgICAgfVxuICAgIH1cbiAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIExSTlNoYXJlZFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdID0gW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXSwgejogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zID0gJ3JhZGl1cyA6IGkzMiwgYmlhcyA6IGYzMiwgYWxwaGEgOiBmMzIsIGJldGEgOiBmMzIsJztcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzI1NiwgMSwgMV07XG4gIG1heEFsbG93UmFkaXVzID0gMTY7XG4gIGVsZW1lbnRzUGVyV29ya2dyb3VwOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoeFNoYXBlOiBudW1iZXJbXSwgcmFkaXVzOiBudW1iZXIpIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgcmFkaXVzIDw9IHRoaXMubWF4QWxsb3dSYWRpdXMsXG4gICAgICAgICgpID0+IGBSYWRpdXMgbXVzdCBiZSBsZXNzIHRoYW4gb3IgZXF1YWwgdG8gJHtcbiAgICAgICAgICAgIHRoaXMubWF4QWxsb3dSYWRpdXN9LCBjdXJyZW50IHJhZGl1cyBpcyAke3JhZGl1c31gKTtcblxuICAgIHRoaXMub3V0cHV0U2hhcGUgPSB4U2hhcGU7XG4gICAgLy8gVGhlIHJlYXNvbiB3aHkgbm90IHVzaW5nIHRoaXMud29ya2dyb3VwU2l6ZVswXSArIDIgKiBtYXhBbGxvd1JhZGl1cyBoZXJlXG4gICAgLy8gaXMgdG8gbWFrZSBzdXJlIHRoYXQgdGhlcmUgaXMgb25seSBvbmUgdGltZSBnbG9iYWwgbWVtb3J5IGxvYWQgYWNjZXNzIGZvclxuICAgIC8vIGVhY2ggdGhyZWFkLlxuICAgIHRoaXMuZWxlbWVudHNQZXJXb3JrZ3JvdXAgPSB0aGlzLndvcmtncm91cFNpemVbMF0gLSAyICogdGhpcy5tYXhBbGxvd1JhZGl1cztcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0ge3g6IFszXSwgeTogWzJdLCB6OiBbMCwgMV19O1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2godGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgW1xuICAgICAgdGhpcy5lbGVtZW50c1Blcldvcmtncm91cCwgdGhpcy53b3JrZ3JvdXBTaXplWzFdLCB0aGlzLndvcmtncm91cFNpemVbMl1cbiAgICBdKTtcbiAgICB0aGlzLnNoYWRlcktleSA9ICdscm5fc2hhcmVkJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgdmFyIDx3b3JrZ3JvdXA+bHJuU3ViOiBhcnJheTxmMzIsICR7dGhpcy53b3JrZ3JvdXBTaXplWzBdfT47XG4gICAgY29uc3QgZWxlbWVudHNQZXJXb3JrZ3JvdXAgPSAke3RoaXMuZWxlbWVudHNQZXJXb3JrZ3JvdXB9O1xuICAgIGNvbnN0IG1heEFsbG93UmFkaXVzID0gJHt0aGlzLm1heEFsbG93UmFkaXVzfTtcblxuICAgICR7bWFpbigpfSB7XG4gICAgICBsZXQgbG9jYWxEZXB0aCA9IGkzMihsb2NhbElkLngpO1xuICAgICAgbGV0IHdvcmtncm91cERlcHRoID0gaTMyKHdvcmtncm91cElkLngpICogZWxlbWVudHNQZXJXb3JrZ3JvdXA7XG4gICAgICBsZXQgeERlcHRoID0gd29ya2dyb3VwRGVwdGggKyBsb2NhbERlcHRoIC0gbWF4QWxsb3dSYWRpdXM7XG4gICAgICBsZXQgYiA9IGkzMihnbG9iYWxJZC56KSAvIHVuaWZvcm1zLnhTaGFwZVsxXTtcbiAgICAgIGxldCByID0gaTMyKGdsb2JhbElkLnopIC0gYiAqIHVuaWZvcm1zLnhTaGFwZVsxXTtcbiAgICAgIGxldCBjID0gaTMyKGdsb2JhbElkLnkpO1xuICAgICAgbGV0IGQgPSB3b3JrZ3JvdXBEZXB0aCArIGxvY2FsRGVwdGg7XG5cbiAgICAgIHZhciB4ID0gMC4wO1xuICAgICAgaWYgKHhEZXB0aCA+PSAwICYmIHhEZXB0aCA8IHVuaWZvcm1zLnhTaGFwZVszXSkge1xuICAgICAgICB4ID0gZ2V0WChiLCByLCBjLCB4RGVwdGgpO1xuICAgICAgfVxuICAgICAgbHJuU3ViW2xvY2FsRGVwdGhdID0geDtcbiAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgICAgaWYgKGxvY2FsRGVwdGggPCBlbGVtZW50c1Blcldvcmtncm91cCAmJiBkIDwgdW5pZm9ybXMub3V0U2hhcGVbM10pIHtcbiAgICAgICAgdmFyIHN1bSA9IDAuMDtcbiAgICAgICAgbGV0IGluZGV4ID0gbG9jYWxEZXB0aCArIG1heEFsbG93UmFkaXVzO1xuICAgICAgICBmb3IgKHZhciBpID0gLXVuaWZvcm1zLnJhZGl1czsgaSA8PSB1bmlmb3Jtcy5yYWRpdXM7IGkgPSBpICsgMSkge1xuICAgICAgICAgIGxldCB6ID0gbHJuU3ViW2luZGV4ICsgaV07XG4gICAgICAgICAgc3VtID0gc3VtICsgeiAqIHo7XG4gICAgICAgIH1cbiAgICAgICAgJHtwb3dPcGVyYXRvclNuaXBwZXR9XG5cbiAgICAgICAgc2V0T3V0cHV0QXRDb29yZHMoYiwgciwgYywgZCwgbHJuU3ViW2luZGV4XSAqIHBvd1ZhbHVlKTtcbiAgICAgIH1cbiAgICB9IGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=