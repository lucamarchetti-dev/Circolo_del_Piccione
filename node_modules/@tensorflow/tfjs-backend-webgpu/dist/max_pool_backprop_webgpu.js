/**
 * @license
 * Copyright 2023 Google LLC.
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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class MaxPool2DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'maxPos'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'maxPool2DBackprop';
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d = coords[3];

        let dyRCCorner = vec2<i32>(coords.yz) - uniforms.pads;
        let dyRCorner = dyRCCorner.x;
        let dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] - 1;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR += uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC += uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);
            let maxPosValue = lastIndex - i32(getMaxPos(batch, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            let curPosValue = wR * uniforms.filterDims[1] + wC;
            let mask = select(0.0, 1.0, maxPosValue == curPosValue);
            dotProd += dyValue * mask;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
export class MaxPool3DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'maxPos'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
      outDepth : i32, outHeight : i32, outWidth : i32`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'maxPool3DBackprop';
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let ch = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyDCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] * uniforms.filterDims[2] - 1;

        for (var wD = 0; wD < uniforms.filterDims[0]; wD++) {
          let dyD = f32(dyDCorner + wD) / f32(uniforms.strides[0]);

          if (dyD < 0.0 || dyD >= f32(uniforms.outDepth) || fract(dyD) > 0.0) {
            continue;
          }
          let idyD = i32(dyD);

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let dyValue = getDy(batch, idyD, idyR, idyC, ch);
              let maxPosValue = lastIndex - i32(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              let curPosValue = wD * uniforms.filterDims[1] * uniforms.filterDims[2] + wR * uniforms.filterDims[2] + wC;
              let mask = select(0.0, 1.0, maxPosValue == curPosValue);
              dotProd += dyValue * mask;
            }
          }
        }

        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4X3Bvb2xfYmFja3Byb3Bfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvbWF4X3Bvb2xfYmFja3Byb3Bfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sd0JBQXdCO0lBWW5DLFlBQVksUUFBaUM7UUFQN0Msa0JBQWEsR0FBRyxDQUFDLElBQUksRUFBRSxRQUFRLENBQUMsQ0FBQztRQUNqQyxhQUFRLEdBQ0o7dUNBQ2lDLENBQUM7UUFDdEMsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUM7UUFFcEMsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxtQkFBbUIsQ0FBQztJQUN2QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQTJDaEIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELE1BQU0sT0FBTyx3QkFBd0I7SUFXbkMsWUFBWSxRQUFpQztRQU43QyxrQkFBYSxHQUFHLENBQUMsSUFBSSxFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBQ2pDLGFBQVEsR0FBRztzREFDeUMsQ0FBQztRQUNyRCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQztRQUVwQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUUzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsU0FBUyxHQUFHLG1CQUFtQixDQUFDO0lBQ3ZDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQXdEaEIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIE1heFBvb2wyREJhY2twcm9wUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsnZHknLCAnbWF4UG9zJ107XG4gIHVuaWZvcm1zID1cbiAgICAgIGBzdHJpZGVzIDogdmVjMjxpMzI+LCBwYWRzIDogdmVjMjxpMzI+LCBkaWxhdGlvbnMgOiB2ZWMyPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMyPGkzMj4sXG4gICAgICAgb3V0SGVpZ2h0IDogaTMyLCBvdXRXaWR0aCA6IGkzMmA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5pblNoYXBlO1xuXG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcblxuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLnNoYWRlcktleSA9ICdtYXhQb29sMkRCYWNrcHJvcCc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgIGxldCBkID0gY29vcmRzWzNdO1xuXG4gICAgICAgIGxldCBkeVJDQ29ybmVyID0gdmVjMjxpMzI+KGNvb3Jkcy55eikgLSB1bmlmb3Jtcy5wYWRzO1xuICAgICAgICBsZXQgZHlSQ29ybmVyID0gZHlSQ0Nvcm5lci54O1xuICAgICAgICBsZXQgZHlDQ29ybmVyID0gZHlSQ0Nvcm5lci55O1xuXG4gICAgICAgIC8vIENvbnZvbHZlIGR5KD8sID8sIGQpIHdpdGggcG9zIG1hc2soOiwgOiwgZCkgdG8gZ2V0IGR4KHhSLCB4QywgZCkuXG4gICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgICB2YXIgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgbGV0IGxhc3RJbmRleCA9IHVuaWZvcm1zLmZpbHRlckRpbXNbMF0gKiB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdIC0gMTtcbiAgICAgICAgZm9yICh2YXIgd1IgPSAwOyB3UiA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMF07IHdSICs9IHVuaWZvcm1zLmRpbGF0aW9uc1swXSkge1xuICAgICAgICAgIGxldCBkeVIgPSBmMzIoZHlSQ29ybmVyICsgd1IpIC8gZjMyKHVuaWZvcm1zLnN0cmlkZXNbMF0pO1xuXG4gICAgICAgICAgaWYgKGR5UiA8IDAuMCB8fCBkeVIgPj0gZjMyKHVuaWZvcm1zLm91dEhlaWdodCkgfHwgZnJhY3QoZHlSKSA+IDAuMCkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuICAgICAgICAgIGxldCBpZHlSID0gaTMyKGR5Uik7XG5cbiAgICAgICAgICBmb3IgKHZhciB3QyA9IDA7IHdDIDwgdW5pZm9ybXMuZmlsdGVyRGltc1sxXTsgd0MgKz0gdW5pZm9ybXMuZGlsYXRpb25zWzFdKSB7XG4gICAgICAgICAgICBsZXQgZHlDID0gZjMyKGR5Q0Nvcm5lciArIHdDKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzFdKTtcblxuICAgICAgICAgICAgaWYgKGR5QyA8IDAuMCB8fCBkeUMgPj0gZjMyKHVuaWZvcm1zLm91dFdpZHRoKSB8fCBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IGlkeUMgPSBpMzIoZHlDKTtcblxuICAgICAgICAgICAgbGV0IGR5VmFsdWUgPSBnZXREeShiYXRjaCwgaWR5UiwgaWR5QywgZCk7XG4gICAgICAgICAgICBsZXQgbWF4UG9zVmFsdWUgPSBsYXN0SW5kZXggLSBpMzIoZ2V0TWF4UG9zKGJhdGNoLCBpZHlSLCBpZHlDLCBkKSk7XG5cbiAgICAgICAgICAgIC8vIEdldCB0aGUgY3VycmVudCB2YWx1ZSwgY2hlY2sgaXQgYWdhaW5zdCB0aGUgdmFsdWUgZnJvbSB0aGVcbiAgICAgICAgICAgIC8vIHBvc2l0aW9uIG1hdHJpeC5cbiAgICAgICAgICAgIGxldCBjdXJQb3NWYWx1ZSA9IHdSICogdW5pZm9ybXMuZmlsdGVyRGltc1sxXSArIHdDO1xuICAgICAgICAgICAgbGV0IG1hc2sgPSBzZWxlY3QoMC4wLCAxLjAsIG1heFBvc1ZhbHVlID09IGN1clBvc1ZhbHVlKTtcbiAgICAgICAgICAgIGRvdFByb2QgKz0gZHlWYWx1ZSAqIG1hc2s7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgTWF4UG9vbDNEQmFja3Byb3BQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydkeScsICdtYXhQb3MnXTtcbiAgdW5pZm9ybXMgPSBgc3RyaWRlcyA6IHZlYzM8aTMyPiwgcGFkcyA6IHZlYzM8aTMyPiwgZmlsdGVyRGltcyA6IHZlYzM8aTMyPixcbiAgICAgIG91dERlcHRoIDogaTMyLCBvdXRIZWlnaHQgOiBpMzIsIG91dFdpZHRoIDogaTMyYDtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLmluU2hhcGU7XG5cbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuXG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ21heFBvb2wzREJhY2twcm9wJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgIGxldCBiYXRjaCA9IGNvb3Jkcy54O1xuICAgICAgICBsZXQgY2ggPSBjb29yZHMudTtcblxuICAgICAgICBsZXQgZHlDb3JuZXIgPSB2ZWMzPGkzMj4oY29vcmRzLnksIGNvb3Jkcy56LCBjb29yZHMudykgLSB1bmlmb3Jtcy5wYWRzO1xuICAgICAgICBsZXQgZHlEQ29ybmVyID0gZHlDb3JuZXIueDtcbiAgICAgICAgbGV0IGR5UkNvcm5lciA9IGR5Q29ybmVyLnk7XG4gICAgICAgIGxldCBkeUNDb3JuZXIgPSBkeUNvcm5lci56O1xuXG4gICAgICAgIC8vIENvbnZvbHZlIGR5KD8sID8sID8sIGNoKSB3aXRoIHBvcyBtYXNrKDosIDosIDosIGQpIHRvIGdldFxuICAgICAgICAvLyBkeCh4RCwgeFIsIHhDLCBjaCkuXG4gICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgICB2YXIgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgbGV0IGxhc3RJbmRleCA9IHVuaWZvcm1zLmZpbHRlckRpbXNbMF0gKiB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdICogdW5pZm9ybXMuZmlsdGVyRGltc1syXSAtIDE7XG5cbiAgICAgICAgZm9yICh2YXIgd0QgPSAwOyB3RCA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMF07IHdEKyspIHtcbiAgICAgICAgICBsZXQgZHlEID0gZjMyKGR5RENvcm5lciArIHdEKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzBdKTtcblxuICAgICAgICAgIGlmIChkeUQgPCAwLjAgfHwgZHlEID49IGYzMih1bmlmb3Jtcy5vdXREZXB0aCkgfHwgZnJhY3QoZHlEKSA+IDAuMCkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuICAgICAgICAgIGxldCBpZHlEID0gaTMyKGR5RCk7XG5cbiAgICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltc1sxXTsgd1IrKykge1xuICAgICAgICAgICAgbGV0IGR5UiA9IGYzMihkeVJDb3JuZXIgKyB3UikgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1sxXSk7XG5cbiAgICAgICAgICAgIGlmIChkeVIgPCAwLjAgfHwgZHlSID49IGYzMih1bmlmb3Jtcy5vdXRIZWlnaHQpIHx8IGZyYWN0KGR5UikgPiAwLjApIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBsZXQgaWR5UiA9IGkzMihkeVIpO1xuXG4gICAgICAgICAgICBmb3IgKHZhciB3QyA9IDA7IHdDIDwgdW5pZm9ybXMuZmlsdGVyRGltc1syXTsgd0MrKykge1xuICAgICAgICAgICAgICBsZXQgZHlDID0gZjMyKGR5Q0Nvcm5lciArIHdDKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzJdKTtcblxuICAgICAgICAgICAgICBpZiAoZHlDIDwgMC4wIHx8IGR5QyA+PSBmMzIodW5pZm9ybXMub3V0V2lkdGgpIHx8IGZyYWN0KGR5QykgPiAwLjApIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICBsZXQgaWR5QyA9IGkzMihkeUMpO1xuXG4gICAgICAgICAgICAgIGxldCBkeVZhbHVlID0gZ2V0RHkoYmF0Y2gsIGlkeUQsIGlkeVIsIGlkeUMsIGNoKTtcbiAgICAgICAgICAgICAgbGV0IG1heFBvc1ZhbHVlID0gbGFzdEluZGV4IC0gaTMyKGdldE1heFBvcyhiYXRjaCwgaWR5RCwgaWR5UiwgaWR5QywgY2gpKTtcblxuICAgICAgICAgICAgICAvLyBHZXQgdGhlIGN1cnJlbnQgdmFsdWUsIGNoZWNrIGl0IGFnYWluc3QgdGhlIHZhbHVlIGZyb20gdGhlXG4gICAgICAgICAgICAgIC8vIHBvc2l0aW9uIG1hdHJpeC5cbiAgICAgICAgICAgICAgbGV0IGN1clBvc1ZhbHVlID0gd0QgKiB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdICogdW5pZm9ybXMuZmlsdGVyRGltc1syXSArIHdSICogdW5pZm9ybXMuZmlsdGVyRGltc1syXSArIHdDO1xuICAgICAgICAgICAgICBsZXQgbWFzayA9IHNlbGVjdCgwLjAsIDEuMCwgbWF4UG9zVmFsdWUgPT0gY3VyUG9zVmFsdWUpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IGR5VmFsdWUgKiBtYXNrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19