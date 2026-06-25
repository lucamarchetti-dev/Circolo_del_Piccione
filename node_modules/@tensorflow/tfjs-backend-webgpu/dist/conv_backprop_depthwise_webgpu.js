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
export class DepthwiseConv2DDerFilterProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'dy'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>, outHeight : i32,
      outWidth : i32, inHeight : i32, inWidth : i32, batchSize : i32, channelMul : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.filterShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthwise_conv2d_backprop_filter`;
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let dm = coords[3];
        let d2 = d1 * uniforms.channelMul + dm;

        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b++) {
          for (var yR = 0; yR < uniforms.outHeight; yR++) {
            let xR = wR + yR * uniforms.strides[0] - uniforms.pads[0];

            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC++) {
              let xC = wC + yC * uniforms.strides[1] - uniforms.pads[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              let dyValue = getDy(b, yR, yC, d2);
              let xValue = getX(b, xR, xC, d1);
              dotProd += xValue * dyValue;
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
export class DepthwiseConv2DDerInputProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'W'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, channelMul : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthwise_conv2d_backprop_input`;
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[3];
        let dyCorner = coords.yz - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }

          let idyR = i32(dyR);
          let wRPerm = uniforms.filterDims[0] - 1 - wR;

          for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }

            let idyC = i32(dyC);
            let wCPerm = uniforms.filterDims[1] - 1 - wC;

            for (var dm = 0; dm < uniforms.channelMul; dm++) {
              let d2 = d1 * uniforms.channelMul + dm;
              let xValue = getDy(batch, idyR, idyC, d2);
              let wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udl9iYWNrcHJvcF9kZXB0aHdpc2Vfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvY29udl9iYWNrcHJvcF9kZXB0aHdpc2Vfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sK0JBQStCO0lBWTFDLFlBQVksUUFBaUM7UUFQN0Msa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM1QixhQUFRLEdBQ0o7d0ZBQ2tGLENBQUM7UUFDdkYsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxXQUFXLENBQUM7UUFFeEMsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxrQ0FBa0MsQ0FBQztJQUN0RCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQWtDaEIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELE1BQU0sT0FBTyw4QkFBOEI7SUFXekMsWUFBWSxRQUFpQztRQU43QyxrQkFBYSxHQUFHLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzVCLGFBQVEsR0FBRzswREFDNkMsQ0FBQztRQUN6RCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQztRQUVwQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUUzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsU0FBUyxHQUFHLGlDQUFpQyxDQUFDO0lBQ3JELENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQXlDaEIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIERlcHRod2lzZUNvbnYyRERlckZpbHRlclByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgYHN0cmlkZXMgOiB2ZWMyPGkzMj4sIHBhZHMgOiB2ZWMyPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMyPGkzMj4sIG91dEhlaWdodCA6IGkzMixcbiAgICAgIG91dFdpZHRoIDogaTMyLCBpbkhlaWdodCA6IGkzMiwgaW5XaWR0aCA6IGkzMiwgYmF0Y2hTaXplIDogaTMyLCBjaGFubmVsTXVsIDogaTMyLGA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5maWx0ZXJTaGFwZTtcblxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgZGVwdGh3aXNlX2NvbnYyZF9iYWNrcHJvcF9maWx0ZXJgO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IHdSID0gY29vcmRzWzBdO1xuICAgICAgICBsZXQgd0MgPSBjb29yZHNbMV07XG4gICAgICAgIGxldCBkMSA9IGNvb3Jkc1syXTtcbiAgICAgICAgbGV0IGRtID0gY29vcmRzWzNdO1xuICAgICAgICBsZXQgZDIgPSBkMSAqIHVuaWZvcm1zLmNoYW5uZWxNdWwgKyBkbTtcblxuICAgICAgICB2YXIgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgZm9yICh2YXIgYiA9IDA7IGIgPCB1bmlmb3Jtcy5iYXRjaFNpemU7IGIrKykge1xuICAgICAgICAgIGZvciAodmFyIHlSID0gMDsgeVIgPCB1bmlmb3Jtcy5vdXRIZWlnaHQ7IHlSKyspIHtcbiAgICAgICAgICAgIGxldCB4UiA9IHdSICsgeVIgKiB1bmlmb3Jtcy5zdHJpZGVzWzBdIC0gdW5pZm9ybXMucGFkc1swXTtcblxuICAgICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSB1bmlmb3Jtcy5pbkhlaWdodCkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZm9yICh2YXIgeUMgPSAwOyB5QyA8IHVuaWZvcm1zLm91dFdpZHRoOyB5QysrKSB7XG4gICAgICAgICAgICAgIGxldCB4QyA9IHdDICsgeUMgKiB1bmlmb3Jtcy5zdHJpZGVzWzFdIC0gdW5pZm9ybXMucGFkc1sxXTtcblxuICAgICAgICAgICAgICBpZiAoeEMgPCAwIHx8IHhDID49IHVuaWZvcm1zLmluV2lkdGgpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGxldCBkeVZhbHVlID0gZ2V0RHkoYiwgeVIsIHlDLCBkMik7XG4gICAgICAgICAgICAgIGxldCB4VmFsdWUgPSBnZXRYKGIsIHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHhWYWx1ZSAqIGR5VmFsdWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgRGVwdGh3aXNlQ29udjJERGVySW5wdXRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydkeScsICdXJ107XG4gIHVuaWZvcm1zID0gYHN0cmlkZXMgOiB2ZWMyPGkzMj4sIHBhZHMgOiB2ZWMyPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMyPGkzMj4sXG4gICAgICAgb3V0SGVpZ2h0IDogaTMyLCBvdXRXaWR0aCA6IGkzMiwgY2hhbm5lbE11bCA6IGkzMixgO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3Rvcihjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gY29udkluZm8uaW5TaGFwZTtcblxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgZGVwdGh3aXNlX2NvbnYyZF9iYWNrcHJvcF9pbnB1dGA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgIGxldCBkMSA9IGNvb3Jkc1szXTtcbiAgICAgICAgbGV0IGR5Q29ybmVyID0gY29vcmRzLnl6IC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgbGV0IGR5UkNvcm5lciA9IGR5Q29ybmVyLng7XG4gICAgICAgIGxldCBkeUNDb3JuZXIgPSBkeUNvcm5lci55O1xuXG4gICAgICAgIHZhciBkb3RQcm9kID0gMC4wO1xuICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltc1swXTsgd1IrKykge1xuICAgICAgICAgIGxldCBkeVIgPSBmMzIoZHlSQ29ybmVyICsgd1IpIC8gZjMyKHVuaWZvcm1zLnN0cmlkZXNbMF0pO1xuXG4gICAgICAgICAgaWYgKGR5UiA8IDAuMCB8fCBkeVIgPj0gZjMyKHVuaWZvcm1zLm91dEhlaWdodCkgfHwgZnJhY3QoZHlSKSA+IDAuMCkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgbGV0IGlkeVIgPSBpMzIoZHlSKTtcbiAgICAgICAgICBsZXQgd1JQZXJtID0gdW5pZm9ybXMuZmlsdGVyRGltc1swXSAtIDEgLSB3UjtcblxuICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdOyB3QysrKSB7XG4gICAgICAgICAgICBsZXQgZHlDID0gZjMyKGR5Q0Nvcm5lciArIHdDKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzFdKTtcblxuICAgICAgICAgICAgaWYgKGR5QyA8IDAuMCB8fCBkeUMgPj0gZjMyKHVuaWZvcm1zLm91dFdpZHRoKSB8fCBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgaWR5QyA9IGkzMihkeUMpO1xuICAgICAgICAgICAgbGV0IHdDUGVybSA9IHVuaWZvcm1zLmZpbHRlckRpbXNbMV0gLSAxIC0gd0M7XG5cbiAgICAgICAgICAgIGZvciAodmFyIGRtID0gMDsgZG0gPCB1bmlmb3Jtcy5jaGFubmVsTXVsOyBkbSsrKSB7XG4gICAgICAgICAgICAgIGxldCBkMiA9IGQxICogdW5pZm9ybXMuY2hhbm5lbE11bCArIGRtO1xuICAgICAgICAgICAgICBsZXQgeFZhbHVlID0gZ2V0RHkoYmF0Y2gsIGlkeVIsIGlkeUMsIGQyKTtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZSA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxLCBkbSk7XG4gICAgICAgICAgICAgIGRvdFByb2QgKz0geFZhbHVlICogd1ZhbHVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBkb3RQcm9kKTtcbiAgICAgIH1cbiAgICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==