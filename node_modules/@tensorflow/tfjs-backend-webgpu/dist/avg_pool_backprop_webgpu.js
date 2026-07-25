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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class AvgPool2DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, avgMultiplier : f32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `avgPool2DBackprop`;
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
        for (var wR = 0; wR < uniforms.filterDims[0]; wR = wR + uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC = wC + uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);

            dotProd = dotProd + dyValue * uniforms.avgMultiplier;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
export class AvgPool3DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
       outDepth : i32, outHeight : i32, outWidth : i32, avgMultiplier : f32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `avgPool3DBackprop`;
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

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
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
              dotProd += dyValue * uniforms.avgMultiplier;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXZnX3Bvb2xfYmFja3Byb3Bfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvYXZnX3Bvb2xfYmFja3Byb3Bfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sd0JBQXdCO0lBWW5DLFlBQVksUUFBaUM7UUFQN0Msa0JBQWEsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLGFBQVEsR0FDSjs2REFDdUQsQ0FBQztRQUM1RCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQztRQUVwQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUUzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsU0FBUyxHQUFHLG1CQUFtQixDQUFDO0lBQ3ZDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0tBcUNoQixDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGO0FBRUQsTUFBTSxPQUFPLHdCQUF3QjtJQVduQyxZQUFZLFFBQWlDO1FBTjdDLGtCQUFhLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN2QixhQUFRLEdBQUc7NkVBQ2dFLENBQUM7UUFDNUUsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUM7UUFFcEMsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxtQkFBbUIsQ0FBQztJQUN2QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7S0ErQ2hCLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBBdmdQb29sMkRCYWNrcHJvcFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ2R5J107XG4gIHVuaWZvcm1zID1cbiAgICAgIGBzdHJpZGVzIDogdmVjMjxpMzI+LCBwYWRzIDogdmVjMjxpMzI+LCBkaWxhdGlvbnMgOiB2ZWMyPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMyPGkzMj4sXG4gICAgICAgb3V0SGVpZ2h0IDogaTMyLCBvdXRXaWR0aCA6IGkzMiwgYXZnTXVsdGlwbGllciA6IGYzMixgO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3Rvcihjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gY29udkluZm8uaW5TaGFwZTtcblxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgYXZnUG9vbDJEQmFja3Byb3BgO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzWzBdO1xuICAgICAgICBsZXQgZCA9IGNvb3Jkc1szXTtcblxuICAgICAgICBsZXQgZHlSQ0Nvcm5lciA9IHZlYzI8aTMyPihjb29yZHMueXopIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgbGV0IGR5UkNvcm5lciA9IGR5UkNDb3JuZXIueDtcbiAgICAgICAgbGV0IGR5Q0Nvcm5lciA9IGR5UkNDb3JuZXIueTtcblxuICAgICAgICAvLyBDb252b2x2ZSBkeSg/LCA/LCBkKSB3aXRoIHBvcyBtYXNrKDosIDosIGQpIHRvIGdldCBkeCh4UiwgeEMsIGQpLlxuICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgICAgdmFyIGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAodmFyIHdSID0gMDsgd1IgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyB3UiA9IHdSICsgdW5pZm9ybXMuZGlsYXRpb25zWzBdKSB7XG4gICAgICAgICAgbGV0IGR5UiA9IGYzMihkeVJDb3JuZXIgKyB3UikgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1swXSk7XG5cbiAgICAgICAgICBpZiAoZHlSIDwgMC4wIHx8IGR5UiA+PSBmMzIodW5pZm9ybXMub3V0SGVpZ2h0KSB8fCBmcmFjdChkeVIpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG4gICAgICAgICAgbGV0IGlkeVIgPSBpMzIoZHlSKTtcblxuICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdOyB3QyA9IHdDICsgdW5pZm9ybXMuZGlsYXRpb25zWzFdKSB7XG4gICAgICAgICAgICBsZXQgZHlDID0gZjMyKGR5Q0Nvcm5lciArIHdDKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzFdKTtcblxuICAgICAgICAgICAgaWYgKGR5QyA8IDAuMCB8fCBkeUMgPj0gZjMyKHVuaWZvcm1zLm91dFdpZHRoKSB8fCBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IGlkeUMgPSBpMzIoZHlDKTtcblxuICAgICAgICAgICAgbGV0IGR5VmFsdWUgPSBnZXREeShiYXRjaCwgaWR5UiwgaWR5QywgZCk7XG5cbiAgICAgICAgICAgIGRvdFByb2QgPSBkb3RQcm9kICsgZHlWYWx1ZSAqIHVuaWZvcm1zLmF2Z011bHRpcGxpZXI7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQXZnUG9vbDNEQmFja3Byb3BQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydkeSddO1xuICB1bmlmb3JtcyA9IGBzdHJpZGVzIDogdmVjMzxpMzI+LCBwYWRzIDogdmVjMzxpMzI+LCBmaWx0ZXJEaW1zIDogdmVjMzxpMzI+LFxuICAgICAgIG91dERlcHRoIDogaTMyLCBvdXRIZWlnaHQgOiBpMzIsIG91dFdpZHRoIDogaTMyLCBhdmdNdWx0aXBsaWVyIDogZjMyLGA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjNESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5pblNoYXBlO1xuXG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcblxuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLnNoYWRlcktleSA9IGBhdmdQb29sM0RCYWNrcHJvcGA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHMueDtcbiAgICAgICAgbGV0IGNoID0gY29vcmRzLnU7XG5cbiAgICAgICAgbGV0IGR5Q29ybmVyID0gdmVjMzxpMzI+KGNvb3Jkcy55LCBjb29yZHMueiwgY29vcmRzLncpIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgbGV0IGR5RENvcm5lciA9IGR5Q29ybmVyLng7XG4gICAgICAgIGxldCBkeVJDb3JuZXIgPSBkeUNvcm5lci55O1xuICAgICAgICBsZXQgZHlDQ29ybmVyID0gZHlDb3JuZXIuejtcblxuICAgICAgICAvLyBDb252b2x2ZSBkeSg/LCA/LCA/LCBkKSB3aXRoIHBvcyBtYXNrKDosIDosIDosIGNoKSB0byBnZXRcbiAgICAgICAgLy8gZHgoeEQsIHhSLCB4QywgY2gpLlxuICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgICAgdmFyIGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAodmFyIHdEID0gMDsgd0QgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyB3RCsrKSB7XG4gICAgICAgICAgbGV0IGR5RCA9IGYzMihkeURDb3JuZXIgKyB3RCkgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1swXSk7XG5cbiAgICAgICAgICBpZiAoZHlEIDwgMC4wIHx8IGR5RCA+PSBmMzIodW5pZm9ybXMub3V0RGVwdGgpIHx8IGZyYWN0KGR5RCkgPiAwLjApIHtcbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cbiAgICAgICAgICBsZXQgaWR5RCA9IGkzMihkeUQpO1xuXG4gICAgICAgICAgZm9yICh2YXIgd1IgPSAwOyB3UiA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMV07IHdSKyspIHtcbiAgICAgICAgICAgIGxldCBkeVIgPSBmMzIoZHlSQ29ybmVyICsgd1IpIC8gZjMyKHVuaWZvcm1zLnN0cmlkZXNbMV0pO1xuXG4gICAgICAgICAgICBpZiAoZHlSIDwgMC4wIHx8IGR5UiA+PSBmMzIodW5pZm9ybXMub3V0SGVpZ2h0KSB8fCBmcmFjdChkeVIpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IGlkeVIgPSBpMzIoZHlSKTtcblxuICAgICAgICAgICAgZm9yICh2YXIgd0MgPSAwOyB3QyA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMl07IHdDKyspIHtcbiAgICAgICAgICAgICAgbGV0IGR5QyA9IGYzMihkeUNDb3JuZXIgKyB3QykgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1syXSk7XG5cbiAgICAgICAgICAgICAgaWYgKGR5QyA8IDAuMCB8fCBkeUMgPj0gZjMyKHVuaWZvcm1zLm91dFdpZHRoKSB8fCBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgbGV0IGlkeUMgPSBpMzIoZHlDKTtcblxuICAgICAgICAgICAgICBsZXQgZHlWYWx1ZSA9IGdldER5KGJhdGNoLCBpZHlELCBpZHlSLCBpZHlDLCBjaCk7XG4gICAgICAgICAgICAgIGRvdFByb2QgKz0gZHlWYWx1ZSAqIHVuaWZvcm1zLmF2Z011bHRpcGxpZXI7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19