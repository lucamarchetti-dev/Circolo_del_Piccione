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
export class Conv3DNaiveProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'filterDims: vec3<i32>, pads: vec3<i32>, strides: vec3<i32>, dilations: vec3<i32>,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `conv3dnaive`;
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords.x;
        let d2 = coords.u;

        let xFRCCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
        let xFCorner = xFRCCorner.x;
        let xRCorner = xFRCCorner.y;
        let xCCorner = xFRCCorner.z;

        let inputDepthNearestVec4 = (uniforms.xShape.u / 4) * 4;
        let inputDepthVec4Remainder = uniforms.xShape.u % 4;

        var dotProd = 0.0;
        for (var wF = 0; wF < uniforms.filterDims[0]; wF++) {
          let xF = xFCorner + wF * uniforms.dilations[0];
          if (xF < 0 || xF >= uniforms.xShape.y) {
            continue;
          }

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let xR = xRCorner + wR * uniforms.dilations[1];
            if (xR < 0 || xR >= uniforms.xShape.z) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let xC = xCCorner + wC * uniforms.dilations[2];
              if (xC < 0 || xC >= uniforms.xShape.w) {
                continue;
              }

              for (var d1 = 0; d1 < inputDepthNearestVec4; d1 += 4) {
                let xValues = vec4<f32>(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                let wValues = vec4<f32>(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (inputDepthVec4Remainder == 1) {
                dotProd += getX(batch, xF, xR, xC, inputDepthNearestVec4) *
                  getW(wF, wR, wC, inputDepthNearestVec4, d2);
              } else if (inputDepthVec4Remainder == 2) {
                let xValues = vec2<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1)
                );
                let wValues = vec2<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (inputDepthVec4Remainder == 3) {
                let xValues = vec3<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 2)
                );
                let wValues = vec3<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }`;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjNkX25haXZlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2NvbnYzZF9uYWl2ZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxrQkFBa0I7SUFXN0IsWUFBWSxRQUFpQztRQU43QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzNCLGFBQVEsR0FDSixtRkFBbUYsQ0FBQztRQUN4RixrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQztRQUNyQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsU0FBUyxHQUFHLGFBQWEsQ0FBQztJQUNqQyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O01BaUZiLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIENvbnYzRE5haXZlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdXJ107XG4gIHVuaWZvcm1zID1cbiAgICAgICdmaWx0ZXJEaW1zOiB2ZWMzPGkzMj4sIHBhZHM6IHZlYzM8aTMyPiwgc3RyaWRlczogdmVjMzxpMzI+LCBkaWxhdGlvbnM6IHZlYzM8aTMyPiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3Rvcihjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYzREluZm8pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gY29udkluZm8ub3V0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgY29udjNkbmFpdmVgO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHMueDtcbiAgICAgICAgbGV0IGQyID0gY29vcmRzLnU7XG5cbiAgICAgICAgbGV0IHhGUkNDb3JuZXIgPSB2ZWMzPGkzMj4oY29vcmRzLnksIGNvb3Jkcy56LCBjb29yZHMudykgKiB1bmlmb3Jtcy5zdHJpZGVzIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgbGV0IHhGQ29ybmVyID0geEZSQ0Nvcm5lci54O1xuICAgICAgICBsZXQgeFJDb3JuZXIgPSB4RlJDQ29ybmVyLnk7XG4gICAgICAgIGxldCB4Q0Nvcm5lciA9IHhGUkNDb3JuZXIuejtcblxuICAgICAgICBsZXQgaW5wdXREZXB0aE5lYXJlc3RWZWM0ID0gKHVuaWZvcm1zLnhTaGFwZS51IC8gNCkgKiA0O1xuICAgICAgICBsZXQgaW5wdXREZXB0aFZlYzRSZW1haW5kZXIgPSB1bmlmb3Jtcy54U2hhcGUudSAlIDQ7XG5cbiAgICAgICAgdmFyIGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAodmFyIHdGID0gMDsgd0YgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyB3RisrKSB7XG4gICAgICAgICAgbGV0IHhGID0geEZDb3JuZXIgKyB3RiAqIHVuaWZvcm1zLmRpbGF0aW9uc1swXTtcbiAgICAgICAgICBpZiAoeEYgPCAwIHx8IHhGID49IHVuaWZvcm1zLnhTaGFwZS55KSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltc1sxXTsgd1IrKykge1xuICAgICAgICAgICAgbGV0IHhSID0geFJDb3JuZXIgKyB3UiAqIHVuaWZvcm1zLmRpbGF0aW9uc1sxXTtcbiAgICAgICAgICAgIGlmICh4UiA8IDAgfHwgeFIgPj0gdW5pZm9ybXMueFNoYXBlLnopIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzJdOyB3QysrKSB7XG4gICAgICAgICAgICAgIGxldCB4QyA9IHhDQ29ybmVyICsgd0MgKiB1bmlmb3Jtcy5kaWxhdGlvbnNbMl07XG4gICAgICAgICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0gdW5pZm9ybXMueFNoYXBlLncpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGZvciAodmFyIGQxID0gMDsgZDEgPCBpbnB1dERlcHRoTmVhcmVzdFZlYzQ7IGQxICs9IDQpIHtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlcyA9IHZlYzQ8ZjMyPihcbiAgICAgICAgICAgICAgICAgIGdldFgoYmF0Y2gsIHhGLCB4UiwgeEMsIGQxKSxcbiAgICAgICAgICAgICAgICAgIGdldFgoYmF0Y2gsIHhGLCB4UiwgeEMsIGQxICsgMSksXG4gICAgICAgICAgICAgICAgICBnZXRYKGJhdGNoLCB4RiwgeFIsIHhDLCBkMSArIDIpLFxuICAgICAgICAgICAgICAgICAgZ2V0WChiYXRjaCwgeEYsIHhSLCB4QywgZDEgKyAzKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgbGV0IHdWYWx1ZXMgPSB2ZWM0PGYzMj4oXG4gICAgICAgICAgICAgICAgICBnZXRXKHdGLCB3Uiwgd0MsIGQxLCBkMiksXG4gICAgICAgICAgICAgICAgICBnZXRXKHdGLCB3Uiwgd0MsIGQxICsgMSwgZDIpLFxuICAgICAgICAgICAgICAgICAgZ2V0Vyh3Riwgd1IsIHdDLCBkMSArIDIsIGQyKSxcbiAgICAgICAgICAgICAgICAgIGdldFcod0YsIHdSLCB3QywgZDEgKyAzLCBkMilcbiAgICAgICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSBkb3QoeFZhbHVlcywgd1ZhbHVlcyk7XG4gICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICBpZiAoaW5wdXREZXB0aFZlYzRSZW1haW5kZXIgPT0gMSkge1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gZ2V0WChiYXRjaCwgeEYsIHhSLCB4QywgaW5wdXREZXB0aE5lYXJlc3RWZWM0KSAqXG4gICAgICAgICAgICAgICAgICBnZXRXKHdGLCB3Uiwgd0MsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCwgZDIpO1xuICAgICAgICAgICAgICB9IGVsc2UgaWYgKGlucHV0RGVwdGhWZWM0UmVtYWluZGVyID09IDIpIHtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlcyA9IHZlYzI8ZjMyPihcbiAgICAgICAgICAgICAgICAgIGdldFgoYmF0Y2gsIHhGLCB4UiwgeEMsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCksXG4gICAgICAgICAgICAgICAgICBnZXRYKGJhdGNoLCB4RiwgeFIsIHhDLCBpbnB1dERlcHRoTmVhcmVzdFZlYzQgKyAxKVxuICAgICAgICAgICAgICAgICk7XG4gICAgICAgICAgICAgICAgbGV0IHdWYWx1ZXMgPSB2ZWMyPGYzMj4oXG4gICAgICAgICAgICAgICAgICBnZXRXKHdGLCB3Uiwgd0MsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCwgZDIpLFxuICAgICAgICAgICAgICAgICAgZ2V0Vyh3Riwgd1IsIHdDLCBpbnB1dERlcHRoTmVhcmVzdFZlYzQgKyAxLCBkMilcbiAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gZG90KHhWYWx1ZXMsIHdWYWx1ZXMpO1xuICAgICAgICAgICAgICB9IGVsc2UgaWYgKGlucHV0RGVwdGhWZWM0UmVtYWluZGVyID09IDMpIHtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlcyA9IHZlYzM8ZjMyPihcbiAgICAgICAgICAgICAgICAgIGdldFgoYmF0Y2gsIHhGLCB4UiwgeEMsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCksXG4gICAgICAgICAgICAgICAgICBnZXRYKGJhdGNoLCB4RiwgeFIsIHhDLCBpbnB1dERlcHRoTmVhcmVzdFZlYzQgKyAxKSxcbiAgICAgICAgICAgICAgICAgIGdldFgoYmF0Y2gsIHhGLCB4UiwgeEMsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCArIDIpXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICBsZXQgd1ZhbHVlcyA9IHZlYzM8ZjMyPihcbiAgICAgICAgICAgICAgICAgIGdldFcod0YsIHdSLCB3QywgaW5wdXREZXB0aE5lYXJlc3RWZWM0LCBkMiksXG4gICAgICAgICAgICAgICAgICBnZXRXKHdGLCB3Uiwgd0MsIGlucHV0RGVwdGhOZWFyZXN0VmVjNCArIDEsIGQyKSxcbiAgICAgICAgICAgICAgICAgIGdldFcod0YsIHdSLCB3QywgaW5wdXREZXB0aE5lYXJlc3RWZWM0ICsgMiwgZDIpXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IGRvdCh4VmFsdWVzLCB3VmFsdWVzKTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBkb3RQcm9kKTtcbiAgICAgIH1cbiAgICB9YDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==