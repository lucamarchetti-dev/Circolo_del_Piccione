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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class TransformProgram {
    constructor(outShape) {
        this.variableNames = ['Image', 'Transforms'];
        this.uniforms = 'interpolationModeId : i32, fillModeId : i32, fillValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'transform';
    }
    getUserCode() {
        const userCode = `
          fn mapCoord(outCoord : f32, len : f32) -> f32{
            var inCoord = outCoord;
            if(uniforms.fillModeId == 2) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  if (inCoord < sz2) {
                    inCoord = sz2 * f32(i32(f32(-inCoord / sz2))) +
                    inCoord;
                  }
                  if (inCoord < -len) {
                    inCoord = inCoord + sz2;
                  } else {
                    inCoord = -inCoord - 1.0;
                  }
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  inCoord = inCoord - sz2 * f32(i32(f32(inCoord / sz2)));
                  if (inCoord >= len) {
                    inCoord = sz2 - inCoord - 1.0;
                  }
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 3) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord + len * (f32(i32(f32(-inCoord / sz))) + 1.0);
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord - len * f32(i32(f32(inCoord / sz)));
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 4) {
              return clamp(outCoord, 0.0, len - 1.0);
            }
            return outCoord;
          }
          fn readWithFillValue(batch : i32, coordY : i32, coordX : i32,
            channel : i32) -> f32 {
            var outputValue : f32;
            if (0 <= coordY && coordY < uniforms.imageShape[1] && 0 <= coordX && coordX < uniforms.imageShape[2]) {
                outputValue = getImage(batch, coordY, coordX, channel);
            } else {
              outputValue = uniforms.fillValue;
            }
            return outputValue;
          }

          ${main('index')} {
            if (index < uniforms.size) {
              let coords = getCoordsFromIndex(index);
              var outputValue : f32;
              let batch = coords[0];
              let x = coords[2];
              let y = coords[1];
              let channel = coords[3];
              let xf = f32(x);
              let yf = f32(y);
              let a1 = getTransforms(batch, 0);
              let a2 = getTransforms(batch, 1);
              let a3 = getTransforms(batch, 2);
              let b1 = getTransforms(batch, 3);
              let b2 = getTransforms(batch, 4);
              let b3 = getTransforms(batch, 5);
              let c1 = getTransforms(batch, 6);
              let c2 = getTransforms(batch, 7);
              let projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = uniforms.fillValue;
              } else {
                let inX = (a1 * xf + a2 * yf + a3) / projection;
                let inY = (b1 * xf + b2 * yf + b3) / projection;
                let mapX = mapCoord(inX, f32(uniforms.imageShape[2]));
                let mapY = mapCoord(inY, f32(uniforms.imageShape[1]));

                if (uniforms.interpolationModeId == 1) {
                  let coordY = i32(round(mapY));
                  let coordX = i32(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  let yFloor = floor(mapY);
                  let xFloor = floor(mapX);
                  let yCeil = yFloor + 1.0;
                  let xCeil = xFloor + 1.0;
                  let valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yFloor), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yFloor), i32(xCeil), channel);
                  let valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yCeil), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yCeil), i32(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutputAtIndex(index, outputValue);
            }
          }
        `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNmb3JtX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3RyYW5zZm9ybV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxnQkFBZ0I7SUFVM0IsWUFBWSxRQUEwQztRQVR0RCxrQkFBYSxHQUFHLENBQUMsT0FBTyxFQUFFLFlBQVksQ0FBQyxDQUFDO1FBRXhDLGFBQVEsR0FBRywrREFBK0QsQ0FBQztRQUkzRSxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDO1FBQzVCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxTQUFTLEdBQUcsV0FBVyxDQUFDO0lBQy9CLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7WUFnRVQsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztTQW9EaEIsQ0FBQztRQUNOLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBUcmFuc2Zvcm1Qcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ0ltYWdlJywgJ1RyYW5zZm9ybXMnXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1bmlmb3JtcyA9ICdpbnRlcnBvbGF0aW9uTW9kZUlkIDogaTMyLCBmaWxsTW9kZUlkIDogaTMyLCBmaWxsVmFsdWUgOiBmMzIsJztcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihvdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3RyYW5zZm9ybSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgICAgIGZuIG1hcENvb3JkKG91dENvb3JkIDogZjMyLCBsZW4gOiBmMzIpIC0+IGYzMntcbiAgICAgICAgICAgIHZhciBpbkNvb3JkID0gb3V0Q29vcmQ7XG4gICAgICAgICAgICBpZih1bmlmb3Jtcy5maWxsTW9kZUlkID09IDIpIHtcbiAgICAgICAgICAgICAgaWYgKGluQ29vcmQgPCAwLjApIHtcbiAgICAgICAgICAgICAgICBpZiAobGVuIDw9IDEuMCkge1xuICAgICAgICAgICAgICAgICAgaW5Db29yZCA9IDAuMDtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgbGV0IHN6MiA9IDIuMCAqIGxlbjtcbiAgICAgICAgICAgICAgICAgIGlmIChpbkNvb3JkIDwgc3oyKSB7XG4gICAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSBzejIgKiBmMzIoaTMyKGYzMigtaW5Db29yZCAvIHN6MikpKSArXG4gICAgICAgICAgICAgICAgICAgIGluQ29vcmQ7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICBpZiAoaW5Db29yZCA8IC1sZW4pIHtcbiAgICAgICAgICAgICAgICAgICAgaW5Db29yZCA9IGluQ29vcmQgKyBzejI7XG4gICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBpbkNvb3JkID0gLWluQ29vcmQgLSAxLjA7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9IGVsc2UgaWYgKGluQ29vcmQgPiBsZW4gLSAxLjApIHtcbiAgICAgICAgICAgICAgICBpZiAobGVuIDw9IDEuMCkge1xuICAgICAgICAgICAgICAgICAgaW5Db29yZCA9IDAuMDtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgbGV0IHN6MiA9IDIuMCAqIGxlbjtcbiAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSBpbkNvb3JkIC0gc3oyICogZjMyKGkzMihmMzIoaW5Db29yZCAvIHN6MikpKTtcbiAgICAgICAgICAgICAgICAgIGlmIChpbkNvb3JkID49IGxlbikge1xuICAgICAgICAgICAgICAgICAgICBpbkNvb3JkID0gc3oyIC0gaW5Db29yZCAtIDEuMDtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgcmV0dXJuIGNsYW1wKGluQ29vcmQsIDAuMCwgbGVuIC0gMS4wKTtcbiAgICAgICAgICAgIH0gZWxzZSBpZiAodW5pZm9ybXMuZmlsbE1vZGVJZCA9PSAzKSB7XG4gICAgICAgICAgICAgIGlmIChpbkNvb3JkIDwgMC4wKSB7XG4gICAgICAgICAgICAgICAgaWYgKGxlbiA8PSAxLjApIHtcbiAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSAwLjA7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgIGxldCBzeiA9IGxlbiAtIDEuMDtcbiAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSBpbkNvb3JkICsgbGVuICogKGYzMihpMzIoZjMyKC1pbkNvb3JkIC8gc3opKSkgKyAxLjApO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgfSBlbHNlIGlmIChpbkNvb3JkID4gbGVuIC0gMS4wKSB7XG4gICAgICAgICAgICAgICAgaWYgKGxlbiA8PSAxLjApIHtcbiAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSAwLjA7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgIGxldCBzeiA9IGxlbiAtIDEuMDtcbiAgICAgICAgICAgICAgICAgIGluQ29vcmQgPSBpbkNvb3JkIC0gbGVuICogZjMyKGkzMihmMzIoaW5Db29yZCAvIHN6KSkpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICByZXR1cm4gY2xhbXAoaW5Db29yZCwgMC4wLCBsZW4gLSAxLjApO1xuICAgICAgICAgICAgfSBlbHNlIGlmICh1bmlmb3Jtcy5maWxsTW9kZUlkID09IDQpIHtcbiAgICAgICAgICAgICAgcmV0dXJuIGNsYW1wKG91dENvb3JkLCAwLjAsIGxlbiAtIDEuMCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gb3V0Q29vcmQ7XG4gICAgICAgICAgfVxuICAgICAgICAgIGZuIHJlYWRXaXRoRmlsbFZhbHVlKGJhdGNoIDogaTMyLCBjb29yZFkgOiBpMzIsIGNvb3JkWCA6IGkzMixcbiAgICAgICAgICAgIGNoYW5uZWwgOiBpMzIpIC0+IGYzMiB7XG4gICAgICAgICAgICB2YXIgb3V0cHV0VmFsdWUgOiBmMzI7XG4gICAgICAgICAgICBpZiAoMCA8PSBjb29yZFkgJiYgY29vcmRZIDwgdW5pZm9ybXMuaW1hZ2VTaGFwZVsxXSAmJiAwIDw9IGNvb3JkWCAmJiBjb29yZFggPCB1bmlmb3Jtcy5pbWFnZVNoYXBlWzJdKSB7XG4gICAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSBnZXRJbWFnZShiYXRjaCwgY29vcmRZLCBjb29yZFgsIGNoYW5uZWwpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSB1bmlmb3Jtcy5maWxsVmFsdWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gb3V0cHV0VmFsdWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgICAgICB2YXIgb3V0cHV0VmFsdWUgOiBmMzI7XG4gICAgICAgICAgICAgIGxldCBiYXRjaCA9IGNvb3Jkc1swXTtcbiAgICAgICAgICAgICAgbGV0IHggPSBjb29yZHNbMl07XG4gICAgICAgICAgICAgIGxldCB5ID0gY29vcmRzWzFdO1xuICAgICAgICAgICAgICBsZXQgY2hhbm5lbCA9IGNvb3Jkc1szXTtcbiAgICAgICAgICAgICAgbGV0IHhmID0gZjMyKHgpO1xuICAgICAgICAgICAgICBsZXQgeWYgPSBmMzIoeSk7XG4gICAgICAgICAgICAgIGxldCBhMSA9IGdldFRyYW5zZm9ybXMoYmF0Y2gsIDApO1xuICAgICAgICAgICAgICBsZXQgYTIgPSBnZXRUcmFuc2Zvcm1zKGJhdGNoLCAxKTtcbiAgICAgICAgICAgICAgbGV0IGEzID0gZ2V0VHJhbnNmb3JtcyhiYXRjaCwgMik7XG4gICAgICAgICAgICAgIGxldCBiMSA9IGdldFRyYW5zZm9ybXMoYmF0Y2gsIDMpO1xuICAgICAgICAgICAgICBsZXQgYjIgPSBnZXRUcmFuc2Zvcm1zKGJhdGNoLCA0KTtcbiAgICAgICAgICAgICAgbGV0IGIzID0gZ2V0VHJhbnNmb3JtcyhiYXRjaCwgNSk7XG4gICAgICAgICAgICAgIGxldCBjMSA9IGdldFRyYW5zZm9ybXMoYmF0Y2gsIDYpO1xuICAgICAgICAgICAgICBsZXQgYzIgPSBnZXRUcmFuc2Zvcm1zKGJhdGNoLCA3KTtcbiAgICAgICAgICAgICAgbGV0IHByb2plY3Rpb24gPSBjMSAqIHhmICsgYzIgKiB5ZiArIDEuMDtcbiAgICAgICAgICAgICAgaWYgKHByb2plY3Rpb24gPT0gMC4wKSB7XG4gICAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSB1bmlmb3Jtcy5maWxsVmFsdWU7XG4gICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IGluWCA9IChhMSAqIHhmICsgYTIgKiB5ZiArIGEzKSAvIHByb2plY3Rpb247XG4gICAgICAgICAgICAgICAgbGV0IGluWSA9IChiMSAqIHhmICsgYjIgKiB5ZiArIGIzKSAvIHByb2plY3Rpb247XG4gICAgICAgICAgICAgICAgbGV0IG1hcFggPSBtYXBDb29yZChpblgsIGYzMih1bmlmb3Jtcy5pbWFnZVNoYXBlWzJdKSk7XG4gICAgICAgICAgICAgICAgbGV0IG1hcFkgPSBtYXBDb29yZChpblksIGYzMih1bmlmb3Jtcy5pbWFnZVNoYXBlWzFdKSk7XG5cbiAgICAgICAgICAgICAgICBpZiAodW5pZm9ybXMuaW50ZXJwb2xhdGlvbk1vZGVJZCA9PSAxKSB7XG4gICAgICAgICAgICAgICAgICBsZXQgY29vcmRZID0gaTMyKHJvdW5kKG1hcFkpKTtcbiAgICAgICAgICAgICAgICAgIGxldCBjb29yZFggPSBpMzIocm91bmQobWFwWCkpO1xuICAgICAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSByZWFkV2l0aEZpbGxWYWx1ZShiYXRjaCwgY29vcmRZLCBjb29yZFgsXG4gICAgICAgICAgICAgICAgICAgIGNoYW5uZWwpO1xuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICBsZXQgeUZsb29yID0gZmxvb3IobWFwWSk7XG4gICAgICAgICAgICAgICAgICBsZXQgeEZsb29yID0gZmxvb3IobWFwWCk7XG4gICAgICAgICAgICAgICAgICBsZXQgeUNlaWwgPSB5Rmxvb3IgKyAxLjA7XG4gICAgICAgICAgICAgICAgICBsZXQgeENlaWwgPSB4Rmxvb3IgKyAxLjA7XG4gICAgICAgICAgICAgICAgICBsZXQgdmFsdWVZRmxvb3IgPSAoeENlaWwgLSBtYXBYKSAqXG4gICAgICAgICAgICAgICAgICByZWFkV2l0aEZpbGxWYWx1ZShiYXRjaCwgaTMyKHlGbG9vciksIGkzMih4Rmxvb3IpLCBjaGFubmVsKSArXG4gICAgICAgICAgICAgICAgICAobWFwWCAtIHhGbG9vcikgKlxuICAgICAgICAgICAgICAgICAgcmVhZFdpdGhGaWxsVmFsdWUoYmF0Y2gsIGkzMih5Rmxvb3IpLCBpMzIoeENlaWwpLCBjaGFubmVsKTtcbiAgICAgICAgICAgICAgICAgIGxldCB2YWx1ZVlDZWlsID0gKHhDZWlsIC0gbWFwWCkgKlxuICAgICAgICAgICAgICAgICAgcmVhZFdpdGhGaWxsVmFsdWUoYmF0Y2gsIGkzMih5Q2VpbCksIGkzMih4Rmxvb3IpLCBjaGFubmVsKSArXG4gICAgICAgICAgICAgICAgICAobWFwWCAtIHhGbG9vcikgKlxuICAgICAgICAgICAgICAgICAgcmVhZFdpdGhGaWxsVmFsdWUoYmF0Y2gsIGkzMih5Q2VpbCksIGkzMih4Q2VpbCksIGNoYW5uZWwpO1xuICAgICAgICAgICAgICAgICAgb3V0cHV0VmFsdWUgPSAoeUNlaWwgLSBtYXBZKSAqIHZhbHVlWUZsb29yICtcbiAgICAgICAgICAgICAgICAgIChtYXBZIC0geUZsb29yKSAqIHZhbHVlWUNlaWw7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIG91dHB1dFZhbHVlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=