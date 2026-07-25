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
export class ReverseProgram {
    constructor(xShape) {
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = ` axis : vec4<i32>,`;
        this.shaderKey = 'reverse';
    }
    getUserCode() {
        const reverseCoordsSnippet = `
      // Using uniform variables as judging conditions, so the function has
      // coherent execution within all threads.
      fn getReverseCoords(coords : vec4<i32>) -> vec4<i32> {
        var reverseCoords = coords;
        if (uniforms.axis[0] == 1) {
          reverseCoords[0] = uniforms.xShape[0] - coords[0] - 1;
        }
        if (uniforms.axis[1] == 1) {
          reverseCoords[1] = uniforms.xShape[1] - coords[1] - 1;
        }
        if (uniforms.axis[2] == 1) {
          reverseCoords[2] = uniforms.xShape[2] - coords[2] - 1;
        }
        if (uniforms.axis[3] == 1) {
          reverseCoords[3] = uniforms.xShape[3] - coords[3] - 1;
        }

        return reverseCoords;
      }
    `;
        const userCode = `
      ${reverseCoordsSnippet}
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let reverseCoords = getReverseCoords(coords);
          setOutputAtIndex(index, getX(reverseCoords[0],
              reverseCoords[1], reverseCoords[2], reverseCoords[3]));
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmV2ZXJzZV93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9yZXZlcnNlX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLGNBQWM7SUFVekIsWUFBWSxNQUF3QztRQUxwRCxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFFdEIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQztRQUMxQixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsUUFBUSxHQUFHLG9CQUFvQixDQUFDO1FBQ3JDLElBQUksQ0FBQyxTQUFTLEdBQUcsU0FBUyxDQUFDO0lBQzdCLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxvQkFBb0IsR0FBRzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7S0FvQjVCLENBQUM7UUFDRixNQUFNLFFBQVEsR0FBRztRQUNiLG9CQUFvQjtRQUNwQixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7OztLQVFoQixDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFJldmVyc2VQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zOiBzdHJpbmc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKHhTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0geFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcbiAgICB0aGlzLnVuaWZvcm1zID0gYCBheGlzIDogdmVjNDxpMzI+LGA7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAncmV2ZXJzZSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHJldmVyc2VDb29yZHNTbmlwcGV0ID0gYFxuICAgICAgLy8gVXNpbmcgdW5pZm9ybSB2YXJpYWJsZXMgYXMganVkZ2luZyBjb25kaXRpb25zLCBzbyB0aGUgZnVuY3Rpb24gaGFzXG4gICAgICAvLyBjb2hlcmVudCBleGVjdXRpb24gd2l0aGluIGFsbCB0aHJlYWRzLlxuICAgICAgZm4gZ2V0UmV2ZXJzZUNvb3Jkcyhjb29yZHMgOiB2ZWM0PGkzMj4pIC0+IHZlYzQ8aTMyPiB7XG4gICAgICAgIHZhciByZXZlcnNlQ29vcmRzID0gY29vcmRzO1xuICAgICAgICBpZiAodW5pZm9ybXMuYXhpc1swXSA9PSAxKSB7XG4gICAgICAgICAgcmV2ZXJzZUNvb3Jkc1swXSA9IHVuaWZvcm1zLnhTaGFwZVswXSAtIGNvb3Jkc1swXSAtIDE7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKHVuaWZvcm1zLmF4aXNbMV0gPT0gMSkge1xuICAgICAgICAgIHJldmVyc2VDb29yZHNbMV0gPSB1bmlmb3Jtcy54U2hhcGVbMV0gLSBjb29yZHNbMV0gLSAxO1xuICAgICAgICB9XG4gICAgICAgIGlmICh1bmlmb3Jtcy5heGlzWzJdID09IDEpIHtcbiAgICAgICAgICByZXZlcnNlQ29vcmRzWzJdID0gdW5pZm9ybXMueFNoYXBlWzJdIC0gY29vcmRzWzJdIC0gMTtcbiAgICAgICAgfVxuICAgICAgICBpZiAodW5pZm9ybXMuYXhpc1szXSA9PSAxKSB7XG4gICAgICAgICAgcmV2ZXJzZUNvb3Jkc1szXSA9IHVuaWZvcm1zLnhTaGFwZVszXSAtIGNvb3Jkc1szXSAtIDE7XG4gICAgICAgIH1cblxuICAgICAgICByZXR1cm4gcmV2ZXJzZUNvb3JkcztcbiAgICAgIH1cbiAgICBgO1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHtyZXZlcnNlQ29vcmRzU25pcHBldH1cbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgbGV0IHJldmVyc2VDb29yZHMgPSBnZXRSZXZlcnNlQ29vcmRzKGNvb3Jkcyk7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgZ2V0WChyZXZlcnNlQ29vcmRzWzBdLFxuICAgICAgICAgICAgICByZXZlcnNlQ29vcmRzWzFdLCByZXZlcnNlQ29vcmRzWzJdLCByZXZlcnNlQ29vcmRzWzNdKSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19