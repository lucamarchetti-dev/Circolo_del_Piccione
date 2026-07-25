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
import { atomicAddSnippet } from './shader_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
const writeSnippet = `
  fn bincount_write(index: i32, value: f32) {
    ${atomicAddSnippet('&result[index]', 'value', 'float32')}
  }
`;
const binaryWriteSnippet = `
  fn bincount_write(index: i32, value: f32) {
    atomicStore(&result[index], bitcast<i32>(value));
  }
`;
export class BincountProgram {
    constructor(shape, hasWeights, binaryOutput = false) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'binCountSize : i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.hasWeights = true;
        this.binaryOutput = false;
        this.outputShape = shape;
        this.rank = shape.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.binaryOutput = binaryOutput;
        if (binaryOutput) {
            this.atomic = false;
        }
        this.hasWeights = hasWeights;
        if (this.hasWeights) {
            this.variableNames.push('w');
        }
        this.shaderKey =
            `bincount_${this.hasWeights}_${this.binaryOutput}_${this.rank}`;
    }
    getUserCode() {
        const userCode = `
    ${this.binaryOutput ? binaryWriteSnippet : writeSnippet}
  ${main('index')} {
    ${this.rank === 1 ?
            `if (index < uniforms.xShape) {
      let indexVal = i32(getX(index));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput ? 1. :
                (this.hasWeights ? 'getW(index)' : '1.')};
        bincount_write(indexVal, value);
      }
    }` :
            `let coord = getCoordsFromIndex(index);
    if (coordsInBounds2D(coord, uniforms.xShape)) {
      let indexVal = i32(getX(coord[0], coord[1]));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput ?
                1. :
                (this.hasWeights ? 'getW(coord[0], coord[1])' : '1.')};
        bincount_write(coord.x * uniforms.binCountSize + indexVal, value);
      }
    }`}
  }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmluY291bnRfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvYmluY291bnRfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUMvQyxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxZQUFZLEdBQUc7O01BRWYsZ0JBQWdCLENBQUMsZ0JBQWdCLEVBQUUsT0FBTyxFQUFFLFNBQVMsQ0FBQzs7Q0FFM0QsQ0FBQztBQUVGLE1BQU0sa0JBQWtCLEdBQUc7Ozs7Q0FJMUIsQ0FBQztBQUVGLE1BQU0sT0FBTyxlQUFlO0lBYTFCLFlBQ0ksS0FBZ0MsRUFBRSxVQUFtQixFQUNyRCxZQUFZLEdBQUcsS0FBSztRQWR4QixnQkFBVyxHQUFhLEVBQUUsQ0FBQztRQUkzQixrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUFHLHFCQUFxQixDQUFDO1FBQ2pDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxXQUFNLEdBQUcsSUFBSSxDQUFDO1FBQ2QsZUFBVSxHQUFHLElBQUksQ0FBQztRQUNsQixpQkFBWSxHQUFHLEtBQUssQ0FBQztRQU1uQixJQUFJLENBQUMsV0FBVyxHQUFHLEtBQUssQ0FBQztRQUN6QixJQUFJLENBQUMsSUFBSSxHQUFHLEtBQUssQ0FBQyxNQUFNLENBQUM7UUFDekIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFlBQVksR0FBRyxZQUFZLENBQUM7UUFDakMsSUFBSSxZQUFZLEVBQUU7WUFDaEIsSUFBSSxDQUFDLE1BQU0sR0FBRyxLQUFLLENBQUM7U0FDckI7UUFDRCxJQUFJLENBQUMsVUFBVSxHQUFHLFVBQVUsQ0FBQztRQUM3QixJQUFJLElBQUksQ0FBQyxVQUFVLEVBQUU7WUFDbkIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7U0FDOUI7UUFDRCxJQUFJLENBQUMsU0FBUztZQUNWLFlBQVksSUFBSSxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUMsWUFBWSxJQUFJLElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUN0RSxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsa0JBQWtCLENBQUMsQ0FBQyxDQUFDLFlBQVk7SUFDdkQsSUFBSSxDQUFDLE9BQU8sQ0FBQztNQUVULElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDYjs7O3NCQUlJLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUNKLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUM7OztNQUd0RSxDQUFDLENBQUM7WUFDSTs7OztzQkFLSSxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUM7Z0JBQ2YsRUFBRSxDQUFDLENBQUM7Z0JBQ0osQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQywwQkFBMEIsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDOzs7TUFHbkU7O0dBRUgsQ0FBQztRQUNBLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHthdG9taWNBZGRTbmlwcGV0fSBmcm9tICcuL3NoYWRlcl91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5jb25zdCB3cml0ZVNuaXBwZXQgPSBgXG4gIGZuIGJpbmNvdW50X3dyaXRlKGluZGV4OiBpMzIsIHZhbHVlOiBmMzIpIHtcbiAgICAke2F0b21pY0FkZFNuaXBwZXQoJyZyZXN1bHRbaW5kZXhdJywgJ3ZhbHVlJywgJ2Zsb2F0MzInKX1cbiAgfVxuYDtcblxuY29uc3QgYmluYXJ5V3JpdGVTbmlwcGV0ID0gYFxuICBmbiBiaW5jb3VudF93cml0ZShpbmRleDogaTMyLCB2YWx1ZTogZjMyKSB7XG4gICAgYXRvbWljU3RvcmUoJnJlc3VsdFtpbmRleF0sIGJpdGNhc3Q8aTMyPih2YWx1ZSkpO1xuICB9XG5gO1xuXG5leHBvcnQgY2xhc3MgQmluY291bnRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgdW5pZm9ybXMgPSAnYmluQ291bnRTaXplIDogaTMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIGF0b21pYyA9IHRydWU7XG4gIGhhc1dlaWdodHMgPSB0cnVlO1xuICBiaW5hcnlPdXRwdXQgPSBmYWxzZTtcbiAgcmFuazogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgc2hhcGU6IFtudW1iZXJdfFtudW1iZXIsIG51bWJlcl0sIGhhc1dlaWdodHM6IGJvb2xlYW4sXG4gICAgICBiaW5hcnlPdXRwdXQgPSBmYWxzZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBzaGFwZTtcbiAgICB0aGlzLnJhbmsgPSBzaGFwZS5sZW5ndGg7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5iaW5hcnlPdXRwdXQgPSBiaW5hcnlPdXRwdXQ7XG4gICAgaWYgKGJpbmFyeU91dHB1dCkge1xuICAgICAgdGhpcy5hdG9taWMgPSBmYWxzZTtcbiAgICB9XG4gICAgdGhpcy5oYXNXZWlnaHRzID0gaGFzV2VpZ2h0cztcbiAgICBpZiAodGhpcy5oYXNXZWlnaHRzKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgndycpO1xuICAgIH1cbiAgICB0aGlzLnNoYWRlcktleSA9XG4gICAgICAgIGBiaW5jb3VudF8ke3RoaXMuaGFzV2VpZ2h0c31fJHt0aGlzLmJpbmFyeU91dHB1dH1fJHt0aGlzLnJhbmt9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHt0aGlzLmJpbmFyeU91dHB1dCA/IGJpbmFyeVdyaXRlU25pcHBldCA6IHdyaXRlU25pcHBldH1cbiAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgJHtcbiAgICAgICAgdGhpcy5yYW5rID09PSAxID9cbiAgICAgICAgICAgIGBpZiAoaW5kZXggPCB1bmlmb3Jtcy54U2hhcGUpIHtcbiAgICAgIGxldCBpbmRleFZhbCA9IGkzMihnZXRYKGluZGV4KSk7XG4gICAgICBpZiAoaW5kZXhWYWwgPCB1bmlmb3Jtcy5iaW5Db3VudFNpemUpIHtcbiAgICAgICAgbGV0IHZhbHVlID0gJHtcbiAgICAgICAgICAgICAgICB0aGlzLmJpbmFyeU91dHB1dCA/IDEuIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICh0aGlzLmhhc1dlaWdodHMgPyAnZ2V0VyhpbmRleCknIDogJzEuJyl9O1xuICAgICAgICBiaW5jb3VudF93cml0ZShpbmRleFZhbCwgdmFsdWUpO1xuICAgICAgfVxuICAgIH1gIDpcbiAgICAgICAgICAgIGBsZXQgY29vcmQgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgIGlmIChjb29yZHNJbkJvdW5kczJEKGNvb3JkLCB1bmlmb3Jtcy54U2hhcGUpKSB7XG4gICAgICBsZXQgaW5kZXhWYWwgPSBpMzIoZ2V0WChjb29yZFswXSwgY29vcmRbMV0pKTtcbiAgICAgIGlmIChpbmRleFZhbCA8IHVuaWZvcm1zLmJpbkNvdW50U2l6ZSkge1xuICAgICAgICBsZXQgdmFsdWUgPSAke1xuICAgICAgICAgICAgICAgIHRoaXMuYmluYXJ5T3V0cHV0ID9cbiAgICAgICAgICAgICAgICAgICAgMS4gOlxuICAgICAgICAgICAgICAgICAgICAodGhpcy5oYXNXZWlnaHRzID8gJ2dldFcoY29vcmRbMF0sIGNvb3JkWzFdKScgOiAnMS4nKX07XG4gICAgICAgIGJpbmNvdW50X3dyaXRlKGNvb3JkLnggKiB1bmlmb3Jtcy5iaW5Db3VudFNpemUgKyBpbmRleFZhbCwgdmFsdWUpO1xuICAgICAgfVxuICAgIH1gfVxuICB9XG4gIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=