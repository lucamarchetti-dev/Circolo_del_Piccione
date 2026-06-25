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
export class Im2ColProgram {
    constructor(outputShape, isChannelsLast) {
        this.variableNames = ['x'];
        this.uniforms = `pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.isChannelsLast = isChannelsLast;
        this.shaderKey = `im2col_${this.isChannelsLast}`;
    }
    getUserCode() {
        const rowDim = this.isChannelsLast ? 1 : 2;
        const colDim = this.isChannelsLast ? 2 : 3;
        const row = this.isChannelsLast ? 'coords[1]' : 'coords[2]';
        const col = this.isChannelsLast ? 'coords[2]' : 'coords[1]';
        const getXSnippet = this.isChannelsLast ? 'getX(batch, xRow, xCol, ch)' :
            'getX(batch, ch, xRow, xCol)';
        const userCode = `
    ${main('index')} {
      let coords = getCoordsFromIndex(index);
      if(index < uniforms.size) {
        let batch = coords[0];
        let row = ${row};
        let col = ${col};
        let offsetY = (row / uniforms.outWidth) * uniforms.strides[0] - uniforms.pads[0];
        let xRow = offsetY + uniforms.dilations[0] * (col / uniforms.itemsPerBlockRow);
        var value = 0.0;
        if(xRow < uniforms.xShape[${rowDim}] && xRow >= 0) {
          let offsetX = (row % uniforms.outWidth) * uniforms.strides[1] -
              uniforms.pads[1];
          let xCol = offsetX + uniforms.dilations[1] * ((col %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
          let ch = col % uniforms.inChannels;
          if(xCol < uniforms.xShape[${colDim}] && xCol >= 0) {
            value = ${getXSnippet};
          }
        }
        setOutputAtIndex(index, value);
      }
    }
   `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW0yY29sX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2ltMmNvbF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxhQUFhO0lBYXhCLFlBQVksV0FBcUIsRUFBRSxjQUF1QjtRQVoxRCxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUNKO3lCQUNtQixDQUFDO1FBS3hCLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLGNBQWMsR0FBRyxjQUFjLENBQUM7UUFDckMsSUFBSSxDQUFDLFNBQVMsR0FBRyxVQUFVLElBQUksQ0FBQyxjQUFjLEVBQUUsQ0FBQztJQUNuRCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNDLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTNDLE1BQU0sR0FBRyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsV0FBVyxDQUFDO1FBQzVELE1BQU0sR0FBRyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsV0FBVyxDQUFDO1FBQzVELE1BQU0sV0FBVyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDL0IsNkJBQTZCLENBQUM7UUFFeEUsTUFBTSxRQUFRLEdBQUc7TUFDZixJQUFJLENBQUMsT0FBTyxDQUFDOzs7O29CQUlDLEdBQUc7b0JBQ0gsR0FBRzs7OztvQ0FJYSxNQUFNOzs7Ozs7c0NBTUosTUFBTTtzQkFDdEIsV0FBVzs7Ozs7O0lBTTdCLENBQUM7UUFDRCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgSW0yQ29sUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zID1cbiAgICAgIGBwYWRzIDogdmVjMjxpMzI+LCBzdHJpZGVzIDogdmVjMjxpMzI+LCBkaWxhdGlvbnMgOiB2ZWMyPGkzMj4sIG91dFdpZHRoIDogaTMyLCBpdGVtc1BlckJsb2NrUm93IDogaTMyLFxuICAgICAgIGluQ2hhbm5lbHMgOiBpMzIsYDtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIGlzQ2hhbm5lbHNMYXN0OiBib29sZWFuO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihvdXRwdXRTaGFwZTogbnVtYmVyW10sIGlzQ2hhbm5lbHNMYXN0OiBib29sZWFuKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcbiAgICB0aGlzLmlzQ2hhbm5lbHNMYXN0ID0gaXNDaGFubmVsc0xhc3Q7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgaW0yY29sXyR7dGhpcy5pc0NoYW5uZWxzTGFzdH1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCByb3dEaW0gPSB0aGlzLmlzQ2hhbm5lbHNMYXN0ID8gMSA6IDI7XG4gICAgY29uc3QgY29sRGltID0gdGhpcy5pc0NoYW5uZWxzTGFzdCA/IDIgOiAzO1xuXG4gICAgY29uc3Qgcm93ID0gdGhpcy5pc0NoYW5uZWxzTGFzdCA/ICdjb29yZHNbMV0nIDogJ2Nvb3Jkc1syXSc7XG4gICAgY29uc3QgY29sID0gdGhpcy5pc0NoYW5uZWxzTGFzdCA/ICdjb29yZHNbMl0nIDogJ2Nvb3Jkc1sxXSc7XG4gICAgY29uc3QgZ2V0WFNuaXBwZXQgPSB0aGlzLmlzQ2hhbm5lbHNMYXN0ID8gJ2dldFgoYmF0Y2gsIHhSb3csIHhDb2wsIGNoKScgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICdnZXRYKGJhdGNoLCBjaCwgeFJvdywgeENvbCknO1xuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgIGxldCByb3cgPSAke3Jvd307XG4gICAgICAgIGxldCBjb2wgPSAke2NvbH07XG4gICAgICAgIGxldCBvZmZzZXRZID0gKHJvdyAvIHVuaWZvcm1zLm91dFdpZHRoKSAqIHVuaWZvcm1zLnN0cmlkZXNbMF0gLSB1bmlmb3Jtcy5wYWRzWzBdO1xuICAgICAgICBsZXQgeFJvdyA9IG9mZnNldFkgKyB1bmlmb3Jtcy5kaWxhdGlvbnNbMF0gKiAoY29sIC8gdW5pZm9ybXMuaXRlbXNQZXJCbG9ja1Jvdyk7XG4gICAgICAgIHZhciB2YWx1ZSA9IDAuMDtcbiAgICAgICAgaWYoeFJvdyA8IHVuaWZvcm1zLnhTaGFwZVske3Jvd0RpbX1dICYmIHhSb3cgPj0gMCkge1xuICAgICAgICAgIGxldCBvZmZzZXRYID0gKHJvdyAlIHVuaWZvcm1zLm91dFdpZHRoKSAqIHVuaWZvcm1zLnN0cmlkZXNbMV0gLVxuICAgICAgICAgICAgICB1bmlmb3Jtcy5wYWRzWzFdO1xuICAgICAgICAgIGxldCB4Q29sID0gb2Zmc2V0WCArIHVuaWZvcm1zLmRpbGF0aW9uc1sxXSAqICgoY29sICVcbiAgICAgICAgICAgICAgdW5pZm9ybXMuaXRlbXNQZXJCbG9ja1JvdykgLyB1bmlmb3Jtcy5pbkNoYW5uZWxzKTtcbiAgICAgICAgICBsZXQgY2ggPSBjb2wgJSB1bmlmb3Jtcy5pbkNoYW5uZWxzO1xuICAgICAgICAgIGlmKHhDb2wgPCB1bmlmb3Jtcy54U2hhcGVbJHtjb2xEaW19XSAmJiB4Q29sID49IDApIHtcbiAgICAgICAgICAgIHZhbHVlID0gJHtnZXRYU25pcHBldH07XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIHZhbHVlKTtcbiAgICAgIH1cbiAgICB9XG4gICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19