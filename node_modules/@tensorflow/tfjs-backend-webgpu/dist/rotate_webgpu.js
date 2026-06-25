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
export class RotateProgram {
    constructor(imageShape, fillValue) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = imageShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = `centerX : f32, centerY : f32, sinRadians : f32,
          cosRadians : f32,`;
        this.shaderKey = 'rotate';
        this.outputShape = imageShape;
        if (typeof fillValue === 'number') {
            this.uniforms += ` fillValue : f32,`;
            this.fillSnippet = `var outputValue = uniforms.fillValue;`;
            this.shaderKey += '_float';
        }
        else {
            this.uniforms += ` fillValue : vec3<f32>,`;
            this.fillSnippet = `var outputValue = uniforms.fillValue[coords[3]];`;
            this.shaderKey += '_vec3';
        }
    }
    getUserCode() {
        const userCode = `
        ${main('index')} {
          if (index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            let coordXFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.cosRadians - (f32(coords[1]) - uniforms.centerY) *
                uniforms.sinRadians;
            let coordYFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.sinRadians + (f32(coords[1]) - uniforms.centerY) *
                uniforms.cosRadians;
            let coordX = i32(round(coordXFloat + uniforms.centerX));
            let coordY = i32(round(coordYFloat + uniforms.centerY));
            ${this.fillSnippet}
            if(coordX >= 0 && coordX < uniforms.xShape[2] && coordY >= 0 &&
                coordY < uniforms.xShape[1]) {
              outputValue = getX(coords[0], coordY, coordX, coords[3]);
            }
            setOutputAtIndex(index, outputValue);
          }
        }
      `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicm90YXRlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3JvdGF0ZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxhQUFhO0lBV3hCLFlBQ0ksVUFBNEMsRUFDNUMsU0FBMEM7UUFaOUMsZ0JBQVcsR0FBYSxFQUFFLENBQUM7UUFJM0Isa0JBQWEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRXRCLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBS1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxVQUFVLENBQUM7UUFDOUIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLFFBQVEsR0FBRzs0QkFDUSxDQUFDO1FBQ3pCLElBQUksQ0FBQyxTQUFTLEdBQUcsUUFBUSxDQUFDO1FBQzFCLElBQUksQ0FBQyxXQUFXLEdBQUcsVUFBVSxDQUFDO1FBRTlCLElBQUksT0FBTyxTQUFTLEtBQUssUUFBUSxFQUFFO1lBQ2pDLElBQUksQ0FBQyxRQUFRLElBQUksbUJBQW1CLENBQUM7WUFDckMsSUFBSSxDQUFDLFdBQVcsR0FBRyx1Q0FBdUMsQ0FBQztZQUMzRCxJQUFJLENBQUMsU0FBUyxJQUFJLFFBQVEsQ0FBQztTQUM1QjthQUFNO1lBQ0wsSUFBSSxDQUFDLFFBQVEsSUFBSSx5QkFBeUIsQ0FBQztZQUMzQyxJQUFJLENBQUMsV0FBVyxHQUFHLGtEQUFrRCxDQUFDO1lBQ3RFLElBQUksQ0FBQyxTQUFTLElBQUksT0FBTyxDQUFDO1NBQzNCO0lBQ0gsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztVQUNYLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7O2NBV1QsSUFBSSxDQUFDLFdBQVc7Ozs7Ozs7O09BUXZCLENBQUM7UUFDSixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgUm90YXRlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zOiBzdHJpbmc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIGZpbGxTbmlwcGV0OiBzdHJpbmc7XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgaW1hZ2VTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBmaWxsVmFsdWU6IG51bWJlcnxbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gaW1hZ2VTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy51bmlmb3JtcyA9IGBjZW50ZXJYIDogZjMyLCBjZW50ZXJZIDogZjMyLCBzaW5SYWRpYW5zIDogZjMyLFxuICAgICAgICAgIGNvc1JhZGlhbnMgOiBmMzIsYDtcbiAgICB0aGlzLnNoYWRlcktleSA9ICdyb3RhdGUnO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBpbWFnZVNoYXBlO1xuXG4gICAgaWYgKHR5cGVvZiBmaWxsVmFsdWUgPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLnVuaWZvcm1zICs9IGAgZmlsbFZhbHVlIDogZjMyLGA7XG4gICAgICB0aGlzLmZpbGxTbmlwcGV0ID0gYHZhciBvdXRwdXRWYWx1ZSA9IHVuaWZvcm1zLmZpbGxWYWx1ZTtgO1xuICAgICAgdGhpcy5zaGFkZXJLZXkgKz0gJ19mbG9hdCc7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudW5pZm9ybXMgKz0gYCBmaWxsVmFsdWUgOiB2ZWMzPGYzMj4sYDtcbiAgICAgIHRoaXMuZmlsbFNuaXBwZXQgPSBgdmFyIG91dHB1dFZhbHVlID0gdW5pZm9ybXMuZmlsbFZhbHVlW2Nvb3Jkc1szXV07YDtcbiAgICAgIHRoaXMuc2hhZGVyS2V5ICs9ICdfdmVjMyc7XG4gICAgfVxuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgICBsZXQgY29vcmRYRmxvYXQgPSAoZjMyKGNvb3Jkc1syXSkgLSB1bmlmb3Jtcy5jZW50ZXJYKSAqXG4gICAgICAgICAgICAgICAgdW5pZm9ybXMuY29zUmFkaWFucyAtIChmMzIoY29vcmRzWzFdKSAtIHVuaWZvcm1zLmNlbnRlclkpICpcbiAgICAgICAgICAgICAgICB1bmlmb3Jtcy5zaW5SYWRpYW5zO1xuICAgICAgICAgICAgbGV0IGNvb3JkWUZsb2F0ID0gKGYzMihjb29yZHNbMl0pIC0gdW5pZm9ybXMuY2VudGVyWCkgKlxuICAgICAgICAgICAgICAgIHVuaWZvcm1zLnNpblJhZGlhbnMgKyAoZjMyKGNvb3Jkc1sxXSkgLSB1bmlmb3Jtcy5jZW50ZXJZKSAqXG4gICAgICAgICAgICAgICAgdW5pZm9ybXMuY29zUmFkaWFucztcbiAgICAgICAgICAgIGxldCBjb29yZFggPSBpMzIocm91bmQoY29vcmRYRmxvYXQgKyB1bmlmb3Jtcy5jZW50ZXJYKSk7XG4gICAgICAgICAgICBsZXQgY29vcmRZID0gaTMyKHJvdW5kKGNvb3JkWUZsb2F0ICsgdW5pZm9ybXMuY2VudGVyWSkpO1xuICAgICAgICAgICAgJHt0aGlzLmZpbGxTbmlwcGV0fVxuICAgICAgICAgICAgaWYoY29vcmRYID49IDAgJiYgY29vcmRYIDwgdW5pZm9ybXMueFNoYXBlWzJdICYmIGNvb3JkWSA+PSAwICYmXG4gICAgICAgICAgICAgICAgY29vcmRZIDwgdW5pZm9ybXMueFNoYXBlWzFdKSB7XG4gICAgICAgICAgICAgIG91dHB1dFZhbHVlID0gZ2V0WChjb29yZHNbMF0sIGNvb3JkWSwgY29vcmRYLCBjb29yZHNbM10pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgb3V0cHV0VmFsdWUpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==