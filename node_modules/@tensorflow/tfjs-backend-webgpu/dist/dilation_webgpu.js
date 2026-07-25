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
export class Dilation2DProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'w'];
        this.uniforms = 'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'dilation2d';
    }
    getUserCode() {
        const userCode = `
       ${main('index')} {
         if (index < uniforms.size) {
           let neg_infinity = -3.4e38;
           let coords = getOutputCoords();
           let batch = coords.x;
           let d1 = coords.w;
           let outTopLeftCorner = coords.yz * uniforms.strides - uniforms.pads;
           let hBeg = outTopLeftCorner.x;
           let wBeg = outTopLeftCorner.y;

           var curVal = neg_infinity;
           for (var h = 0; h < uniforms.filterDims[0]; h = h + 1) {
             let hIn = hBeg + h * uniforms.dilations[0];

             if (hIn >= 0 && hIn < uniforms.xShape[1]) {
               for (var w = 0; w < uniforms.filterDims[1]; w = w + 1) {
                 let wIn = wBeg + w * uniforms.dilations[1];

                 if (wIn >= 0 && wIn < uniforms.xShape[2]) {
                   let val = getX(batch, hIn, wIn, d1) + getW(h, w, d1);
                   if (val > curVal) {
                     curVal = val;
                   }
                 }
               }
             }
           }

           setOutputAtIndex(index, curVal);
         }
       }
     `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGlsYXRpb25fd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvZGlsYXRpb25fd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8saUJBQWlCO0lBVzVCLFlBQVksUUFBaUM7UUFON0Msa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUMzQixhQUFRLEdBQ0osa0ZBQWtGLENBQUM7UUFDdkYsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUM7UUFDckMsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxZQUFZLENBQUM7SUFDaEMsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztTQUNaLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7TUErQmhCLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIERpbGF0aW9uMkRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ3cnXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgJ2ZpbHRlckRpbXM6IHZlYzI8aTMyPiwgcGFkczogdmVjMjxpMzI+LCBzdHJpZGVzOiB2ZWMyPGkzMj4sIGRpbGF0aW9uczogdmVjMjxpMzI+JztcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLm91dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ2RpbGF0aW9uMmQnO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICAgbGV0IG5lZ19pbmZpbml0eSA9IC0zLjRlMzg7XG4gICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzLng7XG4gICAgICAgICAgIGxldCBkMSA9IGNvb3Jkcy53O1xuICAgICAgICAgICBsZXQgb3V0VG9wTGVmdENvcm5lciA9IGNvb3Jkcy55eiAqIHVuaWZvcm1zLnN0cmlkZXMgLSB1bmlmb3Jtcy5wYWRzO1xuICAgICAgICAgICBsZXQgaEJlZyA9IG91dFRvcExlZnRDb3JuZXIueDtcbiAgICAgICAgICAgbGV0IHdCZWcgPSBvdXRUb3BMZWZ0Q29ybmVyLnk7XG5cbiAgICAgICAgICAgdmFyIGN1clZhbCA9IG5lZ19pbmZpbml0eTtcbiAgICAgICAgICAgZm9yICh2YXIgaCA9IDA7IGggPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyBoID0gaCArIDEpIHtcbiAgICAgICAgICAgICBsZXQgaEluID0gaEJlZyArIGggKiB1bmlmb3Jtcy5kaWxhdGlvbnNbMF07XG5cbiAgICAgICAgICAgICBpZiAoaEluID49IDAgJiYgaEluIDwgdW5pZm9ybXMueFNoYXBlWzFdKSB7XG4gICAgICAgICAgICAgICBmb3IgKHZhciB3ID0gMDsgdyA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMV07IHcgPSB3ICsgMSkge1xuICAgICAgICAgICAgICAgICBsZXQgd0luID0gd0JlZyArIHcgKiB1bmlmb3Jtcy5kaWxhdGlvbnNbMV07XG5cbiAgICAgICAgICAgICAgICAgaWYgKHdJbiA+PSAwICYmIHdJbiA8IHVuaWZvcm1zLnhTaGFwZVsyXSkge1xuICAgICAgICAgICAgICAgICAgIGxldCB2YWwgPSBnZXRYKGJhdGNoLCBoSW4sIHdJbiwgZDEpICsgZ2V0VyhoLCB3LCBkMSk7XG4gICAgICAgICAgICAgICAgICAgaWYgKHZhbCA+IGN1clZhbCkge1xuICAgICAgICAgICAgICAgICAgICAgY3VyVmFsID0gdmFsO1xuICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgIH1cbiAgICAgICAgICAgfVxuXG4gICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGN1clZhbCk7XG4gICAgICAgICB9XG4gICAgICAgfVxuICAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19