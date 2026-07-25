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
export class LinSpaceProgram {
    constructor(shape) {
        this.variableNames = [];
        this.outputShape = [];
        this.uniforms = 'start : f32, step : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [shape];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'linSpace';
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          setOutputAtIndex(index, uniforms.start + f32(index) * uniforms.step);
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibGluX3NwYWNlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2xpbl9zcGFjZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxlQUFlO0lBVTFCLFlBQVksS0FBYTtRQVR6QixrQkFBYSxHQUFhLEVBQUUsQ0FBQztRQUM3QixnQkFBVyxHQUFhLEVBQUUsQ0FBQztRQUkzQixhQUFRLEdBQUcsMEJBQTBCLENBQUM7UUFDdEMsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDM0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxVQUFVLENBQUM7SUFDOUIsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7O0tBS2hCLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgTGluU3BhY2VQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXM6IHN0cmluZ1tdID0gW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHVuaWZvcm1zID0gJ3N0YXJ0IDogZjMyLCBzdGVwIDogZjMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBudW1iZXIpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW3NoYXBlXTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLnNoYWRlcktleSA9ICdsaW5TcGFjZSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCB1bmlmb3Jtcy5zdGFydCArIGYzMihpbmRleCkgKiB1bmlmb3Jtcy5zdGVwKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=