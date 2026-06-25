/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { getUnaryOpString } from './unary_op_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class UnaryOpProgram {
    constructor(outputShape, op, uniforms = '') {
        this.variableNames = ['A'];
        this.size = true;
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
        const workgroupSizeX = 128;
        this.workgroupSize = [workgroupSizeX, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.op = op;
        if (uniforms !== '') {
            this.uniforms = uniforms;
        }
        this.shaderKey = `unary_${op}`;
    }
    getUserCode() {
        return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false)}
      }
      ${main('index')} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, unaryOperation(a));
        }
      }
      `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidW5hcnlfb3Bfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvdW5hcnlfb3Bfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxnQkFBZ0IsRUFBYyxNQUFNLGlCQUFpQixDQUFDO0FBQzlELE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sY0FBYztJQVd6QixZQUFZLFdBQXFCLEVBQUUsRUFBZSxFQUFFLFFBQVEsR0FBRyxFQUFFO1FBTmpFLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUl0QixTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsMkVBQTJFO1FBQzNFLE1BQU0sY0FBYyxHQUFHLEdBQUcsQ0FBQztRQUMzQixJQUFJLENBQUMsYUFBYSxHQUFHLENBQUMsY0FBYyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUM1QyxJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQztRQUMvQixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsRUFBRSxHQUFHLEVBQUUsQ0FBQztRQUNiLElBQUksUUFBUSxLQUFLLEVBQUUsRUFBRTtZQUNuQixJQUFJLENBQUMsUUFBUSxHQUFHLFFBQVEsQ0FBQztTQUMxQjtRQUNELElBQUksQ0FBQyxTQUFTLEdBQUcsU0FBUyxFQUFFLEVBQUUsQ0FBQztJQUNqQyxDQUFDO0lBRUQsV0FBVztRQUNULE9BQU87O1VBRUQsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLEVBQUUsRUFBRSxLQUFLLENBQUM7O1FBRWxDLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7OztPQU1kLENBQUM7SUFDTixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0VW5hcnlPcFN0cmluZywgVW5hcnlPcFR5cGV9IGZyb20gJy4vdW5hcnlfb3BfdXRpbCc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFVuYXJ5T3BQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgb3A6IFVuYXJ5T3BUeXBlO1xuICB1bmlmb3Jtcz86IHN0cmluZztcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3Iob3V0cHV0U2hhcGU6IG51bWJlcltdLCBvcDogVW5hcnlPcFR5cGUsIHVuaWZvcm1zID0gJycpIHtcbiAgICAvLyBUT0RPKGppYWppYS5xaW5AaW50ZWwuY29tKTogSGV1cmlzdGljYWxseSBzZWxlY3QgYSBnb29kIHdvcmsgZ3JvdXAgc2l6ZS5cbiAgICBjb25zdCB3b3JrZ3JvdXBTaXplWCA9IDEyODtcbiAgICB0aGlzLndvcmtncm91cFNpemUgPSBbd29ya2dyb3VwU2l6ZVgsIDEsIDFdO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy5vcCA9IG9wO1xuICAgIGlmICh1bmlmb3JtcyAhPT0gJycpIHtcbiAgICAgIHRoaXMudW5pZm9ybXMgPSB1bmlmb3JtcztcbiAgICB9XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgdW5hcnlfJHtvcH1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICByZXR1cm4gYFxuICAgICAgZm4gdW5hcnlPcGVyYXRpb24oYSA6IGYzMikgLT4gZjMyIHtcbiAgICAgICAgJHtnZXRVbmFyeU9wU3RyaW5nKHRoaXMub3AsIGZhbHNlKX1cbiAgICAgIH1cbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IGEgPSBnZXRBQnlPdXRwdXRJbmRleChpbmRleCk7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgdW5hcnlPcGVyYXRpb24oYSkpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBgO1xuICB9XG59XG4iXX0=