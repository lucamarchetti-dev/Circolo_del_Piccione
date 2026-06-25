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
export class SearchSortedProgram {
    constructor(outputShape, side) {
        this.outputShape = [];
        this.variableNames = ['sortedSequence', 'values'];
        this.uniforms = 'numInputs : i32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.side = side;
        this.shaderKey = `search_sorted_${side}`;
    }
    getUserCode() {
        const boundComparator = this.side === 'left' ? '<' : '<=';
        const userCode = `
      fn findBound(batch: i32, value: f32) -> i32 {
        var left = i32(0);
        var right = uniforms.numInputs;
        while (left < right) {
          var mid = (left + right) / 2;
          if (getSortedSequence(batch, mid) ${boundComparator} value) {
            left = mid + 1;
          } else {
            right = mid;
          }
        }
        return right;
      }

      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let value = getValuesByOutputIndex(index);
          setOutputAtIndexI32(index, findBound(coords[0], value));
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2VhcmNoX3NvcnRlZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zZWFyY2hfc29ydGVkX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLG1CQUFtQjtJQVc5QixZQUFZLFdBQTZCLEVBQUUsSUFBb0I7UUFWL0QsZ0JBQVcsR0FBYSxFQUFFLENBQUM7UUFJM0Isa0JBQWEsR0FBRyxDQUFDLGdCQUFnQixFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBQzdDLGFBQVEsR0FBRyxrQkFBa0IsQ0FBQztRQUM5QixrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUlWLElBQUksQ0FBQyxXQUFXLEdBQUcsV0FBVyxDQUFDO1FBQy9CLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRS9ELElBQUksQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2pCLElBQUksQ0FBQyxTQUFTLEdBQUcsaUJBQWlCLElBQUksRUFBRSxDQUFDO0lBQzNDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxlQUFlLEdBQUcsSUFBSSxDQUFDLElBQUksS0FBSyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDO1FBQzFELE1BQU0sUUFBUSxHQUFHOzs7Ozs7OENBTXlCLGVBQWU7Ozs7Ozs7OztRQVNyRCxJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7O0tBT2hCLENBQUM7UUFFRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgU2VhcmNoU29ydGVkUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydzb3J0ZWRTZXF1ZW5jZScsICd2YWx1ZXMnXTtcbiAgdW5pZm9ybXMgPSAnbnVtSW5wdXRzIDogaTMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuICBzaWRlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3Iob3V0cHV0U2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIHNpZGU6ICdsZWZ0J3wncmlnaHQnKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2lkZSA9IHNpZGU7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgc2VhcmNoX3NvcnRlZF8ke3NpZGV9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgYm91bmRDb21wYXJhdG9yID0gdGhpcy5zaWRlID09PSAnbGVmdCcgPyAnPCcgOiAnPD0nO1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgZm4gZmluZEJvdW5kKGJhdGNoOiBpMzIsIHZhbHVlOiBmMzIpIC0+IGkzMiB7XG4gICAgICAgIHZhciBsZWZ0ID0gaTMyKDApO1xuICAgICAgICB2YXIgcmlnaHQgPSB1bmlmb3Jtcy5udW1JbnB1dHM7XG4gICAgICAgIHdoaWxlIChsZWZ0IDwgcmlnaHQpIHtcbiAgICAgICAgICB2YXIgbWlkID0gKGxlZnQgKyByaWdodCkgLyAyO1xuICAgICAgICAgIGlmIChnZXRTb3J0ZWRTZXF1ZW5jZShiYXRjaCwgbWlkKSAke2JvdW5kQ29tcGFyYXRvcn0gdmFsdWUpIHtcbiAgICAgICAgICAgIGxlZnQgPSBtaWQgKyAxO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICByaWdodCA9IG1pZDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJpZ2h0O1xuICAgICAgfVxuXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCB2YWx1ZSA9IGdldFZhbHVlc0J5T3V0cHV0SW5kZXgoaW5kZXgpO1xuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXhJMzIoaW5kZXgsIGZpbmRCb3VuZChjb29yZHNbMF0sIHZhbHVlKSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuXG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=