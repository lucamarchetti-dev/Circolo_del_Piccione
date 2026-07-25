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
export class TileProgram {
    constructor(aShape, reps) {
        this.variableNames = ['A'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[i] * reps[i];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.rank = this.outputShape.length;
        this.shaderKey = 'tile';
    }
    getUserCode() {
        const sourceCoords = getSourceCoords(this.rank, 'uniforms.');
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          setOutputAtIndex(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
function getSourceCoords(rank, uniformPrefix = '') {
    if (rank >= 5) {
        throw Error(`Tile for rank ${rank} is not yet supported`);
    }
    if (rank === 1) {
        return `(resRC % ${uniformPrefix}aShape)`;
    }
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < rank; i++) {
        sourceCoords.push(`(${currentCoords[i]} % ${uniformPrefix}aShape[${i}])`);
    }
    return sourceCoords.join();
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGlsZV93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy90aWxlX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLFdBQVc7SUFVdEIsWUFBWSxNQUFnQixFQUFFLElBQWM7UUFUNUMsa0JBQWEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBS3RCLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBSVYsTUFBTSxXQUFXLEdBQWEsSUFBSSxLQUFLLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3ZELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzNDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ3RDO1FBQ0QsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQztRQUNwQyxJQUFJLENBQUMsU0FBUyxHQUFHLE1BQU0sQ0FBQztJQUMxQixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sWUFBWSxHQUFHLGVBQWUsQ0FBQyxJQUFJLENBQUMsSUFBSSxFQUFFLFdBQVcsQ0FBQyxDQUFDO1FBRTdELE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7O3lDQUdvQixZQUFZOzs7S0FHaEQsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELFNBQVMsZUFBZSxDQUFDLElBQVksRUFBRSxhQUFhLEdBQUcsRUFBRTtJQUN2RCxJQUFJLElBQUksSUFBSSxDQUFDLEVBQUU7UUFDYixNQUFNLEtBQUssQ0FBQyxpQkFBaUIsSUFBSSx1QkFBdUIsQ0FBQyxDQUFDO0tBQzNEO0lBQ0QsSUFBSSxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2QsT0FBTyxZQUFZLGFBQWEsU0FBUyxDQUFDO0tBQzNDO0lBRUQsTUFBTSxhQUFhLEdBQUcsQ0FBQyxTQUFTLEVBQUUsU0FBUyxFQUFFLFNBQVMsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUNuRSxNQUFNLFlBQVksR0FBRyxFQUFFLENBQUM7SUFDeEIsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUM3QixZQUFZLENBQUMsSUFBSSxDQUFDLElBQUksYUFBYSxDQUFDLENBQUMsQ0FBQyxNQUFNLGFBQWEsVUFBVSxDQUFDLElBQUksQ0FBQyxDQUFDO0tBQzNFO0lBQ0QsT0FBTyxZQUFZLENBQUMsSUFBSSxFQUFFLENBQUM7QUFDN0IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBUaWxlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcbiAgcmFuazogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKGFTaGFwZTogbnVtYmVyW10sIHJlcHM6IG51bWJlcltdKSB7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGU6IG51bWJlcltdID0gbmV3IEFycmF5KGFTaGFwZS5sZW5ndGgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0cHV0U2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgIG91dHB1dFNoYXBlW2ldID0gYVNoYXBlW2ldICogcmVwc1tpXTtcbiAgICB9XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcbiAgICB0aGlzLnJhbmsgPSB0aGlzLm91dHB1dFNoYXBlLmxlbmd0aDtcbiAgICB0aGlzLnNoYWRlcktleSA9ICd0aWxlJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3Qgc291cmNlQ29vcmRzID0gZ2V0U291cmNlQ29vcmRzKHRoaXMucmFuaywgJ3VuaWZvcm1zLicpO1xuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCByZXNSQyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgZ2V0QSgke3NvdXJjZUNvb3Jkc30pKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG5cbmZ1bmN0aW9uIGdldFNvdXJjZUNvb3JkcyhyYW5rOiBudW1iZXIsIHVuaWZvcm1QcmVmaXggPSAnJyk6IHN0cmluZyB7XG4gIGlmIChyYW5rID49IDUpIHtcbiAgICB0aHJvdyBFcnJvcihgVGlsZSBmb3IgcmFuayAke3Jhbmt9IGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbiAgaWYgKHJhbmsgPT09IDEpIHtcbiAgICByZXR1cm4gYChyZXNSQyAlICR7dW5pZm9ybVByZWZpeH1hU2hhcGUpYDtcbiAgfVxuXG4gIGNvbnN0IGN1cnJlbnRDb29yZHMgPSBbJ3Jlc1JDLngnLCAncmVzUkMueScsICdyZXNSQy56JywgJ3Jlc1JDLncnXTtcbiAgY29uc3Qgc291cmNlQ29vcmRzID0gW107XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmFuazsgaSsrKSB7XG4gICAgc291cmNlQ29vcmRzLnB1c2goYCgke2N1cnJlbnRDb29yZHNbaV19ICUgJHt1bmlmb3JtUHJlZml4fWFTaGFwZVske2l9XSlgKTtcbiAgfVxuICByZXR1cm4gc291cmNlQ29vcmRzLmpvaW4oKTtcbn1cbiJdfQ==