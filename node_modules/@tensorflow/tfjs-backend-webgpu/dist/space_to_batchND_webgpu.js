/**
 * @license
 * Copyright 2023 Google LLC.
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
import { padCommon } from './pad_webgpu';
import { getSwitchedCoords } from './transpose_webgpu';
import { getCoordsDataType, getCoordsFromIndexSnippet, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class SpaceToBatchNDProgram {
    constructor(xShape, paddedXShape, paddings, reshapedPaddedXShape, newDim, paddedXShapeStridesShapeLength) {
        this.variableNames = ['x'];
        this.outputShape = [];
        this.uniforms = '';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(reshapedPaddedXShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = reshapedPaddedXShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.newDim = newDim;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.xShape = xShape;
        this.paddedXShape = paddedXShape;
        this.uniforms += `reshapedPaddedXShape : ${getCoordsDataType(reshapedPaddedXShape.length)}, paddedXShapeStrides : ${getCoordsDataType(paddedXShapeStridesShapeLength)}, `;
        paddings.map((_, i) => {
            this.uniforms += ` pad${i} : vec2<i32>,`;
        });
        this.shaderKey = `spaceToBatchND_${newDim}`;
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      ${getCoordsFromIndexSnippet(this.paddedXShape, 'PaddedX')}
      ${main('index')} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let switchedIndex = getIndexFromCoords${this.outputShape.length}D(${dtype}(${switched}), uniforms.reshapedPaddedXShape);
          let paddedCoords = getPaddedXCoordsFromIndex(switchedIndex);
          ${padCommon(this.xShape, true)}
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhY2VfdG9fYmF0Y2hORF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zcGFjZV90b19iYXRjaE5EX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3ZDLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBQ3JELE9BQU8sRUFBQyxpQkFBaUIsRUFBRSx5QkFBeUIsRUFBRSxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDMUgsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8scUJBQXFCO0lBYWhDLFlBQ0ksTUFBZ0IsRUFBRSxZQUFzQixFQUN4QyxRQUFpQyxFQUFFLG9CQUE4QixFQUNqRSxNQUFnQixFQUFFLDhCQUFzQztRQWY1RCxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsZ0JBQVcsR0FBYSxFQUFFLENBQUM7UUFJM0IsYUFBUSxHQUFHLEVBQUUsQ0FBQztRQUNkLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUlyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBTVYsTUFBTSxXQUFXLEdBQWEsSUFBSSxLQUFLLENBQUMsb0JBQW9CLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDckUsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDM0MsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLG9CQUFvQixDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ2xEO1FBQ0QsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7UUFDckIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7UUFDckIsSUFBSSxDQUFDLFlBQVksR0FBRyxZQUFZLENBQUM7UUFDakMsSUFBSSxDQUFDLFFBQVEsSUFBSSwwQkFDYixpQkFBaUIsQ0FDYixvQkFBb0IsQ0FBQyxNQUFNLENBQUMsMkJBQ2hDLGlCQUFpQixDQUFDLDhCQUE4QixDQUFDLElBQUksQ0FBQztRQUMxRCxRQUFRLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQ3BCLElBQUksQ0FBQyxRQUFRLElBQUksT0FBTyxDQUFDLGVBQWUsQ0FBQztRQUMzQyxDQUFDLENBQUMsQ0FBQztRQUNILElBQUksQ0FBQyxTQUFTLEdBQUcsa0JBQWtCLE1BQU0sRUFBRSxDQUFDO0lBQzlDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxLQUFLLEdBQUcsaUJBQWlCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN6RCxNQUFNLFFBQVEsR0FBRyxpQkFBaUIsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFaEQsTUFBTSxRQUFRLEdBQUc7UUFDYix5QkFBeUIsQ0FBQyxJQUFJLENBQUMsWUFBWSxFQUFFLFNBQVMsQ0FBQztRQUN2RCxJQUFJLENBQUMsT0FBTyxDQUFDOzs7a0RBRzZCLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxLQUNqRSxLQUFLLElBQUksUUFBUTs7WUFFYixTQUFTLENBQUMsSUFBSSxDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUM7OztLQUduQyxDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge3BhZENvbW1vbn0gZnJvbSAnLi9wYWRfd2ViZ3B1JztcbmltcG9ydCB7Z2V0U3dpdGNoZWRDb29yZHN9IGZyb20gJy4vdHJhbnNwb3NlX3dlYmdwdSc7XG5pbXBvcnQge2dldENvb3Jkc0RhdGFUeXBlLCBnZXRDb29yZHNGcm9tSW5kZXhTbmlwcGV0LCBnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBTcGFjZVRvQmF0Y2hORFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB1bmlmb3JtcyA9ICcnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBuZXdEaW06IG51bWJlcltdO1xuICB4U2hhcGU6IG51bWJlcltdO1xuICBwYWRkZWRYU2hhcGU6IG51bWJlcltdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIHhTaGFwZTogbnVtYmVyW10sIHBhZGRlZFhTaGFwZTogbnVtYmVyW10sXG4gICAgICBwYWRkaW5nczogQXJyYXk8W251bWJlciwgbnVtYmVyXT4sIHJlc2hhcGVkUGFkZGVkWFNoYXBlOiBudW1iZXJbXSxcbiAgICAgIG5ld0RpbTogbnVtYmVyW10sIHBhZGRlZFhTaGFwZVN0cmlkZXNTaGFwZUxlbmd0aDogbnVtYmVyKSB7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGU6IG51bWJlcltdID0gbmV3IEFycmF5KHJlc2hhcGVkUGFkZGVkWFNoYXBlLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBvdXRwdXRTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgICAgb3V0cHV0U2hhcGVbaV0gPSByZXNoYXBlZFBhZGRlZFhTaGFwZVtuZXdEaW1baV1dO1xuICAgIH1cbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5uZXdEaW0gPSBuZXdEaW07XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMueFNoYXBlID0geFNoYXBlO1xuICAgIHRoaXMucGFkZGVkWFNoYXBlID0gcGFkZGVkWFNoYXBlO1xuICAgIHRoaXMudW5pZm9ybXMgKz0gYHJlc2hhcGVkUGFkZGVkWFNoYXBlIDogJHtcbiAgICAgICAgZ2V0Q29vcmRzRGF0YVR5cGUoXG4gICAgICAgICAgICByZXNoYXBlZFBhZGRlZFhTaGFwZS5sZW5ndGgpfSwgcGFkZGVkWFNoYXBlU3RyaWRlcyA6ICR7XG4gICAgICAgIGdldENvb3Jkc0RhdGFUeXBlKHBhZGRlZFhTaGFwZVN0cmlkZXNTaGFwZUxlbmd0aCl9LCBgO1xuICAgIHBhZGRpbmdzLm1hcCgoXywgaSkgPT4ge1xuICAgICAgdGhpcy51bmlmb3JtcyArPSBgIHBhZCR7aX0gOiB2ZWMyPGkzMj4sYDtcbiAgICB9KTtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBzcGFjZVRvQmF0Y2hORF8ke25ld0RpbX1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCBkdHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoKTtcbiAgICBjb25zdCBzd2l0Y2hlZCA9IGdldFN3aXRjaGVkQ29vcmRzKHRoaXMubmV3RGltKTtcblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHtnZXRDb29yZHNGcm9tSW5kZXhTbmlwcGV0KHRoaXMucGFkZGVkWFNoYXBlLCAnUGFkZGVkWCcpfVxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCBzd2l0Y2hlZEluZGV4ID0gZ2V0SW5kZXhGcm9tQ29vcmRzJHt0aGlzLm91dHB1dFNoYXBlLmxlbmd0aH1EKCR7XG4gICAgICAgIGR0eXBlfSgke3N3aXRjaGVkfSksIHVuaWZvcm1zLnJlc2hhcGVkUGFkZGVkWFNoYXBlKTtcbiAgICAgICAgICBsZXQgcGFkZGVkQ29vcmRzID0gZ2V0UGFkZGVkWENvb3Jkc0Zyb21JbmRleChzd2l0Y2hlZEluZGV4KTtcbiAgICAgICAgICAke3BhZENvbW1vbih0aGlzLnhTaGFwZSwgdHJ1ZSl9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19