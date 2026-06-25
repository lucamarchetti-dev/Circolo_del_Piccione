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
import { atomicAddSnippet } from './shader_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class SparseSegmentSumProgram {
    constructor(outShape, sparseSize, outputDtype) {
        this.variableNames = ['input', 'indices', 'segmentIds'];
        this.outputShape = [];
        this.uniforms = 'segmentSize : i32, sparseSize : i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = outShape;
        this.type = outputDtype;
        this.dispatchLayout = flatDispatchLayout([sparseSize]);
        this.dispatch =
            computeDispatch(this.dispatchLayout, [sparseSize], this.workgroupSize);
        this.shaderKey = 'sparseSegmentSum';
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.sparseSize) {
        let indexInSegmentIds = index / uniforms.segmentSize;
        let indexInSegment = index % uniforms.segmentSize;
        let indexInInput = indices[indexInSegmentIds];
        let segmentId = segmentIds[indexInSegmentIds];

        let value = input[indexInInput * uniforms.segmentSize + indexInSegment];
        let outIndex = segmentId * uniforms.segmentSize + indexInSegment;
        ${atomicAddSnippet('&result[outIndex]', 'value', this.type)}
      }
    }
  `;
        return userCode;
    }
}
export class SparseSegmentIdCountProgram {
    constructor(outShape, segmentIdsShape) {
        this.variableNames = ['segmentIds'];
        this.outputShape = [];
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = [outShape];
        this.dispatchLayout = flatDispatchLayout(segmentIdsShape);
        this.dispatch = computeDispatch(this.dispatchLayout, segmentIdsShape, this.workgroupSize);
        this.shaderKey = 'sparseSegmentIdCountProgram';
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.segmentIdsShape) {
        let segmentId = segmentIds[index];
        ${atomicAddSnippet('&result[segmentId]', '1', 'int32')}
      }
    }
  `;
        return userCode;
    }
}
export class SparseSegmentMeanProgram {
    constructor(outShape, outputDtype) {
        this.variableNames = ['segmentSum', 'sameSegmentIdCount'];
        this.outputShape = [];
        this.uniforms = 'segmentSize : i32';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outShape;
        this.type = outputDtype;
        this.dispatchLayout = flatDispatchLayout(outShape);
        this.dispatch =
            computeDispatch(this.dispatchLayout, outShape, this.workgroupSize);
        this.shaderKey = 'sparseSegmentMean';
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let segmentId = index / uniforms.segmentSize;
        let count = sameSegmentIdCount[segmentId];
        if (count != 0) {
          ${this.type === 'float32' ?
            'setOutputAtIndex(index, segmentSum[index] / f32(count));' :
            'setOutputAtIndexI32(index, segmentSum[index] / count);'}
        }
      }
    }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3NlZ21lbnRfcmVkdWNlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3NwYXJzZV9zZWdtZW50X3JlZHVjZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQy9DLE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sdUJBQXVCO0lBV2xDLFlBQVksUUFBa0IsRUFBRSxVQUFrQixFQUFFLFdBQXFCO1FBVnpFLGtCQUFhLEdBQUcsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLFlBQVksQ0FBQyxDQUFDO1FBQ25ELGdCQUFXLEdBQWEsRUFBRSxDQUFDO1FBSTNCLGFBQVEsR0FBRyxzQ0FBc0MsQ0FBQztRQUNsRCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsV0FBTSxHQUFHLElBQUksQ0FBQztRQUlaLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDO1FBQzVCLElBQUksQ0FBQyxJQUFJLEdBQUcsV0FBVyxDQUFDO1FBQ3hCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDO1FBQ3ZELElBQUksQ0FBQyxRQUFRO1lBQ1QsZUFBZSxDQUFDLElBQUksQ0FBQyxjQUFjLEVBQUUsQ0FBQyxVQUFVLENBQUMsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFM0UsSUFBSSxDQUFDLFNBQVMsR0FBRyxrQkFBa0IsQ0FBQztJQUN0QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7O1VBVVgsZ0JBQWdCLENBQ1osbUJBQW1CLEVBQUUsT0FBTyxFQUFFLElBQUksQ0FBQyxJQUEyQixDQUFDOzs7R0FHeEUsQ0FBQztRQUNBLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELE1BQU0sT0FBTywyQkFBMkI7SUFTdEMsWUFBWSxRQUFnQixFQUFFLGVBQXlCO1FBUnZELGtCQUFhLEdBQUcsQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUMvQixnQkFBVyxHQUFhLEVBQUUsQ0FBQztRQUkzQixrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsV0FBTSxHQUFHLElBQUksQ0FBQztRQUdaLElBQUksQ0FBQyxXQUFXLEdBQUcsQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUM5QixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLGVBQWUsQ0FBQyxDQUFDO1FBQzFELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLGVBQWUsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFOUQsSUFBSSxDQUFDLFNBQVMsR0FBRyw2QkFBNkIsQ0FBQztJQUNqRCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7O1VBR1QsZ0JBQWdCLENBQUMsb0JBQW9CLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQzs7O0dBRzNELENBQUM7UUFDQSxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0Y7QUFFRCxNQUFNLE9BQU8sd0JBQXdCO0lBV25DLFlBQVksUUFBa0IsRUFBRSxXQUFxQjtRQVZyRCxrQkFBYSxHQUFHLENBQUMsWUFBWSxFQUFFLG9CQUFvQixDQUFDLENBQUM7UUFDckQsZ0JBQVcsR0FBYSxFQUFFLENBQUM7UUFJM0IsYUFBUSxHQUFHLG1CQUFtQixDQUFDO1FBQy9CLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBSVYsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUM7UUFDNUIsSUFBSSxDQUFDLElBQUksR0FBRyxXQUFXLENBQUM7UUFDeEIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUNuRCxJQUFJLENBQUMsUUFBUTtZQUNULGVBQWUsQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFLFFBQVEsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFdkUsSUFBSSxDQUFDLFNBQVMsR0FBRyxtQkFBbUIsQ0FBQztJQUN2QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO01BQ2YsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7WUFNWCxJQUFJLENBQUMsSUFBSSxLQUFLLFNBQVMsQ0FBQyxDQUFDO1lBQ3JCLDBEQUEwRCxDQUFDLENBQUM7WUFDNUQsd0RBQXdEOzs7O0dBSWpFLENBQUM7UUFDQSxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RGF0YVR5cGV9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7YXRvbWljQWRkU25pcHBldH0gZnJvbSAnLi9zaGFkZXJfdXRpbCc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFNwYXJzZVNlZ21lbnRTdW1Qcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ2lucHV0JywgJ2luZGljZXMnLCAnc2VnbWVudElkcyddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB1bmlmb3JtcyA9ICdzZWdtZW50U2l6ZSA6IGkzMiwgc3BhcnNlU2l6ZSA6IGkzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBhdG9taWMgPSB0cnVlO1xuICB0eXBlOiBEYXRhVHlwZTtcblxuICBjb25zdHJ1Y3RvcihvdXRTaGFwZTogbnVtYmVyW10sIHNwYXJzZVNpemU6IG51bWJlciwgb3V0cHV0RHR5cGU6IERhdGFUeXBlKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dFNoYXBlO1xuICAgIHRoaXMudHlwZSA9IG91dHB1dER0eXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQoW3NwYXJzZVNpemVdKTtcbiAgICB0aGlzLmRpc3BhdGNoID1cbiAgICAgICAgY29tcHV0ZURpc3BhdGNoKHRoaXMuZGlzcGF0Y2hMYXlvdXQsIFtzcGFyc2VTaXplXSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3NwYXJzZVNlZ21lbnRTdW0nO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNwYXJzZVNpemUpIHtcbiAgICAgICAgbGV0IGluZGV4SW5TZWdtZW50SWRzID0gaW5kZXggLyB1bmlmb3Jtcy5zZWdtZW50U2l6ZTtcbiAgICAgICAgbGV0IGluZGV4SW5TZWdtZW50ID0gaW5kZXggJSB1bmlmb3Jtcy5zZWdtZW50U2l6ZTtcbiAgICAgICAgbGV0IGluZGV4SW5JbnB1dCA9IGluZGljZXNbaW5kZXhJblNlZ21lbnRJZHNdO1xuICAgICAgICBsZXQgc2VnbWVudElkID0gc2VnbWVudElkc1tpbmRleEluU2VnbWVudElkc107XG5cbiAgICAgICAgbGV0IHZhbHVlID0gaW5wdXRbaW5kZXhJbklucHV0ICogdW5pZm9ybXMuc2VnbWVudFNpemUgKyBpbmRleEluU2VnbWVudF07XG4gICAgICAgIGxldCBvdXRJbmRleCA9IHNlZ21lbnRJZCAqIHVuaWZvcm1zLnNlZ21lbnRTaXplICsgaW5kZXhJblNlZ21lbnQ7XG4gICAgICAgICR7XG4gICAgICAgIGF0b21pY0FkZFNuaXBwZXQoXG4gICAgICAgICAgICAnJnJlc3VsdFtvdXRJbmRleF0nLCAndmFsdWUnLCB0aGlzLnR5cGUgYXMgJ2Zsb2F0MzInIHwgJ2ludDMyJyl9XG4gICAgICB9XG4gICAgfVxuICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgU3BhcnNlU2VnbWVudElkQ291bnRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3NlZ21lbnRJZHMnXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdID0gW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgYXRvbWljID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihvdXRTaGFwZTogbnVtYmVyLCBzZWdtZW50SWRzU2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IFtvdXRTaGFwZV07XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dChzZWdtZW50SWRzU2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHNlZ21lbnRJZHNTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3NwYXJzZVNlZ21lbnRJZENvdW50UHJvZ3JhbSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2VnbWVudElkc1NoYXBlKSB7XG4gICAgICAgIGxldCBzZWdtZW50SWQgPSBzZWdtZW50SWRzW2luZGV4XTtcbiAgICAgICAgJHthdG9taWNBZGRTbmlwcGV0KCcmcmVzdWx0W3NlZ21lbnRJZF0nLCAnMScsICdpbnQzMicpfVxuICAgICAgfVxuICAgIH1cbiAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIFNwYXJzZVNlZ21lbnRNZWFuUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydzZWdtZW50U3VtJywgJ3NhbWVTZWdtZW50SWRDb3VudCddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB1bmlmb3JtcyA9ICdzZWdtZW50U2l6ZSA6IGkzMic7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuICB0eXBlOiBEYXRhVHlwZTtcblxuICBjb25zdHJ1Y3RvcihvdXRTaGFwZTogbnVtYmVyW10sIG91dHB1dER0eXBlOiBEYXRhVHlwZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRTaGFwZTtcbiAgICB0aGlzLnR5cGUgPSBvdXRwdXREdHlwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KG91dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID1cbiAgICAgICAgY29tcHV0ZURpc3BhdGNoKHRoaXMuZGlzcGF0Y2hMYXlvdXQsIG91dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnc3BhcnNlU2VnbWVudE1lYW4nO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgbGV0IHNlZ21lbnRJZCA9IGluZGV4IC8gdW5pZm9ybXMuc2VnbWVudFNpemU7XG4gICAgICAgIGxldCBjb3VudCA9IHNhbWVTZWdtZW50SWRDb3VudFtzZWdtZW50SWRdO1xuICAgICAgICBpZiAoY291bnQgIT0gMCkge1xuICAgICAgICAgICR7XG4gICAgICAgIHRoaXMudHlwZSA9PT0gJ2Zsb2F0MzInID9cbiAgICAgICAgICAgICdzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBzZWdtZW50U3VtW2luZGV4XSAvIGYzMihjb3VudCkpOycgOlxuICAgICAgICAgICAgJ3NldE91dHB1dEF0SW5kZXhJMzIoaW5kZXgsIHNlZ21lbnRTdW1baW5kZXhdIC8gY291bnQpOyd9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=