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
import { atomicAddSnippet } from './shader_util';
import { dataTypeToGPUType, getCoordsDataType, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ScatterProgram {
    constructor(flattenXShape, sliceDim, indicesRank, updatesRank, strides, shape, outputDtype, sumDupeIndices = true) {
        this.variableNames = ['updates', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = shape;
        this.type = outputDtype;
        this.sumDupeIndices = sumDupeIndices;
        this.dispatchLayout = flatDispatchLayout(flattenXShape);
        // Dispatching based on |updates| shape instead of output shape.
        this.dispatch =
            computeDispatch(this.dispatchLayout, flattenXShape, this.workgroupSize);
        this.sliceDimGreaterThanOne = sliceDim > 1;
        this.shaderKey =
            `scatter_${indicesRank}_${updatesRank}_${this.sliceDimGreaterThanOne}_${outputDtype}_${sumDupeIndices}_${strides.length}`;
        const stridesType = getCoordsDataType(strides.length);
        this.uniforms =
            `sliceDim : i32, strides: ${stridesType}, updatesSize: i32,`;
        this.updatesRank = updatesRank;
        this.indicesRank = indicesRank;
    }
    getUserCode() {
        let indicesString = '';
        if (this.indicesRank === 1) {
            indicesString = 'coords[0]';
        }
        else if (this.indicesRank === 2) {
            indicesString = 'coords[0], j';
        }
        const indicesSnippet = `getIndices(${indicesString})`;
        const strideString = this.sliceDimGreaterThanOne ? 'uniforms.strides[j]' :
            'uniforms.strides';
        let outCoordsString = '';
        let getUpdatesCoordsFromFlatIndex = '';
        if (this.dispatchLayout.x.length === 1) {
            outCoordsString = 'flattenedIndex';
            getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> i32 {
        return index;
      }
      `;
        }
        else if (this.dispatchLayout.x.length === 2) {
            outCoordsString = 'vec2<i32>(flattenedIndex, coords[1])';
            getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> vec2<i32> {
        // N.B. |updates| could be a scalar tensor, conceptually representing a
        // 2D tensor with all values equal to that. By design, its size must be
        // the same as |outShape[1]| in one dimension, and |indicesShape[0]|
        // gives the other.
        let sliceSize = uniforms.outShape[1];
        let d0 = index / sliceSize;
        let d1 = index - d0 * sliceSize;
        return vec2<i32>(d0, d1);
      }
      `;
        }
        const updatesString = Array.from({ length: this.updatesRank }, (_, idx) => `coords[${idx}]`);
        const updatesSnippet = `getUpdates(${updatesString.join(', ')})`;
        const userCode = `
    ${getUpdatesCoordsFromFlatIndex}
      ${main('index')} {
        if (index < uniforms.updatesSize) {
          let coords = getUpdatesCoordsFromFlatIndex(index);
          var flattenedIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexInside = i32(round(${indicesSnippet}));
            flattenedIndex = flattenedIndex + indexInside * ${strideString};
          }
          let updateValue =
              ${dataTypeToGPUType(this.type)}(${updatesSnippet});
          let flatIndex = getOutputIndexFromCoords(${outCoordsString});

          ${this.sumDupeIndices ?
            atomicAddSnippet('&result[flatIndex]', 'updateValue', this.type) :
            `atomicStore(&result[flatIndex], bitcast<i32>(updateValue));`}
        }
      }`;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2NhdHRlcl93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zY2F0dGVyX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDL0MsT0FBTyxFQUFDLGlCQUFpQixFQUFFLGlCQUFpQixFQUFFLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUNsSCxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxjQUFjO0lBZXpCLFlBQ0ksYUFBdUIsRUFBRSxRQUFnQixFQUFFLFdBQW1CLEVBQzlELFdBQW1CLEVBQUUsT0FBaUIsRUFBRSxLQUFlLEVBQ3ZELFdBQXFCLEVBQUUsY0FBYyxHQUFHLElBQUk7UUFqQmhELGtCQUFhLEdBQUcsQ0FBQyxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFPdkMsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBSXJELFdBQU0sR0FBRyxJQUFJLENBQUM7UUFPWixJQUFJLENBQUMsV0FBVyxHQUFHLEtBQUssQ0FBQztRQUN6QixJQUFJLENBQUMsSUFBSSxHQUFHLFdBQVcsQ0FBQztRQUN4QixJQUFJLENBQUMsY0FBYyxHQUFHLGNBQWMsQ0FBQztRQUNyQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQ3hELGdFQUFnRTtRQUNoRSxJQUFJLENBQUMsUUFBUTtZQUNULGVBQWUsQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFLGFBQWEsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDNUUsSUFBSSxDQUFDLHNCQUFzQixHQUFHLFFBQVEsR0FBRyxDQUFDLENBQUM7UUFDM0MsSUFBSSxDQUFDLFNBQVM7WUFDVixXQUFXLFdBQVcsSUFBSSxXQUFXLElBQUksSUFBSSxDQUFDLHNCQUFzQixJQUNoRSxXQUFXLElBQUksY0FBYyxJQUFJLE9BQU8sQ0FBQyxNQUFNLEVBQUUsQ0FBQztRQUMxRCxNQUFNLFdBQVcsR0FBRyxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEQsSUFBSSxDQUFDLFFBQVE7WUFDVCw0QkFBNEIsV0FBVyxxQkFBcUIsQ0FBQztRQUNqRSxJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQztRQUMvQixJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQztJQUNqQyxDQUFDO0lBRUQsV0FBVztRQUNULElBQUksYUFBYSxHQUFHLEVBQUUsQ0FBQztRQUN2QixJQUFJLElBQUksQ0FBQyxXQUFXLEtBQUssQ0FBQyxFQUFFO1lBQzFCLGFBQWEsR0FBRyxXQUFXLENBQUM7U0FDN0I7YUFBTSxJQUFJLElBQUksQ0FBQyxXQUFXLEtBQUssQ0FBQyxFQUFFO1lBQ2pDLGFBQWEsR0FBRyxjQUFjLENBQUM7U0FDaEM7UUFDRCxNQUFNLGNBQWMsR0FBRyxjQUFjLGFBQWEsR0FBRyxDQUFDO1FBRXRELE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLENBQUMscUJBQXFCLENBQUMsQ0FBQztZQUN2QixrQkFBa0IsQ0FBQztRQUV0RSxJQUFJLGVBQWUsR0FBRyxFQUFFLENBQUM7UUFDekIsSUFBSSw2QkFBNkIsR0FBRyxFQUFFLENBQUM7UUFDdkMsSUFBSSxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1lBQ3RDLGVBQWUsR0FBRyxnQkFBZ0IsQ0FBQztZQUNuQyw2QkFBNkIsR0FBRzs7OztPQUkvQixDQUFDO1NBQ0g7YUFBTSxJQUFJLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7WUFDN0MsZUFBZSxHQUFHLHNDQUFzQyxDQUFDO1lBQ3pELDZCQUE2QixHQUFHOzs7Ozs7Ozs7OztPQVcvQixDQUFDO1NBQ0g7UUFDRCxNQUFNLGFBQWEsR0FDZixLQUFLLENBQUMsSUFBSSxDQUFDLEVBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsRUFBRSxDQUFDLFVBQVUsR0FBRyxHQUFHLENBQUMsQ0FBQztRQUN6RSxNQUFNLGNBQWMsR0FBRyxjQUFjLGFBQWEsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQztRQUVqRSxNQUFNLFFBQVEsR0FBRztNQUNmLDZCQUE2QjtRQUMzQixJQUFJLENBQUMsT0FBTyxDQUFDOzs7OzswQ0FLcUIsY0FBYzs4REFDTSxZQUFZOzs7Z0JBRzFELGlCQUFpQixDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxjQUFjO3FEQUNULGVBQWU7O1lBRzVELElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQztZQUNqQixnQkFBZ0IsQ0FDWixvQkFBb0IsRUFBRSxhQUFhLEVBQ25DLElBQUksQ0FBQyxJQUEyQixDQUFDLENBQUMsQ0FBQztZQUN2Qyw2REFBNkQ7O1FBRWpFLENBQUM7UUFDTCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RGF0YVR5cGV9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7YXRvbWljQWRkU25pcHBldH0gZnJvbSAnLi9zaGFkZXJfdXRpbCc7XG5pbXBvcnQge2RhdGFUeXBlVG9HUFVUeXBlLCBnZXRDb29yZHNEYXRhVHlwZSwgZ2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgU2NhdHRlclByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsndXBkYXRlcycsICdpbmRpY2VzJ107XG4gIHVuaWZvcm1zOiBzdHJpbmc7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc3VtRHVwZUluZGljZXM6IGJvb2xlYW47XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgdXBkYXRlc1Jhbms6IG51bWJlcjtcbiAgaW5kaWNlc1Jhbms6IG51bWJlcjtcbiAgc2xpY2VEaW1HcmVhdGVyVGhhbk9uZTogYm9vbGVhbjtcbiAgYXRvbWljID0gdHJ1ZTtcbiAgdHlwZTogRGF0YVR5cGU7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBmbGF0dGVuWFNoYXBlOiBudW1iZXJbXSwgc2xpY2VEaW06IG51bWJlciwgaW5kaWNlc1Jhbms6IG51bWJlcixcbiAgICAgIHVwZGF0ZXNSYW5rOiBudW1iZXIsIHN0cmlkZXM6IG51bWJlcltdLCBzaGFwZTogbnVtYmVyW10sXG4gICAgICBvdXRwdXREdHlwZTogRGF0YVR5cGUsIHN1bUR1cGVJbmRpY2VzID0gdHJ1ZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBzaGFwZTtcbiAgICB0aGlzLnR5cGUgPSBvdXRwdXREdHlwZTtcbiAgICB0aGlzLnN1bUR1cGVJbmRpY2VzID0gc3VtRHVwZUluZGljZXM7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dChmbGF0dGVuWFNoYXBlKTtcbiAgICAvLyBEaXNwYXRjaGluZyBiYXNlZCBvbiB8dXBkYXRlc3wgc2hhcGUgaW5zdGVhZCBvZiBvdXRwdXQgc2hhcGUuXG4gICAgdGhpcy5kaXNwYXRjaCA9XG4gICAgICAgIGNvbXB1dGVEaXNwYXRjaCh0aGlzLmRpc3BhdGNoTGF5b3V0LCBmbGF0dGVuWFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuc2xpY2VEaW1HcmVhdGVyVGhhbk9uZSA9IHNsaWNlRGltID4gMTtcbiAgICB0aGlzLnNoYWRlcktleSA9XG4gICAgICAgIGBzY2F0dGVyXyR7aW5kaWNlc1Jhbmt9XyR7dXBkYXRlc1Jhbmt9XyR7dGhpcy5zbGljZURpbUdyZWF0ZXJUaGFuT25lfV8ke1xuICAgICAgICAgICAgb3V0cHV0RHR5cGV9XyR7c3VtRHVwZUluZGljZXN9XyR7c3RyaWRlcy5sZW5ndGh9YDtcbiAgICBjb25zdCBzdHJpZGVzVHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKHN0cmlkZXMubGVuZ3RoKTtcbiAgICB0aGlzLnVuaWZvcm1zID1cbiAgICAgICAgYHNsaWNlRGltIDogaTMyLCBzdHJpZGVzOiAke3N0cmlkZXNUeXBlfSwgdXBkYXRlc1NpemU6IGkzMixgO1xuICAgIHRoaXMudXBkYXRlc1JhbmsgPSB1cGRhdGVzUmFuaztcbiAgICB0aGlzLmluZGljZXNSYW5rID0gaW5kaWNlc1Jhbms7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGxldCBpbmRpY2VzU3RyaW5nID0gJyc7XG4gICAgaWYgKHRoaXMuaW5kaWNlc1JhbmsgPT09IDEpIHtcbiAgICAgIGluZGljZXNTdHJpbmcgPSAnY29vcmRzWzBdJztcbiAgICB9IGVsc2UgaWYgKHRoaXMuaW5kaWNlc1JhbmsgPT09IDIpIHtcbiAgICAgIGluZGljZXNTdHJpbmcgPSAnY29vcmRzWzBdLCBqJztcbiAgICB9XG4gICAgY29uc3QgaW5kaWNlc1NuaXBwZXQgPSBgZ2V0SW5kaWNlcygke2luZGljZXNTdHJpbmd9KWA7XG5cbiAgICBjb25zdCBzdHJpZGVTdHJpbmcgPSB0aGlzLnNsaWNlRGltR3JlYXRlclRoYW5PbmUgPyAndW5pZm9ybXMuc3RyaWRlc1tqXScgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICd1bmlmb3Jtcy5zdHJpZGVzJztcblxuICAgIGxldCBvdXRDb29yZHNTdHJpbmcgPSAnJztcbiAgICBsZXQgZ2V0VXBkYXRlc0Nvb3Jkc0Zyb21GbGF0SW5kZXggPSAnJztcbiAgICBpZiAodGhpcy5kaXNwYXRjaExheW91dC54Lmxlbmd0aCA9PT0gMSkge1xuICAgICAgb3V0Q29vcmRzU3RyaW5nID0gJ2ZsYXR0ZW5lZEluZGV4JztcbiAgICAgIGdldFVwZGF0ZXNDb29yZHNGcm9tRmxhdEluZGV4ID0gYFxuICAgICAgZm4gZ2V0VXBkYXRlc0Nvb3Jkc0Zyb21GbGF0SW5kZXgoaW5kZXggOiBpMzIpIC0+IGkzMiB7XG4gICAgICAgIHJldHVybiBpbmRleDtcbiAgICAgIH1cbiAgICAgIGA7XG4gICAgfSBlbHNlIGlmICh0aGlzLmRpc3BhdGNoTGF5b3V0LngubGVuZ3RoID09PSAyKSB7XG4gICAgICBvdXRDb29yZHNTdHJpbmcgPSAndmVjMjxpMzI+KGZsYXR0ZW5lZEluZGV4LCBjb29yZHNbMV0pJztcbiAgICAgIGdldFVwZGF0ZXNDb29yZHNGcm9tRmxhdEluZGV4ID0gYFxuICAgICAgZm4gZ2V0VXBkYXRlc0Nvb3Jkc0Zyb21GbGF0SW5kZXgoaW5kZXggOiBpMzIpIC0+IHZlYzI8aTMyPiB7XG4gICAgICAgIC8vIE4uQi4gfHVwZGF0ZXN8IGNvdWxkIGJlIGEgc2NhbGFyIHRlbnNvciwgY29uY2VwdHVhbGx5IHJlcHJlc2VudGluZyBhXG4gICAgICAgIC8vIDJEIHRlbnNvciB3aXRoIGFsbCB2YWx1ZXMgZXF1YWwgdG8gdGhhdC4gQnkgZGVzaWduLCBpdHMgc2l6ZSBtdXN0IGJlXG4gICAgICAgIC8vIHRoZSBzYW1lIGFzIHxvdXRTaGFwZVsxXXwgaW4gb25lIGRpbWVuc2lvbiwgYW5kIHxpbmRpY2VzU2hhcGVbMF18XG4gICAgICAgIC8vIGdpdmVzIHRoZSBvdGhlci5cbiAgICAgICAgbGV0IHNsaWNlU2l6ZSA9IHVuaWZvcm1zLm91dFNoYXBlWzFdO1xuICAgICAgICBsZXQgZDAgPSBpbmRleCAvIHNsaWNlU2l6ZTtcbiAgICAgICAgbGV0IGQxID0gaW5kZXggLSBkMCAqIHNsaWNlU2l6ZTtcbiAgICAgICAgcmV0dXJuIHZlYzI8aTMyPihkMCwgZDEpO1xuICAgICAgfVxuICAgICAgYDtcbiAgICB9XG4gICAgY29uc3QgdXBkYXRlc1N0cmluZyA9XG4gICAgICAgIEFycmF5LmZyb20oe2xlbmd0aDogdGhpcy51cGRhdGVzUmFua30sIChfLCBpZHgpID0+IGBjb29yZHNbJHtpZHh9XWApO1xuICAgIGNvbnN0IHVwZGF0ZXNTbmlwcGV0ID0gYGdldFVwZGF0ZXMoJHt1cGRhdGVzU3RyaW5nLmpvaW4oJywgJyl9KWA7XG5cbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAke2dldFVwZGF0ZXNDb29yZHNGcm9tRmxhdEluZGV4fVxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnVwZGF0ZXNTaXplKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IGdldFVwZGF0ZXNDb29yZHNGcm9tRmxhdEluZGV4KGluZGV4KTtcbiAgICAgICAgICB2YXIgZmxhdHRlbmVkSW5kZXggPSAwO1xuICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgdW5pZm9ybXMuc2xpY2VEaW07IGogPSBqICsgMSkge1xuICAgICAgICAgICAgbGV0IGluZGV4SW5zaWRlID0gaTMyKHJvdW5kKCR7aW5kaWNlc1NuaXBwZXR9KSk7XG4gICAgICAgICAgICBmbGF0dGVuZWRJbmRleCA9IGZsYXR0ZW5lZEluZGV4ICsgaW5kZXhJbnNpZGUgKiAke3N0cmlkZVN0cmluZ307XG4gICAgICAgICAgfVxuICAgICAgICAgIGxldCB1cGRhdGVWYWx1ZSA9XG4gICAgICAgICAgICAgICR7ZGF0YVR5cGVUb0dQVVR5cGUodGhpcy50eXBlKX0oJHt1cGRhdGVzU25pcHBldH0pO1xuICAgICAgICAgIGxldCBmbGF0SW5kZXggPSBnZXRPdXRwdXRJbmRleEZyb21Db29yZHMoJHtvdXRDb29yZHNTdHJpbmd9KTtcblxuICAgICAgICAgICR7XG4gICAgICAgIHRoaXMuc3VtRHVwZUluZGljZXMgP1xuICAgICAgICAgICAgYXRvbWljQWRkU25pcHBldChcbiAgICAgICAgICAgICAgICAnJnJlc3VsdFtmbGF0SW5kZXhdJywgJ3VwZGF0ZVZhbHVlJyxcbiAgICAgICAgICAgICAgICB0aGlzLnR5cGUgYXMgJ2Zsb2F0MzInIHwgJ2ludDMyJykgOlxuICAgICAgICAgICAgYGF0b21pY1N0b3JlKCZyZXN1bHRbZmxhdEluZGV4XSwgYml0Y2FzdDxpMzI+KHVwZGF0ZVZhbHVlKSk7YH1cbiAgICAgICAgfVxuICAgICAgfWA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=