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
import { getCoordsDataType, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class GatherNDProgram {
    constructor(sliceDim, shape) {
        this.variableNames = ['A', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `gathernd_${sliceDim}`;
        this.sliceDim = sliceDim;
        this.uniforms = `sliceDim : i32, strides : ${getCoordsDataType(sliceDim)},`;
    }
    getUserCode() {
        let strideString;
        if (this.sliceDim > 1) {
            strideString = 'uniforms.strides[j]';
        }
        else {
            strideString = 'uniforms.strides';
        }
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          var flattenIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexTemp = i32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }

          setOutputAtIndex(index, getA(flattenIndex, coords[1]));
        }
      }
      `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2F0aGVyX25kX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2dhdGhlcl9uZF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLGlCQUFpQixFQUFFLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUMvRixPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxlQUFlO0lBVTFCLFlBQVksUUFBZ0IsRUFBRSxLQUFlO1FBTDdDLGtCQUFhLEdBQWEsQ0FBQyxHQUFHLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFFM0Msa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLEtBQUssQ0FBQztRQUN6QixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsU0FBUyxHQUFHLFlBQVksUUFBUSxFQUFFLENBQUM7UUFDeEMsSUFBSSxDQUFDLFFBQVEsR0FBRyxRQUFRLENBQUM7UUFDekIsSUFBSSxDQUFDLFFBQVEsR0FBRyw2QkFBNkIsaUJBQWlCLENBQUMsUUFBUSxDQUFDLEdBQUcsQ0FBQztJQUM5RSxDQUFDO0lBRUQsV0FBVztRQUNULElBQUksWUFBWSxDQUFDO1FBQ2pCLElBQUksSUFBSSxDQUFDLFFBQVEsR0FBRyxDQUFDLEVBQUU7WUFDckIsWUFBWSxHQUFHLHFCQUFxQixDQUFDO1NBQ3RDO2FBQU07WUFDTCxZQUFZLEdBQUcsa0JBQWtCLENBQUM7U0FDbkM7UUFDRCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs4QkFNUyxZQUFZOzs7Ozs7O09BT25DLENBQUM7UUFDSixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0Q29vcmRzRGF0YVR5cGUsIGdldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIEdhdGhlck5EUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW10gPSBbJ0EnLCAnaW5kaWNlcyddO1xuICB1bmlmb3Jtczogc3RyaW5nO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcbiAgc2xpY2VEaW06IG51bWJlcjtcbiAgY29uc3RydWN0b3Ioc2xpY2VEaW06IG51bWJlciwgc2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IHNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBnYXRoZXJuZF8ke3NsaWNlRGltfWA7XG4gICAgdGhpcy5zbGljZURpbSA9IHNsaWNlRGltO1xuICAgIHRoaXMudW5pZm9ybXMgPSBgc2xpY2VEaW0gOiBpMzIsIHN0cmlkZXMgOiAke2dldENvb3Jkc0RhdGFUeXBlKHNsaWNlRGltKX0sYDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgbGV0IHN0cmlkZVN0cmluZztcbiAgICBpZiAodGhpcy5zbGljZURpbSA+IDEpIHtcbiAgICAgIHN0cmlkZVN0cmluZyA9ICd1bmlmb3Jtcy5zdHJpZGVzW2pdJztcbiAgICB9IGVsc2Uge1xuICAgICAgc3RyaWRlU3RyaW5nID0gJ3VuaWZvcm1zLnN0cmlkZXMnO1xuICAgIH1cbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgdmFyIGZsYXR0ZW5JbmRleCA9IDA7XG4gICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCB1bmlmb3Jtcy5zbGljZURpbTsgaiA9IGogKyAxKSB7XG4gICAgICAgICAgICBsZXQgaW5kZXhUZW1wID0gaTMyKHJvdW5kKGdldEluZGljZXMoY29vcmRzWzBdLCBqKSkpO1xuICAgICAgICAgICAgbGV0IHN0cmlkZU51bSA9ICR7c3RyaWRlU3RyaW5nfTtcbiAgICAgICAgICAgIGZsYXR0ZW5JbmRleCA9IGZsYXR0ZW5JbmRleCArIGluZGV4VGVtcCAqIHN0cmlkZU51bTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBnZXRBKGZsYXR0ZW5JbmRleCwgY29vcmRzWzFdKSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=