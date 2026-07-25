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
import { getCoordsDataType, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class StridedSliceProgram {
    constructor(destSize) {
        this.variableNames = ['x'];
        // TODO(xing.xu): Increase the workPerThread.
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = destSize;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        const dtype = getCoordsDataType(this.outputShape.length);
        this.uniforms = `begin : ${dtype},  strides : ${dtype}, `;
        this.shaderKey = 'stridedSlice';
    }
    getUserCode() {
        const rank = this.outputShape.length;
        let newCoords = '';
        if (rank === 1) {
            newCoords = 'coords * uniforms.strides + uniforms.begin';
        }
        else {
            let outputAxis = 0;
            newCoords =
                this.outputShape
                    .map((_, i) => {
                    outputAxis++;
                    return this.outputShape.length === 1 ?
                        `coords * uniforms.strides[${i}] + uniforms.begin[${i}]` :
                        `coords[${outputAxis - 1}] * uniforms.strides[${i}] + uniforms.begin[${i}]`;
                })
                    .join(',');
        }
        const userCode = `
       ${main('index')} {
         if (index < uniforms.size) {
           let coords = getCoordsFromIndex(index);
           setOutputAtIndex(index, getX(${newCoords}));
         }
       }
     `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RyaWRlZF9zbGljZV93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zdHJpZGVkX3NsaWNlX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsaUJBQWlCLEVBQUUsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQy9GLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLG1CQUFtQjtJQVk5QixZQUFZLFFBQWtCO1FBWDlCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQU10Qiw2Q0FBNkM7UUFDN0Msa0JBQWEsR0FBRyxDQUFDLENBQUM7UUFDbEIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQztRQUM1QixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLEVBQ3pELENBQUMsSUFBSSxDQUFDLGFBQWEsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVoQyxNQUFNLEtBQUssR0FBRyxpQkFBaUIsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3pELElBQUksQ0FBQyxRQUFRLEdBQUcsV0FBVyxLQUFLLGdCQUFnQixLQUFLLElBQUksQ0FBQztRQUMxRCxJQUFJLENBQUMsU0FBUyxHQUFHLGNBQWMsQ0FBQztJQUNsQyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDO1FBQ3JDLElBQUksU0FBUyxHQUFHLEVBQUUsQ0FBQztRQUNuQixJQUFJLElBQUksS0FBSyxDQUFDLEVBQUU7WUFDZCxTQUFTLEdBQUcsNENBQTRDLENBQUM7U0FDMUQ7YUFBTTtZQUNMLElBQUksVUFBVSxHQUFHLENBQUMsQ0FBQztZQUNuQixTQUFTO2dCQUNMLElBQUksQ0FBQyxXQUFXO3FCQUNYLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtvQkFDWixVQUFVLEVBQUUsQ0FBQztvQkFDYixPQUFPLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxLQUFLLENBQUMsQ0FBQyxDQUFDO3dCQUNsQyw2QkFBNkIsQ0FBQyxzQkFBc0IsQ0FBQyxHQUFHLENBQUMsQ0FBQzt3QkFDMUQsVUFBVSxVQUFVLEdBQUcsQ0FBQyx3QkFDcEIsQ0FBQyxzQkFBc0IsQ0FBQyxHQUFHLENBQUM7Z0JBQ3RDLENBQUMsQ0FBQztxQkFDRCxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7U0FDcEI7UUFFRCxNQUFNLFFBQVEsR0FBRztTQUNaLElBQUksQ0FBQyxPQUFPLENBQUM7OzswQ0FHb0IsU0FBUzs7O01BRzdDLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0Q29vcmRzRGF0YVR5cGUsIGdldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFN0cmlkZWRTbGljZVByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCddO1xuICB1bmlmb3Jtczogc3RyaW5nO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgLy8gVE9ETyh4aW5nLnh1KTogSW5jcmVhc2UgdGhlIHdvcmtQZXJUaHJlYWQuXG4gIHdvcmtQZXJUaHJlYWQgPSAxO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihkZXN0U2l6ZTogbnVtYmVyW10pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gZGVzdFNpemU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIFt0aGlzLndvcmtQZXJUaHJlYWQsIDEsIDFdKTtcblxuICAgIGNvbnN0IGR0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUodGhpcy5vdXRwdXRTaGFwZS5sZW5ndGgpO1xuICAgIHRoaXMudW5pZm9ybXMgPSBgYmVnaW4gOiAke2R0eXBlfSwgIHN0cmlkZXMgOiAke2R0eXBlfSwgYDtcbiAgICB0aGlzLnNoYWRlcktleSA9ICdzdHJpZGVkU2xpY2UnO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCByYW5rID0gdGhpcy5vdXRwdXRTaGFwZS5sZW5ndGg7XG4gICAgbGV0IG5ld0Nvb3JkcyA9ICcnO1xuICAgIGlmIChyYW5rID09PSAxKSB7XG4gICAgICBuZXdDb29yZHMgPSAnY29vcmRzICogdW5pZm9ybXMuc3RyaWRlcyArIHVuaWZvcm1zLmJlZ2luJztcbiAgICB9IGVsc2Uge1xuICAgICAgbGV0IG91dHB1dEF4aXMgPSAwO1xuICAgICAgbmV3Q29vcmRzID1cbiAgICAgICAgICB0aGlzLm91dHB1dFNoYXBlXG4gICAgICAgICAgICAgIC5tYXAoKF8sIGkpID0+IHtcbiAgICAgICAgICAgICAgICBvdXRwdXRBeGlzKys7XG4gICAgICAgICAgICAgICAgcmV0dXJuIHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoID09PSAxID9cbiAgICAgICAgICAgICAgICAgICAgYGNvb3JkcyAqIHVuaWZvcm1zLnN0cmlkZXNbJHtpfV0gKyB1bmlmb3Jtcy5iZWdpblske2l9XWAgOlxuICAgICAgICAgICAgICAgICAgICBgY29vcmRzWyR7b3V0cHV0QXhpcyAtIDF9XSAqIHVuaWZvcm1zLnN0cmlkZXNbJHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGl9XSArIHVuaWZvcm1zLmJlZ2luWyR7aX1dYDtcbiAgICAgICAgICAgICAgfSlcbiAgICAgICAgICAgICAgLmpvaW4oJywnKTtcbiAgICB9XG5cbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGdldFgoJHtuZXdDb29yZHN9KSk7XG4gICAgICAgICB9XG4gICAgICAgfVxuICAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19