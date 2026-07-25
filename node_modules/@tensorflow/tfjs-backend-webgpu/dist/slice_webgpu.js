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
import { getCoordsDataType, getCoordsXYZ, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class SliceProgram {
    constructor(start, destSize) {
        this.variableNames = ['source'];
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = destSize;
        this.rank = destSize.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.start = start;
        this.uniforms = `start : ${getCoordsDataType(start.length)}, `;
        this.shaderKey = 'slice';
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.rank);
        const sourceCoords = getCoords(this.rank);
        let coordSum;
        if (this.start.length === 1) {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc = uniforms.start + coords;`;
            });
        }
        else {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = uniforms.start.${getCoordsXYZ(i)} + coords.${coords[i]};`;
            });
        }
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          var sourceLoc : ${dtype};
          let coords = getCoordsFromIndex(index);
          ${coordSum.join('\n')}
          setOutputAtIndex(index, getSource(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
const coords = ['x', 'y', 'z', 'w', 'u', 'v'];
function getCoords(rank) {
    if (rank === 1) {
        return 'sourceLoc';
    }
    else if (rank <= 6) {
        return coords.slice(0, rank).map(coord => `sourceLoc.${coord}`).join(',');
    }
    else {
        throw Error(`Slicing for rank ${rank} is not yet supported`);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2xpY2Vfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvc2xpY2Vfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxpQkFBaUIsRUFBRSxZQUFZLEVBQUUsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzdHLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLFlBQVk7SUFhdkIsWUFBWSxLQUFlLEVBQUUsUUFBa0I7UUFaL0Msa0JBQWEsR0FBRyxDQUFDLFFBQVEsQ0FBQyxDQUFDO1FBTzNCLGtCQUFhLEdBQUcsQ0FBQyxDQUFDO1FBQ2xCLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUM7UUFDNUIsSUFBSSxDQUFDLElBQUksR0FBRyxRQUFRLENBQUMsTUFBTSxDQUFDO1FBQzVCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsRUFDekQsQ0FBQyxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRWhDLElBQUksQ0FBQyxLQUFLLEdBQUcsS0FBSyxDQUFDO1FBQ25CLElBQUksQ0FBQyxRQUFRLEdBQUcsV0FBVyxpQkFBaUIsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQztRQUMvRCxJQUFJLENBQUMsU0FBUyxHQUFHLE9BQU8sQ0FBQztJQUMzQixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sS0FBSyxHQUFHLGlCQUFpQixDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUMzQyxNQUFNLFlBQVksR0FBRyxTQUFTLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzFDLElBQUksUUFBUSxDQUFDO1FBQ2IsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7WUFDM0IsUUFBUSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFO2dCQUN2QyxPQUFPLHNDQUFzQyxDQUFDO1lBQ2hELENBQUMsQ0FBQyxDQUFDO1NBQ0o7YUFBTTtZQUNMLFFBQVEsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtnQkFDdkMsT0FBTyxhQUFhLE1BQU0sQ0FBQyxDQUFDLENBQUMscUJBQ3pCLFlBQVksQ0FBQyxDQUFDLENBQUMsYUFBYSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQztZQUMvQyxDQUFDLENBQUMsQ0FBQztTQUNKO1FBRUQsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzs0QkFFTyxLQUFLOztZQUVyQixRQUFRLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQzs4Q0FDZSxZQUFZOzs7S0FHckQsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELE1BQU0sTUFBTSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztBQUU5QyxTQUFTLFNBQVMsQ0FBQyxJQUFZO0lBQzdCLElBQUksSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNkLE9BQU8sV0FBVyxDQUFDO0tBQ3BCO1NBQU0sSUFBSSxJQUFJLElBQUksQ0FBQyxFQUFFO1FBQ3BCLE9BQU8sTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsYUFBYSxLQUFLLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztLQUMzRTtTQUFNO1FBQ0wsTUFBTSxLQUFLLENBQUMsb0JBQW9CLElBQUksdUJBQXVCLENBQUMsQ0FBQztLQUM5RDtBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0Q29vcmRzRGF0YVR5cGUsIGdldENvb3Jkc1hZWiwgZ2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgU2xpY2VQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3NvdXJjZSddO1xuICB1bmlmb3Jtczogc3RyaW5nO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICByYW5rOiBudW1iZXI7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB3b3JrUGVyVGhyZWFkID0gMTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc3RhcnQ6IG51bWJlcltdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihzdGFydDogbnVtYmVyW10sIGRlc3RTaXplOiBudW1iZXJbXSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBkZXN0U2l6ZTtcbiAgICB0aGlzLnJhbmsgPSBkZXN0U2l6ZS5sZW5ndGg7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIFt0aGlzLndvcmtQZXJUaHJlYWQsIDEsIDFdKTtcblxuICAgIHRoaXMuc3RhcnQgPSBzdGFydDtcbiAgICB0aGlzLnVuaWZvcm1zID0gYHN0YXJ0IDogJHtnZXRDb29yZHNEYXRhVHlwZShzdGFydC5sZW5ndGgpfSwgYDtcbiAgICB0aGlzLnNoYWRlcktleSA9ICdzbGljZSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IGR0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUodGhpcy5yYW5rKTtcbiAgICBjb25zdCBzb3VyY2VDb29yZHMgPSBnZXRDb29yZHModGhpcy5yYW5rKTtcbiAgICBsZXQgY29vcmRTdW07XG4gICAgaWYgKHRoaXMuc3RhcnQubGVuZ3RoID09PSAxKSB7XG4gICAgICBjb29yZFN1bSA9IHRoaXMub3V0cHV0U2hhcGUubWFwKChfLCBpKSA9PiB7XG4gICAgICAgIHJldHVybiBgc291cmNlTG9jID0gdW5pZm9ybXMuc3RhcnQgKyBjb29yZHM7YDtcbiAgICAgIH0pO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb29yZFN1bSA9IHRoaXMub3V0cHV0U2hhcGUubWFwKChfLCBpKSA9PiB7XG4gICAgICAgIHJldHVybiBgc291cmNlTG9jLiR7Y29vcmRzW2ldfSA9IHVuaWZvcm1zLnN0YXJ0LiR7XG4gICAgICAgICAgICBnZXRDb29yZHNYWVooaSl9ICsgY29vcmRzLiR7Y29vcmRzW2ldfTtgO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIHZhciBzb3VyY2VMb2MgOiAke2R0eXBlfTtcbiAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgICAke2Nvb3JkU3VtLmpvaW4oJ1xcbicpfVxuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGdldFNvdXJjZSgke3NvdXJjZUNvb3Jkc30pKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG5cbmNvbnN0IGNvb3JkcyA9IFsneCcsICd5JywgJ3onLCAndycsICd1JywgJ3YnXTtcblxuZnVuY3Rpb24gZ2V0Q29vcmRzKHJhbms6IG51bWJlcik6IHN0cmluZyB7XG4gIGlmIChyYW5rID09PSAxKSB7XG4gICAgcmV0dXJuICdzb3VyY2VMb2MnO1xuICB9IGVsc2UgaWYgKHJhbmsgPD0gNikge1xuICAgIHJldHVybiBjb29yZHMuc2xpY2UoMCwgcmFuaykubWFwKGNvb3JkID0+IGBzb3VyY2VMb2MuJHtjb29yZH1gKS5qb2luKCcsJyk7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgRXJyb3IoYFNsaWNpbmcgZm9yIHJhbmsgJHtyYW5rfSBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG59XG4iXX0=