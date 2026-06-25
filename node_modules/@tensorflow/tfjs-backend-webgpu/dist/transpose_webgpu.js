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
export class TransposeProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.newDim = newDim;
        this.shaderKey = `transpose_${newDim}`;
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      ${main('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${this.outputShape.length}D(
              ${dtype}(${switched}), uniforms.aShape)]);
          }
        }
      }
    `;
        return userCode;
    }
}
export function getSwitchedCoords(newDim) {
    const rank = newDim.length;
    if (rank > 6) {
        throw Error(`Transpose for rank ${rank} is not yet supported`);
    }
    const switchedCoords = new Array(rank);
    for (let i = 0; i < newDim.length; i++) {
        switchedCoords[newDim[i]] = `coords.${getCoordsXYZ(i)}`;
    }
    return switchedCoords.join();
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNwb3NlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3RyYW5zcG9zZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLGlCQUFpQixFQUFFLFlBQVksRUFBRSxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDN0csT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sZ0JBQWdCO0lBVzNCLFlBQVksTUFBZ0IsRUFBRSxNQUFnQjtRQVY5QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFLdEIsa0JBQWEsR0FBRyxDQUFDLENBQUM7UUFDbEIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBRXJELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixNQUFNLFdBQVcsR0FBYSxJQUFJLEtBQUssQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdkQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDM0MsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNwQztRQUNELElBQUksQ0FBQyxXQUFXLEdBQUcsV0FBVyxDQUFDO1FBQy9CLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsRUFDekQsQ0FBQyxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRWhDLElBQUksQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO1FBQ3JCLElBQUksQ0FBQyxTQUFTLEdBQUcsYUFBYSxNQUFNLEVBQUUsQ0FBQztJQUN6QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sS0FBSyxHQUFHLGlCQUFpQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDekQsTUFBTSxRQUFRLEdBQUcsaUJBQWlCLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRWhELE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs2QkFDUSxJQUFJLENBQUMsYUFBYTtvQ0FDWCxJQUFJLENBQUMsYUFBYTs7OzhEQUk5QyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU07Z0JBQ2YsS0FBSyxJQUFJLFFBQVE7Ozs7S0FJNUIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRjtBQUVELE1BQU0sVUFBVSxpQkFBaUIsQ0FBQyxNQUFnQjtJQUNoRCxNQUFNLElBQUksR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO0lBQzNCLElBQUksSUFBSSxHQUFHLENBQUMsRUFBRTtRQUNaLE1BQU0sS0FBSyxDQUFDLHNCQUFzQixJQUFJLHVCQUF1QixDQUFDLENBQUM7S0FDaEU7SUFDRCxNQUFNLGNBQWMsR0FBRyxJQUFJLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN2QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUN0QyxjQUFjLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsVUFBVSxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztLQUN6RDtJQUVELE9BQU8sY0FBYyxDQUFDLElBQUksRUFBRSxDQUFDO0FBQy9CLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0Q29vcmRzRGF0YVR5cGUsIGdldENvb3Jkc1hZWiwgZ2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgVHJhbnNwb3NlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB3b3JrUGVyVGhyZWFkID0gMTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgbmV3RGltOiBudW1iZXJbXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoYVNoYXBlOiBudW1iZXJbXSwgbmV3RGltOiBudW1iZXJbXSkge1xuICAgIGNvbnN0IG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IG5ldyBBcnJheShhU2hhcGUubGVuZ3RoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dFNoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgICBvdXRwdXRTaGFwZVtpXSA9IGFTaGFwZVtuZXdEaW1baV1dO1xuICAgIH1cbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIFt0aGlzLndvcmtQZXJUaHJlYWQsIDEsIDFdKTtcblxuICAgIHRoaXMubmV3RGltID0gbmV3RGltO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYHRyYW5zcG9zZV8ke25ld0RpbX1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCBkdHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoKTtcbiAgICBjb25zdCBzd2l0Y2hlZCA9IGdldFN3aXRjaGVkQ29vcmRzKHRoaXMubmV3RGltKTtcblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGZvcih2YXIgaSA9IDA7IGkgPCAke3RoaXMud29ya1BlclRocmVhZH07IGkgPSBpICsgMSkge1xuICAgICAgICAgIGxldCBmbGF0SW5kZXggPSBpbmRleCAqICR7dGhpcy53b3JrUGVyVGhyZWFkfSArIGk7XG4gICAgICAgICAgaWYoZmxhdEluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChmbGF0SW5kZXgpO1xuICAgICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChmbGF0SW5kZXgsIEFbZ2V0SW5kZXhGcm9tQ29vcmRzJHtcbiAgICAgICAgdGhpcy5vdXRwdXRTaGFwZS5sZW5ndGh9RChcbiAgICAgICAgICAgICAgJHtkdHlwZX0oJHtzd2l0Y2hlZH0pLCB1bmlmb3Jtcy5hU2hhcGUpXSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFN3aXRjaGVkQ29vcmRzKG5ld0RpbTogbnVtYmVyW10pOiBzdHJpbmcge1xuICBjb25zdCByYW5rID0gbmV3RGltLmxlbmd0aDtcbiAgaWYgKHJhbmsgPiA2KSB7XG4gICAgdGhyb3cgRXJyb3IoYFRyYW5zcG9zZSBmb3IgcmFuayAke3Jhbmt9IGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbiAgY29uc3Qgc3dpdGNoZWRDb29yZHMgPSBuZXcgQXJyYXkocmFuayk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3RGltLmxlbmd0aDsgaSsrKSB7XG4gICAgc3dpdGNoZWRDb29yZHNbbmV3RGltW2ldXSA9IGBjb29yZHMuJHtnZXRDb29yZHNYWVooaSl9YDtcbiAgfVxuXG4gIHJldHVybiBzd2l0Y2hlZENvb3Jkcy5qb2luKCk7XG59XG4iXX0=