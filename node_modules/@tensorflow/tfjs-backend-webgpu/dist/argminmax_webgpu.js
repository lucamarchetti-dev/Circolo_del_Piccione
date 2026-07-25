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
import { backend_util, util } from '@tensorflow/tfjs-core';
import { getCoordsXYZ, getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ArgMinMaxProgram {
    constructor(inputShape, axis, reduceType) {
        this.workgroupSize = [64, 1, 1];
        this.variableNames = ['x'];
        this.uniforms = 'infinityValue : f32,';
        this.size = true;
        const axes = [axis];
        this.op = reduceType === 'min' ? '<' : '>';
        // |outShape| is the shape with the removed axis
        const [outputShape, reduceShape] = backend_util.computeOutAndReduceShapes(inputShape, axes);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        // The shared algorithm is mainly used for large reduce size. It fully
        // utilizes the threads in one workgroup to do the reduction. However,
        // when the reduce size is very small, it's better to use the plain
        // algorithm to reduce the number of workgroups to speedup. The threthold
        // can be further tuned.
        if (util.sizeFromShape(reduceShape) < 32) {
            this.type = 'plain';
            this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        }
        else {
            this.type = 'shared';
            // A work group only outputs a data, so we transfer [1, 1, 1] to compute
            // dispatch size.
            this.dispatch =
                computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);
        }
        this.inputShape = inputShape;
        this.shaderKey = `argMinMax_${this.op}_${this.type}`;
    }
    getUserCode() {
        const workgroupSizeX = this.workgroupSize[0];
        const getInputShapeLastDim = () => {
            if (this.inputShape.length === 1) {
                return 'uniforms.xShape';
            }
            else {
                return `uniforms.xShape.${getCoordsXYZ(this.inputShape.length - 1)}`;
            }
        };
        const splitOutputCoords = () => {
            let snippet = '';
            if (this.outputShape.length === 1) {
                if (this.inputShape.length !== 1) {
                    snippet += 'outputCoords,';
                }
            }
            else {
                for (let i = 0; i < this.outputShape.length; i++) {
                    snippet += `outputCoords.${getCoordsXYZ(i)},`;
                }
            }
            return snippet;
        };
        if (this.type === 'shared') {
            const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<i32, ${workgroupSizeX}>;
      var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
    `;
            const userCode = `
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      ${sharedMemorySnippet}

      ${main('index')} {
        let outputIndex = index / ${workgroupSizeX};
        let reduceLength = ${getInputShapeLastDim()};

        var bestIndex = i32(localId.x);
        var bestValue = uniforms.infinityValue;
        let outputCoords = getCoordsFromIndex(outputIndex);
        for (var k = i32(localId.x); k < reduceLength && outputIndex < uniforms.size;
            k = k + ${workgroupSizeX}) {
          let candidate = getX(${splitOutputCoords()} k);
          if (!isnan(candidate) && candidate ${this.op} bestValue) {
            bestValue = candidate;
            bestIndex = k;
          }
        }
        xBestValues[localId.x] = bestValue;
        xBestIndices[localId.x] = bestIndex;
        workgroupBarrier();

        var reduceSize = min(u32(reduceLength), ${workgroupSizeX}u);
        for (var currentSize = reduceSize / 2u; reduceSize > 1u;
            currentSize = reduceSize / 2u) {
          let interval = DIV_CEIL(reduceSize, 2u);
          if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              xBestValues[localId.x] = bestValue;
              xBestIndices[localId.x] = xBestIndices[localId.x + interval];
            }
          }
          reduceSize = interval;
          workgroupBarrier();
        }

        if (localId.x == 0u && outputIndex < uniforms.size) {
          setOutputAtIndexI32(outputIndex, xBestIndices[localId.x]);
        }
      }
    `;
            return userCode;
        }
        else {
            const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let outputCoords = getCoordsFromIndex(index);
          var bestIndex = 0;
          var bestValue = getX(${splitOutputCoords()} 0);
          let reduceLength = ${getInputShapeLastDim()};
          for (var i = 1; i < reduceLength; i++) {
            let candidate = getX(${splitOutputCoords()} i);
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
          setOutputAtIndexI32(index, bestIndex);
        }
      }
      `;
            return userCode;
        }
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXJnbWlubWF4X3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2FyZ21pbm1heF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUN6RCxPQUFPLEVBQUMsWUFBWSxFQUFFLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUMxRixPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxnQkFBZ0I7SUFjM0IsWUFBWSxVQUFvQixFQUFFLElBQVksRUFBRSxVQUF1QjtRQVR2RSxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsa0JBQWEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3RCLGFBQVEsR0FBRyxzQkFBc0IsQ0FBQztRQUlsQyxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBSVYsTUFBTSxJQUFJLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUVwQixJQUFJLENBQUMsRUFBRSxHQUFHLFVBQVUsS0FBSyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO1FBRTNDLGdEQUFnRDtRQUNoRCxNQUFNLENBQUMsV0FBVyxFQUFFLFdBQVcsQ0FBQyxHQUM1QixZQUFZLENBQUMseUJBQXlCLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBRTdELElBQUksQ0FBQyxXQUFXLEdBQUcsV0FBVyxDQUFDLE1BQU0sS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLFdBQVcsQ0FBQztRQUNoRSxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxzRUFBc0U7UUFDdEUsc0VBQXNFO1FBQ3RFLG1FQUFtRTtRQUNuRSx5RUFBeUU7UUFDekUsd0JBQXdCO1FBQ3hCLElBQUksSUFBSSxDQUFDLGFBQWEsQ0FBQyxXQUFXLENBQUMsR0FBRyxFQUFFLEVBQUU7WUFDeEMsSUFBSSxDQUFDLElBQUksR0FBRyxPQUFPLENBQUM7WUFDcEIsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7U0FDaEU7YUFBTTtZQUNMLElBQUksQ0FBQyxJQUFJLEdBQUcsUUFBUSxDQUFDO1lBQ3JCLHdFQUF3RTtZQUN4RSxpQkFBaUI7WUFDakIsSUFBSSxDQUFDLFFBQVE7Z0JBQ1QsZUFBZSxDQUFDLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUN2RTtRQUVELElBQUksQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDO1FBQzdCLElBQUksQ0FBQyxTQUFTLEdBQUcsYUFBYSxJQUFJLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQztJQUN2RCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sY0FBYyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0MsTUFBTSxvQkFBb0IsR0FBRyxHQUFHLEVBQUU7WUFDaEMsSUFBSSxJQUFJLENBQUMsVUFBVSxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7Z0JBQ2hDLE9BQU8saUJBQWlCLENBQUM7YUFDMUI7aUJBQU07Z0JBQ0wsT0FBTyxtQkFBbUIsWUFBWSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUM7YUFDdEU7UUFDSCxDQUFDLENBQUM7UUFFRixNQUFNLGlCQUFpQixHQUFHLEdBQUcsRUFBRTtZQUM3QixJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUM7WUFDakIsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7Z0JBQ2pDLElBQUksSUFBSSxDQUFDLFVBQVUsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO29CQUNoQyxPQUFPLElBQUksZUFBZSxDQUFDO2lCQUM1QjthQUNGO2lCQUFNO2dCQUNMLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtvQkFDaEQsT0FBTyxJQUFJLGdCQUFnQixZQUFZLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQztpQkFDL0M7YUFDRjtZQUNELE9BQU8sT0FBTyxDQUFDO1FBQ2pCLENBQUMsQ0FBQztRQUVGLElBQUksSUFBSSxDQUFDLElBQUksS0FBSyxRQUFRLEVBQUU7WUFDMUIsTUFBTSxtQkFBbUIsR0FBRztpREFDZSxjQUFjO2dEQUNmLGNBQWM7S0FDekQsQ0FBQztZQUNBLE1BQU0sUUFBUSxHQUFHOzs7OztRQUtmLG1CQUFtQjs7UUFFbkIsSUFBSSxDQUFDLE9BQU8sQ0FBQztvQ0FDZSxjQUFjOzZCQUNyQixvQkFBb0IsRUFBRTs7Ozs7O3NCQU03QixjQUFjO2lDQUNILGlCQUFpQixFQUFFOytDQUNMLElBQUksQ0FBQyxFQUFFOzs7Ozs7Ozs7a0RBU0osY0FBYzs7Ozs7OzRCQU1wQyxJQUFJLENBQUMsRUFBRTs7Ozs7Ozs7Ozs7Ozs7S0FjOUIsQ0FBQztZQUNBLE9BQU8sUUFBUSxDQUFDO1NBQ2pCO2FBQU07WUFDTCxNQUFNLFFBQVEsR0FBRztRQUNmLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7aUNBSVksaUJBQWlCLEVBQUU7K0JBQ3JCLG9CQUFvQixFQUFFOzttQ0FFbEIsaUJBQWlCLEVBQUU7NEJBQzFCLElBQUksQ0FBQyxFQUFFOzs7Ozs7OztPQVE1QixDQUFDO1lBQ0YsT0FBTyxRQUFRLENBQUM7U0FDakI7SUFDSCxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRDb29yZHNYWVosIGdldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIEFyZ01pbk1heFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgdW5pZm9ybXMgPSAnaW5maW5pdHlWYWx1ZSA6IGYzMiwnO1xuICBpbnB1dFNoYXBlOiBudW1iZXJbXTtcbiAgcmVkdWN0aW9uRmFjdG9yOiBudW1iZXI7XG4gIG9wOiBzdHJpbmc7XG4gIHNpemUgPSB0cnVlO1xuICBwcml2YXRlIHR5cGU6IHN0cmluZztcblxuICBjb25zdHJ1Y3RvcihpbnB1dFNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLCByZWR1Y2VUeXBlOiAnbWluJ3wnbWF4Jykge1xuICAgIGNvbnN0IGF4ZXMgPSBbYXhpc107XG5cbiAgICB0aGlzLm9wID0gcmVkdWNlVHlwZSA9PT0gJ21pbicgPyAnPCcgOiAnPic7XG5cbiAgICAvLyB8b3V0U2hhcGV8IGlzIHRoZSBzaGFwZSB3aXRoIHRoZSByZW1vdmVkIGF4aXNcbiAgICBjb25zdCBbb3V0cHV0U2hhcGUsIHJlZHVjZVNoYXBlXSA9XG4gICAgICAgIGJhY2tlbmRfdXRpbC5jb21wdXRlT3V0QW5kUmVkdWNlU2hhcGVzKGlucHV0U2hhcGUsIGF4ZXMpO1xuXG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlLmxlbmd0aCA9PT0gMCA/IFsxXSA6IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgLy8gVGhlIHNoYXJlZCBhbGdvcml0aG0gaXMgbWFpbmx5IHVzZWQgZm9yIGxhcmdlIHJlZHVjZSBzaXplLiBJdCBmdWxseVxuICAgIC8vIHV0aWxpemVzIHRoZSB0aHJlYWRzIGluIG9uZSB3b3JrZ3JvdXAgdG8gZG8gdGhlIHJlZHVjdGlvbi4gSG93ZXZlcixcbiAgICAvLyB3aGVuIHRoZSByZWR1Y2Ugc2l6ZSBpcyB2ZXJ5IHNtYWxsLCBpdCdzIGJldHRlciB0byB1c2UgdGhlIHBsYWluXG4gICAgLy8gYWxnb3JpdGhtIHRvIHJlZHVjZSB0aGUgbnVtYmVyIG9mIHdvcmtncm91cHMgdG8gc3BlZWR1cC4gVGhlIHRocmV0aG9sZFxuICAgIC8vIGNhbiBiZSBmdXJ0aGVyIHR1bmVkLlxuICAgIGlmICh1dGlsLnNpemVGcm9tU2hhcGUocmVkdWNlU2hhcGUpIDwgMzIpIHtcbiAgICAgIHRoaXMudHlwZSA9ICdwbGFpbic7XG4gICAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudHlwZSA9ICdzaGFyZWQnO1xuICAgICAgLy8gQSB3b3JrIGdyb3VwIG9ubHkgb3V0cHV0cyBhIGRhdGEsIHNvIHdlIHRyYW5zZmVyIFsxLCAxLCAxXSB0byBjb21wdXRlXG4gICAgICAvLyBkaXNwYXRjaCBzaXplLlxuICAgICAgdGhpcy5kaXNwYXRjaCA9XG4gICAgICAgICAgY29tcHV0ZURpc3BhdGNoKHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIFsxLCAxLCAxXSk7XG4gICAgfVxuXG4gICAgdGhpcy5pbnB1dFNoYXBlID0gaW5wdXRTaGFwZTtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBhcmdNaW5NYXhfJHt0aGlzLm9wfV8ke3RoaXMudHlwZX1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB3b3JrZ3JvdXBTaXplWCA9IHRoaXMud29ya2dyb3VwU2l6ZVswXTtcbiAgICBjb25zdCBnZXRJbnB1dFNoYXBlTGFzdERpbSA9ICgpID0+IHtcbiAgICAgIGlmICh0aGlzLmlucHV0U2hhcGUubGVuZ3RoID09PSAxKSB7XG4gICAgICAgIHJldHVybiAndW5pZm9ybXMueFNoYXBlJztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBgdW5pZm9ybXMueFNoYXBlLiR7Z2V0Q29vcmRzWFlaKHRoaXMuaW5wdXRTaGFwZS5sZW5ndGggLSAxKX1gO1xuICAgICAgfVxuICAgIH07XG5cbiAgICBjb25zdCBzcGxpdE91dHB1dENvb3JkcyA9ICgpID0+IHtcbiAgICAgIGxldCBzbmlwcGV0ID0gJyc7XG4gICAgICBpZiAodGhpcy5vdXRwdXRTaGFwZS5sZW5ndGggPT09IDEpIHtcbiAgICAgICAgaWYgKHRoaXMuaW5wdXRTaGFwZS5sZW5ndGggIT09IDEpIHtcbiAgICAgICAgICBzbmlwcGV0ICs9ICdvdXRwdXRDb29yZHMsJztcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLm91dHB1dFNoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgc25pcHBldCArPSBgb3V0cHV0Q29vcmRzLiR7Z2V0Q29vcmRzWFlaKGkpfSxgO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICByZXR1cm4gc25pcHBldDtcbiAgICB9O1xuXG4gICAgaWYgKHRoaXMudHlwZSA9PT0gJ3NoYXJlZCcpIHtcbiAgICAgIGNvbnN0IHNoYXJlZE1lbW9yeVNuaXBwZXQgPSBgXG4gICAgICB2YXI8d29ya2dyb3VwPiB4QmVzdEluZGljZXMgOiBhcnJheTxpMzIsICR7d29ya2dyb3VwU2l6ZVh9PjtcbiAgICAgIHZhcjx3b3JrZ3JvdXA+IHhCZXN0VmFsdWVzIDogYXJyYXk8ZjMyLCAke3dvcmtncm91cFNpemVYfT47XG4gICAgYDtcbiAgICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgZm4gRElWX0NFSUwoYSA6IHUzMiwgYiA6IHUzMikgLT4gdTMyIHtcbiAgICAgICAgcmV0dXJuICgoYSAtIDF1KSAvIGIgKyAxdSk7XG4gICAgICB9XG5cbiAgICAgICR7c2hhcmVkTWVtb3J5U25pcHBldH1cblxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGxldCBvdXRwdXRJbmRleCA9IGluZGV4IC8gJHt3b3JrZ3JvdXBTaXplWH07XG4gICAgICAgIGxldCByZWR1Y2VMZW5ndGggPSAke2dldElucHV0U2hhcGVMYXN0RGltKCl9O1xuXG4gICAgICAgIHZhciBiZXN0SW5kZXggPSBpMzIobG9jYWxJZC54KTtcbiAgICAgICAgdmFyIGJlc3RWYWx1ZSA9IHVuaWZvcm1zLmluZmluaXR5VmFsdWU7XG4gICAgICAgIGxldCBvdXRwdXRDb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgob3V0cHV0SW5kZXgpO1xuICAgICAgICBmb3IgKHZhciBrID0gaTMyKGxvY2FsSWQueCk7IGsgPCByZWR1Y2VMZW5ndGggJiYgb3V0cHV0SW5kZXggPCB1bmlmb3Jtcy5zaXplO1xuICAgICAgICAgICAgayA9IGsgKyAke3dvcmtncm91cFNpemVYfSkge1xuICAgICAgICAgIGxldCBjYW5kaWRhdGUgPSBnZXRYKCR7c3BsaXRPdXRwdXRDb29yZHMoKX0gayk7XG4gICAgICAgICAgaWYgKCFpc25hbihjYW5kaWRhdGUpICYmIGNhbmRpZGF0ZSAke3RoaXMub3B9IGJlc3RWYWx1ZSkge1xuICAgICAgICAgICAgYmVzdFZhbHVlID0gY2FuZGlkYXRlO1xuICAgICAgICAgICAgYmVzdEluZGV4ID0gaztcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgeEJlc3RWYWx1ZXNbbG9jYWxJZC54XSA9IGJlc3RWYWx1ZTtcbiAgICAgICAgeEJlc3RJbmRpY2VzW2xvY2FsSWQueF0gPSBiZXN0SW5kZXg7XG4gICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgICAgICB2YXIgcmVkdWNlU2l6ZSA9IG1pbih1MzIocmVkdWNlTGVuZ3RoKSwgJHt3b3JrZ3JvdXBTaXplWH11KTtcbiAgICAgICAgZm9yICh2YXIgY3VycmVudFNpemUgPSByZWR1Y2VTaXplIC8gMnU7IHJlZHVjZVNpemUgPiAxdTtcbiAgICAgICAgICAgIGN1cnJlbnRTaXplID0gcmVkdWNlU2l6ZSAvIDJ1KSB7XG4gICAgICAgICAgbGV0IGludGVydmFsID0gRElWX0NFSUwocmVkdWNlU2l6ZSwgMnUpO1xuICAgICAgICAgIGlmIChsb2NhbElkLnggPCBjdXJyZW50U2l6ZSkge1xuICAgICAgICAgICAgbGV0IGNhbmRpZGF0ZSA9IHhCZXN0VmFsdWVzW2xvY2FsSWQueCArIGludGVydmFsXTtcbiAgICAgICAgICAgIGlmIChjYW5kaWRhdGUgJHt0aGlzLm9wfSBiZXN0VmFsdWUpIHtcbiAgICAgICAgICAgICAgYmVzdFZhbHVlID0gY2FuZGlkYXRlO1xuICAgICAgICAgICAgICB4QmVzdFZhbHVlc1tsb2NhbElkLnhdID0gYmVzdFZhbHVlO1xuICAgICAgICAgICAgICB4QmVzdEluZGljZXNbbG9jYWxJZC54XSA9IHhCZXN0SW5kaWNlc1tsb2NhbElkLnggKyBpbnRlcnZhbF07XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIHJlZHVjZVNpemUgPSBpbnRlcnZhbDtcbiAgICAgICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG4gICAgICAgIH1cblxuICAgICAgICBpZiAobG9jYWxJZC54ID09IDB1ICYmIG91dHB1dEluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXhJMzIob3V0cHV0SW5kZXgsIHhCZXN0SW5kaWNlc1tsb2NhbElkLnhdKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgICByZXR1cm4gdXNlckNvZGU7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICBsZXQgb3V0cHV0Q29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgICB2YXIgYmVzdEluZGV4ID0gMDtcbiAgICAgICAgICB2YXIgYmVzdFZhbHVlID0gZ2V0WCgke3NwbGl0T3V0cHV0Q29vcmRzKCl9IDApO1xuICAgICAgICAgIGxldCByZWR1Y2VMZW5ndGggPSAke2dldElucHV0U2hhcGVMYXN0RGltKCl9O1xuICAgICAgICAgIGZvciAodmFyIGkgPSAxOyBpIDwgcmVkdWNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgIGxldCBjYW5kaWRhdGUgPSBnZXRYKCR7c3BsaXRPdXRwdXRDb29yZHMoKX0gaSk7XG4gICAgICAgICAgICBpZiAoY2FuZGlkYXRlICR7dGhpcy5vcH0gYmVzdFZhbHVlKSB7XG4gICAgICAgICAgICAgIGJlc3RWYWx1ZSA9IGNhbmRpZGF0ZTtcbiAgICAgICAgICAgICAgYmVzdEluZGV4ID0gaTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleEkzMihpbmRleCwgYmVzdEluZGV4KTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgYDtcbiAgICAgIHJldHVybiB1c2VyQ29kZTtcbiAgICB9XG4gIH1cbn1cbiJdfQ==