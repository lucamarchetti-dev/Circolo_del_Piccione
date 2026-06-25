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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class Pool2DProgram {
    constructor(convInfo, poolType, computePositions = false, flattenPositions = false, includeBatchIndex = false) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,`;
        // TODO(jiajia.qin@intel.com): Dynamically choose different workgroupSize for
        // different output shapes.
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.poolType = poolType;
        this.computePositions = computePositions;
        this.flattenPositions = flattenPositions;
        this.includeBatchIndex = includeBatchIndex;
        this.shaderKey = `pool2D_${poolType}_${computePositions}_${flattenPositions}_${includeBatchIndex}`;
    }
    getUserCode() {
        let updateSnippet;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
        }
        else if (this.computePositions) {
            const positionStr = this.flattenPositions ?
                (this.includeBatchIndex ?
                    `((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d` :
                    `(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d`) :
                `wR * uniforms.filterDims.y + wC`;
            updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
        }
        else {
            updateSnippet = `resultValue = max(value, resultValue);`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / max(count, 1.0)`;
        }
        const userCode = `
      ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.strides - uniforms.pads;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          ${this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilations.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilations.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, d);
              ${updateSnippet}
            }
          }

          ${this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
            `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
        return userCode;
    }
}
export class Pool3DProgram {
    constructor(convInfo, poolType, computePositions = false, flattenPositions = false, includeBatchIndex = false) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, convDims : vec3<i32>, filterDims : vec3<i32>,`;
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.poolType = poolType;
        this.computePositions = computePositions;
        this.flattenPositions = flattenPositions;
        this.includeBatchIndex = includeBatchIndex;
        this.shaderKey = `pool3D_${poolType}_${computePositions}_${flattenPositions}_${includeBatchIndex}`;
    }
    getUserCode() {
        let updateSnippet;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue += value; count += 1.0;`;
        }
        else if (this.computePositions) {
            const positionStr = this.flattenPositions ?
                (this.includeBatchIndex ?
                    `(((batch * uniforms.xShape.y + xD) * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch` :
                    `((xD * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch`) :
                `wD * uniforms.filterDims.y * uniforms.filterDims.y + wR * uniforms.filterDims.z + wC`;
            updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
        }
        else {
            updateSnippet = `resultValue = max(value, resultValue);`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / max(count, 1.0)`;
        }
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords.x;
          let ch = coords.u;

          let xCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
          let xDCorner = xCorner.x;
          let xRCorner = xCorner.y;
          let xCCorner = xCorner.z;

          ${this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wD = 0; wD < uniforms.filterDims.x; wD++) {
            let xD = xDCorner + wD;
            if (xD < 0 || xD >= uniforms.convDims.x) {
              continue;
            }

            for (var wR = 0; wR < uniforms.filterDims.y; wR++) {
              let xR = xRCorner + wR;
              if (xR < 0 || xR >= uniforms.convDims.y) {
                continue;
              }

              for (var wC = 0; wC < uniforms.filterDims.z; wC++) {
                let xC = xCCorner + wC;
                if (xC < 0 || xC >= uniforms.convDims.z) {
                  continue;
                }

                let value = getX(batch, xD, xR, xC, ch);
                ${updateSnippet}
              }
            }
          }

          ${this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
            `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9wb29sX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLGFBQWE7SUFpQnhCLFlBQ0ksUUFBaUMsRUFBRSxRQUFxQixFQUN4RCxnQkFBZ0IsR0FBRyxLQUFLLEVBQUUsZ0JBQWdCLEdBQUcsS0FBSyxFQUNsRCxpQkFBaUIsR0FBRyxLQUFLO1FBZjdCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN0QixhQUFRLEdBQ0osNkdBQTZHLENBQUM7UUFDbEgsNkVBQTZFO1FBQzdFLDJCQUEyQjtRQUMzQixrQkFBYSxHQUE2QixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFdEQsU0FBSSxHQUFHLElBQUksQ0FBQztRQVNWLElBQUksUUFBUSxLQUFLLEtBQUssSUFBSSxnQkFBZ0IsRUFBRTtZQUMxQyxNQUFNLElBQUksS0FBSyxDQUFDLDRDQUE0QyxDQUFDLENBQUM7U0FDL0Q7UUFFRCxJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUM7UUFDckMsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxRQUFR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QUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFBvb2wyRFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgYHN0cmlkZXMgOiB2ZWMyPGkzMj4sIHBhZHMgOiB2ZWMyPGkzMj4sIGRpbGF0aW9ucyA6IHZlYzI8aTMyPiwgY29udkRpbXMgOiB2ZWMyPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMyPGkzMj4sYDtcbiAgLy8gVE9ETyhqaWFqaWEucWluQGludGVsLmNvbSk6IER5bmFtaWNhbGx5IGNob29zZSBkaWZmZXJlbnQgd29ya2dyb3VwU2l6ZSBmb3JcbiAgLy8gZGlmZmVyZW50IG91dHB1dCBzaGFwZXMuXG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxMjgsIDEsIDFdO1xuICBwb29sVHlwZTogJ21heCd8J2F2Zyc7XG4gIHNpemUgPSB0cnVlO1xuICBjb21wdXRlUG9zaXRpb25zOiBib29sZWFuO1xuICBmbGF0dGVuUG9zaXRpb25zOiBib29sZWFuO1xuICBpbmNsdWRlQmF0Y2hJbmRleDogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbywgcG9vbFR5cGU6ICdtYXgnfCdhdmcnLFxuICAgICAgY29tcHV0ZVBvc2l0aW9ucyA9IGZhbHNlLCBmbGF0dGVuUG9zaXRpb25zID0gZmFsc2UsXG4gICAgICBpbmNsdWRlQmF0Y2hJbmRleCA9IGZhbHNlKSB7XG4gICAgaWYgKHBvb2xUeXBlID09PSAnYXZnJyAmJiBjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Nhbm5vdCBjb21wdXRlIHBvc2l0aW9ucyBmb3IgYXZlcmFnZSBwb29sLicpO1xuICAgIH1cblxuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5vdXRTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLnBvb2xUeXBlID0gcG9vbFR5cGU7XG4gICAgdGhpcy5jb21wdXRlUG9zaXRpb25zID0gY29tcHV0ZVBvc2l0aW9ucztcbiAgICB0aGlzLmZsYXR0ZW5Qb3NpdGlvbnMgPSBmbGF0dGVuUG9zaXRpb25zO1xuICAgIHRoaXMuaW5jbHVkZUJhdGNoSW5kZXggPSBpbmNsdWRlQmF0Y2hJbmRleDtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBwb29sMkRfJHtwb29sVHlwZX1fJHtjb21wdXRlUG9zaXRpb25zfV8ke1xuICAgICAgICBmbGF0dGVuUG9zaXRpb25zfV8ke2luY2x1ZGVCYXRjaEluZGV4fWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGxldCB1cGRhdGVTbmlwcGV0OiBzdHJpbmc7XG4gICAgaWYgKHRoaXMucG9vbFR5cGUgPT09ICdhdmcnKSB7XG4gICAgICB1cGRhdGVTbmlwcGV0ID0gYHJlc3VsdFZhbHVlID0gcmVzdWx0VmFsdWUgKyB2YWx1ZTsgY291bnQgPSBjb3VudCArIDEuMDtgO1xuICAgIH0gZWxzZSBpZiAodGhpcy5jb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICBjb25zdCBwb3NpdGlvblN0ciA9IHRoaXMuZmxhdHRlblBvc2l0aW9ucyA/XG4gICAgICAgICAgKHRoaXMuaW5jbHVkZUJhdGNoSW5kZXggP1xuICAgICAgICAgICAgICAgYCgoYmF0Y2ggKiB1bmlmb3Jtcy54U2hhcGVbMV0gKyB4UikgKiB1bmlmb3Jtcy54U2hhcGVbMl0gKyB4QykgKiB1bmlmb3Jtcy54U2hhcGVbM10gKyBkYCA6XG4gICAgICAgICAgICAgICBgKHhSICogdW5pZm9ybXMueFNoYXBlWzJdICsgeEMpICogdW5pZm9ybXMueFNoYXBlWzNdICsgZGApIDpcbiAgICAgICAgICBgd1IgKiB1bmlmb3Jtcy5maWx0ZXJEaW1zLnkgKyB3Q2A7XG4gICAgICB1cGRhdGVTbmlwcGV0ID0gYGxldCBjdXJyTWF4VmFsdWUgPSBtaXgodmFsdWUsIG1heFZhbHVlLCBtYXhWYWx1ZUZvdW5kKTtcbiAgICAgIGlmICh2YWx1ZSA+PSBjdXJyTWF4VmFsdWUpIHtcbiAgICAgICAgbWF4VmFsdWUgPSB2YWx1ZTtcbiAgICAgICAgbWF4VmFsdWVGb3VuZCA9IDEuMDtcbiAgICAgICAgbWF4UG9zaXRpb24gPSAke3Bvc2l0aW9uU3RyfTtcbiAgICAgIH1gO1xuICAgIH0gZWxzZSB7XG4gICAgICB1cGRhdGVTbmlwcGV0ID0gYHJlc3VsdFZhbHVlID0gbWF4KHZhbHVlLCByZXN1bHRWYWx1ZSk7YDtcbiAgICB9XG5cbiAgICBsZXQgcmV0dXJuVmFsdWUgPSBgcmVzdWx0VmFsdWVgO1xuICAgIGlmICh0aGlzLnBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgcmV0dXJuVmFsdWUgPSBgcmVzdWx0VmFsdWUgLyBtYXgoY291bnQsIDEuMClgO1xuICAgIH1cblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCBiYXRjaCA9IGNvb3Jkc1swXTtcbiAgICAgICAgICBsZXQgZCA9IGNvb3Jkc1szXTtcbiAgICAgICAgICBsZXQgeFJDQ29ybmVyID0gdmVjMjxpMzI+KGNvb3Jkcy55eikgKiB1bmlmb3Jtcy5zdHJpZGVzIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgICBsZXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgICBsZXQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAgICR7XG4gICAgICAgIHRoaXMuY29tcHV0ZVBvc2l0aW9ucyA/XG4gICAgICAgICAgICBgdmFyIG1heFZhbHVlID0gMC4wO1xuICAgICAgICAgICAgdmFyIG1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICAgICAgICB2YXIgbWF4UG9zaXRpb24gPSAwO2AgOlxuICAgICAgICAgICAgYHZhciByZXN1bHRWYWx1ZSA9ICR7XG4gICAgICAgICAgICAgICAgdGhpcy5wb29sVHlwZSA9PT0gJ2F2ZycgPyAnMC4wJyA6ICctMS4wIC8gcG93KDEwLjAsIC0yMC4wKSd9O2B9XG5cbiAgICAgICAgICB2YXIgY291bnQgPSAwLjA7XG4gICAgICAgICAgZm9yICh2YXIgd1IgPSAwOyB3UiA8IHVuaWZvcm1zLmZpbHRlckRpbXMueDsgd1IgPSB3UiArIHVuaWZvcm1zLmRpbGF0aW9ucy54KSB7XG4gICAgICAgICAgICBsZXQgeFIgPSB4UkNvcm5lciArIHdSO1xuXG4gICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHVuaWZvcm1zLmNvbnZEaW1zLngpIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zLnk7IHdDID0gd0MgKyB1bmlmb3Jtcy5kaWxhdGlvbnMueSkge1xuICAgICAgICAgICAgICBsZXQgeEMgPSB4Q0Nvcm5lciArIHdDO1xuICAgICAgICAgICAgICBpZiAoeEMgPCAwIHx8IHhDID49IHVuaWZvcm1zLmNvbnZEaW1zLnkpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGxldCB2YWx1ZSA9IGdldFgoYmF0Y2gsIHhSLCB4QywgZCk7XG4gICAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG5cbiAgICAgICAgICAke1xuICAgICAgICB0aGlzLmNvbXB1dGVQb3NpdGlvbnMgPyBgc2V0T3V0cHV0QXRJbmRleEkzMihpbmRleCwgbWF4UG9zaXRpb24pO2AgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgJHtyZXR1cm5WYWx1ZX0pO2B9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgUG9vbDNEUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCddO1xuICB1bmlmb3JtcyA9XG4gICAgICBgc3RyaWRlcyA6IHZlYzM8aTMyPiwgcGFkcyA6IHZlYzM8aTMyPiwgY29udkRpbXMgOiB2ZWMzPGkzMj4sIGZpbHRlckRpbXMgOiB2ZWMzPGkzMj4sYDtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEyOCwgMSwgMV07XG4gIHBvb2xUeXBlOiAnbWF4J3wnYXZnJztcbiAgc2l6ZSA9IHRydWU7XG4gIGNvbXB1dGVQb3NpdGlvbnM6IGJvb2xlYW47XG4gIGZsYXR0ZW5Qb3NpdGlvbnM6IGJvb2xlYW47XG4gIGluY2x1ZGVCYXRjaEluZGV4OiBib29sZWFuO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvLCBwb29sVHlwZTogJ21heCd8J2F2ZycsXG4gICAgICBjb21wdXRlUG9zaXRpb25zID0gZmFsc2UsIGZsYXR0ZW5Qb3NpdGlvbnMgPSBmYWxzZSxcbiAgICAgIGluY2x1ZGVCYXRjaEluZGV4ID0gZmFsc2UpIHtcbiAgICBpZiAocG9vbFR5cGUgPT09ICdhdmcnICYmIGNvbXB1dGVQb3NpdGlvbnMpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignQ2Fubm90IGNvbXB1dGUgcG9zaXRpb25zIGZvciBhdmVyYWdlIHBvb2wuJyk7XG4gICAgfVxuXG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLm91dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMucG9vbFR5cGUgPSBwb29sVHlwZTtcbiAgICB0aGlzLmNvbXB1dGVQb3NpdGlvbnMgPSBjb21wdXRlUG9zaXRpb25zO1xuICAgIHRoaXMuZmxhdHRlblBvc2l0aW9ucyA9IGZsYXR0ZW5Qb3NpdGlvbnM7XG4gICAgdGhpcy5pbmNsdWRlQmF0Y2hJbmRleCA9IGluY2x1ZGVCYXRjaEluZGV4O1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYHBvb2wzRF8ke3Bvb2xUeXBlfV8ke2NvbXB1dGVQb3NpdGlvbnN9XyR7XG4gICAgICAgIGZsYXR0ZW5Qb3NpdGlvbnN9XyR7aW5jbHVkZUJhdGNoSW5kZXh9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgbGV0IHVwZGF0ZVNuaXBwZXQ6IHN0cmluZztcbiAgICBpZiAodGhpcy5wb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgIHVwZGF0ZVNuaXBwZXQgPSBgcmVzdWx0VmFsdWUgKz0gdmFsdWU7IGNvdW50ICs9IDEuMDtgO1xuICAgIH0gZWxzZSBpZiAodGhpcy5jb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICBjb25zdCBwb3NpdGlvblN0ciA9IHRoaXMuZmxhdHRlblBvc2l0aW9ucyA/XG4gICAgICAgICAgKHRoaXMuaW5jbHVkZUJhdGNoSW5kZXggP1xuICAgICAgICAgICAgICAgYCgoKGJhdGNoICogdW5pZm9ybXMueFNoYXBlLnkgKyB4RCkgKiB1bmlmb3Jtcy54U2hhcGUueiArIHhSKSAqIHVuaWZvcm1zLnhTaGFwZS53ICsgeEMpICogdW5pZm9ybXMueFNoYXBlLnUgKyBjaGAgOlxuICAgICAgICAgICAgICAgYCgoeEQgKiB1bmlmb3Jtcy54U2hhcGUueiArIHhSKSAqIHVuaWZvcm1zLnhTaGFwZS53ICsgeEMpICogdW5pZm9ybXMueFNoYXBlLnUgKyBjaGApIDpcbiAgICAgICAgICBgd0QgKiB1bmlmb3Jtcy5maWx0ZXJEaW1zLnkgKiB1bmlmb3Jtcy5maWx0ZXJEaW1zLnkgKyB3UiAqIHVuaWZvcm1zLmZpbHRlckRpbXMueiArIHdDYDtcbiAgICAgIHVwZGF0ZVNuaXBwZXQgPSBgbGV0IGN1cnJNYXhWYWx1ZSA9IG1peCh2YWx1ZSwgbWF4VmFsdWUsIG1heFZhbHVlRm91bmQpO1xuICAgICAgaWYgKHZhbHVlID49IGN1cnJNYXhWYWx1ZSkge1xuICAgICAgICBtYXhWYWx1ZSA9IHZhbHVlO1xuICAgICAgICBtYXhWYWx1ZUZvdW5kID0gMS4wO1xuICAgICAgICBtYXhQb3NpdGlvbiA9ICR7cG9zaXRpb25TdHJ9O1xuICAgICAgfWA7XG4gICAgfSBlbHNlIHtcbiAgICAgIHVwZGF0ZVNuaXBwZXQgPSBgcmVzdWx0VmFsdWUgPSBtYXgodmFsdWUsIHJlc3VsdFZhbHVlKTtgO1xuICAgIH1cblxuICAgIGxldCByZXR1cm5WYWx1ZSA9IGByZXN1bHRWYWx1ZWA7XG4gICAgaWYgKHRoaXMucG9vbFR5cGUgPT09ICdhdmcnKSB7XG4gICAgICByZXR1cm5WYWx1ZSA9IGByZXN1bHRWYWx1ZSAvIG1heChjb3VudCwgMS4wKWA7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCBiYXRjaCA9IGNvb3Jkcy54O1xuICAgICAgICAgIGxldCBjaCA9IGNvb3Jkcy51O1xuXG4gICAgICAgICAgbGV0IHhDb3JuZXIgPSB2ZWMzPGkzMj4oY29vcmRzLnksIGNvb3Jkcy56LCBjb29yZHMudykgKiB1bmlmb3Jtcy5zdHJpZGVzIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgICBsZXQgeERDb3JuZXIgPSB4Q29ybmVyLng7XG4gICAgICAgICAgbGV0IHhSQ29ybmVyID0geENvcm5lci55O1xuICAgICAgICAgIGxldCB4Q0Nvcm5lciA9IHhDb3JuZXIuejtcblxuICAgICAgICAgICR7XG4gICAgICAgIHRoaXMuY29tcHV0ZVBvc2l0aW9ucyA/XG4gICAgICAgICAgICBgdmFyIG1heFZhbHVlID0gMC4wO1xuICAgICAgICAgICAgdmFyIG1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICAgICAgICB2YXIgbWF4UG9zaXRpb24gPSAwO2AgOlxuICAgICAgICAgICAgYHZhciByZXN1bHRWYWx1ZSA9ICR7XG4gICAgICAgICAgICAgICAgdGhpcy5wb29sVHlwZSA9PT0gJ2F2ZycgPyAnMC4wJyA6ICctMS4wIC8gcG93KDEwLjAsIC0yMC4wKSd9O2B9XG5cbiAgICAgICAgICB2YXIgY291bnQgPSAwLjA7XG4gICAgICAgICAgZm9yICh2YXIgd0QgPSAwOyB3RCA8IHVuaWZvcm1zLmZpbHRlckRpbXMueDsgd0QrKykge1xuICAgICAgICAgICAgbGV0IHhEID0geERDb3JuZXIgKyB3RDtcbiAgICAgICAgICAgIGlmICh4RCA8IDAgfHwgeEQgPj0gdW5pZm9ybXMuY29udkRpbXMueCkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZm9yICh2YXIgd1IgPSAwOyB3UiA8IHVuaWZvcm1zLmZpbHRlckRpbXMueTsgd1IrKykge1xuICAgICAgICAgICAgICBsZXQgeFIgPSB4UkNvcm5lciArIHdSO1xuICAgICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHVuaWZvcm1zLmNvbnZEaW1zLnkpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zLno7IHdDKyspIHtcbiAgICAgICAgICAgICAgICBsZXQgeEMgPSB4Q0Nvcm5lciArIHdDO1xuICAgICAgICAgICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0gdW5pZm9ybXMuY29udkRpbXMueikge1xuICAgICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgbGV0IHZhbHVlID0gZ2V0WChiYXRjaCwgeEQsIHhSLCB4QywgY2gpO1xuICAgICAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cblxuICAgICAgICAgICR7XG4gICAgICAgIHRoaXMuY29tcHV0ZVBvc2l0aW9ucyA/IGBzZXRPdXRwdXRBdEluZGV4STMyKGluZGV4LCBtYXhQb3NpdGlvbik7YCA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCAke3JldHVyblZhbHVlfSk7YH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=