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
import { util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch } from './webgpu_util';
export class TransposeSharedProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        // Note that the maximum number of workgroup invocations by webgpu is 256.
        this.workgroupSize = [16, 16, 1];
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [0], y: [1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [1, 1, 1]);
        this.shaderKey = 'transposeShared';
    }
    getUserCode() {
        util.assert(this.workgroupSize[0] === this.workgroupSize[1], () => `Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`);
        const tileSize = this.workgroupSize[0];
        const userCode = `
      var<workgroup> tile : array<array<f32, ${this.workgroupSize[0] + 1}>, ${this.workgroupSize[0]}>;
      ${main()} {
        var x = i32(workgroupId.x) * ${tileSize} + i32(localId.x);
        var y = i32(workgroupId.y) * ${tileSize} + i32(localId.y);
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] = f32(A[y * width + x]);
        }
        workgroupBarrier();

        x = i32(workgroupId.y) * ${tileSize} + i32(localId.x);
        y = i32(workgroupId.x) * ${tileSize} + i32(localId.y);
        if (x < height && y < width) {
          setOutputAtIndex((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNwb3NlX3NoYXJlZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy90cmFuc3Bvc2Vfc2hhcmVkX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFDM0MsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRTlDLE1BQU0sT0FBTyxzQkFBc0I7SUFTakMsWUFBWSxNQUFnQixFQUFFLE1BQWdCO1FBUjlDLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUt0QiwwRUFBMEU7UUFDMUUsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBR3BELE1BQU0sV0FBVyxHQUFhLElBQUksS0FBSyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN2RCxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtZQUMzQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQ3BDO1FBQ0QsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxFQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUM7UUFDdkMsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTFFLElBQUksQ0FBQyxTQUFTLEdBQUcsaUJBQWlCLENBQUM7SUFDckMsQ0FBQztJQUVELFdBQVc7UUFDVCxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLEtBQUssSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFDL0MsR0FBRyxFQUFFLENBQUMsZ0RBQ0YsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsTUFBTSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM1RCxNQUFNLFFBQVEsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLE1BQU0sUUFBUSxHQUFHOytDQUMwQixJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsTUFDaEUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7UUFDckIsSUFBSSxFQUFFO3VDQUN5QixRQUFRO3VDQUNSLFFBQVE7Ozs7Ozs7O21DQVFaLFFBQVE7bUNBQ1IsUUFBUTs7Ozs7O0tBTXRDLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7dXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNofSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFRyYW5zcG9zZVNoYXJlZFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsnQSddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIC8vIE5vdGUgdGhhdCB0aGUgbWF4aW11bSBudW1iZXIgb2Ygd29ya2dyb3VwIGludm9jYXRpb25zIGJ5IHdlYmdwdSBpcyAyNTYuXG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxNiwgMTYsIDFdO1xuXG4gIGNvbnN0cnVjdG9yKGFTaGFwZTogbnVtYmVyW10sIG5ld0RpbTogbnVtYmVyW10pIHtcbiAgICBjb25zdCBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBuZXcgQXJyYXkoYVNoYXBlLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBvdXRwdXRTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgICAgb3V0cHV0U2hhcGVbaV0gPSBhU2hhcGVbbmV3RGltW2ldXTtcbiAgICB9XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSB7eDogWzBdLCB5OiBbMV19O1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSwgWzEsIDEsIDFdKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3RyYW5zcG9zZVNoYXJlZCc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB0aGlzLndvcmtncm91cFNpemVbMF0gPT09IHRoaXMud29ya2dyb3VwU2l6ZVsxXSxcbiAgICAgICAgKCkgPT4gYE11c3QgYmUgYSBzcXVhcmUgdGlsZSwgY3VycmVudCB0aWxlIHNoYXBlIGlzICR7XG4gICAgICAgICAgICB0aGlzLndvcmtncm91cFNpemVbMF19IHggJHt0aGlzLndvcmtncm91cFNpemVbMV19YCk7XG4gICAgY29uc3QgdGlsZVNpemUgPSB0aGlzLndvcmtncm91cFNpemVbMF07XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICB2YXI8d29ya2dyb3VwPiB0aWxlIDogYXJyYXk8YXJyYXk8ZjMyLCAke3RoaXMud29ya2dyb3VwU2l6ZVswXSArIDF9PiwgJHtcbiAgICAgICAgdGhpcy53b3JrZ3JvdXBTaXplWzBdfT47XG4gICAgICAke21haW4oKX0ge1xuICAgICAgICB2YXIgeCA9IGkzMih3b3JrZ3JvdXBJZC54KSAqICR7dGlsZVNpemV9ICsgaTMyKGxvY2FsSWQueCk7XG4gICAgICAgIHZhciB5ID0gaTMyKHdvcmtncm91cElkLnkpICogJHt0aWxlU2l6ZX0gKyBpMzIobG9jYWxJZC55KTtcbiAgICAgICAgbGV0IHdpZHRoID0gdW5pZm9ybXMub3V0U2hhcGVbMF07XG4gICAgICAgIGxldCBoZWlnaHQgPSB1bmlmb3Jtcy5vdXRTaGFwZVsxXTtcbiAgICAgICAgaWYgKHggPCB3aWR0aCAmJiB5IDwgaGVpZ2h0KSB7XG4gICAgICAgICAgdGlsZVtsb2NhbElkLnldW2xvY2FsSWQueF0gPSBmMzIoQVt5ICogd2lkdGggKyB4XSk7XG4gICAgICAgIH1cbiAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICAgIHggPSBpMzIod29ya2dyb3VwSWQueSkgKiAke3RpbGVTaXplfSArIGkzMihsb2NhbElkLngpO1xuICAgICAgICB5ID0gaTMyKHdvcmtncm91cElkLngpICogJHt0aWxlU2l6ZX0gKyBpMzIobG9jYWxJZC55KTtcbiAgICAgICAgaWYgKHggPCBoZWlnaHQgJiYgeSA8IHdpZHRoKSB7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleCgoeSAqIGhlaWdodCArIHgpLCB0aWxlW2xvY2FsSWQueF1cbiAgICAgICAgICAgIFtsb2NhbElkLnldKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=