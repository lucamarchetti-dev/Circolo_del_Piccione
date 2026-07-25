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
import { activationFnSnippet } from './activation_util';
import { matMulReadWriteFnSource } from './matmul_packed_webgpu';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch } from './webgpu_util';
export function makeMatMulReduceSource(workgroupSizeX) {
    return `
    var<workgroup> sumValues : array<f32, ${workgroupSizeX}>;
    ${main()} {
      let coords = getOutputCoords();
      let batch = coords[0];
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      let row = coords[1];
      let col = coords[2];
      var sum = 0.0;
      let Length = uniforms.dimInner;
      for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
        let dataA = mm_readA(batchA, row, k);
        let dataB = mm_readB(batchB, k, col);
        sum = sum + dataA * dataB;
      }
      sumValues[localId.x] = sum;
      workgroupBarrier();

      for(var currentSize = ${workgroupSizeX / 2}u; currentSize > 1u;
          currentSize = currentSize / 2u) {
        if (localId.x < currentSize)
        {
          sumValues[localId.x] = sumValues[localId.x] + sumValues[localId.x + currentSize];
        }
        workgroupBarrier();
      }

      if (localId.x == 0u) {
        sum = sumValues[0] + sumValues[1];
        mm_write(batch, row, col, sum);
      }
    }
  `;
}
export class MatMulReduceProgram {
    constructor(outputShape, transposeA = false, transposeB = false, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.workgroupSize = [256, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [], y: [1, 2], z: [0] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        const addBias = bias != null;
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.shaderKey =
            `matMulReduce_${this.activation}_${transposeA}_${transposeB}`;
    }
    getUserCode() {
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
      ${matMulReadWriteFnSource(this.addBias, this.activation, this.transposeA, this.transposeB)}
      ${makeMatMulReduceSource(this.workgroupSize[0])}
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF0bXVsX3JlZHVjZV93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9tYXRtdWxfcmVkdWNlX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUN0RCxPQUFPLEVBQUMsdUJBQXVCLEVBQUMsTUFBTSx3QkFBd0IsQ0FBQztBQUMvRCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFOUMsTUFBTSxVQUFVLHNCQUFzQixDQUFDLGNBQXNCO0lBQzNELE9BQU87NENBQ21DLGNBQWM7TUFDcEQsSUFBSSxFQUFFOzs7Ozs7Ozs7eURBUzZDLGNBQWM7Ozs7Ozs7OzhCQVF6QyxjQUFjLEdBQUcsQ0FBQzs7Ozs7Ozs7Ozs7Ozs7R0FjN0MsQ0FBQztBQUNKLENBQUM7QUFFRCxNQUFNLE9BQU8sbUJBQW1CO0lBYzlCLFlBQ0ksV0FBcUMsRUFBRSxVQUFVLEdBQUcsS0FBSyxFQUN6RCxVQUFVLEdBQUcsS0FBSyxFQUFFLE9BQW1CLElBQUksRUFDM0MsYUFBc0MsSUFBSSxFQUMxQyx5QkFBcUMsSUFBSTtRQWI3QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzNCLGFBQVEsR0FBRyxtREFBbUQsQ0FBQztRQUMvRCxrQkFBYSxHQUE2QixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFZcEQsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUM7UUFDakQsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsTUFBTSxPQUFPLEdBQUcsSUFBSSxJQUFJLElBQUksQ0FBQztRQUM3QixNQUFNLHlCQUF5QixHQUFHLHNCQUFzQixJQUFJLElBQUksQ0FBQztRQUNqRSxJQUFJLE9BQU8sRUFBRTtZQUNYLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1NBQ2pDO1FBRUQsSUFBSSx5QkFBeUIsRUFBRTtZQUM3QixJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDO1NBQ25EO1FBRUQsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7UUFDN0IsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7UUFDN0IsSUFBSSxDQUFDLE9BQU8sR0FBRyxPQUFPLENBQUM7UUFDdkIsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7UUFDN0IsSUFBSSxDQUFDLHlCQUF5QixHQUFHLHlCQUF5QixDQUFDO1FBQzNELElBQUksQ0FBQyxTQUFTO1lBQ1YsZ0JBQWdCLElBQUksQ0FBQyxVQUFVLElBQUksVUFBVSxJQUFJLFVBQVUsRUFBRSxDQUFDO0lBQ3BFLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7UUFDYixtQkFBbUIsQ0FBQyxJQUFJLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyx5QkFBeUIsQ0FBQztRQUVwRSx1QkFBdUIsQ0FDbkIsSUFBSSxDQUFDLE9BQU8sRUFBRSxJQUFJLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyxVQUFVLEVBQUUsSUFBSSxDQUFDLFVBQVUsQ0FBQztRQUNwRSxzQkFBc0IsQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ2hELENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2FjdGl2YXRpb25GblNuaXBwZXR9IGZyb20gJy4vYWN0aXZhdGlvbl91dGlsJztcbmltcG9ydCB7bWF0TXVsUmVhZFdyaXRlRm5Tb3VyY2V9IGZyb20gJy4vbWF0bXVsX3BhY2tlZF93ZWJncHUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2h9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gbWFrZU1hdE11bFJlZHVjZVNvdXJjZSh3b3JrZ3JvdXBTaXplWDogbnVtYmVyKTogc3RyaW5nIHtcbiAgcmV0dXJuIGBcbiAgICB2YXI8d29ya2dyb3VwPiBzdW1WYWx1ZXMgOiBhcnJheTxmMzIsICR7d29ya2dyb3VwU2l6ZVh9PjtcbiAgICAke21haW4oKX0ge1xuICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgbGV0IGJhdGNoID0gY29vcmRzWzBdO1xuICAgICAgbGV0IGJhdGNoQSA9IGJhdGNoICUgdW5pZm9ybXMuYVNoYXBlWzBdO1xuICAgICAgbGV0IGJhdGNoQiA9IGJhdGNoICUgdW5pZm9ybXMuYlNoYXBlWzBdO1xuICAgICAgbGV0IHJvdyA9IGNvb3Jkc1sxXTtcbiAgICAgIGxldCBjb2wgPSBjb29yZHNbMl07XG4gICAgICB2YXIgc3VtID0gMC4wO1xuICAgICAgbGV0IExlbmd0aCA9IHVuaWZvcm1zLmRpbUlubmVyO1xuICAgICAgZm9yICh2YXIgayA9IGkzMihsb2NhbElkLngpOyBrIDwgTGVuZ3RoOyBrID0gayArICR7d29ya2dyb3VwU2l6ZVh9KSB7XG4gICAgICAgIGxldCBkYXRhQSA9IG1tX3JlYWRBKGJhdGNoQSwgcm93LCBrKTtcbiAgICAgICAgbGV0IGRhdGFCID0gbW1fcmVhZEIoYmF0Y2hCLCBrLCBjb2wpO1xuICAgICAgICBzdW0gPSBzdW0gKyBkYXRhQSAqIGRhdGFCO1xuICAgICAgfVxuICAgICAgc3VtVmFsdWVzW2xvY2FsSWQueF0gPSBzdW07XG4gICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgICAgIGZvcih2YXIgY3VycmVudFNpemUgPSAke3dvcmtncm91cFNpemVYIC8gMn11OyBjdXJyZW50U2l6ZSA+IDF1O1xuICAgICAgICAgIGN1cnJlbnRTaXplID0gY3VycmVudFNpemUgLyAydSkge1xuICAgICAgICBpZiAobG9jYWxJZC54IDwgY3VycmVudFNpemUpXG4gICAgICAgIHtcbiAgICAgICAgICBzdW1WYWx1ZXNbbG9jYWxJZC54XSA9IHN1bVZhbHVlc1tsb2NhbElkLnhdICsgc3VtVmFsdWVzW2xvY2FsSWQueCArIGN1cnJlbnRTaXplXTtcbiAgICAgICAgfVxuICAgICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG4gICAgICB9XG5cbiAgICAgIGlmIChsb2NhbElkLnggPT0gMHUpIHtcbiAgICAgICAgc3VtID0gc3VtVmFsdWVzWzBdICsgc3VtVmFsdWVzWzFdO1xuICAgICAgICBtbV93cml0ZShiYXRjaCwgcm93LCBjb2wsIHN1bSk7XG4gICAgICB9XG4gICAgfVxuICBgO1xufVxuXG5leHBvcnQgY2xhc3MgTWF0TXVsUmVkdWNlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXSwgejogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydBJywgJ0InXTtcbiAgdW5pZm9ybXMgPSBgZGltQU91dGVyIDogaTMyLCBkaW1CT3V0ZXIgOiBpMzIsIGRpbUlubmVyIDogaTMyLGA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsyNTYsIDEsIDFdO1xuICB0cmFuc3Bvc2VBOiBib29sZWFuO1xuICB0cmFuc3Bvc2VCOiBib29sZWFuO1xuICBhZGRCaWFzOiBib29sZWFuO1xuICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbjtcbiAgaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIG91dHB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRyYW5zcG9zZUEgPSBmYWxzZSxcbiAgICAgIHRyYW5zcG9zZUIgPSBmYWxzZSwgYmlhczogVGVuc29ySW5mbyA9IG51bGwsXG4gICAgICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbiA9IG51bGwsXG4gICAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzOiBUZW5zb3JJbmZvID0gbnVsbCkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0ge3g6IFtdLCB5OiBbMSwgMl0sIHo6IFswXX07XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIGNvbnN0IGFkZEJpYXMgPSBiaWFzICE9IG51bGw7XG4gICAgY29uc3QgaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyA9IHByZWx1QWN0aXZhdGlvbldlaWdodHMgIT0gbnVsbDtcbiAgICBpZiAoYWRkQmlhcykge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ2JpYXMnKTtcbiAgICB9XG5cbiAgICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cykge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ3ByZWx1QWN0aXZhdGlvbldlaWdodHMnKTtcbiAgICB9XG5cbiAgICB0aGlzLnRyYW5zcG9zZUEgPSB0cmFuc3Bvc2VBO1xuICAgIHRoaXMudHJhbnNwb3NlQiA9IHRyYW5zcG9zZUI7XG4gICAgdGhpcy5hZGRCaWFzID0gYWRkQmlhcztcbiAgICB0aGlzLmFjdGl2YXRpb24gPSBhY3RpdmF0aW9uO1xuICAgIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyA9IGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHM7XG4gICAgdGhpcy5zaGFkZXJLZXkgPVxuICAgICAgICBgbWF0TXVsUmVkdWNlXyR7dGhpcy5hY3RpdmF0aW9ufV8ke3RyYW5zcG9zZUF9XyR7dHJhbnNwb3NlQn1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7YWN0aXZhdGlvbkZuU25pcHBldCh0aGlzLmFjdGl2YXRpb24sIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyl9XG4gICAgICAke1xuICAgICAgICBtYXRNdWxSZWFkV3JpdGVGblNvdXJjZShcbiAgICAgICAgICAgIHRoaXMuYWRkQmlhcywgdGhpcy5hY3RpdmF0aW9uLCB0aGlzLnRyYW5zcG9zZUEsIHRoaXMudHJhbnNwb3NlQil9XG4gICAgICAke21ha2VNYXRNdWxSZWR1Y2VTb3VyY2UodGhpcy53b3JrZ3JvdXBTaXplWzBdKX1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19