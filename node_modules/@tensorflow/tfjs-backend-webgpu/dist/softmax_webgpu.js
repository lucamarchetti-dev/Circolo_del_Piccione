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
import { getMainHeaderString as main } from './webgpu_program';
import { flatDispatchLayout } from './webgpu_util';
export class SoftmaxProgram {
    constructor(outputShape) {
        this.variableNames = ['logits'];
        this.outputShape = outputShape; // [rows, cols]
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [this.outputShape[0], 1, 1];
        if (this.outputShape[1] >= 4096) {
            this.workgroupSize = [256, 1, 1];
        }
        else {
            this.workgroupSize = [64, 1, 1];
        }
        this.shaderKey = 'softmax';
    }
    getUserCode() {
        const userCode = `
    var<workgroup> buf : array<f32, ${this.workgroupSize[0]}>;
    var<workgroup> rowMaxShared : f32;
    var<workgroup> rowSumShared : f32;
    const blockSize = ${this.workgroupSize[0]};
    ${main('index')} {
      let row = index / blockSize;
      let tid = i32(localId.x);
      let cols = uniforms.outShape[1];

      var threadMax = -3.402823e+38f;
      for (var col = tid; col < cols; col += blockSize) {
        let value = getLogits(row, col);
        threadMax = max(threadMax, value);
      }
      if (tid < cols) {
        buf[tid] = threadMax;
      }
      workgroupBarrier();

      var reduceSize = min(cols, blockSize);
      for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
        reduceSize = currSize + (reduceSize & 1);
        if (tid < currSize) {
          buf[tid] = max(buf[tid], buf[tid + reduceSize]);
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowMaxShared = buf[0];
      }
      workgroupBarrier();

      var threadSum = 0.0;
      for (var col = tid; col < cols; col += blockSize) {
        let subExp = exp(getLogits(row, col) - rowMaxShared);
        threadSum += subExp;
      }
      buf[tid] = threadSum;
      workgroupBarrier();

      for (var currSize = blockSize >> 1;  currSize > 0; currSize = currSize >> 1) {
        if (tid < currSize) {
          buf[tid] = buf[tid] + buf[tid + currSize];
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowSumShared = buf[0];
      }
      workgroupBarrier();

      for (var col = tid; col < cols; col += blockSize) {
        let value = exp(getLogits(row, col) - rowMaxShared) / rowSumShared;
        setOutputAtCoords(row, col, value);
      }
  }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic29mdG1heF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zb2Z0bWF4X3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVqRCxNQUFNLE9BQU8sY0FBYztJQVF6QixZQUFZLFdBQXFCO1FBUGpDLGtCQUFhLEdBQUcsQ0FBQyxRQUFRLENBQUMsQ0FBQztRQVF6QixJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQyxDQUFFLGVBQWU7UUFDaEQsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzVDLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLEVBQUU7WUFDL0IsSUFBSSxDQUFDLGFBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7U0FDbEM7YUFBTTtZQUNMLElBQUksQ0FBQyxhQUFhLEdBQUcsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1NBQ2pDO1FBQ0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxTQUFTLENBQUM7SUFDN0IsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztzQ0FDaUIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7Ozt3QkFHbkMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7TUFDdkMsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0tBc0RkLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7ZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFNvZnRtYXhQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ2xvZ2l0cyddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuXG4gIGNvbnN0cnVjdG9yKG91dHB1dFNoYXBlOiBudW1iZXJbXSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZTsgIC8vIFtyb3dzLCBjb2xzXVxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IFt0aGlzLm91dHB1dFNoYXBlWzBdLCAxLCAxXTtcbiAgICBpZiAodGhpcy5vdXRwdXRTaGFwZVsxXSA+PSA0MDk2KSB7XG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSBbMjU2LCAxLCAxXTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy53b3JrZ3JvdXBTaXplID0gWzY0LCAxLCAxXTtcbiAgICB9XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnc29mdG1heCc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgIHZhcjx3b3JrZ3JvdXA+IGJ1ZiA6IGFycmF5PGYzMiwgJHt0aGlzLndvcmtncm91cFNpemVbMF19PjtcbiAgICB2YXI8d29ya2dyb3VwPiByb3dNYXhTaGFyZWQgOiBmMzI7XG4gICAgdmFyPHdvcmtncm91cD4gcm93U3VtU2hhcmVkIDogZjMyO1xuICAgIGNvbnN0IGJsb2NrU2l6ZSA9ICR7dGhpcy53b3JrZ3JvdXBTaXplWzBdfTtcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGxldCByb3cgPSBpbmRleCAvIGJsb2NrU2l6ZTtcbiAgICAgIGxldCB0aWQgPSBpMzIobG9jYWxJZC54KTtcbiAgICAgIGxldCBjb2xzID0gdW5pZm9ybXMub3V0U2hhcGVbMV07XG5cbiAgICAgIHZhciB0aHJlYWRNYXggPSAtMy40MDI4MjNlKzM4ZjtcbiAgICAgIGZvciAodmFyIGNvbCA9IHRpZDsgY29sIDwgY29sczsgY29sICs9IGJsb2NrU2l6ZSkge1xuICAgICAgICBsZXQgdmFsdWUgPSBnZXRMb2dpdHMocm93LCBjb2wpO1xuICAgICAgICB0aHJlYWRNYXggPSBtYXgodGhyZWFkTWF4LCB2YWx1ZSk7XG4gICAgICB9XG4gICAgICBpZiAodGlkIDwgY29scykge1xuICAgICAgICBidWZbdGlkXSA9IHRocmVhZE1heDtcbiAgICAgIH1cbiAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgICAgdmFyIHJlZHVjZVNpemUgPSBtaW4oY29scywgYmxvY2tTaXplKTtcbiAgICAgIGZvciAodmFyIGN1cnJTaXplID0gcmVkdWNlU2l6ZSA+PiAxOyAgY3VyclNpemUgPiAwOyBjdXJyU2l6ZSA9IHJlZHVjZVNpemUgPj4gMSkge1xuICAgICAgICByZWR1Y2VTaXplID0gY3VyclNpemUgKyAocmVkdWNlU2l6ZSAmIDEpO1xuICAgICAgICBpZiAodGlkIDwgY3VyclNpemUpIHtcbiAgICAgICAgICBidWZbdGlkXSA9IG1heChidWZbdGlkXSwgYnVmW3RpZCArIHJlZHVjZVNpemVdKTtcbiAgICAgICAgfVxuICAgICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG4gICAgICB9XG5cbiAgICAgIGlmICh0aWQgPT0gMCkge1xuICAgICAgICByb3dNYXhTaGFyZWQgPSBidWZbMF07XG4gICAgICB9XG4gICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgICAgIHZhciB0aHJlYWRTdW0gPSAwLjA7XG4gICAgICBmb3IgKHZhciBjb2wgPSB0aWQ7IGNvbCA8IGNvbHM7IGNvbCArPSBibG9ja1NpemUpIHtcbiAgICAgICAgbGV0IHN1YkV4cCA9IGV4cChnZXRMb2dpdHMocm93LCBjb2wpIC0gcm93TWF4U2hhcmVkKTtcbiAgICAgICAgdGhyZWFkU3VtICs9IHN1YkV4cDtcbiAgICAgIH1cbiAgICAgIGJ1Zlt0aWRdID0gdGhyZWFkU3VtO1xuICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICBmb3IgKHZhciBjdXJyU2l6ZSA9IGJsb2NrU2l6ZSA+PiAxOyAgY3VyclNpemUgPiAwOyBjdXJyU2l6ZSA9IGN1cnJTaXplID4+IDEpIHtcbiAgICAgICAgaWYgKHRpZCA8IGN1cnJTaXplKSB7XG4gICAgICAgICAgYnVmW3RpZF0gPSBidWZbdGlkXSArIGJ1Zlt0aWQgKyBjdXJyU2l6ZV07XG4gICAgICAgIH1cbiAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuICAgICAgfVxuXG4gICAgICBpZiAodGlkID09IDApIHtcbiAgICAgICAgcm93U3VtU2hhcmVkID0gYnVmWzBdO1xuICAgICAgfVxuICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICBmb3IgKHZhciBjb2wgPSB0aWQ7IGNvbCA8IGNvbHM7IGNvbCArPSBibG9ja1NpemUpIHtcbiAgICAgICAgbGV0IHZhbHVlID0gZXhwKGdldExvZ2l0cyhyb3csIGNvbCkgLSByb3dNYXhTaGFyZWQpIC8gcm93U3VtU2hhcmVkO1xuICAgICAgICBzZXRPdXRwdXRBdENvb3Jkcyhyb3csIGNvbCwgdmFsdWUpO1xuICAgICAgfVxuICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==