/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import { activationFnSnippet, biasActivationSnippet } from './activation_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch } from './webgpu_util';
export class DepthwiseConv2DNCHWSharedProgram {
    constructor(outputShape, filterHeight, filterWidth, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = `pads : vec2<i32>, inDims : vec2<i32>,`;
        this.workgroupSize = [16, 16, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [3], y: [2], z: [0, 1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivation = hasPreluActivation;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.shaderKey = `depthwiseNCHW_${this.activation}_${this.filterHeight}_${this.filterWidth}`;
    }
    getUserCode() {
        const filterSize = this.filterWidth * this.filterHeight;
        const flatWorkgroupSize = this.workgroupSize[0] * this.workgroupSize[1] * this.workgroupSize[2];
        const tileAHeight = this.workgroupSize[1] + this.filterHeight - 1;
        const tileAWidth = this.workgroupSize[0] + this.filterWidth - 1;
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, false, 4)}

      var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHeight}>;
      var<workgroup> mm_Bsub : array<array<f32, ${this.filterWidth}>, ${this.filterHeight}>;
      fn readX(batch : i32, channel : i32, row : i32, col : i32) -> f32 {
        var value = 0.0;
        if (row >=0 && row < uniforms.inDims[0] && col >=0 && col < uniforms.inDims[1])
        {
          value = getX(batch, channel, row, col);
        }
        return value;
      }

      ${main()} {
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.zw) - uniforms.pads;
        let channelMul = uniforms.wShape[3];
        let d1 = coords[1] / channelMul;
        let q = coords[1] % channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;

        let localRow = i32(localId.y);
        let localCol = i32(localId.x);

        // Load one tile of X into local memory.
        for (var inputRow = localRow; inputRow < ${tileAHeight}; inputRow = inputRow + ${this.workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${tileAWidth}; inputCol = inputCol + ${this.workgroupSize[0]}) {
            let rowOffset = inputRow - localRow;
            let colOffset = inputCol - localCol;
            mm_Asub[inputRow][inputCol] = readX(batch, d1, inputRowStart + rowOffset, inputColStart + colOffset);
          }
        }

        // Load one tile of W into local memory.
        var wIndex = i32(localIndex);
        ${filterSize < flatWorkgroupSize ?
            `if (wIndex < ${filterSize})` :
            `for(; wIndex < ${filterSize}; wIndex = wIndex + ${flatWorkgroupSize})`}

        {
          let wRow = wIndex / ${this.filterWidth};
          let wCol = wIndex % ${this.filterWidth};
          mm_Bsub[wRow][wCol] = getW(wRow, wCol, d1, q);
        }

        workgroupBarrier();

        var value = 0.0;
        for (var wR = 0; wR < ${this.filterHeight}; wR = wR + 1) {
          for (var wC = 0; wC < ${this.filterWidth}; wC = wC + 1) {
            let xVal = mm_Asub[localRow + wR][localCol + wC];
            let wVal = mm_Bsub[wR][wC];
            value = fma(xVal, wVal, value);
          }
        }
        ${biasActivationSnippet(this.addBias, this.activation)}
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZF9uY2h3X3NoYXJlZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9kZXB0aHdpc2VfY29udjJkX25jaHdfc2hhcmVkX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsbUJBQW1CLEVBQUUscUJBQXFCLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUM3RSxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFOUMsTUFBTSxPQUFPLGdDQUFnQztJQWMzQyxZQUNJLFdBQXFCLEVBQUUsWUFBb0IsRUFBRSxXQUFtQixFQUNoRSxPQUFPLEdBQUcsS0FBSyxFQUFFLGFBQXNDLElBQUksRUFDM0Qsa0JBQWtCLEdBQUcsS0FBSztRQVo5QixrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzNCLGFBQVEsR0FBRyx1Q0FBdUMsQ0FBQztRQUNuRCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFXcEQsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxFQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBQyxDQUFDO1FBQ2xELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRS9ELElBQUksT0FBTyxFQUFFO1lBQ1gsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7U0FDakM7UUFDRCxJQUFJLGtCQUFrQixFQUFFO1lBQ3RCLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUFDLENBQUM7U0FDbkQ7UUFFRCxJQUFJLENBQUMsT0FBTyxHQUFHLE9BQU8sQ0FBQztRQUN2QixJQUFJLENBQUMsVUFBVSxHQUFHLFVBQVUsQ0FBQztRQUM3QixJQUFJLENBQUMsa0JBQWtCLEdBQUcsa0JBQWtCLENBQUM7UUFDN0MsSUFBSSxDQUFDLFlBQVksR0FBRyxZQUFZLENBQUM7UUFDakMsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLFNBQVMsR0FBRyxpQkFBaUIsSUFBSSxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUMsWUFBWSxJQUNsRSxJQUFJLENBQUMsV0FBVyxFQUFFLENBQUM7SUFDekIsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFVBQVUsR0FBRyxJQUFJLENBQUMsV0FBVyxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUM7UUFDeEQsTUFBTSxpQkFBaUIsR0FDbkIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDMUUsTUFBTSxXQUFXLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsWUFBWSxHQUFHLENBQUMsQ0FBQztRQUNsRSxNQUFNLFVBQVUsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxXQUFXLEdBQUcsQ0FBQyxDQUFDO1FBRWhFLE1BQU0sUUFBUSxHQUFHO1FBQ2IsbUJBQW1CLENBQUMsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMsa0JBQWtCLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQzs7a0RBRTdCLFVBQVUsTUFBTSxXQUFXO2tEQUMzQixJQUFJLENBQUMsV0FBVyxNQUMxRCxJQUFJLENBQUMsWUFBWTs7Ozs7Ozs7OztRQVVqQixJQUFJLEVBQUU7Ozs7Ozs7Ozs7Ozs7OzttREFnQk4sV0FBVywyQkFBMkIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7cURBRTNELFVBQVUsMkJBQTJCLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDOzs7Ozs7Ozs7VUFVMUQsVUFBVSxHQUFHLGlCQUFpQixDQUFDLENBQUM7WUFDNUIsZ0JBQWdCLFVBQVUsR0FBRyxDQUFDLENBQUM7WUFDL0Isa0JBQWtCLFVBQVUsdUJBQ3hCLGlCQUFpQixHQUFHOzs7Z0NBR0osSUFBSSxDQUFDLFdBQVc7Z0NBQ2hCLElBQUksQ0FBQyxXQUFXOzs7Ozs7O2dDQU9oQixJQUFJLENBQUMsWUFBWTtrQ0FDZixJQUFJLENBQUMsV0FBVzs7Ozs7O1VBTXhDLHFCQUFxQixDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLFVBQVUsQ0FBQzs7Ozs7S0FLekQsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7YWN0aXZhdGlvbkZuU25pcHBldCwgYmlhc0FjdGl2YXRpb25TbmlwcGV0fSBmcm9tICcuL2FjdGl2YXRpb25fdXRpbCc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBEZXB0aHdpc2VDb252MkROQ0hXU2hhcmVkUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXSwgejogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ1cnXTtcbiAgdW5pZm9ybXMgPSBgcGFkcyA6IHZlYzI8aTMyPiwgaW5EaW1zIDogdmVjMjxpMzI+LGA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxNiwgMTYsIDFdO1xuICBhZGRCaWFzOiBib29sZWFuO1xuICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbjtcbiAgaGFzUHJlbHVBY3RpdmF0aW9uOiBib29sZWFuO1xuICBmaWx0ZXJIZWlnaHQ6IG51bWJlcjtcbiAgZmlsdGVyV2lkdGg6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIG91dHB1dFNoYXBlOiBudW1iZXJbXSwgZmlsdGVySGVpZ2h0OiBudW1iZXIsIGZpbHRlcldpZHRoOiBudW1iZXIsXG4gICAgICBhZGRCaWFzID0gZmFsc2UsIGFjdGl2YXRpb246IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uID0gbnVsbCxcbiAgICAgIGhhc1ByZWx1QWN0aXZhdGlvbiA9IGZhbHNlKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSB7eDogWzNdLCB5OiBbMl0sIHo6IFswLCAxXX07XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIGlmIChhZGRCaWFzKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgnYmlhcycpO1xuICAgIH1cbiAgICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgncHJlbHVBY3RpdmF0aW9uV2VpZ2h0cycpO1xuICAgIH1cblxuICAgIHRoaXMuYWRkQmlhcyA9IGFkZEJpYXM7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gYWN0aXZhdGlvbjtcbiAgICB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbiA9IGhhc1ByZWx1QWN0aXZhdGlvbjtcbiAgICB0aGlzLmZpbHRlckhlaWdodCA9IGZpbHRlckhlaWdodDtcbiAgICB0aGlzLmZpbHRlcldpZHRoID0gZmlsdGVyV2lkdGg7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgZGVwdGh3aXNlTkNIV18ke3RoaXMuYWN0aXZhdGlvbn1fJHt0aGlzLmZpbHRlckhlaWdodH1fJHtcbiAgICAgICAgdGhpcy5maWx0ZXJXaWR0aH1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCBmaWx0ZXJTaXplID0gdGhpcy5maWx0ZXJXaWR0aCAqIHRoaXMuZmlsdGVySGVpZ2h0O1xuICAgIGNvbnN0IGZsYXRXb3JrZ3JvdXBTaXplID1cbiAgICAgICAgdGhpcy53b3JrZ3JvdXBTaXplWzBdICogdGhpcy53b3JrZ3JvdXBTaXplWzFdICogdGhpcy53b3JrZ3JvdXBTaXplWzJdO1xuICAgIGNvbnN0IHRpbGVBSGVpZ2h0ID0gdGhpcy53b3JrZ3JvdXBTaXplWzFdICsgdGhpcy5maWx0ZXJIZWlnaHQgLSAxO1xuICAgIGNvbnN0IHRpbGVBV2lkdGggPSB0aGlzLndvcmtncm91cFNpemVbMF0gKyB0aGlzLmZpbHRlcldpZHRoIC0gMTtcblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHthY3RpdmF0aW9uRm5TbmlwcGV0KHRoaXMuYWN0aXZhdGlvbiwgdGhpcy5oYXNQcmVsdUFjdGl2YXRpb24sIGZhbHNlLCA0KX1cblxuICAgICAgdmFyPHdvcmtncm91cD4gbW1fQXN1YiA6IGFycmF5PGFycmF5PGYzMiwgJHt0aWxlQVdpZHRofT4sICR7dGlsZUFIZWlnaHR9PjtcbiAgICAgIHZhcjx3b3JrZ3JvdXA+IG1tX0JzdWIgOiBhcnJheTxhcnJheTxmMzIsICR7dGhpcy5maWx0ZXJXaWR0aH0+LCAke1xuICAgICAgICB0aGlzLmZpbHRlckhlaWdodH0+O1xuICAgICAgZm4gcmVhZFgoYmF0Y2ggOiBpMzIsIGNoYW5uZWwgOiBpMzIsIHJvdyA6IGkzMiwgY29sIDogaTMyKSAtPiBmMzIge1xuICAgICAgICB2YXIgdmFsdWUgPSAwLjA7XG4gICAgICAgIGlmIChyb3cgPj0wICYmIHJvdyA8IHVuaWZvcm1zLmluRGltc1swXSAmJiBjb2wgPj0wICYmIGNvbCA8IHVuaWZvcm1zLmluRGltc1sxXSlcbiAgICAgICAge1xuICAgICAgICAgIHZhbHVlID0gZ2V0WChiYXRjaCwgY2hhbm5lbCwgcm93LCBjb2wpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgIH1cblxuICAgICAgJHttYWluKCl9IHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgIGxldCB4UkNDb3JuZXIgPSB2ZWMyPGkzMj4oY29vcmRzLnp3KSAtIHVuaWZvcm1zLnBhZHM7XG4gICAgICAgIGxldCBjaGFubmVsTXVsID0gdW5pZm9ybXMud1NoYXBlWzNdO1xuICAgICAgICBsZXQgZDEgPSBjb29yZHNbMV0gLyBjaGFubmVsTXVsO1xuICAgICAgICBsZXQgcSA9IGNvb3Jkc1sxXSAlIGNoYW5uZWxNdWw7XG5cbiAgICAgICAgbGV0IGlucHV0Um93U3RhcnQgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgbGV0IGlucHV0Q29sU3RhcnQgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICBsZXQgbG9jYWxSb3cgPSBpMzIobG9jYWxJZC55KTtcbiAgICAgICAgbGV0IGxvY2FsQ29sID0gaTMyKGxvY2FsSWQueCk7XG5cbiAgICAgICAgLy8gTG9hZCBvbmUgdGlsZSBvZiBYIGludG8gbG9jYWwgbWVtb3J5LlxuICAgICAgICBmb3IgKHZhciBpbnB1dFJvdyA9IGxvY2FsUm93OyBpbnB1dFJvdyA8ICR7XG4gICAgICAgIHRpbGVBSGVpZ2h0fTsgaW5wdXRSb3cgPSBpbnB1dFJvdyArICR7dGhpcy53b3JrZ3JvdXBTaXplWzFdfSkge1xuICAgICAgICAgIGZvciAodmFyIGlucHV0Q29sID0gbG9jYWxDb2w7IGlucHV0Q29sIDwgJHtcbiAgICAgICAgdGlsZUFXaWR0aH07IGlucHV0Q29sID0gaW5wdXRDb2wgKyAke3RoaXMud29ya2dyb3VwU2l6ZVswXX0pIHtcbiAgICAgICAgICAgIGxldCByb3dPZmZzZXQgPSBpbnB1dFJvdyAtIGxvY2FsUm93O1xuICAgICAgICAgICAgbGV0IGNvbE9mZnNldCA9IGlucHV0Q29sIC0gbG9jYWxDb2w7XG4gICAgICAgICAgICBtbV9Bc3ViW2lucHV0Um93XVtpbnB1dENvbF0gPSByZWFkWChiYXRjaCwgZDEsIGlucHV0Um93U3RhcnQgKyByb3dPZmZzZXQsIGlucHV0Q29sU3RhcnQgKyBjb2xPZmZzZXQpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIC8vIExvYWQgb25lIHRpbGUgb2YgVyBpbnRvIGxvY2FsIG1lbW9yeS5cbiAgICAgICAgdmFyIHdJbmRleCA9IGkzMihsb2NhbEluZGV4KTtcbiAgICAgICAgJHtcbiAgICAgICAgZmlsdGVyU2l6ZSA8IGZsYXRXb3JrZ3JvdXBTaXplID9cbiAgICAgICAgICAgIGBpZiAod0luZGV4IDwgJHtmaWx0ZXJTaXplfSlgIDpcbiAgICAgICAgICAgIGBmb3IoOyB3SW5kZXggPCAke2ZpbHRlclNpemV9OyB3SW5kZXggPSB3SW5kZXggKyAke1xuICAgICAgICAgICAgICAgIGZsYXRXb3JrZ3JvdXBTaXplfSlgfVxuXG4gICAgICAgIHtcbiAgICAgICAgICBsZXQgd1JvdyA9IHdJbmRleCAvICR7dGhpcy5maWx0ZXJXaWR0aH07XG4gICAgICAgICAgbGV0IHdDb2wgPSB3SW5kZXggJSAke3RoaXMuZmlsdGVyV2lkdGh9O1xuICAgICAgICAgIG1tX0JzdWJbd1Jvd11bd0NvbF0gPSBnZXRXKHdSb3csIHdDb2wsIGQxLCBxKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgICAgICB2YXIgdmFsdWUgPSAwLjA7XG4gICAgICAgIGZvciAodmFyIHdSID0gMDsgd1IgPCAke3RoaXMuZmlsdGVySGVpZ2h0fTsgd1IgPSB3UiArIDEpIHtcbiAgICAgICAgICBmb3IgKHZhciB3QyA9IDA7IHdDIDwgJHt0aGlzLmZpbHRlcldpZHRofTsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICAgIGxldCB4VmFsID0gbW1fQXN1Yltsb2NhbFJvdyArIHdSXVtsb2NhbENvbCArIHdDXTtcbiAgICAgICAgICAgIGxldCB3VmFsID0gbW1fQnN1Ylt3Ul1bd0NdO1xuICAgICAgICAgICAgdmFsdWUgPSBmbWEoeFZhbCwgd1ZhbCwgdmFsdWUpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICAke2JpYXNBY3RpdmF0aW9uU25pcHBldCh0aGlzLmFkZEJpYXMsIHRoaXMuYWN0aXZhdGlvbil9XG4gICAgICAgIGlmIChjb29yZHNJbkJvdW5kczREKGNvb3JkcywgdW5pZm9ybXMub3V0U2hhcGUpKSB7XG4gICAgICAgICAgc2V0T3V0cHV0QXRDb29yZHMoY29vcmRzWzBdLCBjb29yZHNbMV0sIGNvb3Jkc1syXSwgY29vcmRzWzNdLCB2YWx1ZSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19