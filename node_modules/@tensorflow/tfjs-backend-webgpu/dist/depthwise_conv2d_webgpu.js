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
import { activationFnSnippet, biasActivationSnippet } from './activation_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class DepthwiseConv2DProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = `pads : vec2<i32>, inDims : vec2<i32>, filterHeight : i32,
      filterWidth : i32, strides : vec2<i32>, dilations : vec2<i32>,`;
        // This is an experimental value.
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        this.convInfo = convInfo;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivation = hasPreluActivation;
        this.shaderKey = `depthwise_${this.activation}_${this.isChannelsLast}`;
    }
    getUserCode() {
        const getXSnippet = this.isChannelsLast ? 'getX(batch, xR, xC, d1);' :
            'getX(batch, d1, xR, xC);';
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, false, 4)}

      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let batch = coords[0];
          let xRCCorner = vec2<i32>(coords.${this.isChannelsLast ? 'yz' : 'zw'}) * uniforms.strides - uniforms.pads;
          let d2 = coords[${this.isChannelsLast ? 3 : 1}];
          let channelMul = uniforms.wShape[3];
          let d1 = d2 / channelMul;
          let q = d2 % channelMul;

          let inputRowStart = xRCCorner.x;
          let inputColStart = xRCCorner.y;
          let inputRowEnd = inputRowStart + uniforms.filterHeight *
              uniforms.dilations[0];
          let inputColEnd = inputColStart + uniforms.filterWidth *
              uniforms.dilations[1];

          // Convolve x(?, ?, d1)|x(d1, ?, ?) with w(:, :, d1, q) to get
          // y(yR, yC, d2)|y(d2, yR, yC). ? = to be determined. : = across all
          // values in that axis. x(?, ?, d1) and y(yR, yC, d2) is for NHWC.
          // x(d1, ?, ?) and y(d2, yR, yC) is for NCHW.
          var value = 0.0;

          // Extract if checking out of for loop for performance.
          if (inputRowStart >= 0 && inputColStart >= 0 &&
            inputRowEnd < uniforms.inDims[0] &&
                inputColEnd < uniforms.inDims[1]) {
              for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
                let xR = inputRowStart + wR * uniforms.dilations[0];

                for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                  let xC = inputColStart + wC * uniforms.dilations[1];

                  let xVal = ${getXSnippet};
                  let wVal = getW(wR, wC, d1, q);
                  value = value + xVal * wVal;
                }
              }
            } else {
              for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
                let xR = inputRowStart + wR * uniforms.dilations[0];

                if (xR < 0 || xR >= uniforms.inDims[0]) {
                  continue;
                }

                for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                  let xC = inputColStart + wC * uniforms.dilations[1];

                  if (xC < 0 || xC >= uniforms.inDims[1]) {
                    continue;
                  }

                  let xVal = ${getXSnippet};
                  let wVal = getW(wR, wC, d1, q);
                  value = value + xVal * wVal;
                }
              }
            }
            ${biasActivationSnippet(this.addBias, this.activation)}
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9kZXB0aHdpc2VfY29udjJkX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsbUJBQW1CLEVBQUUscUJBQXFCLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUM3RSxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLHNCQUFzQjtJQWlCakMsWUFDSSxRQUFpQyxFQUFFLE9BQU8sR0FBRyxLQUFLLEVBQ2xELGFBQXNDLElBQUksRUFBRSxrQkFBa0IsR0FBRyxLQUFLO1FBZDFFLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDM0IsYUFBUSxHQUFHO3FFQUN3RCxDQUFDO1FBQ3BFLGlDQUFpQztRQUNqQyxrQkFBYSxHQUE2QixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFNdEQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUtWLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQztRQUNyQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsY0FBYyxHQUFHLFFBQVEsQ0FBQyxVQUFVLEtBQUssY0FBYyxDQUFDO1FBRTdELElBQUksT0FBTyxFQUFFO1lBQ1gsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7U0FDakM7UUFDRCxJQUFJLGtCQUFrQixFQUFFO1lBQ3RCLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUFDLENBQUM7U0FDbkQ7UUFFRCxJQUFJLENBQUMsUUFBUSxHQUFHLFFBQVEsQ0FBQztRQUN6QixJQUFJLENBQUMsT0FBTyxHQUFHLE9BQU8sQ0FBQztRQUN2QixJQUFJLENBQUMsVUFBVSxHQUFHLFVBQVUsQ0FBQztRQUM3QixJQUFJLENBQUMsa0JBQWtCLEdBQUcsa0JBQWtCLENBQUM7UUFDN0MsSUFBSSxDQUFDLFNBQVMsR0FBRyxhQUFhLElBQUksQ0FBQyxVQUFVLElBQUksSUFBSSxDQUFDLGNBQWMsRUFBRSxDQUFDO0lBQ3pFLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxXQUFXLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsMEJBQTBCLENBQUMsQ0FBQztZQUM1QiwwQkFBMEIsQ0FBQztRQUVyRSxNQUFNLFFBQVEsR0FBRztRQUNiLG1CQUFtQixDQUFDLElBQUksQ0FBQyxVQUFVLEVBQUUsSUFBSSxDQUFDLGtCQUFrQixFQUFFLEtBQUssRUFBRSxDQUFDLENBQUM7O1FBRXZFLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7NkNBS2IsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJOzRCQUNiLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OzsrQkE0QnhCLFdBQVc7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OytCQW9CWCxXQUFXOzs7Ozs7Y0FNNUIscUJBQXFCLENBQUMsSUFBSSxDQUFDLE9BQU8sRUFBRSxJQUFJLENBQUMsVUFBVSxDQUFDOzs7O0tBSTdELENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2FjdGl2YXRpb25GblNuaXBwZXQsIGJpYXNBY3RpdmF0aW9uU25pcHBldH0gZnJvbSAnLi9hY3RpdmF0aW9uX3V0aWwnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBEZXB0aHdpc2VDb252MkRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW10sIHk/OiBudW1iZXJbXSwgej86IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdXJ107XG4gIHVuaWZvcm1zID0gYHBhZHMgOiB2ZWMyPGkzMj4sIGluRGltcyA6IHZlYzI8aTMyPiwgZmlsdGVySGVpZ2h0IDogaTMyLFxuICAgICAgZmlsdGVyV2lkdGggOiBpMzIsIHN0cmlkZXMgOiB2ZWMyPGkzMj4sIGRpbGF0aW9ucyA6IHZlYzI8aTMyPixgO1xuICAvLyBUaGlzIGlzIGFuIGV4cGVyaW1lbnRhbCB2YWx1ZS5cbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzI1NiwgMSwgMV07XG4gIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbztcbiAgYWRkQmlhczogYm9vbGVhbjtcbiAgYWN0aXZhdGlvbjogYmFja2VuZF91dGlsLkFjdGl2YXRpb247XG4gIGhhc1ByZWx1QWN0aXZhdGlvbjogYm9vbGVhbjtcbiAgaXNDaGFubmVsc0xhc3Q6IGJvb2xlYW47XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvLCBhZGRCaWFzID0gZmFsc2UsXG4gICAgICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbiA9IG51bGwsIGhhc1ByZWx1QWN0aXZhdGlvbiA9IGZhbHNlKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLm91dFNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcbiAgICB0aGlzLmlzQ2hhbm5lbHNMYXN0ID0gY29udkluZm8uZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCc7XG5cbiAgICBpZiAoYWRkQmlhcykge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ2JpYXMnKTtcbiAgICB9XG4gICAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbikge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ3ByZWx1QWN0aXZhdGlvbldlaWdodHMnKTtcbiAgICB9XG5cbiAgICB0aGlzLmNvbnZJbmZvID0gY29udkluZm87XG4gICAgdGhpcy5hZGRCaWFzID0gYWRkQmlhcztcbiAgICB0aGlzLmFjdGl2YXRpb24gPSBhY3RpdmF0aW9uO1xuICAgIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uID0gaGFzUHJlbHVBY3RpdmF0aW9uO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGRlcHRod2lzZV8ke3RoaXMuYWN0aXZhdGlvbn1fJHt0aGlzLmlzQ2hhbm5lbHNMYXN0fWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IGdldFhTbmlwcGV0ID0gdGhpcy5pc0NoYW5uZWxzTGFzdCA/ICdnZXRYKGJhdGNoLCB4UiwgeEMsIGQxKTsnIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnZ2V0WChiYXRjaCwgZDEsIHhSLCB4Qyk7JztcblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHthY3RpdmF0aW9uRm5TbmlwcGV0KHRoaXMuYWN0aXZhdGlvbiwgdGhpcy5oYXNQcmVsdUFjdGl2YXRpb24sIGZhbHNlLCA0KX1cblxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzWzBdO1xuICAgICAgICAgIGxldCB4UkNDb3JuZXIgPSB2ZWMyPGkzMj4oY29vcmRzLiR7XG4gICAgICAgIHRoaXMuaXNDaGFubmVsc0xhc3QgPyAneXonIDogJ3p3J30pICogdW5pZm9ybXMuc3RyaWRlcyAtIHVuaWZvcm1zLnBhZHM7XG4gICAgICAgICAgbGV0IGQyID0gY29vcmRzWyR7dGhpcy5pc0NoYW5uZWxzTGFzdCA/IDMgOiAxfV07XG4gICAgICAgICAgbGV0IGNoYW5uZWxNdWwgPSB1bmlmb3Jtcy53U2hhcGVbM107XG4gICAgICAgICAgbGV0IGQxID0gZDIgLyBjaGFubmVsTXVsO1xuICAgICAgICAgIGxldCBxID0gZDIgJSBjaGFubmVsTXVsO1xuXG4gICAgICAgICAgbGV0IGlucHV0Um93U3RhcnQgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgICBsZXQgaW5wdXRDb2xTdGFydCA9IHhSQ0Nvcm5lci55O1xuICAgICAgICAgIGxldCBpbnB1dFJvd0VuZCA9IGlucHV0Um93U3RhcnQgKyB1bmlmb3Jtcy5maWx0ZXJIZWlnaHQgKlxuICAgICAgICAgICAgICB1bmlmb3Jtcy5kaWxhdGlvbnNbMF07XG4gICAgICAgICAgbGV0IGlucHV0Q29sRW5kID0gaW5wdXRDb2xTdGFydCArIHVuaWZvcm1zLmZpbHRlcldpZHRoICpcbiAgICAgICAgICAgICAgdW5pZm9ybXMuZGlsYXRpb25zWzFdO1xuXG4gICAgICAgICAgLy8gQ29udm9sdmUgeCg/LCA/LCBkMSl8eChkMSwgPywgPykgd2l0aCB3KDosIDosIGQxLCBxKSB0byBnZXRcbiAgICAgICAgICAvLyB5KHlSLCB5QywgZDIpfHkoZDIsIHlSLCB5QykuID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbFxuICAgICAgICAgIC8vIHZhbHVlcyBpbiB0aGF0IGF4aXMuIHgoPywgPywgZDEpIGFuZCB5KHlSLCB5QywgZDIpIGlzIGZvciBOSFdDLlxuICAgICAgICAgIC8vIHgoZDEsID8sID8pIGFuZCB5KGQyLCB5UiwgeUMpIGlzIGZvciBOQ0hXLlxuICAgICAgICAgIHZhciB2YWx1ZSA9IDAuMDtcblxuICAgICAgICAgIC8vIEV4dHJhY3QgaWYgY2hlY2tpbmcgb3V0IG9mIGZvciBsb29wIGZvciBwZXJmb3JtYW5jZS5cbiAgICAgICAgICBpZiAoaW5wdXRSb3dTdGFydCA+PSAwICYmIGlucHV0Q29sU3RhcnQgPj0gMCAmJlxuICAgICAgICAgICAgaW5wdXRSb3dFbmQgPCB1bmlmb3Jtcy5pbkRpbXNbMF0gJiZcbiAgICAgICAgICAgICAgICBpbnB1dENvbEVuZCA8IHVuaWZvcm1zLmluRGltc1sxXSkge1xuICAgICAgICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVySGVpZ2h0OyB3UiA9IHdSICsgMSkge1xuICAgICAgICAgICAgICAgIGxldCB4UiA9IGlucHV0Um93U3RhcnQgKyB3UiAqIHVuaWZvcm1zLmRpbGF0aW9uc1swXTtcblxuICAgICAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJXaWR0aDsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICAgICAgICAgIGxldCB4QyA9IGlucHV0Q29sU3RhcnQgKyB3QyAqIHVuaWZvcm1zLmRpbGF0aW9uc1sxXTtcblxuICAgICAgICAgICAgICAgICAgbGV0IHhWYWwgPSAke2dldFhTbmlwcGV0fTtcbiAgICAgICAgICAgICAgICAgIGxldCB3VmFsID0gZ2V0Vyh3Uiwgd0MsIGQxLCBxKTtcbiAgICAgICAgICAgICAgICAgIHZhbHVlID0gdmFsdWUgKyB4VmFsICogd1ZhbDtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGZvciAodmFyIHdSID0gMDsgd1IgPCB1bmlmb3Jtcy5maWx0ZXJIZWlnaHQ7IHdSID0gd1IgKyAxKSB7XG4gICAgICAgICAgICAgICAgbGV0IHhSID0gaW5wdXRSb3dTdGFydCArIHdSICogdW5pZm9ybXMuZGlsYXRpb25zWzBdO1xuXG4gICAgICAgICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSB1bmlmb3Jtcy5pbkRpbXNbMF0pIHtcbiAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJXaWR0aDsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICAgICAgICAgIGxldCB4QyA9IGlucHV0Q29sU3RhcnQgKyB3QyAqIHVuaWZvcm1zLmRpbGF0aW9uc1sxXTtcblxuICAgICAgICAgICAgICAgICAgaWYgKHhDIDwgMCB8fCB4QyA+PSB1bmlmb3Jtcy5pbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICAgIGxldCB4VmFsID0gJHtnZXRYU25pcHBldH07XG4gICAgICAgICAgICAgICAgICBsZXQgd1ZhbCA9IGdldFcod1IsIHdDLCBkMSwgcSk7XG4gICAgICAgICAgICAgICAgICB2YWx1ZSA9IHZhbHVlICsgeFZhbCAqIHdWYWw7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICAke2JpYXNBY3RpdmF0aW9uU25pcHBldCh0aGlzLmFkZEJpYXMsIHRoaXMuYWN0aXZhdGlvbil9XG4gICAgICAgICAgc2V0T3V0cHV0QXRDb29yZHMoY29vcmRzWzBdLCBjb29yZHNbMV0sIGNvb3Jkc1syXSwgY29vcmRzWzNdLCB2YWx1ZSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19