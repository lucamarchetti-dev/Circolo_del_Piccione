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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class Conv2DDerInputProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'W'];
        this.uniforms = 'filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, outBackprop : vec4<i32>,';
        this.workgroupSize = [64, 1, 1];
        this.size = false;
        this.isVec4 = false;
        this.workPerThread = 1;
        this.outputShape = convInfo.inShape;
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        this.isVec4 = this.isChannelsLast && convInfo.outChannels % 4 === 0 &&
            convInfo.inChannels % 4 === 0;
        if (this.isVec4) {
            // TODO: Expand to any value.
            this.workPerThread = 2;
            this.outputComponent = 4;
            this.workgroupSize = [4, 4, 4];
            this.dispatchLayout = { x: [3], y: [2], z: [0, 1] };
            this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [4, this.workPerThread, 1]);
        }
        else {
            this.size = true;
            this.workPerThread = 1;
            this.workgroupSize = [64, 1, 1];
            this.dispatchLayout = flatDispatchLayout(this.outputShape);
            this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        }
        this.shaderKey = `conv2DDerInput_${this.isChannelsLast}_${this.isVec4}_${this.workPerThread}`;
    }
    getUserCode() {
        const rowDim = this.isChannelsLast ? 1 : 2;
        const colDim = this.isChannelsLast ? 2 : 3;
        const channelDim = this.isChannelsLast ? 3 : 1;
        const vec4Snippet = `
    ${main()} {
      let batch = i32(globalId.z) / uniforms.outShape[1];
      let r = i32(globalId.z) % uniforms.outShape[1];
      let c = i32(globalId.y) * ${this.workPerThread};
      let d1 = i32(globalId.x) * 4;

      let dyCorner = vec2<i32>(r, c) - uniforms.pads;

      // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
      // ? = to be determined. : = across all values in that axis.
      var dotProd: array<vec4<f32>, ${this.workPerThread}>;
      for (var i = 0; i < ${this.workPerThread}; i++) {
        dotProd[i] = vec4<f32>(0.0);
      }
      for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + 1) {
        let dyR = f32(dyCorner.x + wR) / f32(uniforms.strides.x);
        let wRPerm = uniforms.filterDims.x - 1 - wR;
        if (dyR < 0.0 || dyR >= f32(uniforms.outBackprop[1]) ||
            fract(dyR) > 0.0) {
          continue;
        }
        let idyR = i32(dyR);

        for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + 1) {
          let dyC = f32(dyCorner.y + wC) / f32(uniforms.strides.y);
          let dyC2 = f32(dyCorner.y + 1 + wC) / f32(uniforms.strides.y);
          let wCPerm = uniforms.filterDims.y - 1 - wC;
          var bDyCVal = true;
          var bDyCVal2 = true;
          if (dyC < 0.0 || dyC >= f32(uniforms.outBackprop[2]) ||
              fract(dyC) > 0.0) {
            bDyCVal = false;
          }
          if (dyC2 < 0.0 || dyC2 >= f32(uniforms.outBackprop[2]) ||
              fract(dyC2) > 0.0) {
            bDyCVal2 = false;
          }

          let idyC = i32(dyC);
          let idyC2 = i32(dyC2);
          if (bDyCVal && bDyCVal2) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[0] = dotProd[0] + tmpval;
              xValue = getDy(batch, idyR, idyC2, d2);
              dotProd[1] = dotProd[1] + vec4<f32>(dot(xValue, wValue0),
                                                  dot(xValue, wValue1),
                                                  dot(xValue, wValue2),
                                                  dot(xValue, wValue3));
            }
          } else if (bDyCVal) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[0] = dotProd[0] + tmpval;
            }
          } else if (bDyCVal2) {
            let d2Length = uniforms.outBackprop[3];
            for (var d2 = 0; d2 < d2Length; d2 = d2 + 4) {
              let wValue0 = getW(wRPerm, wCPerm, d1, d2);
              let wValue1 = getW(wRPerm, wCPerm, d1 + 1, d2);
              let wValue2 = getW(wRPerm, wCPerm, d1 + 2, d2);
              let wValue3 = getW(wRPerm, wCPerm, d1 + 3, d2);
              var xValue =  getDy(batch, idyR, idyC2, d2);
              let tmpval = vec4<f32>(dot(xValue, wValue0),
                                     dot(xValue, wValue1),
                                     dot(xValue, wValue2),
                                     dot(xValue, wValue3));
              dotProd[1] = dotProd[1] + tmpval;
            }
          }
        }
      }

      for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
        let coords = vec4<i32>(batch, r, c + i, d1);
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
        }
      }
    }
    `;
        return this.isVec4 ?
            `
    ${vec4Snippet}
    ` :
            `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[${channelDim}];

        let dyCorner = vec2<i32>(coords[${rowDim}], coords[${colDim}]) - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + 1) {
          let dyR = (f32(dyRCorner) + f32(wR)) / f32(uniforms.strides.x);
          let wRPerm = uniforms.filterDims.x - 1 - wR;
          if (dyR < 0.0 || dyR >= f32(uniforms.outBackprop[1]) || fract(dyR) > 0.0 ||
              wRPerm < 0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + 1) {
            let dyC = (f32(dyCCorner) + f32(wC)) / f32(uniforms.strides.y);
            let wCPerm = uniforms.filterDims.y - 1 - wC;
            if (dyC < 0.0 || dyC >= f32(uniforms.outBackprop[2]) ||
                fract(dyC) > 0.0 || wCPerm < 0) {
              continue;
            }
            let idyC = i32(dyC);

            for (var d2 = 0; d2 < uniforms.outBackprop[3]; d2 = d2 + 1) {
              let xValue = ${this.isChannelsLast ? 'getDy(batch, idyR, idyC, d2)' :
                'getDy(batch, d2, idyR, idyC)'};
              let wValue = getW(wRPerm, wCPerm, d1, d2);
              dotProd = dotProd + xValue * wValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
    }
}
export class Conv2DDerFilterProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'dy'];
        this.uniforms = 'pads : vec2<i32>, strides : vec2<i32>, batchSize : i32, outHeight : i32, outWidth : i32, inHeight : i32, inWidth : i32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.filterShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        this.shaderKey = `conv2DDerFilter_${this.isChannelsLast}`;
    }
    getUserCode() {
        return `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let d2 = coords[3];

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b = b + 1) {
          for (var yR = 0; yR < uniforms.outHeight; yR = yR + 1) {
            let xR = wR + yR * uniforms.strides[0] - uniforms.pads[0];
            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC = yC + 1) {
              let xC = wC + yC * uniforms.strides[1] - uniforms.pads[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              if (${this.isChannelsLast}) {
                let dyValue = getDy(b, yR, yC, d2);
                let xValue = getX(b, xR, xC, d1);
                dotProd = dotProd + xValue * dyValue;
              } else {
                let dyValue = getDy(b, d2, yR, yC);
                let xValue = getX(b, d1, xR, xC);
                dotProd = dotProd + xValue * dyValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
    }
}
export class Conv3DDerFilterProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'dy'];
        this.uniforms = `pads : vec3<i32>, strides : vec3<i32>, batchSize : i32, outDepth : i32,
       outHeight : i32, outWidth : i32, inDepth : i32, inHeight : i32, inWidth : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.filterShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `conv3DDerFilter`;
    }
    getUserCode() {
        return `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wF = coords.x;
        let wR = coords.y;
        let wC = coords.z;
        let d1 = coords.w;
        let d2 = coords.u;

        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b++) {
          for (var yF = 0; yF < uniforms.outDepth; yF++) {
            let xF = wF + yF * uniforms.strides[0] - uniforms.pads[0];
            if (xF < 0 || xF >= uniforms.inDepth) {
              continue;
            }

            for (var yR = 0; yR < uniforms.outHeight; yR++) {
              let xR = wR + yR * uniforms.strides[1] - uniforms.pads[1];
              if (xR < 0 || xR >= uniforms.inHeight) {
                continue;
              }

              for (var yC = 0; yC < uniforms.outWidth; yC++) {
                let xC = wC + yC * uniforms.strides[2] - uniforms.pads[2];
                if (xC < 0 || xC >= uniforms.inWidth) {
                  continue;
                }

                let dyValue = getDy(b, yF, yR, yC, d2);
                let xValue = getX(b, xF, xR, xC, d1);
                dotProd += xValue * dyValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
    }
}
export class Conv3DDerInputProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'W'];
        this.uniforms = `filterDims : vec3<i32>, pads : vec3<i32>, strides : vec3<i32>,
      outDepth : i32, outHeight : i32, outWidth : i32, outChannels : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `conv3DDerInput`;
    }
    getUserCode() {
        return `
    ${main('index')} {
      if(index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let d1 = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyFCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        var dotProd = 0.0;
        for (var wF = 0; wF < uniforms.filterDims[0]; wF++) {
          let dyF = f32(dyFCorner + wF) / f32(uniforms.strides[0]);
          if (dyF < 0.0 || dyF >= f32(uniforms.outDepth) || fract(dyF) > 0.0) {
            continue;
          }
          let idyF = i32(dyF);

          let wFPerm = uniforms.filterDims[0] - 1 - wF;

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            let wRPerm = uniforms.filterDims[1] - 1 - wR;

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let wCPerm = uniforms.filterDims[2] - 1 - wC;

              for (var d2 = 0; d2 < uniforms.outChannels; d2++) {
                let xValue = getDy(batch, idyF, idyR, idyC, d2);
                let wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
  `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udl9iYWNrcHJvcF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9jb252X2JhY2twcm9wX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLHFCQUFxQjtJQWVoQyxZQUFZLFFBQWlDO1FBZDdDLGtCQUFhLEdBQUcsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDNUIsYUFBUSxHQUNKLHlGQUF5RixDQUFDO1FBSzlGLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsS0FBSyxDQUFDO1FBQ2IsV0FBTSxHQUFHLEtBQUssQ0FBQztRQUNmLGtCQUFhLEdBQUcsQ0FBQyxDQUFDO1FBSWhCLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQztRQUNwQyxJQUFJLENBQUMsY0FBYyxHQUFHLFFBQVEsQ0FBQyxVQUFVLEtBQUssY0FBYyxDQUFDO1FBQzdELElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLGNBQWMsSUFBSSxRQUFRLENBQUMsV0FBVyxHQUFHLENBQUMsS0FBSyxDQUFDO1lBQy9ELFFBQVEsQ0FBQyxVQUFVLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNsQyxJQUFJLElBQUksQ0FBQyxNQUFNLEVBQUU7WUFDZiw2QkFBNkI7WUFDN0IsSUFBSSxDQUFDLGFBQWEsR0FBRyxDQUFDLENBQUM7WUFDdkIsSUFBSSxDQUFDLGVBQWUsR0FBRyxDQUFDLENBQUM7WUFDekIsSUFBSSxD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Ozs7Ozs7Ozs7Ozs7Ozs7O0dBdUNoQixDQUFDO0lBQ0YsQ0FBQztDQUNGO0FBRUQsTUFBTSxPQUFPLHFCQUFxQjtJQVdoQyxZQUFZLFFBQWlDO1FBVjdDLGtCQUFhLEdBQUcsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDNUIsYUFBUSxHQUFHOzBFQUM2RCxDQUFDO1FBS3pFLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDO1FBQ3BDLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxTQUFTLEdBQUcsZ0JBQWdCLENBQUM7SUFDcEMsQ0FBQztJQUVELFdBQVc7UUFDVCxPQUFPO01BQ0wsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9EaEIsQ0FBQztJQUNGLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIENvbnYyRERlcklucHV0UHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydkeScsICdXJ107XG4gIHVuaWZvcm1zID1cbiAgICAgICdmaWx0ZXJEaW1zIDogdmVjMjxpMzI+LCBwYWRzIDogdmVjMjxpMzI+LCBzdHJpZGVzIDogdmVjMjxpMzI+LCBvdXRCYWNrcHJvcCA6IHZlYzQ8aTMyPiwnO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5PzogbnVtYmVyW10sIHo/OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIGlzQ2hhbm5lbHNMYXN0OiBib29sZWFuO1xuICBzaXplID0gZmFsc2U7XG4gIGlzVmVjNCA9IGZhbHNlO1xuICB3b3JrUGVyVGhyZWFkID0gMTtcbiAgb3V0cHV0Q29tcG9uZW50OiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLmluU2hhcGU7XG4gICAgdGhpcy5pc0NoYW5uZWxzTGFzdCA9IGNvbnZJbmZvLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnO1xuICAgIHRoaXMuaXNWZWM0ID0gdGhpcy5pc0NoYW5uZWxzTGFzdCAmJiBjb252SW5mby5vdXRDaGFubmVscyAlIDQgPT09IDAgJiZcbiAgICAgICAgY29udkluZm8uaW5DaGFubmVscyAlIDQgPT09IDA7XG4gICAgaWYgKHRoaXMuaXNWZWM0KSB7XG4gICAgICAvLyBUT0RPOiBFeHBhbmQgdG8gYW55IHZhbHVlLlxuICAgICAgdGhpcy53b3JrUGVyVGhyZWFkID0gMjtcbiAgICAgIHRoaXMub3V0cHV0Q29tcG9uZW50ID0gNDtcbiAgICAgIHRoaXMud29ya2dyb3VwU2l6ZSA9IFs0LCA0LCA0XTtcbiAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSB7eDogWzNdLCB5OiBbMl0sIHo6IFswLCAxXX07XG4gICAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSxcbiAgICAgICAgICBbNCwgdGhpcy53b3JrUGVyVGhyZWFkLCAxXSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuc2l6ZSA9IHRydWU7XG4gICAgICB0aGlzLndvcmtQZXJUaHJlYWQgPSAxO1xuICAgICAgdGhpcy53b3JrZ3JvdXBTaXplID0gWzY0LCAxLCAxXTtcbiAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgfVxuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGNvbnYyRERlcklucHV0XyR7dGhpcy5pc0NoYW5uZWxzTGFzdH1fJHt0aGlzLmlzVmVjNH1fJHtcbiAgICAgICAgdGhpcy53b3JrUGVyVGhyZWFkfWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHJvd0RpbSA9IHRoaXMuaXNDaGFubmVsc0xhc3QgPyAxIDogMjtcbiAgICBjb25zdCBjb2xEaW0gPSB0aGlzLmlzQ2hhbm5lbHNMYXN0ID8gMiA6IDM7XG4gICAgY29uc3QgY2hhbm5lbERpbSA9IHRoaXMuaXNDaGFubmVsc0xhc3QgPyAzIDogMTtcblxuICAgIGNvbnN0IHZlYzRTbmlwcGV0ID0gYFxuICAgICR7bWFpbigpfSB7XG4gICAgICBsZXQgYmF0Y2ggPSBpMzIoZ2xvYmFsSWQueikgLyB1bmlmb3Jtcy5vdXRTaGFwZVsxXTtcbiAgICAgIGxldCByID0gaTMyKGdsb2JhbElkLnopICUgdW5pZm9ybXMub3V0U2hhcGVbMV07XG4gICAgICBsZXQgYyA9IGkzMihnbG9iYWxJZC55KSAqICR7dGhpcy53b3JrUGVyVGhyZWFkfTtcbiAgICAgIGxldCBkMSA9IGkzMihnbG9iYWxJZC54KSAqIDQ7XG5cbiAgICAgIGxldCBkeUNvcm5lciA9IHZlYzI8aTMyPihyLCBjKSAtIHVuaWZvcm1zLnBhZHM7XG5cbiAgICAgIC8vIENvbnZvbHZlIGR5KD8sID8sIGQyKSB3aXRoIHcoOiwgOiwgZDEsIGQyKSB0byBjb21wdXRlIGR4KHhSLCB4QywgZDEpLlxuICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICB2YXIgZG90UHJvZDogYXJyYXk8dmVjNDxmMzI+LCAke3RoaXMud29ya1BlclRocmVhZH0+O1xuICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCAke3RoaXMud29ya1BlclRocmVhZH07IGkrKykge1xuICAgICAgICBkb3RQcm9kW2ldID0gdmVjNDxmMzI+KDAuMCk7XG4gICAgICB9XG4gICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltcy54OyB3UiA9IHdSICsgMSkge1xuICAgICAgICBsZXQgZHlSID0gZjMyKGR5Q29ybmVyLnggKyB3UikgLyBmMzIodW5pZm9ybXMuc3RyaWRlcy54KTtcbiAgICAgICAgbGV0IHdSUGVybSA9IHVuaWZvcm1zLmZpbHRlckRpbXMueCAtIDEgLSB3UjtcbiAgICAgICAgaWYgKGR5UiA8IDAuMCB8fCBkeVIgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzFdKSB8fFxuICAgICAgICAgICAgZnJhY3QoZHlSKSA+IDAuMCkge1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIGxldCBpZHlSID0gaTMyKGR5Uik7XG5cbiAgICAgICAgZm9yICh2YXIgd0MgPSAwOyB3QyA8IHVuaWZvcm1zLmZpbHRlckRpbXMueTsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICBsZXQgZHlDID0gZjMyKGR5Q29ybmVyLnkgKyB3QykgLyBmMzIodW5pZm9ybXMuc3RyaWRlcy55KTtcbiAgICAgICAgICBsZXQgZHlDMiA9IGYzMihkeUNvcm5lci55ICsgMSArIHdDKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzLnkpO1xuICAgICAgICAgIGxldCB3Q1Blcm0gPSB1bmlmb3Jtcy5maWx0ZXJEaW1zLnkgLSAxIC0gd0M7XG4gICAgICAgICAgdmFyIGJEeUNWYWwgPSB0cnVlO1xuICAgICAgICAgIHZhciBiRHlDVmFsMiA9IHRydWU7XG4gICAgICAgICAgaWYgKGR5QyA8IDAuMCB8fCBkeUMgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzJdKSB8fFxuICAgICAgICAgICAgICBmcmFjdChkeUMpID4gMC4wKSB7XG4gICAgICAgICAgICBiRHlDVmFsID0gZmFsc2U7XG4gICAgICAgICAgfVxuICAgICAgICAgIGlmIChkeUMyIDwgMC4wIHx8IGR5QzIgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzJdKSB8fFxuICAgICAgICAgICAgICBmcmFjdChkeUMyKSA+IDAuMCkge1xuICAgICAgICAgICAgYkR5Q1ZhbDIgPSBmYWxzZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBsZXQgaWR5QyA9IGkzMihkeUMpO1xuICAgICAgICAgIGxldCBpZHlDMiA9IGkzMihkeUMyKTtcbiAgICAgICAgICBpZiAoYkR5Q1ZhbCAmJiBiRHlDVmFsMikge1xuICAgICAgICAgICAgbGV0IGQyTGVuZ3RoID0gdW5pZm9ybXMub3V0QmFja3Byb3BbM107XG4gICAgICAgICAgICBmb3IgKHZhciBkMiA9IDA7IGQyIDwgZDJMZW5ndGg7IGQyID0gZDIgKyA0KSB7XG4gICAgICAgICAgICAgIGxldCB3VmFsdWUwID0gZ2V0Vyh3UlBlcm0sIHdDUGVybSwgZDEsIGQyKTtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZTEgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMSArIDEsIGQyKTtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZTIgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMSArIDIsIGQyKTtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZTMgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMSArIDMsIGQyKTtcbiAgICAgICAgICAgICAgdmFyIHhWYWx1ZSA9ICBnZXREeShiYXRjaCwgaWR5UiwgaWR5QywgZDIpO1xuICAgICAgICAgICAgICBsZXQgdG1wdmFsID0gdmVjNDxmMzI+KGRvdCh4VmFsdWUsIHdWYWx1ZTApLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTEpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTIpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTMpKTtcbiAgICAgICAgICAgICAgZG90UHJvZFswXSA9IGRvdFByb2RbMF0gKyB0bXB2YWw7XG4gICAgICAgICAgICAgIHhWYWx1ZSA9IGdldER5KGJhdGNoLCBpZHlSLCBpZHlDMiwgZDIpO1xuICAgICAgICAgICAgICBkb3RQcm9kWzFdID0gZG90UHJvZFsxXSArIHZlYzQ8ZjMyPihkb3QoeFZhbHVlLCB3VmFsdWUwKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG90KHhWYWx1ZSwgd1ZhbHVlMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTIpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkb3QoeFZhbHVlLCB3VmFsdWUzKSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSBlbHNlIGlmIChiRHlDVmFsKSB7XG4gICAgICAgICAgICBsZXQgZDJMZW5ndGggPSB1bmlmb3Jtcy5vdXRCYWNrcHJvcFszXTtcbiAgICAgICAgICAgIGZvciAodmFyIGQyID0gMDsgZDIgPCBkMkxlbmd0aDsgZDIgPSBkMiArIDQpIHtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZTAgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMSwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMSA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMSwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMiA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMiwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMyA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMywgZDIpO1xuICAgICAgICAgICAgICB2YXIgeFZhbHVlID0gIGdldER5KGJhdGNoLCBpZHlSLCBpZHlDLCBkMik7XG4gICAgICAgICAgICAgIGxldCB0bXB2YWwgPSB2ZWM0PGYzMj4oZG90KHhWYWx1ZSwgd1ZhbHVlMCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG90KHhWYWx1ZSwgd1ZhbHVlMSksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG90KHhWYWx1ZSwgd1ZhbHVlMiksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZG90KHhWYWx1ZSwgd1ZhbHVlMykpO1xuICAgICAgICAgICAgICBkb3RQcm9kWzBdID0gZG90UHJvZFswXSArIHRtcHZhbDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9IGVsc2UgaWYgKGJEeUNWYWwyKSB7XG4gICAgICAgICAgICBsZXQgZDJMZW5ndGggPSB1bmlmb3Jtcy5vdXRCYWNrcHJvcFszXTtcbiAgICAgICAgICAgIGZvciAodmFyIGQyID0gMDsgZDIgPCBkMkxlbmd0aDsgZDIgPSBkMiArIDQpIHtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZTAgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMSwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMSA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMSwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMiA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMiwgZDIpO1xuICAgICAgICAgICAgICBsZXQgd1ZhbHVlMyA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxICsgMywgZDIpO1xuICAgICAgICAgICAgICB2YXIgeFZhbHVlID0gIGdldER5KGJhdGNoLCBpZHlSLCBpZHlDMiwgZDIpO1xuICAgICAgICAgICAgICBsZXQgdG1wdmFsID0gdmVjNDxmMzI+KGRvdCh4VmFsdWUsIHdWYWx1ZTApLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTEpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTIpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGRvdCh4VmFsdWUsIHdWYWx1ZTMpKTtcbiAgICAgICAgICAgICAgZG90UHJvZFsxXSA9IGRvdFByb2RbMV0gKyB0bXB2YWw7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgJHt0aGlzLndvcmtQZXJUaHJlYWR9OyBpID0gaSArIDEpIHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IHZlYzQ8aTMyPihiYXRjaCwgciwgYyArIGksIGQxKTtcbiAgICAgICAgaWYgKGNvb3Jkc0luQm91bmRzNEQoY29vcmRzLCB1bmlmb3Jtcy5vdXRTaGFwZSkpIHtcbiAgICAgICAgICBzZXRPdXRwdXRBdENvb3Jkcyhjb29yZHNbMF0sIGNvb3Jkc1sxXSwgY29vcmRzWzJdLCBjb29yZHNbM10sIGRvdFByb2RbaV0pO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHRoaXMuaXNWZWM0ID9cbiAgICAgICAgYFxuICAgICR7dmVjNFNuaXBwZXR9XG4gICAgYCA6XG4gICAgICAgIGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzWzBdO1xuICAgICAgICBsZXQgZDEgPSBjb29yZHNbJHtjaGFubmVsRGltfV07XG5cbiAgICAgICAgbGV0IGR5Q29ybmVyID0gdmVjMjxpMzI+KGNvb3Jkc1ske3Jvd0RpbX1dLCBjb29yZHNbJHtcbiAgICAgICAgICAgIGNvbERpbX1dKSAtIHVuaWZvcm1zLnBhZHM7XG4gICAgICAgIGxldCBkeVJDb3JuZXIgPSBkeUNvcm5lci54O1xuICAgICAgICBsZXQgZHlDQ29ybmVyID0gZHlDb3JuZXIueTtcblxuICAgICAgICAvLyBDb252b2x2ZSBkeSg/LCA/LCBkMikgd2l0aCB3KDosIDosIGQxLCBkMikgdG8gY29tcHV0ZSBkeCh4UiwgeEMsIGQxKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICAgIHZhciBkb3RQcm9kID0gMC4wO1xuICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltcy54OyB3UiA9IHdSICsgMSkge1xuICAgICAgICAgIGxldCBkeVIgPSAoZjMyKGR5UkNvcm5lcikgKyBmMzIod1IpKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzLngpO1xuICAgICAgICAgIGxldCB3UlBlcm0gPSB1bmlmb3Jtcy5maWx0ZXJEaW1zLnggLSAxIC0gd1I7XG4gICAgICAgICAgaWYgKGR5UiA8IDAuMCB8fCBkeVIgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzFdKSB8fCBmcmFjdChkeVIpID4gMC4wIHx8XG4gICAgICAgICAgICAgIHdSUGVybSA8IDApIHtcbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cbiAgICAgICAgICBsZXQgaWR5UiA9IGkzMihkeVIpO1xuXG4gICAgICAgICAgZm9yICh2YXIgd0MgPSAwOyB3QyA8IHVuaWZvcm1zLmZpbHRlckRpbXMueTsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICAgIGxldCBkeUMgPSAoZjMyKGR5Q0Nvcm5lcikgKyBmMzIod0MpKSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzLnkpO1xuICAgICAgICAgICAgbGV0IHdDUGVybSA9IHVuaWZvcm1zLmZpbHRlckRpbXMueSAtIDEgLSB3QztcbiAgICAgICAgICAgIGlmIChkeUMgPCAwLjAgfHwgZHlDID49IGYzMih1bmlmb3Jtcy5vdXRCYWNrcHJvcFsyXSkgfHxcbiAgICAgICAgICAgICAgICBmcmFjdChkeUMpID4gMC4wIHx8IHdDUGVybSA8IDApIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBsZXQgaWR5QyA9IGkzMihkeUMpO1xuXG4gICAgICAgICAgICBmb3IgKHZhciBkMiA9IDA7IGQyIDwgdW5pZm9ybXMub3V0QmFja3Byb3BbM107IGQyID0gZDIgKyAxKSB7XG4gICAgICAgICAgICAgIGxldCB4VmFsdWUgPSAke1xuICAgICAgICAgICAgdGhpcy5pc0NoYW5uZWxzTGFzdCA/ICdnZXREeShiYXRjaCwgaWR5UiwgaWR5QywgZDIpJyA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2dldER5KGJhdGNoLCBkMiwgaWR5UiwgaWR5QyknfTtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZSA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQxLCBkMik7XG4gICAgICAgICAgICAgIGRvdFByb2QgPSBkb3RQcm9kICsgeFZhbHVlICogd1ZhbHVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBkb3RQcm9kKTtcbiAgICAgIH1cbiAgICB9XG4gIGA7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIENvbnYyRERlckZpbHRlclByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdkeSddO1xuICB1bmlmb3JtcyA9XG4gICAgICAncGFkcyA6IHZlYzI8aTMyPiwgc3RyaWRlcyA6IHZlYzI8aTMyPiwgYmF0Y2hTaXplIDogaTMyLCBvdXRIZWlnaHQgOiBpMzIsIG91dFdpZHRoIDogaTMyLCBpbkhlaWdodCA6IGkzMiwgaW5XaWR0aCA6IGkzMiwnO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgaXNDaGFubmVsc0xhc3Q6IGJvb2xlYW47XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5maWx0ZXJTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy5pc0NoYW5uZWxzTGFzdCA9IGNvbnZJbmZvLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGNvbnYyRERlckZpbHRlcl8ke3RoaXMuaXNDaGFubmVsc0xhc3R9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgcmV0dXJuIGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IHdSID0gY29vcmRzWzBdO1xuICAgICAgICBsZXQgd0MgPSBjb29yZHNbMV07XG4gICAgICAgIGxldCBkMSA9IGNvb3Jkc1syXTtcbiAgICAgICAgbGV0IGQyID0gY29vcmRzWzNdO1xuXG4gICAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggZHkoOiwgOiwgZDIpIHRvIGdldCBkdyh3Uiwgd0MsIGQxLCBkMikuXG4gICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgICB2YXIgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgZm9yICh2YXIgYiA9IDA7IGIgPCB1bmlmb3Jtcy5iYXRjaFNpemU7IGIgPSBiICsgMSkge1xuICAgICAgICAgIGZvciAodmFyIHlSID0gMDsgeVIgPCB1bmlmb3Jtcy5vdXRIZWlnaHQ7IHlSID0geVIgKyAxKSB7XG4gICAgICAgICAgICBsZXQgeFIgPSB3UiArIHlSICogdW5pZm9ybXMuc3RyaWRlc1swXSAtIHVuaWZvcm1zLnBhZHNbMF07XG4gICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHVuaWZvcm1zLmluSGVpZ2h0KSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmb3IgKHZhciB5QyA9IDA7IHlDIDwgdW5pZm9ybXMub3V0V2lkdGg7IHlDID0geUMgKyAxKSB7XG4gICAgICAgICAgICAgIGxldCB4QyA9IHdDICsgeUMgKiB1bmlmb3Jtcy5zdHJpZGVzWzFdIC0gdW5pZm9ybXMucGFkc1sxXTtcblxuICAgICAgICAgICAgICBpZiAoeEMgPCAwIHx8IHhDID49IHVuaWZvcm1zLmluV2lkdGgpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGlmICgke3RoaXMuaXNDaGFubmVsc0xhc3R9KSB7XG4gICAgICAgICAgICAgICAgbGV0IGR5VmFsdWUgPSBnZXREeShiLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlID0gZ2V0WChiLCB4UiwgeEMsIGQxKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kID0gZG90UHJvZCArIHhWYWx1ZSAqIGR5VmFsdWU7XG4gICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgbGV0IGR5VmFsdWUgPSBnZXREeShiLCBkMiwgeVIsIHlDKTtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlID0gZ2V0WChiLCBkMSwgeFIsIHhDKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kID0gZG90UHJvZCArIHhWYWx1ZSAqIGR5VmFsdWU7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgZG90UHJvZCk7XG4gICAgICB9XG4gICAgfVxuICBgO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBDb252M0REZXJGaWx0ZXJQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgYHBhZHMgOiB2ZWMzPGkzMj4sIHN0cmlkZXMgOiB2ZWMzPGkzMj4sIGJhdGNoU2l6ZSA6IGkzMiwgb3V0RGVwdGggOiBpMzIsXG4gICAgICAgb3V0SGVpZ2h0IDogaTMyLCBvdXRXaWR0aCA6IGkzMiwgaW5EZXB0aCA6IGkzMiwgaW5IZWlnaHQgOiBpMzIsIGluV2lkdGggOiBpMzIsYDtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjNESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5maWx0ZXJTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgY29udjNERGVyRmlsdGVyYDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgcmV0dXJuIGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IHdGID0gY29vcmRzLng7XG4gICAgICAgIGxldCB3UiA9IGNvb3Jkcy55O1xuICAgICAgICBsZXQgd0MgPSBjb29yZHMuejtcbiAgICAgICAgbGV0IGQxID0gY29vcmRzLnc7XG4gICAgICAgIGxldCBkMiA9IGNvb3Jkcy51O1xuXG4gICAgICAgIHZhciBkb3RQcm9kID0gMC4wO1xuICAgICAgICBmb3IgKHZhciBiID0gMDsgYiA8IHVuaWZvcm1zLmJhdGNoU2l6ZTsgYisrKSB7XG4gICAgICAgICAgZm9yICh2YXIgeUYgPSAwOyB5RiA8IHVuaWZvcm1zLm91dERlcHRoOyB5RisrKSB7XG4gICAgICAgICAgICBsZXQgeEYgPSB3RiArIHlGICogdW5pZm9ybXMuc3RyaWRlc1swXSAtIHVuaWZvcm1zLnBhZHNbMF07XG4gICAgICAgICAgICBpZiAoeEYgPCAwIHx8IHhGID49IHVuaWZvcm1zLmluRGVwdGgpIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZvciAodmFyIHlSID0gMDsgeVIgPCB1bmlmb3Jtcy5vdXRIZWlnaHQ7IHlSKyspIHtcbiAgICAgICAgICAgICAgbGV0IHhSID0gd1IgKyB5UiAqIHVuaWZvcm1zLnN0cmlkZXNbMV0gLSB1bmlmb3Jtcy5wYWRzWzFdO1xuICAgICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHVuaWZvcm1zLmluSGVpZ2h0KSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICBmb3IgKHZhciB5QyA9IDA7IHlDIDwgdW5pZm9ybXMub3V0V2lkdGg7IHlDKyspIHtcbiAgICAgICAgICAgICAgICBsZXQgeEMgPSB3QyArIHlDICogdW5pZm9ybXMuc3RyaWRlc1syXSAtIHVuaWZvcm1zLnBhZHNbMl07XG4gICAgICAgICAgICAgICAgaWYgKHhDIDwgMCB8fCB4QyA+PSB1bmlmb3Jtcy5pbldpZHRoKSB7XG4gICAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBsZXQgZHlWYWx1ZSA9IGdldER5KGIsIHlGLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgICAgICAgICBsZXQgeFZhbHVlID0gZ2V0WChiLCB4RiwgeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSB4VmFsdWUgKiBkeVZhbHVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGRvdFByb2QpO1xuICAgICAgfVxuICAgIH1cbiAgYDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQ29udjNERGVySW5wdXRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ2R5JywgJ1cnXTtcbiAgdW5pZm9ybXMgPSBgZmlsdGVyRGltcyA6IHZlYzM8aTMyPiwgcGFkcyA6IHZlYzM8aTMyPiwgc3RyaWRlcyA6IHZlYzM8aTMyPixcbiAgICAgIG91dERlcHRoIDogaTMyLCBvdXRIZWlnaHQgOiBpMzIsIG91dFdpZHRoIDogaTMyLCBvdXRDaGFubmVscyA6IGkzMixgO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLmluU2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGNvbnYzRERlcklucHV0YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgcmV0dXJuIGBcbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzLng7XG4gICAgICAgIGxldCBkMSA9IGNvb3Jkcy51O1xuXG4gICAgICAgIGxldCBkeUNvcm5lciA9IHZlYzM8aTMyPihjb29yZHMueSwgY29vcmRzLnosIGNvb3Jkcy53KSAtIHVuaWZvcm1zLnBhZHM7XG4gICAgICAgIGxldCBkeUZDb3JuZXIgPSBkeUNvcm5lci54O1xuICAgICAgICBsZXQgZHlSQ29ybmVyID0gZHlDb3JuZXIueTtcbiAgICAgICAgbGV0IGR5Q0Nvcm5lciA9IGR5Q29ybmVyLno7XG5cbiAgICAgICAgdmFyIGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAodmFyIHdGID0gMDsgd0YgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyB3RisrKSB7XG4gICAgICAgICAgbGV0IGR5RiA9IGYzMihkeUZDb3JuZXIgKyB3RikgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1swXSk7XG4gICAgICAgICAgaWYgKGR5RiA8IDAuMCB8fCBkeUYgPj0gZjMyKHVuaWZvcm1zLm91dERlcHRoKSB8fCBmcmFjdChkeUYpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG4gICAgICAgICAgbGV0IGlkeUYgPSBpMzIoZHlGKTtcblxuICAgICAgICAgIGxldCB3RlBlcm0gPSB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdIC0gMSAtIHdGO1xuXG4gICAgICAgICAgZm9yICh2YXIgd1IgPSAwOyB3UiA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMV07IHdSKyspIHtcbiAgICAgICAgICAgIGxldCBkeVIgPSBmMzIoZHlSQ29ybmVyICsgd1IpIC8gZjMyKHVuaWZvcm1zLnN0cmlkZXNbMV0pO1xuXG4gICAgICAgICAgICBpZiAoZHlSIDwgMC4wIHx8IGR5UiA+PSBmMzIodW5pZm9ybXMub3V0SGVpZ2h0KSB8fCBmcmFjdChkeVIpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgbGV0IGlkeVIgPSBpMzIoZHlSKTtcblxuICAgICAgICAgICAgbGV0IHdSUGVybSA9IHVuaWZvcm1zLmZpbHRlckRpbXNbMV0gLSAxIC0gd1I7XG5cbiAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzJdOyB3QysrKSB7XG4gICAgICAgICAgICAgIGxldCBkeUMgPSBmMzIoZHlDQ29ybmVyICsgd0MpIC8gZjMyKHVuaWZvcm1zLnN0cmlkZXNbMl0pO1xuXG4gICAgICAgICAgICAgIGlmIChkeUMgPCAwLjAgfHwgZHlDID49IGYzMih1bmlmb3Jtcy5vdXRXaWR0aCkgfHwgZnJhY3QoZHlDKSA+IDAuMCkge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGxldCBpZHlDID0gaTMyKGR5Qyk7XG5cbiAgICAgICAgICAgICAgbGV0IHdDUGVybSA9IHVuaWZvcm1zLmZpbHRlckRpbXNbMl0gLSAxIC0gd0M7XG5cbiAgICAgICAgICAgICAgZm9yICh2YXIgZDIgPSAwOyBkMiA8IHVuaWZvcm1zLm91dENoYW5uZWxzOyBkMisrKSB7XG4gICAgICAgICAgICAgICAgbGV0IHhWYWx1ZSA9IGdldER5KGJhdGNoLCBpZHlGLCBpZHlSLCBpZHlDLCBkMik7XG4gICAgICAgICAgICAgICAgbGV0IHdWYWx1ZSA9IGdldFcod0ZQZXJtLCB3UlBlcm0sIHdDUGVybSwgZDEsIGQyKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHhWYWx1ZSAqIHdWYWx1ZTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBkb3RQcm9kKTtcbiAgICAgIH1cbiAgICB9XG4gIGA7XG4gIH1cbn1cbiJdfQ==