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
import { util } from '@tensorflow/tfjs-core';
import { activationFnSnippet, biasActivationSnippet } from './activation_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class DepthwiseConv2DVec4Program {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'pads : vec2<i32>, inDims : vec2<i32>, virtualWidth : i32,';
        this.workgroupSize = [64, 1, 1];
        this.workPerThread = 4;
        this.outputComponent = 4;
        this.outputShape = convInfo.outShape;
        this.virtualWidth = Math.ceil(this.outputShape[2] / this.workPerThread) *
            this.workPerThread;
        const virtualOutputShape = [
            this.outputShape[0], this.outputShape[1], this.virtualWidth,
            this.outputShape[3]
        ];
        this.dispatchLayout = flatDispatchLayout(virtualOutputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, virtualOutputShape, this.workgroupSize, [this.outputComponent * this.workPerThread, 1, 1]);
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
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
        this.shaderKey =
            `depthwiseVec4_${activation}_${this.convInfo.filterHeight}_${this.convInfo.filterWidth}_${this.convInfo.strideHeight}_${this.convInfo.strideWidth}_${this.workPerThread}`;
    }
    getUserCode() {
        const xNumber = (this.workPerThread - 1) * this.convInfo.strideWidth +
            this.convInfo.filterWidth;
        const strideHeight = this.convInfo.strideHeight;
        const strideWidth = this.convInfo.strideWidth;
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, true, 4)}
      fn readX(batch : i32, row : i32, col : i32, channel : i32) -> vec4<f32> {
        var value = vec4<f32>(0.0);
        if (col >=0 && col < uniforms.inDims[1]) {
          value = getX(batch, row, col, channel);
        }
        return value;
      }

      ${main('index')} {
        let width0 = uniforms.outShape[3] / ${this.outputComponent};
        let d1 = (index % width0) * ${this.outputComponent};
        var index1 = index / width0;
        let width1 = uniforms.virtualWidth / ${this.workPerThread};
        let c = (index1 % width1) * ${this.workPerThread};
        index1 = index1 / width1;
        let r = index1 % uniforms.outShape[1];
        let batch = index1 / uniforms.outShape[1];

        let xRCCorner = vec2<i32>(r, c) * vec2<i32>(${strideHeight}, ${strideWidth}) - uniforms.pads;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;
        var xVals : array<vec4<f32>, ${xNumber}>;
        var dotProd : array<vec4<f32>, ${this.workPerThread}>;
        for (var i = 0; i < ${this.workPerThread}; i++) {
          dotProd[i] = vec4<f32>(0.0);
        }

        // Use constant instead of uniform can give better performance.
        for (var wR = 0; wR < ${this.convInfo.filterHeight}; wR = wR + 1) {
          let xR = xRCorner + wR;
          if (xR >=0 && xR < uniforms.inDims[0]) {
            for (var i = 0; i < ${xNumber}; i++) {
              xVals[i] = readX(batch, xR, xCCorner + i, d1);
            }
            for (var wC = 0; wC < ${this.convInfo.filterWidth}; wC = wC + 1) {
              let wValue = getW(wR, wC, d1, 0);
              for (var i = 0; i < ${this.workPerThread}; i++) {
                dotProd[i] = fma(xVals[i * ${strideWidth} + wC], wValue, dotProd[i]);
              }
            }
          }
        }

        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let coords = vec4<i32>(batch, r, c + i, d1);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
            var value = dotProd[i];
            ${biasActivationSnippet(this.addBias, this.activation)}
            setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
          }
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZF92ZWM0X3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2RlcHRod2lzZV9jb252MmRfdmVjNF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ3pELE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxxQkFBcUIsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzdFLE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sMEJBQTBCO0lBZ0JyQyxZQUNJLFFBQWlDLEVBQUUsT0FBTyxHQUFHLEtBQUssRUFDbEQsYUFBc0MsSUFBSSxFQUFFLGtCQUFrQixHQUFHLEtBQUs7UUFiMUUsa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUMzQixhQUFRLEdBQUcsMkRBQTJELENBQUM7UUFDdkUsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELGtCQUFhLEdBQUcsQ0FBQyxDQUFDO1FBS2xCLG9CQUFlLEdBQUcsQ0FBQyxDQUFDO1FBTWxCLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQztRQUNyQyxJQUFJLENBQUMsWUFBWSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDO1lBQ25FLElBQUksQ0FBQyxhQUFhLENBQUM7UUFDdkIsTUFBTSxrQkFBa0IsR0FBRztZQUN6QixJ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bGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHthY3RpdmF0aW9uRm5TbmlwcGV0LCBiaWFzQWN0aXZhdGlvblNuaXBwZXR9IGZyb20gJy4vYWN0aXZhdGlvbl91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgRGVwdGh3aXNlQ29udjJEVmVjNFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnVyddO1xuICB1bmlmb3JtcyA9ICdwYWRzIDogdmVjMjxpMzI+LCBpbkRpbXMgOiB2ZWMyPGkzMj4sIHZpcnR1YWxXaWR0aCA6IGkzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICB3b3JrUGVyVGhyZWFkID0gNDtcbiAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvO1xuICBhZGRCaWFzOiBib29sZWFuO1xuICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbjtcbiAgaGFzUHJlbHVBY3RpdmF0aW9uOiBib29sZWFuO1xuICBvdXRwdXRDb21wb25lbnQgPSA0O1xuICB2aXJ0dWFsV2lkdGg6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbywgYWRkQmlhcyA9IGZhbHNlLFxuICAgICAgYWN0aXZhdGlvbjogYmFja2VuZF91dGlsLkFjdGl2YXRpb24gPSBudWxsLCBoYXNQcmVsdUFjdGl2YXRpb24gPSBmYWxzZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5vdXRTaGFwZTtcbiAgICB0aGlzLnZpcnR1YWxXaWR0aCA9IE1hdGguY2VpbCh0aGlzLm91dHB1dFNoYXBlWzJdIC8gdGhpcy53b3JrUGVyVGhyZWFkKSAqXG4gICAgICAgIHRoaXMud29ya1BlclRocmVhZDtcbiAgICBjb25zdCB2aXJ0dWFsT3V0cHV0U2hhcGUgPSBbXG4gICAgICB0aGlzLm91dHB1dFNoYXBlWzBdLCB0aGlzLm91dHB1dFNoYXBlWzFdLCB0aGlzLnZpcnR1YWxXaWR0aCxcbiAgICAgIHRoaXMub3V0cHV0U2hhcGVbM11cbiAgICBdO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodmlydHVhbE91dHB1dFNoYXBlKTtcblxuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHZpcnR1YWxPdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplLFxuICAgICAgICBbdGhpcy5vdXRwdXRDb21wb25lbnQgKiB0aGlzLndvcmtQZXJUaHJlYWQsIDEsIDFdKTtcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjb252SW5mby5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JyxcbiAgICAgICAgKCkgPT4gJ1RPRE86IE5DSFcgaXMgdW5pbXBsZW1lbnRlZCcpO1xuXG4gICAgaWYgKGFkZEJpYXMpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdiaWFzJyk7XG4gICAgfVxuICAgIGlmIChoYXNQcmVsdUFjdGl2YXRpb24pIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdwcmVsdUFjdGl2YXRpb25XZWlnaHRzJyk7XG4gICAgfVxuXG4gICAgdGhpcy5jb252SW5mbyA9IGNvbnZJbmZvO1xuICAgIHRoaXMuYWRkQmlhcyA9IGFkZEJpYXM7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gYWN0aXZhdGlvbjtcbiAgICB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbiA9IGhhc1ByZWx1QWN0aXZhdGlvbjtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID1cbiAgICAgICAgYGRlcHRod2lzZVZlYzRfJHthY3RpdmF0aW9ufV8ke3RoaXMuY29udkluZm8uZmlsdGVySGVpZ2h0fV8ke1xuICAgICAgICAgICAgdGhpcy5jb252SW5mby5maWx0ZXJXaWR0aH1fJHt0aGlzLmNvbnZJbmZvLnN0cmlkZUhlaWdodH1fJHtcbiAgICAgICAgICAgIHRoaXMuY29udkluZm8uc3RyaWRlV2lkdGh9XyR7dGhpcy53b3JrUGVyVGhyZWFkfWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHhOdW1iZXIgPSAodGhpcy53b3JrUGVyVGhyZWFkIC0gMSkgKiB0aGlzLmNvbnZJbmZvLnN0cmlkZVdpZHRoICtcbiAgICAgICAgdGhpcy5jb252SW5mby5maWx0ZXJXaWR0aDtcbiAgICBjb25zdCBzdHJpZGVIZWlnaHQgPSB0aGlzLmNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgICBjb25zdCBzdHJpZGVXaWR0aCA9IHRoaXMuY29udkluZm8uc3RyaWRlV2lkdGg7XG5cbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7YWN0aXZhdGlvbkZuU25pcHBldCh0aGlzLmFjdGl2YXRpb24sIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uLCB0cnVlLCA0KX1cbiAgICAgIGZuIHJlYWRYKGJhdGNoIDogaTMyLCByb3cgOiBpMzIsIGNvbCA6IGkzMiwgY2hhbm5lbCA6IGkzMikgLT4gdmVjNDxmMzI+IHtcbiAgICAgICAgdmFyIHZhbHVlID0gdmVjNDxmMzI+KDAuMCk7XG4gICAgICAgIGlmIChjb2wgPj0wICYmIGNvbCA8IHVuaWZvcm1zLmluRGltc1sxXSkge1xuICAgICAgICAgIHZhbHVlID0gZ2V0WChiYXRjaCwgcm93LCBjb2wsIGNoYW5uZWwpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgIH1cblxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGxldCB3aWR0aDAgPSB1bmlmb3Jtcy5vdXRTaGFwZVszXSAvICR7dGhpcy5vdXRwdXRDb21wb25lbnR9O1xuICAgICAgICBsZXQgZDEgPSAoaW5kZXggJSB3aWR0aDApICogJHt0aGlzLm91dHB1dENvbXBvbmVudH07XG4gICAgICAgIHZhciBpbmRleDEgPSBpbmRleCAvIHdpZHRoMDtcbiAgICAgICAgbGV0IHdpZHRoMSA9IHVuaWZvcm1zLnZpcnR1YWxXaWR0aCAvICR7dGhpcy53b3JrUGVyVGhyZWFkfTtcbiAgICAgICAgbGV0IGMgPSAoaW5kZXgxICUgd2lkdGgxKSAqICR7dGhpcy53b3JrUGVyVGhyZWFkfTtcbiAgICAgICAgaW5kZXgxID0gaW5kZXgxIC8gd2lkdGgxO1xuICAgICAgICBsZXQgciA9IGluZGV4MSAlIHVuaWZvcm1zLm91dFNoYXBlWzFdO1xuICAgICAgICBsZXQgYmF0Y2ggPSBpbmRleDEgLyB1bmlmb3Jtcy5vdXRTaGFwZVsxXTtcblxuICAgICAgICBsZXQgeFJDQ29ybmVyID0gdmVjMjxpMzI+KHIsIGMpICogdmVjMjxpMzI+KCR7c3RyaWRlSGVpZ2h0fSwgJHtcbiAgICAgICAgc3RyaWRlV2lkdGh9KSAtIHVuaWZvcm1zLnBhZHM7XG5cbiAgICAgICAgbGV0IHhSQ29ybmVyID0geFJDQ29ybmVyLng7XG4gICAgICAgIGxldCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuICAgICAgICB2YXIgeFZhbHMgOiBhcnJheTx2ZWM0PGYzMj4sICR7eE51bWJlcn0+O1xuICAgICAgICB2YXIgZG90UHJvZCA6IGFycmF5PHZlYzQ8ZjMyPiwgJHt0aGlzLndvcmtQZXJUaHJlYWR9PjtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCAke3RoaXMud29ya1BlclRocmVhZH07IGkrKykge1xuICAgICAgICAgIGRvdFByb2RbaV0gPSB2ZWM0PGYzMj4oMC4wKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFVzZSBjb25zdGFudCBpbnN0ZWFkIG9mIHVuaWZvcm0gY2FuIGdpdmUgYmV0dGVyIHBlcmZvcm1hbmNlLlxuICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgJHt0aGlzLmNvbnZJbmZvLmZpbHRlckhlaWdodH07IHdSID0gd1IgKyAxKSB7XG4gICAgICAgICAgbGV0IHhSID0geFJDb3JuZXIgKyB3UjtcbiAgICAgICAgICBpZiAoeFIgPj0wICYmIHhSIDwgdW5pZm9ybXMuaW5EaW1zWzBdKSB7XG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8ICR7eE51bWJlcn07IGkrKykge1xuICAgICAgICAgICAgICB4VmFsc1tpXSA9IHJlYWRYKGJhdGNoLCB4UiwgeENDb3JuZXIgKyBpLCBkMSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBmb3IgKHZhciB3QyA9IDA7IHdDIDwgJHt0aGlzLmNvbnZJbmZvLmZpbHRlcldpZHRofTsgd0MgPSB3QyArIDEpIHtcbiAgICAgICAgICAgICAgbGV0IHdWYWx1ZSA9IGdldFcod1IsIHdDLCBkMSwgMCk7XG4gICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgJHt0aGlzLndvcmtQZXJUaHJlYWR9OyBpKyspIHtcbiAgICAgICAgICAgICAgICBkb3RQcm9kW2ldID0gZm1hKHhWYWxzW2kgKiAke1xuICAgICAgICBzdHJpZGVXaWR0aH0gKyB3Q10sIHdWYWx1ZSwgZG90UHJvZFtpXSk7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8ICR7dGhpcy53b3JrUGVyVGhyZWFkfTsgaSA9IGkgKyAxKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IHZlYzQ8aTMyPihiYXRjaCwgciwgYyArIGksIGQxKTtcbiAgICAgICAgICBpZiAoY29vcmRzSW5Cb3VuZHM0RChjb29yZHMsIHVuaWZvcm1zLm91dFNoYXBlKSkge1xuICAgICAgICAgICAgdmFyIHZhbHVlID0gZG90UHJvZFtpXTtcbiAgICAgICAgICAgICR7Ymlhc0FjdGl2YXRpb25TbmlwcGV0KHRoaXMuYWRkQmlhcywgdGhpcy5hY3RpdmF0aW9uKX1cbiAgICAgICAgICAgIHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkc1swXSwgY29vcmRzWzFdLCBjb29yZHNbMl0sIGNvb3Jkc1szXSwgdmFsdWUpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=