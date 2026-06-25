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
import { getMainHeaderString as main, PixelsOpType } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class DrawProgram {
    constructor(outShape, type, textureFormat) {
        this.variableNames = ['Image'];
        this.uniforms = 'alpha: f32,';
        this.workgroupSize = [64, 1, 1];
        this.pixelsOpType = PixelsOpType.DRAW;
        this.size = true;
        this.outputShape = outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.type = type;
        this.textureFormat = textureFormat;
        this.shaderKey = `draw_${type}_${textureFormat}`;
    }
    getUserCode() {
        let calculateResult;
        const value = this.type === 'float32' ? 'value' : 'value / 255.0';
        calculateResult = `
      if (uniforms.numChannels == 1) {
        rgba[0] = ${value};
        rgba[1] = ${value};
        rgba[2] = ${value};
      } else {
        rgba[d] = ${value};
      }`;
        const userCode = `
       @group(0) @binding(0) var outImage : texture_storage_2d<${this.textureFormat}, write>;
       ${main('index')} {
         if (index < uniforms.size) {
           var rgba = vec4<f32>(0.0, 0.0, 0.0, uniforms.alpha);
           for (var d = 0; d < uniforms.numChannels; d = d + 1) {
             let value = f32(inBuf[index * uniforms.numChannels + d]);
             ${calculateResult}
           }
           rgba.x = rgba.x * rgba.w;
           rgba.y = rgba.y * rgba.w;
           rgba.z = rgba.z * rgba.w;
           let coords = getCoordsFromIndex(index);
           textureStore(outImage, vec2<i32>(coords.yx), rgba);
         }
       }
      `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZHJhd193ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9kcmF3X3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFFLFlBQVksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUMxRixPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxXQUFXO0lBYXRCLFlBQ0ksUUFBa0IsRUFBRSxJQUFjLEVBQUUsYUFBK0I7UUFidkUsa0JBQWEsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQzFCLGFBQVEsR0FBRyxhQUFhLENBQUM7UUFLekIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBR3JELGlCQUFZLEdBQUcsWUFBWSxDQUFDLElBQUksQ0FBQztRQUNqQyxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBSVYsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUM7UUFDNUIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7UUFDakIsSUFBSSxDQUFDLGFBQWEsR0FBRyxhQUFhLENBQUM7UUFDbkMsSUFBSSxDQUFDLFNBQVMsR0FBRyxRQUFRLElBQUksSUFBSSxhQUFhLEVBQUUsQ0FBQztJQUNuRCxDQUFDO0lBRUQsV0FBVztRQUNULElBQUksZUFBZSxDQUFDO1FBQ3BCLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxJQUFJLEtBQUssU0FBUyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLGVBQWUsQ0FBQztRQUNsRSxlQUFlLEdBQUc7O29CQUVGLEtBQUs7b0JBQ0wsS0FBSztvQkFDTCxLQUFLOztvQkFFTCxLQUFLO1FBQ2pCLENBQUM7UUFFTCxNQUFNLFFBQVEsR0FBRztpRUFFYixJQUFJLENBQUMsYUFBYTtTQUNqQixJQUFJLENBQUMsT0FBTyxDQUFDOzs7OztlQUtQLGVBQWU7Ozs7Ozs7OztPQVN2QixDQUFDO1FBQ0osT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0RhdGFUeXBlfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgUGl4ZWxzT3BUeXBlLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgRHJhd1Byb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsnSW1hZ2UnXTtcbiAgdW5pZm9ybXMgPSAnYWxwaGE6IGYzMiwnO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgdHlwZTogRGF0YVR5cGU7XG4gIHRleHR1cmVGb3JtYXQ6IEdQVVRleHR1cmVGb3JtYXQ7XG4gIHBpeGVsc09wVHlwZSA9IFBpeGVsc09wVHlwZS5EUkFXO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIG91dFNoYXBlOiBudW1iZXJbXSwgdHlwZTogRGF0YVR5cGUsIHRleHR1cmVGb3JtYXQ6IEdQVVRleHR1cmVGb3JtYXQpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMudHlwZSA9IHR5cGU7XG4gICAgdGhpcy50ZXh0dXJlRm9ybWF0ID0gdGV4dHVyZUZvcm1hdDtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBkcmF3XyR7dHlwZX1fJHt0ZXh0dXJlRm9ybWF0fWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGxldCBjYWxjdWxhdGVSZXN1bHQ7XG4gICAgY29uc3QgdmFsdWUgPSB0aGlzLnR5cGUgPT09ICdmbG9hdDMyJyA/ICd2YWx1ZScgOiAndmFsdWUgLyAyNTUuMCc7XG4gICAgY2FsY3VsYXRlUmVzdWx0ID0gYFxuICAgICAgaWYgKHVuaWZvcm1zLm51bUNoYW5uZWxzID09IDEpIHtcbiAgICAgICAgcmdiYVswXSA9ICR7dmFsdWV9O1xuICAgICAgICByZ2JhWzFdID0gJHt2YWx1ZX07XG4gICAgICAgIHJnYmFbMl0gPSAke3ZhbHVlfTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJnYmFbZF0gPSAke3ZhbHVlfTtcbiAgICAgIH1gO1xuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAgQGdyb3VwKDApIEBiaW5kaW5nKDApIHZhciBvdXRJbWFnZSA6IHRleHR1cmVfc3RvcmFnZV8yZDwke1xuICAgICAgICB0aGlzLnRleHR1cmVGb3JtYXR9LCB3cml0ZT47XG4gICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgIHZhciByZ2JhID0gdmVjNDxmMzI+KDAuMCwgMC4wLCAwLjAsIHVuaWZvcm1zLmFscGhhKTtcbiAgICAgICAgICAgZm9yICh2YXIgZCA9IDA7IGQgPCB1bmlmb3Jtcy5udW1DaGFubmVsczsgZCA9IGQgKyAxKSB7XG4gICAgICAgICAgICAgbGV0IHZhbHVlID0gZjMyKGluQnVmW2luZGV4ICogdW5pZm9ybXMubnVtQ2hhbm5lbHMgKyBkXSk7XG4gICAgICAgICAgICAgJHtjYWxjdWxhdGVSZXN1bHR9XG4gICAgICAgICAgIH1cbiAgICAgICAgICAgcmdiYS54ID0gcmdiYS54ICogcmdiYS53O1xuICAgICAgICAgICByZ2JhLnkgPSByZ2JhLnkgKiByZ2JhLnc7XG4gICAgICAgICAgIHJnYmEueiA9IHJnYmEueiAqIHJnYmEudztcbiAgICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgIHRleHR1cmVTdG9yZShvdXRJbWFnZSwgdmVjMjxpMzI+KGNvb3Jkcy55eCksIHJnYmEpO1xuICAgICAgICAgfVxuICAgICAgIH1cbiAgICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=