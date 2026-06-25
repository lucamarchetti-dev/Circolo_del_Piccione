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
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class LRNGradProgram {
    constructor(inputShape) {
        this.outputShape = [];
        this.variableNames = ['inputImage', 'outputImage', 'dy'];
        this.uniforms = 'depthRadius : i32, bias : f32, alpha : f32, beta : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = inputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'lrn_grad';
    }
    getUserCode() {
        const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];

        let MIN_DEPTH_BEGIN = 0;
        let MAX_DEPTH_END = uniforms.outShape[3];
        var result = 0.0;
        for (var d = MIN_DEPTH_BEGIN; d < MAX_DEPTH_END; d++) {
          let depthBegin = max(MIN_DEPTH_BEGIN, d - uniforms.depthRadius);
          let depthEnd = min(MAX_DEPTH_END, d + uniforms.depthRadius + 1);

          var norm = 0.0;
          for (var k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; k++) {
            if (k < depthBegin) {
              continue;
            } else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            } else {
              break;
            }
          }

          norm = uniforms.alpha * norm + uniforms.bias;

          for (var k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; k++) {
            if (k < depthBegin) {
              continue;
            } else if (k >= depthBegin && k < depthEnd) {
              var dyi = -2.0 * uniforms.alpha * uniforms.beta
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d) / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * uniforms.beta);
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            } else {
              break;
            }
          }
        }

        setOutputAtIndex(index, result);
      }
    }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibHJuX2dyYWRfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvbHJuX2dyYWRfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sY0FBYztJQVV6QixZQUFZLFVBQW9CO1FBVGhDLGdCQUFXLEdBQWEsRUFBRSxDQUFDO1FBSTNCLGtCQUFhLEdBQUcsQ0FBQyxZQUFZLEVBQUUsYUFBYSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ3BELGFBQVEsR0FBRyx5REFBeUQsQ0FBQztRQUNyRSxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsVUFBVSxDQUFDO1FBQzlCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxTQUFTLEdBQUcsVUFBVSxDQUFDO0lBQzlCLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7TUFDZixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBaURoQixDQUFDO1FBQ0EsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIExSTkdyYWRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ2lucHV0SW1hZ2UnLCAnb3V0cHV0SW1hZ2UnLCAnZHknXTtcbiAgdW5pZm9ybXMgPSAnZGVwdGhSYWRpdXMgOiBpMzIsIGJpYXMgOiBmMzIsIGFscGhhIDogZjMyLCBiZXRhIDogZjMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGlucHV0U2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGlucHV0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ2xybl9ncmFkJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgbGV0IGIgPSBjb29yZHNbMF07XG4gICAgICAgIGxldCByID0gY29vcmRzWzFdO1xuICAgICAgICBsZXQgYyA9IGNvb3Jkc1syXTtcblxuICAgICAgICBsZXQgTUlOX0RFUFRIX0JFR0lOID0gMDtcbiAgICAgICAgbGV0IE1BWF9ERVBUSF9FTkQgPSB1bmlmb3Jtcy5vdXRTaGFwZVszXTtcbiAgICAgICAgdmFyIHJlc3VsdCA9IDAuMDtcbiAgICAgICAgZm9yICh2YXIgZCA9IE1JTl9ERVBUSF9CRUdJTjsgZCA8IE1BWF9ERVBUSF9FTkQ7IGQrKykge1xuICAgICAgICAgIGxldCBkZXB0aEJlZ2luID0gbWF4KE1JTl9ERVBUSF9CRUdJTiwgZCAtIHVuaWZvcm1zLmRlcHRoUmFkaXVzKTtcbiAgICAgICAgICBsZXQgZGVwdGhFbmQgPSBtaW4oTUFYX0RFUFRIX0VORCwgZCArIHVuaWZvcm1zLmRlcHRoUmFkaXVzICsgMSk7XG5cbiAgICAgICAgICB2YXIgbm9ybSA9IDAuMDtcbiAgICAgICAgICBmb3IgKHZhciBrID0gTUlOX0RFUFRIX0JFR0lOOyBrIDwgTUFYX0RFUFRIX0VORDsgaysrKSB7XG4gICAgICAgICAgICBpZiAoayA8IGRlcHRoQmVnaW4pIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9IGVsc2UgaWYgKGsgPj0gZGVwdGhCZWdpbiAmJiBrIDwgZGVwdGhFbmQpIHtcbiAgICAgICAgICAgICAgbm9ybSArPSBnZXRJbnB1dEltYWdlKGIsIHIsIGMsIGspICogZ2V0SW5wdXRJbWFnZShiLCByLCBjLCBrKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cblxuICAgICAgICAgIG5vcm0gPSB1bmlmb3Jtcy5hbHBoYSAqIG5vcm0gKyB1bmlmb3Jtcy5iaWFzO1xuXG4gICAgICAgICAgZm9yICh2YXIgayA9IE1JTl9ERVBUSF9CRUdJTjsgayA8IE1BWF9ERVBUSF9FTkQ7IGsrKykge1xuICAgICAgICAgICAgaWYgKGsgPCBkZXB0aEJlZ2luKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfSBlbHNlIGlmIChrID49IGRlcHRoQmVnaW4gJiYgayA8IGRlcHRoRW5kKSB7XG4gICAgICAgICAgICAgIHZhciBkeWkgPSAtMi4wICogdW5pZm9ybXMuYWxwaGEgKiB1bmlmb3Jtcy5iZXRhXG4gICAgICAgICAgICAgICAgKiBnZXRJbnB1dEltYWdlKGIsIHIsIGMsIGspICogZ2V0T3V0cHV0SW1hZ2UoYiwgciwgYywgZCkgLyBub3JtO1xuICAgICAgICAgICAgICBpZiAoayA9PSBkKSB7XG4gICAgICAgICAgICAgICAgZHlpICs9IHBvdyhub3JtLCAtMS4wICogdW5pZm9ybXMuYmV0YSk7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgaWYgKGsgPT0gY29vcmRzWzNdKSB7XG4gICAgICAgICAgICAgICAgZHlpICo9IGdldER5KGIsIHIsIGMsIGQpO1xuICAgICAgICAgICAgICAgIHJlc3VsdCArPSBkeWk7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIHJlc3VsdCk7XG4gICAgICB9XG4gICAgfVxuICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19