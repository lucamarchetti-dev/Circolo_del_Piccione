/**
 * @license
 * Copyright 2022 Google LLC.
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
export class FFTProgram {
    constructor(component, shape) {
        this.variableNames = ['real', 'imag'];
        this.outputShape = [];
        this.uniforms = 'exponentMultiplier : f32, denominator: f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.component = component;
        this.shaderKey = `fft_${component}`;
    }
    getUserCode() {
        const opString = this.component === 'real' ?
            'return real * expR - imag * expI;' :
            'return real * expI + imag * expR;';
        const userCode = `
    fn unaryOpComplex(real: f32, expR: f32, imag: f32, expI: f32) -> f32 {
      ${opString}
    }

    fn mulMatDFT(batch: i32, index: i32) -> f32 {
      let indexRatio = f32(index) / f32(uniforms.realShape[1]);
      let exponentMultiplierTimesIndexRatio =
          uniforms.exponentMultiplier * indexRatio;

      var result = 0.0;

      for (var i = 0; i < uniforms.realShape[1]; i = i + 1) {
        // x = (-2|2 * PI / N) * index * i;
        let x = exponentMultiplierTimesIndexRatio * f32(i);
        let expR = cos(x);
        let expI = sin(x);
        let real = getReal(batch, i);
        let imag = getImag(batch, i);

        result = result +
            unaryOpComplex(real, expR, imag, expI) / uniforms.denominator;
      }

      return result;
    }

    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        setOutputAtIndex(index, mulMatDFT(coords[0], coords[1]));
      }
    }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZmZ0X3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2ZmdF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxVQUFVO0lBV3JCLFlBQVksU0FBd0IsRUFBRSxLQUF1QjtRQVY3RCxrQkFBYSxHQUFhLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQzNDLGdCQUFXLEdBQWEsRUFBRSxDQUFDO1FBSTNCLGFBQVEsR0FBRyw2Q0FBNkMsQ0FBQztRQUN6RCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUlWLElBQUksQ0FBQyxXQUFXLEdBQUcsS0FBSyxDQUFDO1FBQ3pCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRS9ELElBQUksQ0FBQyxTQUFTLEdBQUcsU0FBUyxDQUFDO1FBQzNCLElBQUksQ0FBQyxTQUFTLEdBQUcsT0FBTyxTQUFTLEVBQUUsQ0FBQztJQUN0QyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxTQUFTLEtBQUssTUFBTSxDQUFDLENBQUM7WUFDeEMsbUNBQW1DLENBQUMsQ0FBQztZQUNyQyxtQ0FBbUMsQ0FBQztRQUN4QyxNQUFNLFFBQVEsR0FBRzs7UUFFYixRQUFROzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O01BeUJWLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7OztHQU1oQixDQUFDO1FBQ0EsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIEZGVFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW10gPSBbJ3JlYWwnLCAnaW1hZyddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB1bmlmb3JtcyA9ICdleHBvbmVudE11bHRpcGxpZXIgOiBmMzIsIGRlbm9taW5hdG9yOiBmMzIsJztcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG4gIGNvbXBvbmVudDogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKGNvbXBvbmVudDogJ3JlYWwnfCdpbWFnJywgc2hhcGU6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gc2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5jb21wb25lbnQgPSBjb21wb25lbnQ7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgZmZ0XyR7Y29tcG9uZW50fWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IG9wU3RyaW5nID0gdGhpcy5jb21wb25lbnQgPT09ICdyZWFsJyA/XG4gICAgICAgICdyZXR1cm4gcmVhbCAqIGV4cFIgLSBpbWFnICogZXhwSTsnIDpcbiAgICAgICAgJ3JldHVybiByZWFsICogZXhwSSArIGltYWcgKiBleHBSOyc7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgZm4gdW5hcnlPcENvbXBsZXgocmVhbDogZjMyLCBleHBSOiBmMzIsIGltYWc6IGYzMiwgZXhwSTogZjMyKSAtPiBmMzIge1xuICAgICAgJHtvcFN0cmluZ31cbiAgICB9XG5cbiAgICBmbiBtdWxNYXRERlQoYmF0Y2g6IGkzMiwgaW5kZXg6IGkzMikgLT4gZjMyIHtcbiAgICAgIGxldCBpbmRleFJhdGlvID0gZjMyKGluZGV4KSAvIGYzMih1bmlmb3Jtcy5yZWFsU2hhcGVbMV0pO1xuICAgICAgbGV0IGV4cG9uZW50TXVsdGlwbGllclRpbWVzSW5kZXhSYXRpbyA9XG4gICAgICAgICAgdW5pZm9ybXMuZXhwb25lbnRNdWx0aXBsaWVyICogaW5kZXhSYXRpbztcblxuICAgICAgdmFyIHJlc3VsdCA9IDAuMDtcblxuICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB1bmlmb3Jtcy5yZWFsU2hhcGVbMV07IGkgPSBpICsgMSkge1xuICAgICAgICAvLyB4ID0gKC0yfDIgKiBQSSAvIE4pICogaW5kZXggKiBpO1xuICAgICAgICBsZXQgeCA9IGV4cG9uZW50TXVsdGlwbGllclRpbWVzSW5kZXhSYXRpbyAqIGYzMihpKTtcbiAgICAgICAgbGV0IGV4cFIgPSBjb3MoeCk7XG4gICAgICAgIGxldCBleHBJID0gc2luKHgpO1xuICAgICAgICBsZXQgcmVhbCA9IGdldFJlYWwoYmF0Y2gsIGkpO1xuICAgICAgICBsZXQgaW1hZyA9IGdldEltYWcoYmF0Y2gsIGkpO1xuXG4gICAgICAgIHJlc3VsdCA9IHJlc3VsdCArXG4gICAgICAgICAgICB1bmFyeU9wQ29tcGxleChyZWFsLCBleHBSLCBpbWFnLCBleHBJKSAvIHVuaWZvcm1zLmRlbm9taW5hdG9yO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cblxuICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIG11bE1hdERGVChjb29yZHNbMF0sIGNvb3Jkc1sxXSkpO1xuICAgICAgfVxuICAgIH1cbiAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==