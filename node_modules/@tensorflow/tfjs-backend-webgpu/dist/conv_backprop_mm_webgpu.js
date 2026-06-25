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
import { makeMatMulPackedSource, makeMatMulPackedVec4Source } from './matmul_packed_webgpu';
import { typeSnippet } from './webgpu_program';
import { computeDispatch, computeWorkgroupSizeForConv2d, computeWorkPerThreadForConv2d } from './webgpu_util';
function conv2dTransposeCommonSnippet(innerElementSize = 4) {
    const getWSnippet = (innerElementSize) => {
        switch (innerElementSize) {
            case 1:
                return 'return W[getIndexFromCoords4D(coord, uniforms.wShape)];';
            case 4:
                return `
            let coord1 = vec4<i32>(coordX, coordY, col + 1, rowInner);
            let coord2 = vec4<i32>(coordX, coordY, col + 2, rowInner);
            let coord3 = vec4<i32>(coordX, coordY, col + 3, rowInner);
            let v0 = W[getIndexFromCoords4D(coord, uniforms.wShape)];
            let v1 = W[getIndexFromCoords4D(coord1, uniforms.wShape)];
            let v2 = W[getIndexFromCoords4D(coord2, uniforms.wShape)];
            let v3 = W[getIndexFromCoords4D(coord3, uniforms.wShape)];
            return vec4<f32>(v0, v1, v2, v3);
            `;
            default:
                throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
    };
    const readASnippet = `
      let outRow = row / uniforms.outShape[2];
      let outCol = row % uniforms.outShape[2];

      let WRow = col / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
      let WCol = col / uniforms.outBackprop[3] % uniforms.filterDims[1];
      let xR = f32(outRow - uniforms.pads[0] + WRow) / f32(uniforms.strides[0]);
      let xC = f32(outCol - uniforms.pads[1] + WCol) / f32(uniforms.strides[1]);
      if (xR < 0.0 || xR >= f32(uniforms.outBackprop[1]) || fract(xR) > 0.0) {
        return ${typeSnippet(innerElementSize)}(0.0);
      }
      if (xC < 0.0 || xC >= f32(uniforms.outBackprop[2]) || fract(xC) > 0.0) {
        return ${typeSnippet(innerElementSize)}(0.0);
      }
      let coord = vec4<i32>(
          batch,
          i32(xR),
          i32(xC),
          col % uniforms.outBackprop[3]);
      return x[getIndexFromCoords4D(coord, uniforms.xShape)/${innerElementSize}];`;
    const sampleA = `if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readASnippet}
      }
      return ${typeSnippet(innerElementSize)}(0.0);`;
    const userCode = `
  fn mm_readA(batch: i32, row : i32, col : i32) -> ${typeSnippet(innerElementSize)} {
    ${sampleA}
  }

  fn mm_readB(batch: i32, row : i32, col : i32) -> ${typeSnippet(innerElementSize)} {
    let coordX = uniforms.filterDims.x - 1 -
        row / (uniforms.filterDims[1] * uniforms.outBackprop[3]);
    let coordY = uniforms.filterDims.y - 1 -
        (row / uniforms.outBackprop[3]) % uniforms.filterDims[1];
    if (row < uniforms.dimInner && col < uniforms.dimBOuter &&
        coordX >= 0 && coordY >= 0) {
      let rowInner = row % uniforms.outBackprop[3];
      let coord = vec4<i32>(coordX, coordY, col, rowInner);
      ${getWSnippet(innerElementSize)}
    }
    return ${typeSnippet(innerElementSize)}(0.0);
  }

  fn mm_write(batch: i32, row : i32, col : i32, valueInput : ${typeSnippet(innerElementSize)}) {
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
      var value = valueInput;
      let outCoord = vec4<i32>(
          batch,
          row / uniforms.outShape[2],
          row % uniforms.outShape[2],
          col);
      result[getIndexFromCoords4D(outCoord, uniforms.outShape)/${innerElementSize}] = value;
    }
  }`;
    return userCode;
}
export class Conv2DDerInputMMProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, outBackprop : vec4<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,';
        this.outputShape = convInfo.inShape;
        util.assert(convInfo.dataFormat === 'channelsLast', () => 'TODO: NCHW is unimplemented');
        this.isVec4 =
            convInfo.inChannels % 4 === 0 && convInfo.outChannels % 4 === 0;
        this.dispatchLayout = { x: [3], y: [1, 2], z: [0] };
        this.workgroupSize = computeWorkgroupSizeForConv2d(this.dispatchLayout, this.outputShape, this.isVec4);
        this.elementsPerThread = computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape, this.isVec4);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, this.elementsPerThread);
        if (this.isVec4) {
            this.outputComponent = 4;
            this.variableComponents = [4, 1];
        }
        this.shaderKey =
            `conv2DDerInputMM_${this.isVec4}_${this.elementsPerThread}`;
    }
    getUserCode() {
        const matMulSource = this.isVec4 ?
            makeMatMulPackedVec4Source(this.elementsPerThread, this.workgroupSize) :
            makeMatMulPackedSource(this.elementsPerThread, this.workgroupSize);
        const userCode = `
    ${conv2dTransposeCommonSnippet(this.isVec4 ? 4 : 1)}
    ${matMulSource}
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udl9iYWNrcHJvcF9tbV93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9jb252X2JhY2twcm9wX21tX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQWUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFekQsT0FBTyxFQUFDLHNCQUFzQixFQUFFLDBCQUEwQixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDMUYsT0FBTyxFQUFDLFdBQVcsRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RCxPQUFPLEVBQUMsZUFBZSxFQUFFLDZCQUE2QixFQUFFLDZCQUE2QixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRTVHLFNBQVMsNEJBQTRCLENBQUMsZ0JBQWdCLEdBQUcsQ0FBQztJQUN4RCxNQUFNLFdBQVcsR0FBRyxDQUFDLGdCQUF3QixFQUFFLEVBQUU7UUFDL0MsUUFBUSxnQkFBZ0IsRUFBRTtZQUN4QixLQUFLLENBQUM7Z0JBQ0osT0FBTyx5REFBeUQsQ0FBQztZQUNuRSxLQUFLLENBQUM7Z0JBQ0osT0FBTzs7Ozs7Ozs7O2FBU0YsQ0FBQztZQUNSO2dCQUNFLE1BQU0sSUFBSSxLQUFLLENBQ1gsb0JBQW9CLGdCQUFnQixvQkFBb0IsQ0FBQyxDQUFDO1NBQ2pFO0lBQ0gsQ0FBQyxDQUFDO0lBRUYsTUFBTSxZQUFZLEdBQUc7Ozs7Ozs7OztpQkFTTixXQUFXLENBQUMsZ0JBQWdCLENBQUM7OztpQkFHN0IsV0FBVyxDQUFDLGdCQUFnQixDQUFDOzs7Ozs7OzhEQVF4QyxnQkFBZ0IsSUFBSSxDQUFDO0lBRXpCLE1BQU0sT0FBTyxHQUFHO1VBQ1IsWUFBWTs7ZUFFUCxXQUFXLENBQUMsZ0JBQWdCLENBQUMsUUFBUSxDQUFDO0lBRW5ELE1BQU0sUUFBUSxHQUFHO3FEQUViLFdBQVcsQ0FBQyxnQkFBZ0IsQ0FBQztNQUM3QixPQUFPOzs7cURBSVAsV0FBVyxDQUFDLGdCQUFnQixDQUFDOzs7Ozs7Ozs7UUFTM0IsV0FBVyxDQUFDLGdCQUFnQixDQUFDOzthQUV4QixXQUFXLENBQUMsZ0JBQWdCLENBQUM7OzsrREFJcEMsV0FBVyxDQUFDLGdCQUFnQixDQUFDOzs7Ozs7OztpRUFTN0IsZ0JBQWdCOztJQUVsQixDQUFDO0lBQ0gsT0FBTyxRQUFRLENBQUM7QUFDbEIsQ0FBQztBQUVELE1BQU0sT0FBTyx1QkFBdUI7SUFjbEMsWUFBWSxRQUFpQztRQVQ3QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBRTNCLGFBQVEsR0FDSiwySUFBMkksQ0FBQztRQU85SSxJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUM7UUFFcEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsVUFBVSxLQUFLLGNBQWMsRUFDdEMsR0FBRyxFQUFFLENBQUMsNkJBQTZCLENBQUMsQ0FBQztRQUN6QyxJQUFJLENBQUMsTUFBTTtZQUNQLFFBQVEsQ0FBQyxVQUFVLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsV0FBVyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDcEUsSUFBSSxDQUFDLGNBQWMsR0FBRyxFQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDO1FBQ2xELElBQUksQ0FBQyxhQUFhLEdBQUcsNkJBQTZCLENBQzlDLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDeEQsSUFBSSxDQUFDLGlCQUFpQixHQUFHLDZCQUE2QixDQUNsRCxJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRXhELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsRUFDekQsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUM7UUFFNUIsSUFBSSxJQUFJLENBQUMsTUFBTSxFQUFFO1lBQ2YsSUFBSSxDQUFDLGVBQWUsR0FBRyxDQUFDLENBQUM7WUFDekIsSUFBSSxDQUFDLGtCQUFrQixHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1NBQ2xDO1FBRUQsSUFBSSxDQUFDLFNBQVM7WUFDVixvQkFBb0IsSUFBSSxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsaUJBQWlCLEVBQUUsQ0FBQztJQUNsRSxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sWUFBWSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUM5QiwwQkFBMEIsQ0FBQyxJQUFJLENBQUMsaUJBQWlCLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7WUFDeEUsc0JBQXNCLENBQUMsSUFBSSxDQUFDLGlCQUFpQixFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUN2RSxNQUFNLFFBQVEsR0FBRztNQUNmLDRCQUE0QixDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO01BQ2pELFlBQVk7S0FDYixDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHttYWtlTWF0TXVsUGFja2VkU291cmNlLCBtYWtlTWF0TXVsUGFja2VkVmVjNFNvdXJjZX0gZnJvbSAnLi9tYXRtdWxfcGFja2VkX3dlYmdwdSc7XG5pbXBvcnQge3R5cGVTbmlwcGV0LCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBjb21wdXRlV29ya2dyb3VwU2l6ZUZvckNvbnYyZCwgY29tcHV0ZVdvcmtQZXJUaHJlYWRGb3JDb252MmR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5mdW5jdGlvbiBjb252MmRUcmFuc3Bvc2VDb21tb25TbmlwcGV0KGlubmVyRWxlbWVudFNpemUgPSA0KSB7XG4gIGNvbnN0IGdldFdTbmlwcGV0ID0gKGlubmVyRWxlbWVudFNpemU6IG51bWJlcikgPT4ge1xuICAgIHN3aXRjaCAoaW5uZXJFbGVtZW50U2l6ZSkge1xuICAgICAgY2FzZSAxOlxuICAgICAgICByZXR1cm4gJ3JldHVybiBXW2dldEluZGV4RnJvbUNvb3JkczREKGNvb3JkLCB1bmlmb3Jtcy53U2hhcGUpXTsnO1xuICAgICAgY2FzZSA0OlxuICAgICAgICByZXR1cm4gYFxuICAgICAgICAgICAgbGV0IGNvb3JkMSA9IHZlYzQ8aTMyPihjb29yZFgsIGNvb3JkWSwgY29sICsgMSwgcm93SW5uZXIpO1xuICAgICAgICAgICAgbGV0IGNvb3JkMiA9IHZlYzQ8aTMyPihjb29yZFgsIGNvb3JkWSwgY29sICsgMiwgcm93SW5uZXIpO1xuICAgICAgICAgICAgbGV0IGNvb3JkMyA9IHZlYzQ8aTMyPihjb29yZFgsIGNvb3JkWSwgY29sICsgMywgcm93SW5uZXIpO1xuICAgICAgICAgICAgbGV0IHYwID0gV1tnZXRJbmRleEZyb21Db29yZHM0RChjb29yZCwgdW5pZm9ybXMud1NoYXBlKV07XG4gICAgICAgICAgICBsZXQgdjEgPSBXW2dldEluZGV4RnJvbUNvb3JkczREKGNvb3JkMSwgdW5pZm9ybXMud1NoYXBlKV07XG4gICAgICAgICAgICBsZXQgdjIgPSBXW2dldEluZGV4RnJvbUNvb3JkczREKGNvb3JkMiwgdW5pZm9ybXMud1NoYXBlKV07XG4gICAgICAgICAgICBsZXQgdjMgPSBXW2dldEluZGV4RnJvbUNvb3JkczREKGNvb3JkMywgdW5pZm9ybXMud1NoYXBlKV07XG4gICAgICAgICAgICByZXR1cm4gdmVjNDxmMzI+KHYwLCB2MSwgdjIsIHYzKTtcbiAgICAgICAgICAgIGA7XG4gICAgICBkZWZhdWx0OlxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgaW5uZXJFbGVtZW50U2l6ZSAke2lubmVyRWxlbWVudFNpemV9IGlzIG5vdCBzdXBwb3J0ZWQuYCk7XG4gICAgfVxuICB9O1xuXG4gIGNvbnN0IHJlYWRBU25pcHBldCA9IGBcbiAgICAgIGxldCBvdXRSb3cgPSByb3cgLyB1bmlmb3Jtcy5vdXRTaGFwZVsyXTtcbiAgICAgIGxldCBvdXRDb2wgPSByb3cgJSB1bmlmb3Jtcy5vdXRTaGFwZVsyXTtcblxuICAgICAgbGV0IFdSb3cgPSBjb2wgLyAodW5pZm9ybXMuZmlsdGVyRGltc1sxXSAqIHVuaWZvcm1zLm91dEJhY2twcm9wWzNdKTtcbiAgICAgIGxldCBXQ29sID0gY29sIC8gdW5pZm9ybXMub3V0QmFja3Byb3BbM10gJSB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdO1xuICAgICAgbGV0IHhSID0gZjMyKG91dFJvdyAtIHVuaWZvcm1zLnBhZHNbMF0gKyBXUm93KSAvIGYzMih1bmlmb3Jtcy5zdHJpZGVzWzBdKTtcbiAgICAgIGxldCB4QyA9IGYzMihvdXRDb2wgLSB1bmlmb3Jtcy5wYWRzWzFdICsgV0NvbCkgLyBmMzIodW5pZm9ybXMuc3RyaWRlc1sxXSk7XG4gICAgICBpZiAoeFIgPCAwLjAgfHwgeFIgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzFdKSB8fCBmcmFjdCh4UikgPiAwLjApIHtcbiAgICAgICAgcmV0dXJuICR7dHlwZVNuaXBwZXQoaW5uZXJFbGVtZW50U2l6ZSl9KDAuMCk7XG4gICAgICB9XG4gICAgICBpZiAoeEMgPCAwLjAgfHwgeEMgPj0gZjMyKHVuaWZvcm1zLm91dEJhY2twcm9wWzJdKSB8fCBmcmFjdCh4QykgPiAwLjApIHtcbiAgICAgICAgcmV0dXJuICR7dHlwZVNuaXBwZXQoaW5uZXJFbGVtZW50U2l6ZSl9KDAuMCk7XG4gICAgICB9XG4gICAgICBsZXQgY29vcmQgPSB2ZWM0PGkzMj4oXG4gICAgICAgICAgYmF0Y2gsXG4gICAgICAgICAgaTMyKHhSKSxcbiAgICAgICAgICBpMzIoeEMpLFxuICAgICAgICAgIGNvbCAlIHVuaWZvcm1zLm91dEJhY2twcm9wWzNdKTtcbiAgICAgIHJldHVybiB4W2dldEluZGV4RnJvbUNvb3JkczREKGNvb3JkLCB1bmlmb3Jtcy54U2hhcGUpLyR7XG4gICAgICBpbm5lckVsZW1lbnRTaXplfV07YDtcblxuICBjb25zdCBzYW1wbGVBID0gYGlmIChyb3cgPCB1bmlmb3Jtcy5kaW1BT3V0ZXIgJiYgY29sIDwgdW5pZm9ybXMuZGltSW5uZXIpIHtcbiAgICAgICAgJHtyZWFkQVNuaXBwZXR9XG4gICAgICB9XG4gICAgICByZXR1cm4gJHt0eXBlU25pcHBldChpbm5lckVsZW1lbnRTaXplKX0oMC4wKTtgO1xuXG4gIGNvbnN0IHVzZXJDb2RlID0gYFxuICBmbiBtbV9yZWFkQShiYXRjaDogaTMyLCByb3cgOiBpMzIsIGNvbCA6IGkzMikgLT4gJHtcbiAgICAgIHR5cGVTbmlwcGV0KGlubmVyRWxlbWVudFNpemUpfSB7XG4gICAgJHtzYW1wbGVBfVxuICB9XG5cbiAgZm4gbW1fcmVhZEIoYmF0Y2g6IGkzMiwgcm93IDogaTMyLCBjb2wgOiBpMzIpIC0+ICR7XG4gICAgICB0eXBlU25pcHBldChpbm5lckVsZW1lbnRTaXplKX0ge1xuICAgIGxldCBjb29yZFggPSB1bmlmb3Jtcy5maWx0ZXJEaW1zLnggLSAxIC1cbiAgICAgICAgcm93IC8gKHVuaWZvcm1zLmZpbHRlckRpbXNbMV0gKiB1bmlmb3Jtcy5vdXRCYWNrcHJvcFszXSk7XG4gICAgbGV0IGNvb3JkWSA9IHVuaWZvcm1zLmZpbHRlckRpbXMueSAtIDEgLVxuICAgICAgICAocm93IC8gdW5pZm9ybXMub3V0QmFja3Byb3BbM10pICUgdW5pZm9ybXMuZmlsdGVyRGltc1sxXTtcbiAgICBpZiAocm93IDwgdW5pZm9ybXMuZGltSW5uZXIgJiYgY29sIDwgdW5pZm9ybXMuZGltQk91dGVyICYmXG4gICAgICAgIGNvb3JkWCA+PSAwICYmIGNvb3JkWSA+PSAwKSB7XG4gICAgICBsZXQgcm93SW5uZXIgPSByb3cgJSB1bmlmb3Jtcy5vdXRCYWNrcHJvcFszXTtcbiAgICAgIGxldCBjb29yZCA9IHZlYzQ8aTMyPihjb29yZFgsIGNvb3JkWSwgY29sLCByb3dJbm5lcik7XG4gICAgICAke2dldFdTbmlwcGV0KGlubmVyRWxlbWVudFNpemUpfVxuICAgIH1cbiAgICByZXR1cm4gJHt0eXBlU25pcHBldChpbm5lckVsZW1lbnRTaXplKX0oMC4wKTtcbiAgfVxuXG4gIGZuIG1tX3dyaXRlKGJhdGNoOiBpMzIsIHJvdyA6IGkzMiwgY29sIDogaTMyLCB2YWx1ZUlucHV0IDogJHtcbiAgICAgIHR5cGVTbmlwcGV0KGlubmVyRWxlbWVudFNpemUpfSkge1xuICAgIGlmIChyb3cgPCB1bmlmb3Jtcy5kaW1BT3V0ZXIgJiYgY29sIDwgdW5pZm9ybXMuZGltQk91dGVyKSB7XG4gICAgICB2YXIgdmFsdWUgPSB2YWx1ZUlucHV0O1xuICAgICAgbGV0IG91dENvb3JkID0gdmVjNDxpMzI+KFxuICAgICAgICAgIGJhdGNoLFxuICAgICAgICAgIHJvdyAvIHVuaWZvcm1zLm91dFNoYXBlWzJdLFxuICAgICAgICAgIHJvdyAlIHVuaWZvcm1zLm91dFNoYXBlWzJdLFxuICAgICAgICAgIGNvbCk7XG4gICAgICByZXN1bHRbZ2V0SW5kZXhGcm9tQ29vcmRzNEQob3V0Q29vcmQsIHVuaWZvcm1zLm91dFNoYXBlKS8ke1xuICAgICAgaW5uZXJFbGVtZW50U2l6ZX1dID0gdmFsdWU7XG4gICAgfVxuICB9YDtcbiAgcmV0dXJuIHVzZXJDb2RlO1xufVxuXG5leHBvcnQgY2xhc3MgQ29udjJERGVySW5wdXRNTVByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXSwgeTogbnVtYmVyW10sIHo6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdXJ107XG4gIHZhcmlhYmxlQ29tcG9uZW50czogbnVtYmVyW107XG4gIHVuaWZvcm1zID1cbiAgICAgICdmaWx0ZXJEaW1zIDogdmVjMjxpMzI+LCBwYWRzIDogdmVjMjxpMzI+LCBzdHJpZGVzIDogdmVjMjxpMzI+LCBvdXRCYWNrcHJvcCA6IHZlYzQ8aTMyPiwgZGltQU91dGVyIDogaTMyLCBkaW1CT3V0ZXIgOiBpMzIsIGRpbUlubmVyIDogaTMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgZWxlbWVudHNQZXJUaHJlYWQ6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgaXNWZWM0PzogYm9vbGVhbjtcbiAgb3V0cHV0Q29tcG9uZW50OiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLmluU2hhcGU7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgY29udkluZm8uZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcsXG4gICAgICAgICgpID0+ICdUT0RPOiBOQ0hXIGlzIHVuaW1wbGVtZW50ZWQnKTtcbiAgICB0aGlzLmlzVmVjNCA9XG4gICAgICAgIGNvbnZJbmZvLmluQ2hhbm5lbHMgJSA0ID09PSAwICYmIGNvbnZJbmZvLm91dENoYW5uZWxzICUgNCA9PT0gMDtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0ge3g6IFszXSwgeTogWzEsIDJdLCB6OiBbMF19O1xuICAgIHRoaXMud29ya2dyb3VwU2l6ZSA9IGNvbXB1dGVXb3JrZ3JvdXBTaXplRm9yQ29udjJkKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLmlzVmVjNCk7XG4gICAgdGhpcy5lbGVtZW50c1BlclRocmVhZCA9IGNvbXB1dGVXb3JrUGVyVGhyZWFkRm9yQ29udjJkKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLmlzVmVjNCk7XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIHRoaXMuZWxlbWVudHNQZXJUaHJlYWQpO1xuXG4gICAgaWYgKHRoaXMuaXNWZWM0KSB7XG4gICAgICB0aGlzLm91dHB1dENvbXBvbmVudCA9IDQ7XG4gICAgICB0aGlzLnZhcmlhYmxlQ29tcG9uZW50cyA9IFs0LCAxXTtcbiAgICB9XG5cbiAgICB0aGlzLnNoYWRlcktleSA9XG4gICAgICAgIGBjb252MkREZXJJbnB1dE1NXyR7dGhpcy5pc1ZlYzR9XyR7dGhpcy5lbGVtZW50c1BlclRocmVhZH1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCBtYXRNdWxTb3VyY2UgPSB0aGlzLmlzVmVjNCA/XG4gICAgICAgIG1ha2VNYXRNdWxQYWNrZWRWZWM0U291cmNlKHRoaXMuZWxlbWVudHNQZXJUaHJlYWQsIHRoaXMud29ya2dyb3VwU2l6ZSkgOlxuICAgICAgICBtYWtlTWF0TXVsUGFja2VkU291cmNlKHRoaXMuZWxlbWVudHNQZXJUaHJlYWQsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHtjb252MmRUcmFuc3Bvc2VDb21tb25TbmlwcGV0KHRoaXMuaXNWZWM0ID8gNCA6IDEpfVxuICAgICR7bWF0TXVsU291cmNlfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=