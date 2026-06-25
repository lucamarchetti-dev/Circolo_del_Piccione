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
export function makeMatMulSmallOutputSizeSource(workgroupSize) {
    const tileAOuter = workgroupSize[1];
    const tileBOuter = workgroupSize[0];
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    return `
  var<workgroup> mm_Asub : array<array<f32, ${tileInner}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Read data from global memory to registers firstly, then store them into
  // shared memory, so it is instruction-Level parallelism for arithmetic
  // operations and others handle IO operations between barrier api, makes ALU
  // and load/store units work simultaneously, could improves the performance.
  ${main()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${tileInner} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = 0;
    var regA = mm_readA(batchA, globalRow, globalColA);
    var regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
    var regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
    globalColA = globalColA + ${tileInner};
    globalRowB = globalRowB + ${tileInner};

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = regA;
      mm_Bsub[2 * tileRow][tileCol] = regB0;
      mm_Bsub[2 * tileRow + 1][tileCol] = regB1;

      workgroupBarrier();

      regA = mm_readA(batchA, globalRow, globalColA);
      regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
      regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
      globalColA = globalColA + ${tileInner};
      globalRowB = globalRowB + ${tileInner};

      for (var k = 0; k < ${tileInner}; k = k + 1) {
        acc = acc + mm_Asub[tileRow][k] * mm_Bsub[k][tileCol];
      }
      workgroupBarrier();
    }

    mm_write(batch, globalRow, globalCol, acc);
  }
  `;
}
export class MatMulSmallOutputSizeProgram {
    constructor(aShape, bShape, outputShape, transposeA = false, transposeB = false, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.workgroupSize = [16, 8, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [2], y: [1], z: [0] };
        this.dispatch = [
            Math.ceil(outputShape[2] / this.workgroupSize[0]),
            Math.ceil(outputShape[1] / this.workgroupSize[1]), outputShape[0]
        ];
        const addBias = bias != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.shaderKey =
            `matMulSmallOutputSize_${this.activation}_${transposeA}_${transposeB}`;
    }
    getUserCode() {
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
      ${matMulReadWriteFnSource(this.addBias, this.activation, this.transposeA, this.transposeB)}
      ${makeMatMulSmallOutputSizeSource(this.workgroupSize)}
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF0bXVsX3NtYWxsX291dHB1dF9zaXplX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL21hdG11bF9zbWFsbF9vdXRwdXRfc2l6ZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDdEQsT0FBTyxFQUFDLHVCQUF1QixFQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUU1RSxNQUFNLFVBQVUsK0JBQStCLENBQzNDLGFBQXVDO0lBQ3pDLE1BQU0sVUFBVSxHQUFHLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQyxNQUFNLFVBQVUsR0FBRyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDcEMsTUFBTSxTQUFTLEdBQUcsVUFBVSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUM7SUFDcEUsT0FBTzs4Q0FDcUMsU0FBUyxNQUFNLFVBQVU7OENBQ3pCLFVBQVUsTUFBTSxTQUFTOzs7Ozs7OztJQVFuRSxJQUFJLEVBQUU7Ozs7Ozs7Ozs7K0NBVXFDLFNBQVM7Ozs7Ozs7O2dDQVF4QixTQUFTO2dDQUNULFNBQVM7Ozs7Ozs7Ozs7OztrQ0FZUCxTQUFTO2tDQUNULFNBQVM7OzRCQUVmLFNBQVM7Ozs7Ozs7O0dBUWxDLENBQUM7QUFDSixDQUFDO0FBRUQsTUFBTSxPQUFPLDRCQUE0QjtJQWN2QyxZQUNJLE1BQWdDLEVBQUUsTUFBZ0MsRUFDbEUsV0FBcUMsRUFBRSxVQUFVLEdBQUcsS0FBSyxFQUN6RCxVQUFVLEdBQUcsS0FBSyxFQUFFLE9BQW1CLElBQUksRUFDM0MsYUFBc0MsSUFBSSxFQUMxQyx5QkFBcUMsSUFBSTtRQWQ3QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzNCLGFBQVEsR0FBRyxtREFBbUQsQ0FBQztRQUMvRCxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFhbkQsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFFL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxFQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUM7UUFDL0MsSUFBSSxDQUFDLFFBQVEsR0FBRztZQUNkLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakQsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUM7U0FDbEUsQ0FBQztRQUVGLE1BQU0sT0FBTyxHQUFHLElBQUksSUFBSSxJQUFJLENBQUM7UUFDN0IsSUFBSSxPQUFPLEVBQUU7WUFDWCxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztTQUNqQztRQUVELE1BQU0seUJBQXlCLEdBQUcsc0JBQXNCLElBQUksSUFBSSxDQUFDO1FBQ2pFLElBQUkseUJBQXlCLEVBQUU7WUFDN0IsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQztTQUNuRDtRQUVELElBQUksQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDO1FBQzdCLElBQUksQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDO1FBQzdCLElBQUksQ0FBQyxPQUFPLEdBQUcsT0FBTyxDQUFDO1FBQ3ZCLElBQUksQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDO1FBQzdCLElBQUksQ0FBQyx5QkFBeUIsR0FBRyx5QkFBeUIsQ0FBQztRQUMzRCxJQUFJLENBQUMsU0FBUztZQUNWLHlCQUF5QixJQUFJLENBQUMsVUFBVSxJQUFJLFVBQVUsSUFBSSxVQUFVLEVBQUUsQ0FBQztJQUM3RSxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsbUJBQW1CLENBQUMsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMseUJBQXlCLENBQUM7UUFFcEUsdUJBQXVCLENBQ25CLElBQUksQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyxVQUFVLENBQUM7UUFDcEUsK0JBQStCLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQztLQUN0RCxDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7YWN0aXZhdGlvbkZuU25pcHBldH0gZnJvbSAnLi9hY3RpdmF0aW9uX3V0aWwnO1xuaW1wb3J0IHttYXRNdWxSZWFkV3JpdGVGblNvdXJjZX0gZnJvbSAnLi9tYXRtdWxfcGFja2VkX3dlYmdwdSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlTWF0TXVsU21hbGxPdXRwdXRTaXplU291cmNlKFxuICAgIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGNvbnN0IHRpbGVBT3V0ZXIgPSB3b3JrZ3JvdXBTaXplWzFdO1xuICBjb25zdCB0aWxlQk91dGVyID0gd29ya2dyb3VwU2l6ZVswXTtcbiAgY29uc3QgdGlsZUlubmVyID0gdGlsZUFPdXRlciA+IHRpbGVCT3V0ZXIgPyB0aWxlQU91dGVyIDogdGlsZUJPdXRlcjtcbiAgcmV0dXJuIGBcbiAgdmFyPHdvcmtncm91cD4gbW1fQXN1YiA6IGFycmF5PGFycmF5PGYzMiwgJHt0aWxlSW5uZXJ9PiwgJHt0aWxlQU91dGVyfT47XG4gIHZhcjx3b3JrZ3JvdXA+IG1tX0JzdWIgOiBhcnJheTxhcnJheTxmMzIsICR7dGlsZUJPdXRlcn0+LCAke3RpbGVJbm5lcn0+O1xuXG4gIC8vIElmIHRoZSBvdXRwdXQgc2l6ZSBpcyBzbWFsbCBmb3IgbWF0cml4IG11bHRpcGxpY2F0aW9uLCBhdm9pZCB0byB1c2UgdmVjNFxuICAvLyBhbmQgaGFuZGxlIHNvbWUgZWxlbWVudHMgcGVyIHRocmVhZCB0byBvcHRpbWFsbHkgdXRpbGl6ZSB0aGUgQUxVLlxuICAvLyBSZWFkIGRhdGEgZnJvbSBnbG9iYWwgbWVtb3J5IHRvIHJlZ2lzdGVycyBmaXJzdGx5LCB0aGVuIHN0b3JlIHRoZW0gaW50b1xuICAvLyBzaGFyZWQgbWVtb3J5LCBzbyBpdCBpcyBpbnN0cnVjdGlvbi1MZXZlbCBwYXJhbGxlbGlzbSBmb3IgYXJpdGhtZXRpY1xuICAvLyBvcGVyYXRpb25zIGFuZCBvdGhlcnMgaGFuZGxlIElPIG9wZXJhdGlvbnMgYmV0d2VlbiBiYXJyaWVyIGFwaSwgbWFrZXMgQUxVXG4gIC8vIGFuZCBsb2FkL3N0b3JlIHVuaXRzIHdvcmsgc2ltdWx0YW5lb3VzbHksIGNvdWxkIGltcHJvdmVzIHRoZSBwZXJmb3JtYW5jZS5cbiAgJHttYWluKCl9IHtcbiAgICBsZXQgdGlsZVJvdyA9IGkzMihsb2NhbElkLnkpO1xuICAgIGxldCB0aWxlQ29sID0gaTMyKGxvY2FsSWQueCk7XG4gICAgbGV0IGdsb2JhbFJvdyA9IGkzMihnbG9iYWxJZC55KTtcbiAgICBsZXQgZ2xvYmFsQ29sID0gaTMyKGdsb2JhbElkLngpO1xuICAgIGxldCBiYXRjaCA9IGkzMihnbG9iYWxJZC56KTtcbiAgICBsZXQgYmF0Y2hBID0gYmF0Y2ggJSB1bmlmb3Jtcy5hU2hhcGVbMF07XG4gICAgbGV0IGJhdGNoQiA9IGJhdGNoICUgdW5pZm9ybXMuYlNoYXBlWzBdO1xuXG4gICAgLy8gdW5pZm9ybXMuZGltSW5uZXIgc2hvdWxkIGJlIGdyZWF0ZXIgdGhhbiAwLlxuICAgIGxldCBudW1UaWxlcyA9ICh1bmlmb3Jtcy5kaW1Jbm5lciAtIDEpIC8gJHt0aWxlSW5uZXJ9ICsgMTtcbiAgICB2YXIgYWNjID0gMC4wO1xuXG4gICAgdmFyIGdsb2JhbENvbEEgPSB0aWxlQ29sO1xuICAgIHZhciBnbG9iYWxSb3dCID0gMDtcbiAgICB2YXIgcmVnQSA9IG1tX3JlYWRBKGJhdGNoQSwgZ2xvYmFsUm93LCBnbG9iYWxDb2xBKTtcbiAgICB2YXIgcmVnQjAgPSBtbV9yZWFkQihiYXRjaEIsIGdsb2JhbFJvd0IgKyAyICogdGlsZVJvdywgZ2xvYmFsQ29sKTtcbiAgICB2YXIgcmVnQjEgPSBtbV9yZWFkQihiYXRjaEIsIGdsb2JhbFJvd0IgKyAyICogdGlsZVJvdyArIDEsIGdsb2JhbENvbCk7XG4gICAgZ2xvYmFsQ29sQSA9IGdsb2JhbENvbEEgKyAke3RpbGVJbm5lcn07XG4gICAgZ2xvYmFsUm93QiA9IGdsb2JhbFJvd0IgKyAke3RpbGVJbm5lcn07XG5cbiAgICBmb3IgKHZhciB0ID0gMDsgdCA8IG51bVRpbGVzOyB0ID0gdCArIDEpIHtcbiAgICAgIG1tX0FzdWJbdGlsZVJvd11bdGlsZUNvbF0gPSByZWdBO1xuICAgICAgbW1fQnN1YlsyICogdGlsZVJvd11bdGlsZUNvbF0gPSByZWdCMDtcbiAgICAgIG1tX0JzdWJbMiAqIHRpbGVSb3cgKyAxXVt0aWxlQ29sXSA9IHJlZ0IxO1xuXG4gICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgICAgIHJlZ0EgPSBtbV9yZWFkQShiYXRjaEEsIGdsb2JhbFJvdywgZ2xvYmFsQ29sQSk7XG4gICAgICByZWdCMCA9IG1tX3JlYWRCKGJhdGNoQiwgZ2xvYmFsUm93QiArIDIgKiB0aWxlUm93LCBnbG9iYWxDb2wpO1xuICAgICAgcmVnQjEgPSBtbV9yZWFkQihiYXRjaEIsIGdsb2JhbFJvd0IgKyAyICogdGlsZVJvdyArIDEsIGdsb2JhbENvbCk7XG4gICAgICBnbG9iYWxDb2xBID0gZ2xvYmFsQ29sQSArICR7dGlsZUlubmVyfTtcbiAgICAgIGdsb2JhbFJvd0IgPSBnbG9iYWxSb3dCICsgJHt0aWxlSW5uZXJ9O1xuXG4gICAgICBmb3IgKHZhciBrID0gMDsgayA8ICR7dGlsZUlubmVyfTsgayA9IGsgKyAxKSB7XG4gICAgICAgIGFjYyA9IGFjYyArIG1tX0FzdWJbdGlsZVJvd11ba10gKiBtbV9Cc3ViW2tdW3RpbGVDb2xdO1xuICAgICAgfVxuICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuICAgIH1cblxuICAgIG1tX3dyaXRlKGJhdGNoLCBnbG9iYWxSb3csIGdsb2JhbENvbCwgYWNjKTtcbiAgfVxuICBgO1xufVxuXG5leHBvcnQgY2xhc3MgTWF0TXVsU21hbGxPdXRwdXRTaXplUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXSwgejogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWydBJywgJ0InXTtcbiAgdW5pZm9ybXMgPSBgZGltQU91dGVyIDogaTMyLCBkaW1CT3V0ZXIgOiBpMzIsIGRpbUlubmVyIDogaTMyLGA7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxNiwgOCwgMV07XG4gIHRyYW5zcG9zZUE6IGJvb2xlYW47XG4gIHRyYW5zcG9zZUI6IGJvb2xlYW47XG4gIGFkZEJpYXM6IGJvb2xlYW47XG4gIGFjdGl2YXRpb246IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uO1xuICBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzOiBib29sZWFuO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgYVNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgb3V0cHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgdHJhbnNwb3NlQSA9IGZhbHNlLFxuICAgICAgdHJhbnNwb3NlQiA9IGZhbHNlLCBiaWFzOiBUZW5zb3JJbmZvID0gbnVsbCxcbiAgICAgIGFjdGl2YXRpb246IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uID0gbnVsbCxcbiAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHM6IFRlbnNvckluZm8gPSBudWxsKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dFNoYXBlO1xuXG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IHt4OiBbMl0sIHk6IFsxXSwgejogWzBdfTtcbiAgICB0aGlzLmRpc3BhdGNoID0gW1xuICAgICAgTWF0aC5jZWlsKG91dHB1dFNoYXBlWzJdIC8gdGhpcy53b3JrZ3JvdXBTaXplWzBdKSxcbiAgICAgIE1hdGguY2VpbChvdXRwdXRTaGFwZVsxXSAvIHRoaXMud29ya2dyb3VwU2l6ZVsxXSksIG91dHB1dFNoYXBlWzBdXG4gICAgXTtcblxuICAgIGNvbnN0IGFkZEJpYXMgPSBiaWFzICE9IG51bGw7XG4gICAgaWYgKGFkZEJpYXMpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdiaWFzJyk7XG4gICAgfVxuXG4gICAgY29uc3QgaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyA9IHByZWx1QWN0aXZhdGlvbldlaWdodHMgIT0gbnVsbDtcbiAgICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cykge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ3ByZWx1QWN0aXZhdGlvbldlaWdodHMnKTtcbiAgICB9XG5cbiAgICB0aGlzLnRyYW5zcG9zZUEgPSB0cmFuc3Bvc2VBO1xuICAgIHRoaXMudHJhbnNwb3NlQiA9IHRyYW5zcG9zZUI7XG4gICAgdGhpcy5hZGRCaWFzID0gYWRkQmlhcztcbiAgICB0aGlzLmFjdGl2YXRpb24gPSBhY3RpdmF0aW9uO1xuICAgIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyA9IGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHM7XG4gICAgdGhpcy5zaGFkZXJLZXkgPVxuICAgICAgICBgbWF0TXVsU21hbGxPdXRwdXRTaXplXyR7dGhpcy5hY3RpdmF0aW9ufV8ke3RyYW5zcG9zZUF9XyR7dHJhbnNwb3NlQn1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7YWN0aXZhdGlvbkZuU25pcHBldCh0aGlzLmFjdGl2YXRpb24sIHRoaXMuaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyl9XG4gICAgICAke1xuICAgICAgICBtYXRNdWxSZWFkV3JpdGVGblNvdXJjZShcbiAgICAgICAgICAgIHRoaXMuYWRkQmlhcywgdGhpcy5hY3RpdmF0aW9uLCB0aGlzLnRyYW5zcG9zZUEsIHRoaXMudHJhbnNwb3NlQil9XG4gICAgICAke21ha2VNYXRNdWxTbWFsbE91dHB1dFNpemVTb3VyY2UodGhpcy53b3JrZ3JvdXBTaXplKX1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19