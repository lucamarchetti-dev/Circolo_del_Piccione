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
import { util } from '@tensorflow/tfjs-core';
import { activationFnSnippet, biasActivationSnippet } from './activation_util';
import { getMainHeaderString as main, typeSnippet } from './webgpu_program';
import { computeDispatch, computeWorkgroupInfoForMatMul } from './webgpu_util';
export function matMulReadFnSource(transposeA, transposeB, fitAOuter = false, fitBOuter = false, fitInner = false, component = 1) {
    util.assert(transposeA && component === 1 || !transposeA, () => `transposeA ${transposeA} is not compatible with component size ${component}`);
    const sampleA = `
      ${transposeA ? `value = getA(batch, col, row);` :
        `value = getA(batch, row, col);`}

    `;
    const sampleB = transposeB ? `value = getB(batch, col, row);` :
        `value = getB(batch, row, col);`;
    return `
  fn mm_readA(batch: i32, row: i32, col: i32) -> ${typeSnippet(component)} {
    var value = ${typeSnippet(component)}(0.0);
    ${fitAOuter && fitInner ?
        sampleA :
        `
    ${transposeA ?
            `if(row < uniforms.dimAOuter && col < uniforms.dimInner)` :
            `if(row < uniforms.aShape[1] && col < uniforms.aShape[2])`}
    {
      ${sampleA}
    }
    `}
    return value;
  }

  fn mm_readB(batch: i32, row: i32, col: i32) -> ${typeSnippet(component)} {
    var value = ${typeSnippet(component)}(0.0);
    ${sampleB}
    return value;
  }
  `;
}
export function matMulReadWriteFnSource(hasBias, activation, transposeA, transposeB, fitAOuter = false, fitBOuter = false, fitInner = false, component = 1) {
    return `
  ${matMulReadFnSource(transposeA, transposeB, fitAOuter, fitBOuter, fitInner, component)}
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: ${typeSnippet(component)}) {
    ${fitAOuter && fitBOuter ?
        '' :
        'if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)'}
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      ${biasActivationSnippet(hasBias, activation)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  `;
}
const writeDataToSubAVec4Snippet = (transpose, innerElementSize) => {
    if (transpose) {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol * ${innerElementSize});
        `;
    }
    else {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * ${innerElementSize});
        `;
    }
};
const calculateResultSnippet = (transposeA, innerElementSize, rowPerThread, tileInner) => {
    if (transposeA) {
        return `
      for (var k = 0; k < ${tileInner}; k++) {
        let BCached0 = mm_Bsub[k][tileCol];
        let ACached0 = mm_Asub[k][localRow];
        for (var i = 0; i < ${rowPerThread}; i++) {
          acc[i] = fma(BCached0, vec4<f32>(ACached0[i]), acc[i]);
        }
      }`;
    }
    else {
        let bCachedStr = '';
        let accStr = '';
        for (let i = 0; i < innerElementSize; i++) {
            bCachedStr += `let BCached${i} = mm_Bsub[k * ${innerElementSize} + ${i}][tileCol];`;
            accStr +=
                `acc[i] = fma(BCached${i}, vec4<f32>(ACached[${i}]), acc[i]);`;
        }
        return `
      for (var k = 0; k < ${tileInner / innerElementSize}; k++) {
        ${bCachedStr}
        for (var i = 0; i < ${rowPerThread}; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          ${accStr}
        }
      }`;
    }
};
export function makeMatMulPackedVec4Source(workPerThread, workgroupSize, transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32, broadcastBatch = false) {
    const tileAOuter = workgroupSize[1] * workPerThread[1];
    const tileBOuter = workgroupSize[0] * workPerThread[0];
    const tileAWidth = transposeA ? tileAOuter : tileInner;
    const tileAHight = transposeA ? tileInner : tileAOuter;
    const innerElementSize = tileAWidth / workgroupSize[0];
    const rowPerThreadB = tileInner / workgroupSize[1];
    const rowPerThread = workPerThread[1];
    const colPerThread = workPerThread[0];
    util.assert(((transposeA && innerElementSize === 4 && workPerThread[1] === 4) ||
        (!transposeA && (innerElementSize === 3 || innerElementSize === 4))) &&
        tileAWidth % workgroupSize[0] === 0 &&
        tileInner % workgroupSize[1] === 0 && workPerThread[0] === 4, () => `If transposeA ${transposeA} is true, innerElementSize ${innerElementSize} and workPerThread[1] ${workPerThread[1]} must be 4.
          Otherwise, innerElementSize ${innerElementSize} must be 3 or 4.
      tileAWidth ${tileAWidth} must be divisible by workgroupSize[0]${workgroupSize[0]}. tileInner ${tileInner} must be divisible by workgroupSize[1] ${workgroupSize[1]}. colPerThread ${workPerThread[0]} must be 4.`);
    return `
  var<workgroup> mm_Asub : array<array<vec${innerElementSize}<f32>, ${tileAWidth / innerElementSize}>, ${tileAHight}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${tileBOuter / workPerThread[0]}>, ${tileInner}>;

  ${main()} {
    let localRow = i32(localId.y);
    let tileRow = localRow * ${rowPerThread};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * ${rowPerThread};
    let globalCol = i32(globalId.x) * ${colPerThread};
    let batch = ${splitK ? '0' : 'i32(globalId.z)'};
    let batchA = ${splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.aShape[0]'};
    let batchB = ${splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.bShape[0]'};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

    let numTiles = ${splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
        `(uniforms.dimInner - 1) / ${tileInner} + 1`};
    var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

    var acc: array<vec4<f32>, ${rowPerThread}>;

    // Loop over shared dimension.
    let tileRowB = localRow * ${rowPerThreadB};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${writeDataToSubAVec4Snippet(transposeA, innerElementSize)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + ${tileInner};
        workgroupBarrier();

        // Compute acc values for a single thread.
        ${calculateResultSnippet(transposeA, innerElementSize, rowPerThread, tileInner)}
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`;
}
const writeDataToSubASnippet = (transpose) => {
    if (transpose) {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol);
        `;
    }
    else {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRowStart + inputRow,
          kStart + inputCol);
        `;
    }
};
const readDataFromSubASnippet = (transposeA) => {
    return transposeA ? 'let ACached = mm_Asub[k][tileRow + innerRow];' :
        'let ACached = mm_Asub[tileRow + innerRow][k];';
};
// sequentialAccessByThreads means sequential data in memory is accessed by
// threads, instead of a single thread (default behavior).
export function makeMatMulPackedSource(workPerThread, workgroupSize, transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32, sequentialAccessByThreads = false, broadcastBatch = false) {
    const tileAOuter = workPerThread[1] * workgroupSize[1];
    const tileBOuter = workPerThread[0] * workgroupSize[0];
    const tileAWidth = transposeA ? tileAOuter : tileInner;
    const tileAHight = transposeA ? tileInner : tileAOuter;
    util.assert(tileAHight % workgroupSize[1] === 0 &&
        tileAWidth % workgroupSize[0] === 0 &&
        tileInner % workgroupSize[1] === 0, () => `tileAHight ${tileAHight} must be divisible by workgroupSize[1]${workgroupSize[1]}, tileAWidth ${tileAWidth} must be divisible by workgroupSize[0]${workgroupSize[0]}, tileInner ${tileInner} must be divisible by workgroupSize[1]${workgroupSize[1]}`);
    const rowPerThreadA = tileAHight / workgroupSize[1];
    const colPerThreadA = tileAWidth / workgroupSize[0];
    const rowPerThreadB = tileInner / workgroupSize[1];
    const rowPerThread = workPerThread[1];
    const colPerThread = workPerThread[0];
    const matmulSnippet = sequentialAccessByThreads ?
        `
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
      let globalColStart = i32(workgroupId.x) * ${tileBOuter};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var inputRow = localRow; inputRow < ${tileAHight}; inputRow = inputRow + ${workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${tileAWidth}; inputCol = inputCol + ${workgroupSize[0]}) {
            ${writeDataToSubASnippet(transposeA)}
          }
        }
        // Load one tile of B into local memory.
        for (var inputRow = localRow; inputRow < ${tileInner}; inputRow = inputRow + ${workgroupSize[1]}) {
              for (var inputCol = localCol; inputCol < ${tileBOuter}; inputCol = inputCol + ${workgroupSize[0]}) {
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
              kStart + inputRow,
              globalColStart + inputCol);
          }
        }
        kStart = kStart + ${tileInner};
        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ${colPerThread}>;
        for (var k = 0; k < ${tileInner}; k++) {
          for (var inner = 0; inner < ${colPerThread}; inner++) {
            BCached[inner] = mm_Bsub[k][localCol + inner * ${workgroupSize[0]}];
          }
          for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
            let ACached = ${transposeA ?
            `mm_Asub[k][localRow + innerRow * ${workgroupSize[1]}];` :
            `mm_Asub[localRow + innerRow * ${workgroupSize[1]}][k];`}
            for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
              acc[innerRow][innerCol] =
                  fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        let gRow = globalRowStart + localRow + innerRow * ${workgroupSize[1]};
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          let gCol = globalColStart + localCol + innerCol * ${workgroupSize[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      ` :
        `
  let tileRow = i32(localId.y) * ${rowPerThread};
  let tileCol = i32(localId.x) * ${colPerThread};

  let globalRow = i32(globalId.y) * ${rowPerThread};
  let globalCol = i32(globalId.x) * ${colPerThread};
  let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

  let tileRowA = i32(localId.y) * ${rowPerThreadA};
  let tileColA = i32(localId.x) * ${colPerThreadA};
  let tileRowB = i32(localId.y) * ${rowPerThreadB};
  // Loop over shared dimension.
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < ${rowPerThreadA}; innerRow++) {
      for (var innerCol = 0; innerCol < ${colPerThreadA}; innerCol++) {
        let inputRow = tileRowA + innerRow;
        let inputCol = tileColA + innerCol;
        ${writeDataToSubASnippet(transposeA)}
      }
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < ${rowPerThreadB}; innerRow++) {
      for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol + innerCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
          kStart + inputRow,
          globalCol + innerCol);
      }
    }
    kStart = kStart + ${tileInner};
    workgroupBarrier();

    // Compute acc values for a single thread.
    var BCached : array<f32, ${colPerThread}>;
    for (var k = 0; k < ${tileInner}; k++) {
      for (var inner = 0; inner < ${colPerThread}; inner++) {
        BCached[inner] = mm_Bsub[k][tileCol + inner];
      }

      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        ${readDataFromSubASnippet(transposeA)}
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          acc[innerRow][innerCol] =
              fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
    for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;
    return `
    var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHight}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;

    ${main()} {
      let batch = ${splitK ? '0' : 'i32(globalId.z)'};
      let batchA = ${splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.aShape[0]'};
      let batchB = ${splitK || !broadcastBatch ? 'batch' : 'batch % uniforms.bShape[0]'};
      let numTiles = ${splitK ? `${Math.ceil(splitedDimInner / tileInner)}` :
        `(uniforms.dimInner - 1) / ${tileInner} + 1`};
      var kStart = ${splitK ? `i32(globalId.z) * ${splitedDimInner}` : '0'};

      var acc : array<array<f32, ${colPerThread}>, ${rowPerThread}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${rowPerThread}; innerRow++) {
        for (var innerCol = 0; innerCol < ${colPerThread}; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${matmulSnippet}
    }
  `;
}
const readVectorASnippet = (transpose) => {
    return transpose ? `
      mm_readA(batchA, colA, globalRow),
      mm_readA(batchA, colA + 1, globalRow),
      mm_readA(batchA, colA + 2, globalRow),
      mm_readA(batchA, colA + 3, globalRow)
  ` :
        `
      mm_readA(batchA, globalRow, colA),
      mm_readA(batchA, globalRow, colA + 1),
      mm_readA(batchA, globalRow, colA + 2),
      mm_readA(batchA, globalRow, colA + 3)
  `;
};
export function makeVectorMatrixProductSource(workgroupSize, transposeA = false) {
    util.assert(workgroupSize[1] === 1 && workgroupSize[2] === 1, () => `A linear work group size is required. But got ${workgroupSize}.`);
    const tileSize = workgroupSize[0] * 4;
    return `
    var<workgroup> mm_Asub : array<vec4<f32>, ${workgroupSize[0]}>;

    ${main()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / ${tileSize} + 1;
      let batch = i32(globalId.z);
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * ${tileSize} + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${readVectorASnippet(transposeA)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < ${tileSize / 4}; k++) {
          let rowB = t * ${tileSize} + k * 4;
          let BCached = vec4<f32>(mm_readB(batchB, rowB, globalCol),
                              mm_readB(batchB, rowB + 1, globalCol),
                              mm_readB(batchB, rowB + 2, globalCol),
                              mm_readB(batchB, rowB + 3, globalCol));

          let ACached = mm_Asub[k];
          acc = acc + dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      mm_write(batch, globalRow, globalCol, acc);
    }
  `;
}
export class MatMulPackedProgram {
    constructor(aShape, outputShape, transposeA = false, transposeB = false, bias = null, activation = null, preluActivationWeights = null, sequentialAccessByThreads = false) {
        this.variableNames = ['A', 'B'];
        this.uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [2], y: [1], z: [0] };
        const dimInner = transposeA ? aShape[1] : aShape[2];
        this.isVec4 = ((dimInner % 4 === 0 && !transposeA) ||
            (outputShape[1] % 4 === 0 && transposeA)) &&
            outputShape[2] % 4 === 0 && !transposeB;
        this.outputComponent = this.isVec4 ? 4 : 1;
        this.isVectorA = outputShape[1] === 1 && !transposeA;
        if (!this.isVec4 && this.isVectorA) {
            // For makeVectorMatrixProductSource
            this.elementsPerThread = [1, 1, 1];
            this.workgroupSize = [32, 1, 1];
        }
        else {
            const workgroupInfo = computeWorkgroupInfoForMatMul(outputShape[1], dimInner, outputShape[2], transposeA);
            this.workgroupSize = workgroupInfo.workgroupSize;
            this.elementsPerThread = workgroupInfo.elementsPerThread;
        }
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, this.elementsPerThread);
        const addBias = bias != null;
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.sequentialAccessByThreads = sequentialAccessByThreads;
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        [this.fitAOuter, this.fitBOuter, this.fitInner] =
            this.getShapeFit(outputShape[1], outputShape[2], dimInner);
        this.shaderKey = `matMulPacked_${this.elementsPerThread}_${transposeA}_${transposeB}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.isVectorA}_${this.sequentialAccessByThreads}`;
    }
    getShapeFit(dimAOuter, dimBOuter, dimInner) {
        const tileAOuter = this.workgroupSize[1] * this.elementsPerThread[1];
        const tileBOuter = this.workgroupSize[0] * this.elementsPerThread[0];
        if (!this.isVec4 && this.isVectorA) {
            // For makeVectorMatrixProductSource
            this.tileInner = this.workgroupSize[0] * 4;
        }
        else {
            this.tileInner = tileBOuter;
        }
        const fitAOuter = dimAOuter % tileAOuter === 0;
        const fitBOuter = dimBOuter % tileBOuter === 0;
        const fitInner = dimInner % this.tileInner === 0;
        return [fitAOuter, fitBOuter, fitInner];
    }
    getUserCode() {
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights, this.isVec4)}
      ${matMulReadWriteFnSource(this.addBias, this.activation, false /* transposeA is implemented in makeMatMulPackedSource */, this.transposeB, this.fitAOuter, this.fitBOuter, this.fitInner, this.isVec4 ? 4 : 1)}
      ${this.isVec4 ?
            makeMatMulPackedVec4Source(this.elementsPerThread, this.workgroupSize, this.transposeA, this.tileInner, false, null, true) :
            (this.isVectorA ? makeVectorMatrixProductSource(this.workgroupSize, this.transposeA) :
                makeMatMulPackedSource(this.elementsPerThread, this.workgroupSize, this.transposeA, this.tileInner, false, null, this.sequentialAccessByThreads, true))}
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF0bXVsX3BhY2tlZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9tYXRtdWxfcGFja2VkX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXJFLE9BQU8sRUFBQyxtQkFBbUIsRUFBRSxxQkFBcUIsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQzdFLE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQUUsV0FBVyxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQ3pGLE9BQU8sRUFBQyxlQUFlLEVBQUUsNkJBQTZCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFN0UsTUFBTSxVQUFVLGtCQUFrQixDQUM5QixVQUFtQixFQUFFLFVBQW1CLEVBQUUsU0FBUyxHQUFHLEtBQUssRUFDM0QsU0FBUyxHQUFHLEtBQUssRUFBRSxRQUFRLEdBQUcsS0FBSyxFQUFFLFNBQVMsR0FBRyxDQUFDO0lBQ3BELElBQUksQ0FBQyxNQUFNLENBQ1AsVUFBVSxJQUFJLFNBQVMsS0FBSyxDQUFDLElBQUksQ0FBQyxVQUFVLEVBQzVDLEdBQUcsRUFBRSxDQUFDLGNBQWMsVUFBVSwwQ0FDMUIsU0FBUyxFQUFFLENBQUMsQ0FBQztJQUNyQixNQUFNLE9BQU8sR0FBRztRQUVaLFVBQVUsQ0FBQyxDQUFDLENBQUMsZ0NBQWdDLENBQUMsQ0FBQztRQUNsQyxnQ0FBZ0M7O0tBRTlDLENBQUM7SUFDSixNQUFNLE9BQU8sR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLGdDQUFnQyxDQUFDLENBQUM7UUFDbEMsZ0NBQWdDLENBQUM7SUFFOUQsT0FBTzttREFDMEMsV0FBVyxDQUFDLFNBQVMsQ0FBQztrQkFDdkQsV0FBVyxDQUFDLFNBQVMsQ0FBQztNQUVsQyxTQUFTLElBQUks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QyxTQUFTLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ3hDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsNkJBQTZCLENBQ3pCLElBQUksQ0FBQyxhQUFhLEVBQUUsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUM7Z0JBQzFDLHNCQUFzQixDQUNsQixJQUFJLENBQUMsaUJBQWlCLEVBQUUsSUFBSSxDQUFDLGFBQWEsRUFDMUMsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMsU0FBUyxFQUFFLEtBQUssRUFBRSxJQUFJLEVBQzVDLElBQUksQ0FBQyx5QkFBeUIsRUFBRSxJQUFJLENBQUMsQ0FBQztLQUNuRSxDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHthY3RpdmF0aW9uRm5TbmlwcGV0LCBiaWFzQWN0aXZhdGlvblNuaXBwZXR9IGZyb20gJy4vYWN0aXZhdGlvbl91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCB0eXBlU25pcHBldCwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgY29tcHV0ZVdvcmtncm91cEluZm9Gb3JNYXRNdWx9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gbWF0TXVsUmVhZEZuU291cmNlKFxuICAgIHRyYW5zcG9zZUE6IGJvb2xlYW4sIHRyYW5zcG9zZUI6IGJvb2xlYW4sIGZpdEFPdXRlciA9IGZhbHNlLFxuICAgIGZpdEJPdXRlciA9IGZhbHNlLCBmaXRJbm5lciA9IGZhbHNlLCBjb21wb25lbnQgPSAxKSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgdHJhbnNwb3NlQSAmJiBjb21wb25lbnQgPT09IDEgfHwgIXRyYW5zcG9zZUEsXG4gICAgICAoKSA9PiBgdHJhbnNwb3NlQSAke3RyYW5zcG9zZUF9IGlzIG5vdCBjb21wYXRpYmxlIHdpdGggY29tcG9uZW50IHNpemUgJHtcbiAgICAgICAgICBjb21wb25lbnR9YCk7XG4gIGNvbnN0IHNhbXBsZUEgPSBgXG4gICAgICAke1xuICAgICAgdHJhbnNwb3NlQSA/IGB2YWx1ZSA9IGdldEEoYmF0Y2gsIGNvbCwgcm93KTtgIDpcbiAgICAgICAgICAgICAgICAgICBgdmFsdWUgPSBnZXRBKGJhdGNoLCByb3csIGNvbCk7YH1cblxuICAgIGA7XG4gIGNvbnN0IHNhbXBsZUIgPSB0cmFuc3Bvc2VCID8gYHZhbHVlID0gZ2V0QihiYXRjaCwgY29sLCByb3cpO2AgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGB2YWx1ZSA9IGdldEIoYmF0Y2gsIHJvdywgY29sKTtgO1xuXG4gIHJldHVybiBgXG4gIGZuIG1tX3JlYWRBKGJhdGNoOiBpMzIsIHJvdzogaTMyLCBjb2w6IGkzMikgLT4gJHt0eXBlU25pcHBldChjb21wb25lbnQpfSB7XG4gICAgdmFyIHZhbHVlID0gJHt0eXBlU25pcHBldChjb21wb25lbnQpfSgwLjApO1xuICAgICR7XG4gICAgICBmaXRBT3V0ZXIgJiYgZml0SW5uZXIgP1xuICAgICAgICAgIHNhbXBsZUEgOlxuICAgICAgICAgIGBcbiAgICAke1xuICAgICAgICAgICAgICB0cmFuc3Bvc2VBID9cbiAgICAgICAgICAgICAgICAgIGBpZihyb3cgPCB1bmlmb3Jtcy5kaW1BT3V0ZXIgJiYgY29sIDwgdW5pZm9ybXMuZGltSW5uZXIpYCA6XG4gICAgICAgICAgICAgICAgICBgaWYocm93IDwgdW5pZm9ybXMuYVNoYXBlWzFdICYmIGNvbCA8IHVuaWZvcm1zLmFTaGFwZVsyXSlgfVxuICAgIHtcbiAgICAgICR7c2FtcGxlQX1cbiAgICB9XG4gICAgYH1cbiAgICByZXR1cm4gdmFsdWU7XG4gIH1cblxuICBmbiBtbV9yZWFkQihiYXRjaDogaTMyLCByb3c6IGkzMiwgY29sOiBpMzIpIC0+ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0ge1xuICAgIHZhciB2YWx1ZSA9ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0oMC4wKTtcbiAgICAke3NhbXBsZUJ9XG4gICAgcmV0dXJuIHZhbHVlO1xuICB9XG4gIGA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYXRNdWxSZWFkV3JpdGVGblNvdXJjZShcbiAgICBoYXNCaWFzOiBib29sZWFuLCBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbiwgdHJhbnNwb3NlQTogYm9vbGVhbixcbiAgICB0cmFuc3Bvc2VCOiBib29sZWFuLCBmaXRBT3V0ZXIgPSBmYWxzZSwgZml0Qk91dGVyID0gZmFsc2UsIGZpdElubmVyID0gZmFsc2UsXG4gICAgY29tcG9uZW50ID0gMSkge1xuICByZXR1cm4gYFxuICAke1xuICAgICAgbWF0TXVsUmVhZEZuU291cmNlKFxuICAgICAgICAgIHRyYW5zcG9zZUEsIHRyYW5zcG9zZUIsIGZpdEFPdXRlciwgZml0Qk91dGVyLCBmaXRJbm5lciwgY29tcG9uZW50KX1cbiAgZm4gbW1fd3JpdGUoYmF0Y2g6IGkzMiwgcm93OiBpMzIsIGNvbDogaTMyLCB2YWx1ZUluOiAke1xuICAgICAgdHlwZVNuaXBwZXQoY29tcG9uZW50KX0pIHtcbiAgICAke1xuICAgICAgZml0QU91dGVyICYmIGZpdEJPdXRlciA/XG4gICAgICAgICAgJycgOlxuICAgICAgICAgICdpZiAocm93IDwgdW5pZm9ybXMuZGltQU91dGVyICYmIGNvbCA8IHVuaWZvcm1zLmRpbUJPdXRlciknfVxuICAgIHtcbiAgICAgIHZhciB2YWx1ZSA9IHZhbHVlSW47XG4gICAgICBsZXQgY29vcmRzID0gdmVjMzxpMzI+KGJhdGNoLCByb3csIGNvbCk7XG4gICAgICAke2JpYXNBY3RpdmF0aW9uU25pcHBldChoYXNCaWFzLCBhY3RpdmF0aW9uKX1cbiAgICAgIHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkc1swXSwgY29vcmRzWzFdLCBjb29yZHNbMl0sIHZhbHVlKTtcbiAgICB9XG4gIH1cbiAgYDtcbn1cblxuY29uc3Qgd3JpdGVEYXRhVG9TdWJBVmVjNFNuaXBwZXQgPVxuICAgICh0cmFuc3Bvc2U6IGJvb2xlYW4sIGlubmVyRWxlbWVudFNpemU6IG51bWJlcikgPT4ge1xuICAgICAgaWYgKHRyYW5zcG9zZSkge1xuICAgICAgICByZXR1cm4gYFxuICAgICAgICBtbV9Bc3ViW2lucHV0Um93XVtpbnB1dENvbF0gPSBtbV9yZWFkQShiYXRjaEEsXG4gICAgICAgICAga1N0YXJ0ICsgaW5wdXRSb3csXG4gICAgICAgICAgZ2xvYmFsUm93U3RhcnQgKyBpbnB1dENvbCAqICR7aW5uZXJFbGVtZW50U2l6ZX0pO1xuICAgICAgICBgO1xuXG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gYFxuICAgICAgICBtbV9Bc3ViW2lucHV0Um93XVtpbnB1dENvbF0gPSBtbV9yZWFkQShiYXRjaEEsXG4gICAgICAgICAgZ2xvYmFsUm93ICsgaW5uZXJSb3csXG4gICAgICAgICAga1N0YXJ0ICsgaW5wdXRDb2wgKiAke2lubmVyRWxlbWVudFNpemV9KTtcbiAgICAgICAgYDtcbiAgICAgIH1cbiAgICB9O1xuXG5jb25zdCBjYWxjdWxhdGVSZXN1bHRTbmlwcGV0ID1cbiAgICAodHJhbnNwb3NlQTogYm9vbGVhbiwgaW5uZXJFbGVtZW50U2l6ZTogbnVtYmVyLCByb3dQZXJUaHJlYWQ6IG51bWJlcixcbiAgICAgdGlsZUlubmVyOiBudW1iZXIpID0+IHtcbiAgICAgIGlmICh0cmFuc3Bvc2VBKSB7XG4gICAgICAgIHJldHVybiBgXG4gICAgICBmb3IgKHZhciBrID0gMDsgayA8ICR7dGlsZUlubmVyfTsgaysrKSB7XG4gICAgICAgIGxldCBCQ2FjaGVkMCA9IG1tX0JzdWJba11bdGlsZUNvbF07XG4gICAgICAgIGxldCBBQ2FjaGVkMCA9IG1tX0FzdWJba11bbG9jYWxSb3ddO1xuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8ICR7cm93UGVyVGhyZWFkfTsgaSsrKSB7XG4gICAgICAgICAgYWNjW2ldID0gZm1hKEJDYWNoZWQwLCB2ZWM0PGYzMj4oQUNhY2hlZDBbaV0pLCBhY2NbaV0pO1xuICAgICAgICB9XG4gICAgICB9YDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGxldCBiQ2FjaGVkU3RyID0gJyc7XG4gICAgICAgIGxldCBhY2NTdHIgPSAnJztcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBpbm5lckVsZW1lbnRTaXplOyBpKyspIHtcbiAgICAgICAgICBiQ2FjaGVkU3RyICs9IGBsZXQgQkNhY2hlZCR7aX0gPSBtbV9Cc3ViW2sgKiAke2lubmVyRWxlbWVudFNpemV9ICsgJHtcbiAgICAgICAgICAgICAgaX1dW3RpbGVDb2xdO2A7XG4gICAgICAgICAgYWNjU3RyICs9XG4gICAgICAgICAgICAgIGBhY2NbaV0gPSBmbWEoQkNhY2hlZCR7aX0sIHZlYzQ8ZjMyPihBQ2FjaGVkWyR7aX1dKSwgYWNjW2ldKTtgO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBgXG4gICAgICBmb3IgKHZhciBrID0gMDsgayA8ICR7dGlsZUlubmVyIC8gaW5uZXJFbGVtZW50U2l6ZX07IGsrKykge1xuICAgICAgICAke2JDYWNoZWRTdHJ9XG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgJHtyb3dQZXJUaHJlYWR9OyBpKyspIHtcbiAgICAgICAgICBsZXQgQUNhY2hlZCA9IG1tX0FzdWJbdGlsZVJvdyArIGldW2tdO1xuICAgICAgICAgICR7YWNjU3RyfVxuICAgICAgICB9XG4gICAgICB9YDtcbiAgICAgIH1cbiAgICB9O1xuXG5leHBvcnQgZnVuY3Rpb24gbWFrZU1hdE11bFBhY2tlZFZlYzRTb3VyY2UoXG4gICAgd29ya1BlclRocmVhZDogbnVtYmVyW10sIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICB0cmFuc3Bvc2VBID0gZmFsc2UsIHRpbGVJbm5lciA9IDMyLCBzcGxpdEsgPSBmYWxzZSwgc3BsaXRlZERpbUlubmVyID0gMzIsXG4gICAgYnJvYWRjYXN0QmF0Y2ggPSBmYWxzZSk6IHN0cmluZyB7XG4gIGNvbnN0IHRpbGVBT3V0ZXIgPSB3b3JrZ3JvdXBTaXplWzFdICogd29ya1BlclRocmVhZFsxXTtcbiAgY29uc3QgdGlsZUJPdXRlciA9IHdvcmtncm91cFNpemVbMF0gKiB3b3JrUGVyVGhyZWFkWzBdO1xuICBjb25zdCB0aWxlQVdpZHRoID0gdHJhbnNwb3NlQSA/IHRpbGVBT3V0ZXIgOiB0aWxlSW5uZXI7XG4gIGNvbnN0IHRpbGVBSGlnaHQgPSB0cmFuc3Bvc2VBID8gdGlsZUlubmVyIDogdGlsZUFPdXRlcjtcbiAgY29uc3QgaW5uZXJFbGVtZW50U2l6ZSA9IHRpbGVBV2lkdGggLyB3b3JrZ3JvdXBTaXplWzBdO1xuICBjb25zdCByb3dQZXJUaHJlYWRCID0gdGlsZUlubmVyIC8gd29ya2dyb3VwU2l6ZVsxXTtcbiAgY29uc3Qgcm93UGVyVGhyZWFkID0gd29ya1BlclRocmVhZFsxXTtcbiAgY29uc3QgY29sUGVyVGhyZWFkID0gd29ya1BlclRocmVhZFswXTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAoKHRyYW5zcG9zZUEgJiYgaW5uZXJFbGVtZW50U2l6ZSA9PT0gNCAmJiB3b3JrUGVyVGhyZWFkWzFdID09PSA0KSB8fFxuICAgICAgICghdHJhbnNwb3NlQSAmJiAoaW5uZXJFbGVtZW50U2l6ZSA9PT0gMyB8fCBpbm5lckVsZW1lbnRTaXplID09PSA0KSkpICYmXG4gICAgICAgICAgdGlsZUFXaWR0aCAlIHdvcmtncm91cFNpemVbMF0gPT09IDAgJiZcbiAgICAgICAgICB0aWxlSW5uZXIgJSB3b3JrZ3JvdXBTaXplWzFdID09PSAwICYmIHdvcmtQZXJUaHJlYWRbMF0gPT09IDQsXG4gICAgICAoKSA9PiBgSWYgdHJhbnNwb3NlQSAke3RyYW5zcG9zZUF9IGlzIHRydWUsIGlubmVyRWxlbWVudFNpemUgJHtcbiAgICAgICAgICBpbm5lckVsZW1lbnRTaXplfSBhbmQgd29ya1BlclRocmVhZFsxXSAke3dvcmtQZXJUaHJlYWRbMV19IG11c3QgYmUgNC5cbiAgICAgICAgICBPdGhlcndpc2UsIGlubmVyRWxlbWVudFNpemUgJHtpbm5lckVsZW1lbnRTaXplfSBtdXN0IGJlIDMgb3IgNC5cbiAgICAgIHRpbGVBV2lkdGggJHt0aWxlQVdpZHRofSBtdXN0IGJlIGRpdmlzaWJsZSBieSB3b3JrZ3JvdXBTaXplWzBdJHtcbiAgICAgICAgICB3b3JrZ3JvdXBTaXplWzBdfS4gdGlsZUlubmVyICR7XG4gICAgICAgICAgdGlsZUlubmVyfSBtdXN0IGJlIGRpdmlzaWJsZSBieSB3b3JrZ3JvdXBTaXplWzFdICR7XG4gICAgICAgICAgd29ya2dyb3VwU2l6ZVsxXX0uIGNvbFBlclRocmVhZCAke3dvcmtQZXJUaHJlYWRbMF19IG11c3QgYmUgNC5gKTtcbiAgcmV0dXJuIGBcbiAgdmFyPHdvcmtncm91cD4gbW1fQXN1YiA6IGFycmF5PGFycmF5PHZlYyR7aW5uZXJFbGVtZW50U2l6ZX08ZjMyPiwgJHtcbiAgICAgIHRpbGVBV2lkdGggLyBpbm5lckVsZW1lbnRTaXplfT4sICR7dGlsZUFIaWdodH0+O1xuICB2YXI8d29ya2dyb3VwPiBtbV9Cc3ViIDogYXJyYXk8YXJyYXk8dmVjNDxmMzI+LCAke1xuICAgICAgdGlsZUJPdXRlciAvIHdvcmtQZXJUaHJlYWRbMF19PiwgJHt0aWxlSW5uZXJ9PjtcblxuICAke21haW4oKX0ge1xuICAgIGxldCBsb2NhbFJvdyA9IGkzMihsb2NhbElkLnkpO1xuICAgIGxldCB0aWxlUm93ID0gbG9jYWxSb3cgKiAke3Jvd1BlclRocmVhZH07XG4gICAgbGV0IHRpbGVDb2wgPSBpMzIobG9jYWxJZC54KTtcblxuICAgIGxldCBnbG9iYWxSb3cgPSBpMzIoZ2xvYmFsSWQueSkgKiAke3Jvd1BlclRocmVhZH07XG4gICAgbGV0IGdsb2JhbENvbCA9IGkzMihnbG9iYWxJZC54KSAqICR7Y29sUGVyVGhyZWFkfTtcbiAgICBsZXQgYmF0Y2ggPSAke3NwbGl0SyA/ICcwJyA6ICdpMzIoZ2xvYmFsSWQueiknfTtcbiAgICBsZXQgYmF0Y2hBID0gJHtcbiAgICAgIHNwbGl0SyB8fCAhYnJvYWRjYXN0QmF0Y2ggPyAnYmF0Y2gnIDogJ2JhdGNoICUgdW5pZm9ybXMuYVNoYXBlWzBdJ307XG4gICAgbGV0IGJhdGNoQiA9ICR7XG4gICAgICBzcGxpdEsgfHwgIWJyb2FkY2FzdEJhdGNoID8gJ2JhdGNoJyA6ICdiYXRjaCAlIHVuaWZvcm1zLmJTaGFwZVswXSd9O1xuICAgIGxldCBnbG9iYWxSb3dTdGFydCA9IGkzMih3b3JrZ3JvdXBJZC55KSAqICR7dGlsZUFPdXRlcn07XG5cbiAgICBsZXQgbnVtVGlsZXMgPSAke1xuICAgICAgc3BsaXRLID8gYCR7TWF0aC5jZWlsKHNwbGl0ZWREaW1Jbm5lciAvIHRpbGVJbm5lcil9YCA6XG4gICAgICAgICAgICAgICBgKHVuaWZvcm1zLmRpbUlubmVyIC0gMSkgLyAke3RpbGVJbm5lcn0gKyAxYH07XG4gICAgdmFyIGtTdGFydCA9ICR7c3BsaXRLID8gYGkzMihnbG9iYWxJZC56KSAqICR7c3BsaXRlZERpbUlubmVyfWAgOiAnMCd9O1xuXG4gICAgdmFyIGFjYzogYXJyYXk8dmVjNDxmMzI+LCAke3Jvd1BlclRocmVhZH0+O1xuXG4gICAgLy8gTG9vcCBvdmVyIHNoYXJlZCBkaW1lbnNpb24uXG4gICAgbGV0IHRpbGVSb3dCID0gbG9jYWxSb3cgKiAke3Jvd1BlclRocmVhZEJ9O1xuICAgIGZvciAodmFyIHQgPSAwOyB0IDwgbnVtVGlsZXM7IHQrKykge1xuICAgICAgICAvLyBMb2FkIG9uZSB0aWxlIG9mIEEgaW50byBsb2NhbCBtZW1vcnkuXG4gICAgICAgIGZvciAodmFyIGlubmVyUm93ID0gMDsgaW5uZXJSb3cgPCAke3Jvd1BlclRocmVhZH07IGlubmVyUm93KyspIHtcbiAgICAgICAgICAgIGxldCBpbnB1dFJvdyA9IHRpbGVSb3cgKyBpbm5lclJvdztcbiAgICAgICAgICAgIGxldCBpbnB1dENvbCA9IHRpbGVDb2w7XG4gICAgICAgICAgICAke3dyaXRlRGF0YVRvU3ViQVZlYzRTbmlwcGV0KHRyYW5zcG9zZUEsIGlubmVyRWxlbWVudFNpemUpfVxuICAgICAgICB9XG5cbiAgICAgICAgLy8gTG9hZCBvbmUgdGlsZSBvZiBCIGludG8gbG9jYWwgbWVtb3J5LlxuICAgICAgICBmb3IgKHZhciBpbm5lclJvdyA9IDA7IGlubmVyUm93IDwgJHtyb3dQZXJUaHJlYWRCfTsgaW5uZXJSb3crKykge1xuICAgICAgICAgICAgbGV0IGlucHV0Um93ID0gdGlsZVJvd0IgKyBpbm5lclJvdztcbiAgICAgICAgICAgIGxldCBpbnB1dENvbCA9IHRpbGVDb2w7XG4gICAgICAgICAgICBtbV9Cc3ViW2lucHV0Um93XVtpbnB1dENvbF0gPSBtbV9yZWFkQihiYXRjaEIsIGtTdGFydCArIGlucHV0Um93LCBnbG9iYWxDb2wpO1xuICAgICAgICB9XG4gICAgICAgIGtTdGFydCA9IGtTdGFydCArICR7dGlsZUlubmVyfTtcbiAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICAgIC8vIENvbXB1dGUgYWNjIHZhbHVlcyBmb3IgYSBzaW5nbGUgdGhyZWFkLlxuICAgICAgICAke1xuICAgICAgY2FsY3VsYXRlUmVzdWx0U25pcHBldChcbiAgICAgICAgICB0cmFuc3Bvc2VBLCBpbm5lckVsZW1lbnRTaXplLCByb3dQZXJUaHJlYWQsIHRpbGVJbm5lcil9XG4gICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcbiAgICB9XG5cbiAgICBmb3IgKHZhciBpbm5lclJvdyA9IDA7IGlubmVyUm93IDwgJHtyb3dQZXJUaHJlYWR9OyBpbm5lclJvdysrKSB7XG4gICAgICAgIG1tX3dyaXRlKGJhdGNoLCBnbG9iYWxSb3cgKyBpbm5lclJvdywgZ2xvYmFsQ29sLCBhY2NbaW5uZXJSb3ddKTtcbiAgICB9XG4gIH1gO1xufVxuXG5jb25zdCB3cml0ZURhdGFUb1N1YkFTbmlwcGV0ID0gKHRyYW5zcG9zZTogYm9vbGVhbikgPT4ge1xuICBpZiAodHJhbnNwb3NlKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgICAgbW1fQXN1YltpbnB1dFJvd11baW5wdXRDb2xdID0gbW1fcmVhZEEoYmF0Y2hBLFxuICAgICAgICAgIGtTdGFydCArIGlucHV0Um93LFxuICAgICAgICAgIGdsb2JhbFJvd1N0YXJ0ICsgaW5wdXRDb2wpO1xuICAgICAgICBgO1xuXG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIGBcbiAgICAgICAgbW1fQXN1YltpbnB1dFJvd11baW5wdXRDb2xdID0gbW1fcmVhZEEoYmF0Y2hBLFxuICAgICAgICAgIGdsb2JhbFJvd1N0YXJ0ICsgaW5wdXRSb3csXG4gICAgICAgICAga1N0YXJ0ICsgaW5wdXRDb2wpO1xuICAgICAgICBgO1xuICB9XG59O1xuXG5jb25zdCByZWFkRGF0YUZyb21TdWJBU25pcHBldCA9ICh0cmFuc3Bvc2VBOiBib29sZWFuKSA9PiB7XG4gIHJldHVybiB0cmFuc3Bvc2VBID8gJ2xldCBBQ2FjaGVkID0gbW1fQXN1YltrXVt0aWxlUm93ICsgaW5uZXJSb3ddOycgOlxuXG4gICAgICAgICAgICAgICAgICAgICAgJ2xldCBBQ2FjaGVkID0gbW1fQXN1Ylt0aWxlUm93ICsgaW5uZXJSb3ddW2tdOyc7XG59O1xuXG4vLyBzZXF1ZW50aWFsQWNjZXNzQnlUaHJlYWRzIG1lYW5zIHNlcXVlbnRpYWwgZGF0YSBpbiBtZW1vcnkgaXMgYWNjZXNzZWQgYnlcbi8vIHRocmVhZHMsIGluc3RlYWQgb2YgYSBzaW5nbGUgdGhyZWFkIChkZWZhdWx0IGJlaGF2aW9yKS5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlTWF0TXVsUGFja2VkU291cmNlKFxuICAgIHdvcmtQZXJUaHJlYWQ6IG51bWJlcltdLCB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgdHJhbnNwb3NlQSA9IGZhbHNlLCB0aWxlSW5uZXIgPSAzMiwgc3BsaXRLID0gZmFsc2UsIHNwbGl0ZWREaW1Jbm5lciA9IDMyLFxuICAgIHNlcXVlbnRpYWxBY2Nlc3NCeVRocmVhZHMgPSBmYWxzZSwgYnJvYWRjYXN0QmF0Y2ggPSBmYWxzZSk6IHN0cmluZyB7XG4gIGNvbnN0IHRpbGVBT3V0ZXIgPSB3b3JrUGVyVGhyZWFkWzFdICogd29ya2dyb3VwU2l6ZVsxXTtcbiAgY29uc3QgdGlsZUJPdXRlciA9IHdvcmtQZXJUaHJlYWRbMF0gKiB3b3JrZ3JvdXBTaXplWzBdO1xuICBjb25zdCB0aWxlQVdpZHRoID0gdHJhbnNwb3NlQSA/IHRpbGVBT3V0ZXIgOiB0aWxlSW5uZXI7XG4gIGNvbnN0IHRpbGVBSGlnaHQgPSB0cmFuc3Bvc2VBID8gdGlsZUlubmVyIDogdGlsZUFPdXRlcjtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB0aWxlQUhpZ2h0ICUgd29ya2dyb3VwU2l6ZVsxXSA9PT0gMCAmJlxuICAgICAgICAgIHRpbGVBV2lkdGggJSB3b3JrZ3JvdXBTaXplWzBdID09PSAwICYmXG4gICAgICAgICAgdGlsZUlubmVyICUgd29ya2dyb3VwU2l6ZVsxXSA9PT0gMCxcbiAgICAgICgpID0+IGB0aWxlQUhpZ2h0ICR7dGlsZUFIaWdodH0gbXVzdCBiZSBkaXZpc2libGUgYnkgd29ya2dyb3VwU2l6ZVsxXSR7XG4gICAgICAgICAgd29ya2dyb3VwU2l6ZVsxXX0sIHRpbGVBV2lkdGggJHtcbiAgICAgICAgICB0aWxlQVdpZHRofSBtdXN0IGJlIGRpdmlzaWJsZSBieSB3b3JrZ3JvdXBTaXplWzBdJHtcbiAgICAgICAgICB3b3JrZ3JvdXBTaXplWzBdfSwgdGlsZUlubmVyICR7XG4gICAgICAgICAgdGlsZUlubmVyfSBtdXN0IGJlIGRpdmlzaWJsZSBieSB3b3JrZ3JvdXBTaXplWzFdJHt3b3JrZ3JvdXBTaXplWzFdfWApO1xuICBjb25zdCByb3dQZXJUaHJlYWRBID0gdGlsZUFIaWdodCAvIHdvcmtncm91cFNpemVbMV07XG4gIGNvbnN0IGNvbFBlclRocmVhZEEgPSB0aWxlQVdpZHRoIC8gd29ya2dyb3VwU2l6ZVswXTtcbiAgY29uc3Qgcm93UGVyVGhyZWFkQiA9IHRpbGVJbm5lciAvIHdvcmtncm91cFNpemVbMV07XG4gIGNvbnN0IHJvd1BlclRocmVhZCA9IHdvcmtQZXJUaHJlYWRbMV07XG4gIGNvbnN0IGNvbFBlclRocmVhZCA9IHdvcmtQZXJUaHJlYWRbMF07XG4gIGNvbnN0IG1hdG11bFNuaXBwZXQgPSBzZXF1ZW50aWFsQWNjZXNzQnlUaHJlYWRzID9cbiAgICAgIGBcbiAgICAgIGxldCBsb2NhbFJvdyA9IGkzMihsb2NhbElkLnkpO1xuICAgICAgbGV0IGxvY2FsQ29sID0gaTMyKGxvY2FsSWQueCk7XG4gICAgICBsZXQgZ2xvYmFsUm93U3RhcnQgPSBpMzIod29ya2dyb3VwSWQueSkgKiAke3RpbGVBT3V0ZXJ9O1xuICAgICAgbGV0IGdsb2JhbENvbFN0YXJ0ID0gaTMyKHdvcmtncm91cElkLngpICogJHt0aWxlQk91dGVyfTtcblxuICAgICAgLy8gTG9vcCBvdmVyIHNoYXJlZCBkaW1lbnNpb24uXG4gICAgICBmb3IgKHZhciB0ID0gMDsgdCA8IG51bVRpbGVzOyB0KyspIHtcbiAgICAgICAgLy8gTG9hZCBvbmUgdGlsZSBvZiBBIGludG8gbG9jYWwgbWVtb3J5LlxuICAgICAgICBmb3IgKHZhciBpbnB1dFJvdyA9IGxvY2FsUm93OyBpbnB1dFJvdyA8ICR7XG4gICAgICAgICAgdGlsZUFIaWdodH07IGlucHV0Um93ID0gaW5wdXRSb3cgKyAke3dvcmtncm91cFNpemVbMV19KSB7XG4gICAgICAgICAgZm9yICh2YXIgaW5wdXRDb2wgPSBsb2NhbENvbDsgaW5wdXRDb2wgPCAke1xuICAgICAgICAgIHRpbGVBV2lkdGh9OyBpbnB1dENvbCA9IGlucHV0Q29sICsgJHt3b3JrZ3JvdXBTaXplWzBdfSkge1xuICAgICAgICAgICAgJHt3cml0ZURhdGFUb1N1YkFTbmlwcGV0KHRyYW5zcG9zZUEpfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICAvLyBMb2FkIG9uZSB0aWxlIG9mIEIgaW50byBsb2NhbCBtZW1vcnkuXG4gICAgICAgIGZvciAodmFyIGlucHV0Um93ID0gbG9jYWxSb3c7IGlucHV0Um93IDwgJHtcbiAgICAgICAgICB0aWxlSW5uZXJ9OyBpbnB1dFJvdyA9IGlucHV0Um93ICsgJHt3b3JrZ3JvdXBTaXplWzFdfSkge1xuICAgICAgICAgICAgICBmb3IgKHZhciBpbnB1dENvbCA9IGxvY2FsQ29sOyBpbnB1dENvbCA8ICR7XG4gICAgICAgICAgdGlsZUJPdXRlcn07IGlucHV0Q29sID0gaW5wdXRDb2wgKyAke3dvcmtncm91cFNpemVbMF19KSB7XG4gICAgICAgICAgICBtbV9Cc3ViW2lucHV0Um93XVtpbnB1dENvbF0gPSBtbV9yZWFkQihiYXRjaEIsXG4gICAgICAgICAgICAgIGtTdGFydCArIGlucHV0Um93LFxuICAgICAgICAgICAgICBnbG9iYWxDb2xTdGFydCArIGlucHV0Q29sKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAga1N0YXJ0ID0ga1N0YXJ0ICsgJHt0aWxlSW5uZXJ9O1xuICAgICAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgICAgICAgLy8gQ29tcHV0ZSBhY2MgdmFsdWVzIGZvciBhIHNpbmdsZSB0aHJlYWQuXG4gICAgICAgIHZhciBCQ2FjaGVkIDogYXJyYXk8ZjMyLCAke2NvbFBlclRocmVhZH0+O1xuICAgICAgICBmb3IgKHZhciBrID0gMDsgayA8ICR7dGlsZUlubmVyfTsgaysrKSB7XG4gICAgICAgICAgZm9yICh2YXIgaW5uZXIgPSAwOyBpbm5lciA8ICR7Y29sUGVyVGhyZWFkfTsgaW5uZXIrKykge1xuICAgICAgICAgICAgQkNhY2hlZFtpbm5lcl0gPSBtbV9Cc3ViW2tdW2xvY2FsQ29sICsgaW5uZXIgKiAke3dvcmtncm91cFNpemVbMF19XTtcbiAgICAgICAgICB9XG4gICAgICAgICAgZm9yICh2YXIgaW5uZXJSb3cgPSAwOyBpbm5lclJvdyA8ICR7cm93UGVyVGhyZWFkfTsgaW5uZXJSb3crKykge1xuICAgICAgICAgICAgbGV0IEFDYWNoZWQgPSAke1xuICAgICAgICAgIHRyYW5zcG9zZUEgP1xuICAgICAgICAgICAgICBgbW1fQXN1YltrXVtsb2NhbFJvdyArIGlubmVyUm93ICogJHt3b3JrZ3JvdXBTaXplWzFdfV07YCA6XG4gICAgICAgICAgICAgIGBtbV9Bc3ViW2xvY2FsUm93ICsgaW5uZXJSb3cgKiAke3dvcmtncm91cFNpemVbMV19XVtrXTtgfVxuICAgICAgICAgICAgZm9yICh2YXIgaW5uZXJDb2wgPSAwOyBpbm5lckNvbCA8ICR7Y29sUGVyVGhyZWFkfTsgaW5uZXJDb2wrKykge1xuICAgICAgICAgICAgICBhY2NbaW5uZXJSb3ddW2lubmVyQ29sXSA9XG4gICAgICAgICAgICAgICAgICBmbWEoQUNhY2hlZCwgQkNhY2hlZFtpbm5lckNvbF0sIGFjY1tpbm5lclJvd11baW5uZXJDb2xdKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuICAgICAgfVxuICAgICAgZm9yICh2YXIgaW5uZXJSb3cgPSAwOyBpbm5lclJvdyA8ICR7cm93UGVyVGhyZWFkfTsgaW5uZXJSb3crKykge1xuICAgICAgICBsZXQgZ1JvdyA9IGdsb2JhbFJvd1N0YXJ0ICsgbG9jYWxSb3cgKyBpbm5lclJvdyAqICR7d29ya2dyb3VwU2l6ZVsxXX07XG4gICAgICAgIGZvciAodmFyIGlubmVyQ29sID0gMDsgaW5uZXJDb2wgPCAke2NvbFBlclRocmVhZH07IGlubmVyQ29sKyspIHtcbiAgICAgICAgICBsZXQgZ0NvbCA9IGdsb2JhbENvbFN0YXJ0ICsgbG9jYWxDb2wgKyBpbm5lckNvbCAqICR7d29ya2dyb3VwU2l6ZVswXX07XG4gICAgICAgICAgbW1fd3JpdGUoYmF0Y2gsIGdSb3csIGdDb2wsIGFjY1tpbm5lclJvd11baW5uZXJDb2xdKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgYCA6XG4gICAgICBgXG4gIGxldCB0aWxlUm93ID0gaTMyKGxvY2FsSWQueSkgKiAke3Jvd1BlclRocmVhZH07XG4gIGxldCB0aWxlQ29sID0gaTMyKGxvY2FsSWQueCkgKiAke2NvbFBlclRocmVhZH07XG5cbiAgbGV0IGdsb2JhbFJvdyA9IGkzMihnbG9iYWxJZC55KSAqICR7cm93UGVyVGhyZWFkfTtcbiAgbGV0IGdsb2JhbENvbCA9IGkzMihnbG9iYWxJZC54KSAqICR7Y29sUGVyVGhyZWFkfTtcbiAgbGV0IGdsb2JhbFJvd1N0YXJ0ID0gaTMyKHdvcmtncm91cElkLnkpICogJHt0aWxlQU91dGVyfTtcblxuICBsZXQgdGlsZVJvd0EgPSBpMzIobG9jYWxJZC55KSAqICR7cm93UGVyVGhyZWFkQX07XG4gIGxldCB0aWxlQ29sQSA9IGkzMihsb2NhbElkLngpICogJHtjb2xQZXJUaHJlYWRBfTtcbiAgbGV0IHRpbGVSb3dCID0gaTMyKGxvY2FsSWQueSkgKiAke3Jvd1BlclRocmVhZEJ9O1xuICAvLyBMb29wIG92ZXIgc2hhcmVkIGRpbWVuc2lvbi5cbiAgZm9yICh2YXIgdCA9IDA7IHQgPCBudW1UaWxlczsgdCsrKSB7XG4gICAgLy8gTG9hZCBvbmUgdGlsZSBvZiBBIGludG8gbG9jYWwgbWVtb3J5LlxuICAgIGZvciAodmFyIGlubmVyUm93ID0gMDsgaW5uZXJSb3cgPCAke3Jvd1BlclRocmVhZEF9OyBpbm5lclJvdysrKSB7XG4gICAgICBmb3IgKHZhciBpbm5lckNvbCA9IDA7IGlubmVyQ29sIDwgJHtjb2xQZXJUaHJlYWRBfTsgaW5uZXJDb2wrKykge1xuICAgICAgICBsZXQgaW5wdXRSb3cgPSB0aWxlUm93QSArIGlubmVyUm93O1xuICAgICAgICBsZXQgaW5wdXRDb2wgPSB0aWxlQ29sQSArIGlubmVyQ29sO1xuICAgICAgICAke3dyaXRlRGF0YVRvU3ViQVNuaXBwZXQodHJhbnNwb3NlQSl9XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gTG9hZCBvbmUgdGlsZSBvZiBCIGludG8gbG9jYWwgbWVtb3J5LlxuICAgIGZvciAodmFyIGlubmVyUm93ID0gMDsgaW5uZXJSb3cgPCAke3Jvd1BlclRocmVhZEJ9OyBpbm5lclJvdysrKSB7XG4gICAgICBmb3IgKHZhciBpbm5lckNvbCA9IDA7IGlubmVyQ29sIDwgJHtjb2xQZXJUaHJlYWR9OyBpbm5lckNvbCsrKSB7XG4gICAgICAgIGxldCBpbnB1dFJvdyA9IHRpbGVSb3dCICsgaW5uZXJSb3c7XG4gICAgICAgIGxldCBpbnB1dENvbCA9IHRpbGVDb2wgKyBpbm5lckNvbDtcbiAgICAgICAgbW1fQnN1YltpbnB1dFJvd11baW5wdXRDb2xdID0gbW1fcmVhZEIoYmF0Y2hCLFxuICAgICAgICAgIGtTdGFydCArIGlucHV0Um93LFxuICAgICAgICAgIGdsb2JhbENvbCArIGlubmVyQ29sKTtcbiAgICAgIH1cbiAgICB9XG4gICAga1N0YXJ0ID0ga1N0YXJ0ICsgJHt0aWxlSW5uZXJ9O1xuICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgIC8vIENvbXB1dGUgYWNjIHZhbHVlcyBmb3IgYSBzaW5nbGUgdGhyZWFkLlxuICAgIHZhciBCQ2FjaGVkIDogYXJyYXk8ZjMyLCAke2NvbFBlclRocmVhZH0+O1xuICAgIGZvciAodmFyIGsgPSAwOyBrIDwgJHt0aWxlSW5uZXJ9OyBrKyspIHtcbiAgICAgIGZvciAodmFyIGlubmVyID0gMDsgaW5uZXIgPCAke2NvbFBlclRocmVhZH07IGlubmVyKyspIHtcbiAgICAgICAgQkNhY2hlZFtpbm5lcl0gPSBtbV9Cc3ViW2tdW3RpbGVDb2wgKyBpbm5lcl07XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIGlubmVyUm93ID0gMDsgaW5uZXJSb3cgPCAke3Jvd1BlclRocmVhZH07IGlubmVyUm93KyspIHtcbiAgICAgICAgJHtyZWFkRGF0YUZyb21TdWJBU25pcHBldCh0cmFuc3Bvc2VBKX1cbiAgICAgICAgZm9yICh2YXIgaW5uZXJDb2wgPSAwOyBpbm5lckNvbCA8ICR7Y29sUGVyVGhyZWFkfTsgaW5uZXJDb2wrKykge1xuICAgICAgICAgIGFjY1tpbm5lclJvd11baW5uZXJDb2xdID1cbiAgICAgICAgICAgICAgZm1hKEFDYWNoZWQsIEJDYWNoZWRbaW5uZXJDb2xdLCBhY2NbaW5uZXJSb3ddW2lubmVyQ29sXSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG4gIH1cblxuICBmb3IgKHZhciBpbm5lclJvdyA9IDA7IGlubmVyUm93IDwgJHtyb3dQZXJUaHJlYWR9OyBpbm5lclJvdysrKSB7XG4gICAgZm9yICh2YXIgaW5uZXJDb2wgPSAwOyBpbm5lckNvbCA8ICR7Y29sUGVyVGhyZWFkfTsgaW5uZXJDb2wrKykge1xuICAgICAgbW1fd3JpdGUoYmF0Y2gsIGdsb2JhbFJvdyArIGlubmVyUm93LCBnbG9iYWxDb2wgKyBpbm5lckNvbCxcbiAgICAgICAgICBhY2NbaW5uZXJSb3ddW2lubmVyQ29sXSk7XG4gICAgfVxuICB9XG4gIGA7XG5cbiAgcmV0dXJuIGBcbiAgICB2YXI8d29ya2dyb3VwPiBtbV9Bc3ViIDogYXJyYXk8YXJyYXk8ZjMyLCAke3RpbGVBV2lkdGh9PiwgJHt0aWxlQUhpZ2h0fT47XG4gICAgdmFyPHdvcmtncm91cD4gbW1fQnN1YiA6IGFycmF5PGFycmF5PGYzMiwgJHt0aWxlQk91dGVyfT4sICR7dGlsZUlubmVyfT47XG5cbiAgICAke21haW4oKX0ge1xuICAgICAgbGV0IGJhdGNoID0gJHtzcGxpdEsgPyAnMCcgOiAnaTMyKGdsb2JhbElkLnopJ307XG4gICAgICBsZXQgYmF0Y2hBID0gJHtcbiAgICAgIHNwbGl0SyB8fCAhYnJvYWRjYXN0QmF0Y2ggPyAnYmF0Y2gnIDogJ2JhdGNoICUgdW5pZm9ybXMuYVNoYXBlWzBdJ307XG4gICAgICBsZXQgYmF0Y2hCID0gJHtcbiAgICAgIHNwbGl0SyB8fCAhYnJvYWRjYXN0QmF0Y2ggPyAnYmF0Y2gnIDogJ2JhdGNoICUgdW5pZm9ybXMuYlNoYXBlWzBdJ307XG4gICAgICBsZXQgbnVtVGlsZXMgPSAke1xuICAgICAgc3BsaXRLID8gYCR7TWF0aC5jZWlsKHNwbGl0ZWREaW1Jbm5lciAvIHRpbGVJbm5lcil9YCA6XG4gICAgICAgICAgICAgICBgKHVuaWZvcm1zLmRpbUlubmVyIC0gMSkgLyAke3RpbGVJbm5lcn0gKyAxYH07XG4gICAgICB2YXIga1N0YXJ0ID0gJHtzcGxpdEsgPyBgaTMyKGdsb2JhbElkLnopICogJHtzcGxpdGVkRGltSW5uZXJ9YCA6ICcwJ307XG5cbiAgICAgIHZhciBhY2MgOiBhcnJheTxhcnJheTxmMzIsICR7Y29sUGVyVGhyZWFkfT4sICR7cm93UGVyVGhyZWFkfT47XG5cbiAgICAgIC8vIFdpdGhvdXQgdGhpcyBpbml0aWFsaXphdGlvbiBzdHJhbmdlIHZhbHVlcyBzaG93IHVwIGluIGFjYy5cbiAgICAgIGZvciAodmFyIGlubmVyUm93ID0gMDsgaW5uZXJSb3cgPCAke3Jvd1BlclRocmVhZH07IGlubmVyUm93KyspIHtcbiAgICAgICAgZm9yICh2YXIgaW5uZXJDb2wgPSAwOyBpbm5lckNvbCA8ICR7Y29sUGVyVGhyZWFkfTsgaW5uZXJDb2wrKykge1xuICAgICAgICAgIGFjY1tpbm5lclJvd11baW5uZXJDb2xdID0gMC4wO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICAke21hdG11bFNuaXBwZXR9XG4gICAgfVxuICBgO1xufVxuXG5jb25zdCByZWFkVmVjdG9yQVNuaXBwZXQgPSAodHJhbnNwb3NlOiBib29sZWFuKSA9PiB7XG4gIHJldHVybiB0cmFuc3Bvc2UgPyBgXG4gICAgICBtbV9yZWFkQShiYXRjaEEsIGNvbEEsIGdsb2JhbFJvdyksXG4gICAgICBtbV9yZWFkQShiYXRjaEEsIGNvbEEgKyAxLCBnbG9iYWxSb3cpLFxuICAgICAgbW1fcmVhZEEoYmF0Y2hBLCBjb2xBICsgMiwgZ2xvYmFsUm93KSxcbiAgICAgIG1tX3JlYWRBKGJhdGNoQSwgY29sQSArIDMsIGdsb2JhbFJvdylcbiAgYCA6XG4gICAgICAgICAgICAgICAgICAgICBgXG4gICAgICBtbV9yZWFkQShiYXRjaEEsIGdsb2JhbFJvdywgY29sQSksXG4gICAgICBtbV9yZWFkQShiYXRjaEEsIGdsb2JhbFJvdywgY29sQSArIDEpLFxuICAgICAgbW1fcmVhZEEoYmF0Y2hBLCBnbG9iYWxSb3csIGNvbEEgKyAyKSxcbiAgICAgIG1tX3JlYWRBKGJhdGNoQSwgZ2xvYmFsUm93LCBjb2xBICsgMylcbiAgYDtcbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlVmVjdG9yTWF0cml4UHJvZHVjdFNvdXJjZShcbiAgICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRyYW5zcG9zZUEgPSBmYWxzZSk6IHN0cmluZyB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgd29ya2dyb3VwU2l6ZVsxXSA9PT0gMSAmJiB3b3JrZ3JvdXBTaXplWzJdID09PSAxLFxuICAgICAgKCkgPT4gYEEgbGluZWFyIHdvcmsgZ3JvdXAgc2l6ZSBpcyByZXF1aXJlZC4gQnV0IGdvdCAke3dvcmtncm91cFNpemV9LmApO1xuICBjb25zdCB0aWxlU2l6ZSA9IHdvcmtncm91cFNpemVbMF0gKiA0O1xuICByZXR1cm4gYFxuICAgIHZhcjx3b3JrZ3JvdXA+IG1tX0FzdWIgOiBhcnJheTx2ZWM0PGYzMj4sICR7d29ya2dyb3VwU2l6ZVswXX0+O1xuXG4gICAgJHttYWluKCl9IHtcbiAgICAgIGxldCB0aWxlQ29sID0gaTMyKGxvY2FsSWQueCk7XG4gICAgICBsZXQgZ2xvYmFsQ29sID0gaTMyKGdsb2JhbElkLngpO1xuICAgICAgbGV0IGdsb2JhbFJvdyA9IGkzMihnbG9iYWxJZC55KTtcblxuICAgICAgbGV0IG51bVRpbGVzID0gKHVuaWZvcm1zLmRpbUlubmVyIC0gMSkgLyAke3RpbGVTaXplfSArIDE7XG4gICAgICBsZXQgYmF0Y2ggPSBpMzIoZ2xvYmFsSWQueik7XG4gICAgICBsZXQgYmF0Y2hBID0gYmF0Y2ggJSB1bmlmb3Jtcy5hU2hhcGVbMF07XG4gICAgICBsZXQgYmF0Y2hCID0gYmF0Y2ggJSB1bmlmb3Jtcy5iU2hhcGVbMF07XG4gICAgICAvLyBXaXRob3V0IHRoaXMgaW5pdGlhbGl6YXRpb24gc3RyYW5nZSB2YWx1ZXMgc2hvdyB1cCBpbiBhY2MuXG4gICAgICB2YXIgYWNjID0gMC4wO1xuXG4gICAgICAvLyBMb29wIG92ZXIgc2hhcmVkIGRpbWVuc2lvbi5cbiAgICAgIGZvciAodmFyIHQgPSAwOyB0IDwgbnVtVGlsZXM7IHQrKykge1xuICAgICAgICAvLyBMb2FkIG9uZSB0aWxlIG9mIEEgaW50byBsb2NhbCBtZW1vcnkuXG4gICAgICAgIGxldCBjb2xBID0gdCAqICR7dGlsZVNpemV9ICsgdGlsZUNvbCAqIDQ7XG4gICAgICAgIG1tX0FzdWJbdGlsZUNvbF0gPSB2ZWM0PGYzMj4oJHtyZWFkVmVjdG9yQVNuaXBwZXQodHJhbnNwb3NlQSl9KTtcbiAgICAgICAgd29ya2dyb3VwQmFycmllcigpO1xuXG4gICAgICAgIC8vIENvbXB1dGUgYWNjIHZhbHVlcyBmb3IgYSBzaW5nbGUgdGhyZWFkLlxuICAgICAgICBmb3IgKHZhciBrID0gMDsgayA8ICR7dGlsZVNpemUgLyA0fTsgaysrKSB7XG4gICAgICAgICAgbGV0IHJvd0IgPSB0ICogJHt0aWxlU2l6ZX0gKyBrICogNDtcbiAgICAgICAgICBsZXQgQkNhY2hlZCA9IHZlYzQ8ZjMyPihtbV9yZWFkQihiYXRjaEIsIHJvd0IsIGdsb2JhbENvbCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtbV9yZWFkQihiYXRjaEIsIHJvd0IgKyAxLCBnbG9iYWxDb2wpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbW1fcmVhZEIoYmF0Y2hCLCByb3dCICsgMiwgZ2xvYmFsQ29sKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1tX3JlYWRCKGJhdGNoQiwgcm93QiArIDMsIGdsb2JhbENvbCkpO1xuXG4gICAgICAgICAgbGV0IEFDYWNoZWQgPSBtbV9Bc3ViW2tdO1xuICAgICAgICAgIGFjYyA9IGFjYyArIGRvdChBQ2FjaGVkLCBCQ2FjaGVkKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcbiAgICAgIH1cblxuICAgICAgbW1fd3JpdGUoYmF0Y2gsIGdsb2JhbFJvdywgZ2xvYmFsQ29sLCBhY2MpO1xuICAgIH1cbiAgYDtcbn1cblxuZXhwb3J0IGNsYXNzIE1hdE11bFBhY2tlZFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXSwgeTogbnVtYmVyW10sIHo6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsnQScsICdCJ107XG4gIHVuaWZvcm1zID0gYGRpbUFPdXRlciA6IGkzMiwgZGltQk91dGVyIDogaTMyLCBkaW1Jbm5lciA6IGkzMixgO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGVsZW1lbnRzUGVyVGhyZWFkOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHRyYW5zcG9zZUE6IGJvb2xlYW47XG4gIHRyYW5zcG9zZUI6IGJvb2xlYW47XG4gIGFkZEJpYXM6IGJvb2xlYW47XG4gIGFjdGl2YXRpb246IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uO1xuICBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzOiBib29sZWFuO1xuICBmaXRBT3V0ZXI6IGJvb2xlYW47XG4gIGZpdEJPdXRlcjogYm9vbGVhbjtcbiAgZml0SW5uZXI6IGJvb2xlYW47XG4gIHRpbGVJbm5lcjogbnVtYmVyO1xuICBpc1ZlY3RvckE6IGJvb2xlYW47XG4gIGlzVmVjNDogYm9vbGVhbjtcbiAgb3V0cHV0Q29tcG9uZW50OiBudW1iZXI7XG4gIHByaXZhdGUgc2VxdWVudGlhbEFjY2Vzc0J5VGhyZWFkczogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGFTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBvdXRwdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgdHJhbnNwb3NlQSA9IGZhbHNlLCB0cmFuc3Bvc2VCID0gZmFsc2UsIGJpYXM6IFRlbnNvckluZm8gPSBudWxsLFxuICAgICAgYWN0aXZhdGlvbjogYmFja2VuZF91dGlsLkFjdGl2YXRpb24gPSBudWxsLFxuICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogVGVuc29ySW5mbyA9IG51bGwsXG4gICAgICBzZXF1ZW50aWFsQWNjZXNzQnlUaHJlYWRzID0gZmFsc2UpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IHt4OiBbMl0sIHk6IFsxXSwgejogWzBdfTtcbiAgICBjb25zdCBkaW1Jbm5lciA9IHRyYW5zcG9zZUEgPyBhU2hhcGVbMV0gOiBhU2hhcGVbMl07XG4gICAgdGhpcy5pc1ZlYzQgPSAoKGRpbUlubmVyICUgNCA9PT0gMCAmJiAhdHJhbnNwb3NlQSkgfHxcbiAgICAgICAgICAgICAgICAgICAob3V0cHV0U2hhcGVbMV0gJSA0ID09PSAwICYmIHRyYW5zcG9zZUEpKSAmJlxuICAgICAgICBvdXRwdXRTaGFwZVsyXSAlIDQgPT09IDAgJiYgIXRyYW5zcG9zZUI7XG4gICAgdGhpcy5vdXRwdXRDb21wb25lbnQgPSB0aGlzLmlzVmVjNCA/IDQgOiAxO1xuICAgIHRoaXMuaXNWZWN0b3JBID0gb3V0cHV0U2hhcGVbMV0gPT09IDEgJiYgIXRyYW5zcG9zZUE7XG5cbiAgICBpZiAoIXRoaXMuaXNWZWM0ICYmIHRoaXMuaXNWZWN0b3JBKSB7XG4gICAgICAvLyBGb3IgbWFrZVZlY3Rvck1hdHJpeFByb2R1Y3RTb3VyY2VcbiAgICAgIHRoaXMuZWxlbWVudHNQZXJUaHJlYWQgPSBbMSwgMSwgMV07XG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSBbMzIsIDEsIDFdO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB3b3JrZ3JvdXBJbmZvID0gY29tcHV0ZVdvcmtncm91cEluZm9Gb3JNYXRNdWwoXG4gICAgICAgICAgb3V0cHV0U2hhcGVbMV0sIGRpbUlubmVyLCBvdXRwdXRTaGFwZVsyXSwgdHJhbnNwb3NlQSk7XG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSB3b3JrZ3JvdXBJbmZvLndvcmtncm91cFNpemU7XG4gICAgICB0aGlzLmVsZW1lbnRzUGVyVGhyZWFkID0gd29ya2dyb3VwSW5mby5lbGVtZW50c1BlclRocmVhZDtcbiAgICB9XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIHRoaXMuZWxlbWVudHNQZXJUaHJlYWQpO1xuXG4gICAgY29uc3QgYWRkQmlhcyA9IGJpYXMgIT0gbnVsbDtcbiAgICBjb25zdCBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyAhPSBudWxsO1xuICAgIGlmIChhZGRCaWFzKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgnYmlhcycpO1xuICAgIH1cblxuICAgIGlmIChoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgncHJlbHVBY3RpdmF0aW9uV2VpZ2h0cycpO1xuICAgIH1cblxuICAgIHRoaXMuc2VxdWVudGlhbEFjY2Vzc0J5VGhyZWFkcyA9IHNlcXVlbnRpYWxBY2Nlc3NCeVRocmVhZHM7XG4gICAgdGhpcy50cmFuc3Bvc2VBID0gdHJhbnNwb3NlQTtcbiAgICB0aGlzLnRyYW5zcG9zZUIgPSB0cmFuc3Bvc2VCO1xuICAgIHRoaXMuYWRkQmlhcyA9IGFkZEJpYXM7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gYWN0aXZhdGlvbjtcbiAgICB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzO1xuICAgIFt0aGlzLmZpdEFPdXRlciwgdGhpcy5maXRCT3V0ZXIsIHRoaXMuZml0SW5uZXJdID1cbiAgICAgICAgdGhpcy5nZXRTaGFwZUZpdChvdXRwdXRTaGFwZVsxXSwgb3V0cHV0U2hhcGVbMl0sIGRpbUlubmVyKTtcbiAgICB0aGlzLnNoYWRlcktleSA9IGBtYXRNdWxQYWNrZWRfJHt0aGlzLmVsZW1lbnRzUGVyVGhyZWFkfV8ke3RyYW5zcG9zZUF9XyR7XG4gICAgICAgIHRyYW5zcG9zZUJ9XyR7dGhpcy5hY3RpdmF0aW9ufV8ke3RoaXMuZml0QU91dGVyfV8ke3RoaXMuZml0Qk91dGVyfV8ke1xuICAgICAgICB0aGlzLmZpdElubmVyfV8ke3RoaXMuaXNWZWM0fV8ke3RoaXMuaXNWZWN0b3JBfV8ke1xuICAgICAgICB0aGlzLnNlcXVlbnRpYWxBY2Nlc3NCeVRocmVhZHN9YDtcbiAgfVxuXG4gIGdldFNoYXBlRml0KGRpbUFPdXRlcjogbnVtYmVyLCBkaW1CT3V0ZXI6IG51bWJlciwgZGltSW5uZXI6IG51bWJlcik6XG4gICAgICBib29sZWFuW10ge1xuICAgIGNvbnN0IHRpbGVBT3V0ZXIgPSB0aGlzLndvcmtncm91cFNpemVbMV0gKiB0aGlzLmVsZW1lbnRzUGVyVGhyZWFkWzFdO1xuICAgIGNvbnN0IHRpbGVCT3V0ZXIgPSB0aGlzLndvcmtncm91cFNpemVbMF0gKiB0aGlzLmVsZW1lbnRzUGVyVGhyZWFkWzBdO1xuXG4gICAgaWYgKCF0aGlzLmlzVmVjNCAmJiB0aGlzLmlzVmVjdG9yQSkge1xuICAgICAgLy8gRm9yIG1ha2VWZWN0b3JNYXRyaXhQcm9kdWN0U291cmNlXG4gICAgICB0aGlzLnRpbGVJbm5lciA9IHRoaXMud29ya2dyb3VwU2l6ZVswXSAqIDQ7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudGlsZUlubmVyID0gdGlsZUJPdXRlcjtcbiAgICB9XG5cbiAgICBjb25zdCBmaXRBT3V0ZXIgPSBkaW1BT3V0ZXIgJSB0aWxlQU91dGVyID09PSAwO1xuICAgIGNvbnN0IGZpdEJPdXRlciA9IGRpbUJPdXRlciAlIHRpbGVCT3V0ZXIgPT09IDA7XG4gICAgY29uc3QgZml0SW5uZXIgPSBkaW1Jbm5lciAlIHRoaXMudGlsZUlubmVyID09PSAwO1xuICAgIHJldHVybiBbZml0QU91dGVyLCBmaXRCT3V0ZXIsIGZpdElubmVyXTtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke1xuICAgICAgICBhY3RpdmF0aW9uRm5TbmlwcGV0KFxuICAgICAgICAgICAgdGhpcy5hY3RpdmF0aW9uLCB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMsIHRoaXMuaXNWZWM0KX1cbiAgICAgICR7XG4gICAgICAgIG1hdE11bFJlYWRXcml0ZUZuU291cmNlKFxuICAgICAgICAgICAgdGhpcy5hZGRCaWFzLCB0aGlzLmFjdGl2YXRpb24sXG4gICAgICAgICAgICBmYWxzZSAvKiB0cmFuc3Bvc2VBIGlzIGltcGxlbWVudGVkIGluIG1ha2VNYXRNdWxQYWNrZWRTb3VyY2UgKi8sXG4gICAgICAgICAgICB0aGlzLnRyYW5zcG9zZUIsIHRoaXMuZml0QU91dGVyLCB0aGlzLmZpdEJPdXRlciwgdGhpcy5maXRJbm5lcixcbiAgICAgICAgICAgIHRoaXMuaXNWZWM0ID8gNCA6IDEpfVxuICAgICAgJHtcbiAgICAgICAgdGhpcy5pc1ZlYzQgP1xuICAgICAgICAgICAgbWFrZU1hdE11bFBhY2tlZFZlYzRTb3VyY2UoXG4gICAgICAgICAgICAgICAgdGhpcy5lbGVtZW50c1BlclRocmVhZCwgdGhpcy53b3JrZ3JvdXBTaXplLCB0aGlzLnRyYW5zcG9zZUEsXG4gICAgICAgICAgICAgICAgdGhpcy50aWxlSW5uZXIsIGZhbHNlLCBudWxsLCB0cnVlKSA6XG4gICAgICAgICAgICAodGhpcy5pc1ZlY3RvckEgPyBtYWtlVmVjdG9yTWF0cml4UHJvZHVjdFNvdXJjZShcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLndvcmtncm91cFNpemUsIHRoaXMudHJhbnNwb3NlQSkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWFrZU1hdE11bFBhY2tlZFNvdXJjZShcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmVsZW1lbnRzUGVyVGhyZWFkLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy50cmFuc3Bvc2VBLCB0aGlzLnRpbGVJbm5lciwgZmFsc2UsIG51bGwsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5zZXF1ZW50aWFsQWNjZXNzQnlUaHJlYWRzLCB0cnVlKSl9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==