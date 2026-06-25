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
import { backend_util, env, util } from '@tensorflow/tfjs-core';
import { symbolicallyComputeStrides } from './shader_util';
export var PixelsOpType;
(function (PixelsOpType) {
    PixelsOpType[PixelsOpType["FROM_PIXELS"] = 0] = "FROM_PIXELS";
    PixelsOpType[PixelsOpType["DRAW"] = 1] = "DRAW";
})(PixelsOpType || (PixelsOpType = {}));
export const compileProgram = (device, program, inputsData, output, parallelCompilation) => {
    const outputData = { dtype: output.dtype, shape: output.shape };
    const source = makeShader(inputsData, outputData, program);
    const module = device.createShaderModule({ code: source, label: program.constructor.name });
    let printShaderString = env().get('WEBGPU_PRINT_SHADER');
    if (printShaderString !== '') {
        printShaderString = printShaderString.toLowerCase();
        const printShaderArray = printShaderString.split(',');
        if (printShaderString === 'all' ||
            printShaderArray.some(item => program.shaderKey.toLowerCase().includes(item))) {
            console.group(program.shaderKey);
            console.debug(source);
            console.groupEnd();
        }
    }
    if (parallelCompilation) {
        return device.createComputePipelineAsync({
            compute: { module, entryPoint: '_start' },
            label: program.constructor.name,
            layout: 'auto'
        });
    }
    else {
        return device.createComputePipeline({
            compute: { module, entryPoint: '_start' },
            label: program.constructor.name,
            layout: 'auto'
        });
    }
};
export const typeSnippet = (component, type = 'f32') => {
    switch (component) {
        case 1:
            return `${type}`;
        case 2:
            return `vec2<${type}>`;
        case 3:
            return `vec3<${type}>`;
        case 4:
            return `vec4<${type}>`;
        default:
            throw new Error(`${component}-component ${type} is not supported.`);
    }
};
export function getCoordsDataType(rank) {
    if (rank <= 1) {
        return 'i32';
    }
    else if (rank === 2) {
        return `vec2<i32>`;
    }
    else if (rank === 3) {
        return `vec3<i32>`;
    }
    else if (rank === 4) {
        return `vec4<i32>`;
    }
    else if (rank === 5) {
        return `vec5`;
    }
    else if (rank === 6) {
        return `vec6`;
    }
    else {
        throw Error(`GPU for rank ${rank} is not yet supported`);
    }
}
export function getCoordsXYZ(index) {
    if (index === 0) {
        return 'x';
    }
    else if (index === 1) {
        return 'y';
    }
    else if (index === 2) {
        return 'z';
    }
    else if (index === 3) {
        return 'w';
    }
    else if (index === 4) {
        return 'u';
    }
    else if (index === 5) {
        return 'v';
    }
    else {
        throw Error(`Index ${index} is not yet supported`);
    }
}
export function getMainHeaderString(...params) {
    let snippet;
    switch (params.length) {
        case 0:
            snippet = `
        fn main()
      `;
            break;
        case 1:
            snippet = `
        fn main(${params[0]} : i32)
      `;
            break;
        default:
            throw Error('Unreachable');
    }
    return snippet;
}
export function getStartHeaderString(useGlobalIndex, program) {
    let snippet;
    snippet = `
     ${getWorkgroupSizeString(program)}
      fn _start(@builtin(local_invocation_id) LocalId : vec3<u32>,
                @builtin(global_invocation_id) GlobalId : vec3<u32>,
                @builtin(local_invocation_index) LocalIndex: u32,
                @builtin(workgroup_id) WorkgroupId : vec3<u32>,
                @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
        localId = LocalId;
        localIndex = LocalIndex;
        globalId = GlobalId;
        numWorkgroups = NumWorkgroups;
        workgroupId = WorkgroupId;
        ${useGlobalIndex ? `main(getGlobalIndex());` : `main();`};
      }
    `;
    return snippet;
}
export function getWorkgroupSizeString(program) {
    return `
  @compute @workgroup_size(${program.workgroupSize[0]}, ${program.workgroupSize[1]}, ${program.workgroupSize[2]})
`;
}
function makeShader(inputInfo, outputData, program) {
    const prefixSnippets = [];
    const flatWorkgroupSize = program.workgroupSize[0] *
        program.workgroupSize[1] * program.workgroupSize[2];
    program.outputComponent =
        program.outputComponent ? program.outputComponent : 1;
    prefixSnippets.push(`

      var<private> localId: vec3<u32>;
      var<private> localIndex: u32;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;
      var<private> workgroupId: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
        ${isFlatDispatch(program) ?
        `  return i32(globalId.x);` :
        `  return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
                workgroupId.y * numWorkgroups.x + workgroupId.x) * ${flatWorkgroupSize}u +
                localIndex);
        `}
      }
    `);
    if (program.pixelsOpType != null) {
        const inoutSnippet = program.pixelsOpType === PixelsOpType.FROM_PIXELS ?
            `@group(0) @binding(0) var<storage, read_write> result: array<${dataTypeToGPUType(outputData.dtype, program.outputComponent)}>;` :
            `@group(0) @binding(1) var<storage, read> inBuf : array<${dataTypeToGPUType(inputInfo[0].dtype, program.outputComponent)}>;`;
        const outShapeStridesType = outputData.shape.length === 3 ? 'vec2<i32>' : 'i32';
        prefixSnippets.push(`
        struct Uniform {
          outShapeStrides : ${outShapeStridesType},
          size            : i32,
          numChannels     : i32,
          alpha           : f32,
        };

        ${inoutSnippet}
        @group(0) @binding(2) var<uniform> uniforms: Uniform;
      `);
        const useGlobalIndex = isFlatDispatchLayout(program);
        return [
            commonSnippet,
            prefixSnippets.join('\n'),
            getCoordsFromIndexSnippet(outputData.shape),
            program.getUserCode(),
            getStartHeaderString(useGlobalIndex, program),
        ].join('\n');
    }
    let stridesLength;
    let stridesDataType;
    let uniformDeclaration = 'struct Uniforms { NAN : f32, INFINITY : f32, ';
    program.variableNames.forEach((x, i) => {
        const perDataType = getCoordsDataType(inputInfo[i].shape.length);
        uniformDeclaration +=
            `${x.charAt(0).toLowerCase() + x.slice(1)}Shape : ${perDataType}, `;
        stridesLength = inputInfo[i].shape.length - 1;
        stridesDataType = getCoordsDataType(stridesLength);
        uniformDeclaration +=
            `${x.charAt(0).toLowerCase() + x.slice(1)}ShapeStrides: ${stridesDataType}, `;
    });
    const outputDataType = getCoordsDataType(outputData.shape.length);
    uniformDeclaration += `outShape : ${outputDataType}, `;
    stridesLength = outputData.shape.length - 1;
    stridesDataType = getCoordsDataType(stridesLength);
    uniformDeclaration += `
         outShapeStrides: ${stridesDataType}, `;
    if (program.size) {
        uniformDeclaration += 'size : i32, ';
    }
    if (program.uniforms) {
        uniformDeclaration += program.uniforms;
    }
    uniformDeclaration += '};';
    uniformDeclaration = insertAlignment(uniformDeclaration);
    prefixSnippets.push(uniformDeclaration);
    // Output buffer.
    if (program.atomic) {
        prefixSnippets.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<atomic<i32>>;
    `);
    }
    else {
        prefixSnippets.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<${dataTypeToGPUType(outputData.dtype, program.outputComponent)}>;
    `);
    }
    program.variableNames.forEach((x, i) => {
        prefixSnippets.push(`
      @group(0) @binding(${1 + i}) var<storage, read> ${x}: array<${program.variableComponents ?
            dataTypeToGPUType(inputInfo[i].dtype, program.variableComponents[i]) :
            dataTypeToGPUType(inputInfo[i].dtype, program.outputComponent)}>;
        `);
    });
    if (uniformDeclaration !== '') {
        prefixSnippets.push(`
      @group(0) @binding(${1 + program.variableNames.length}) var<uniform> uniforms: Uniforms;
      `);
    }
    const coordsSnippet = getOutputCoordsSnippet(outputData.shape, program.dispatchLayout);
    const sources = [
        commonSnippet, prefixSnippets.join('\n') + isInfSnippet,
        getCoordsFromIndexSnippet(outputData.shape), coordsSnippet,
        getOutputIndexFromCoordsSnippet(outputData.shape.length)
    ];
    if (!program.atomic) {
        sources.push(setOutputSnippet(outputData.shape, outputData.dtype, program.outputComponent));
    }
    program.variableNames.forEach((x, i) => {
        sources.push(`${getCoordsFromIndexSnippet(inputInfo[i].shape, x)}`);
    });
    const inputSnippet = inputInfo
        .map((x, i) => getInputSnippet(x, outputData.shape, program.variableComponents ? program.variableComponents[i] :
        program.outputComponent, program.dispatchLayout.x.length === outputData.shape.length))
        .join('\n');
    sources.push(inputSnippet);
    sources.push(program.getUserCode());
    const useGlobalIndex = isFlatDispatchLayout(program);
    sources.push(getStartHeaderString(useGlobalIndex, program));
    const source = sources.join('\n');
    return source;
}
export function makeShaderKey(program, inputsData, output) {
    let key = program.shaderKey;
    if (program.pixelsOpType != null) {
        return key;
    }
    const shapes = [];
    const types = [];
    inputsData.forEach(element => {
        shapes.push(element.shape);
        types.push(element.dtype);
    });
    shapes.push(output.shape);
    types.push(output.dtype);
    const broadcastDims = inputsData.map(d => backend_util.getBroadcastDims(d.shape, output.shape));
    const inputShapesEqualsOutShape = inputsData.map(d => util.arraysEqual(d.shape, output.shape)).join('_');
    const broadcastDimsKey = broadcastDims.map(d => d.join('_')).join(';');
    const flatDispatchString = isFlatDispatch(program) ? 'flatDispatch' : '';
    key += '_' + (program.workgroupSize ? program.workgroupSize.join(',') : '') +
        shapes.map(shape => shape.length).join(',') + types.join(',') +
        program.variableNames.join(',') + broadcastDimsKey +
        inputShapesEqualsOutShape + flatDispatchString;
    return key;
}
const commonSnippet = `
  struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
  struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) && all(coord < shape);
  }

  fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
    return coord;
  }
  fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return dot(coords, vec2<i32>(shape.y, 1));
  }
  fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
  }
  fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return dot(coords, vec4<i32>(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
  }
  fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
    let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
  }
  fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
    let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
  }

  // NaN defination in IEEE 754-1985 is :
  //   - sign = either 0 or 1.
  //   - biased exponent = all 1 bits.
  //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
  // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
  fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
  }
  fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
    let floatToUint: vec4<u32> = bitcast<vec4<u32>>(val);
    return (floatToUint & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
  }
`;
const isInfSnippet = `
  fn isinf(val: f32) -> bool {
    return abs(val) == uniforms.INFINITY;
  }
`;
/**
 * Derives logical coordinates from a flat index. Performs integer division
 * with each stride and decrements the index until the index equals the final
 * dimension coordinate.
 */
export function getCoordsFromIndexSnippet(shape, name = '') {
    const rank = shape.length;
    const funcName = name !== '' ?
        `get${name.charAt(0).toUpperCase() + name.slice(1)}CoordsFromIndex` :
        'getCoordsFromIndex';
    const stridesName = name !== '' ?
        `${name.charAt(0).toLowerCase() + name.slice(1)}ShapeStrides` :
        `outShapeStrides`;
    if (rank <= 1) {
        return `fn ${funcName}(index : i32) -> i32 { return index; }`;
    }
    const strides = util.computeStrides(shape);
    const dtype = getCoordsDataType(rank);
    const coords = [];
    for (let i = 0; i < rank; i++) {
        coords.push(`d${i}`);
    }
    if (strides.length === 1) {
        return `    fn ${funcName}(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.${stridesName}; let d1 = index - d0 * uniforms.${stridesName};
      return vec2<i32>(d0, d1);
    }`;
    }
    let snippet;
    snippet = 'var index2 = index;' +
        strides
            .map((_, i) => {
            const line1 = `let ${coords[i]} = index2 / uniforms.${stridesName}.${getCoordsXYZ(i)}`;
            const line2 = i === strides.length - 1 ?
                `let ${coords[i + 1]} = index2 - ${coords[i]} * uniforms.${stridesName}.${getCoordsXYZ(i)}` :
                `index2 = index2 - ${coords[i]} * uniforms.${stridesName}.${getCoordsXYZ(i)}`;
            return `${line1}; ${line2};`;
        })
            .join('');
    return `
    fn ${funcName}(index : i32) -> ${dtype} {
      ${snippet}
      return ${dtype}(${coords.join(',')});
    }
  `;
}
function getInputAtCoordsSnippet(inputInfo, component) {
    const texName = inputInfo.name;
    const rank = inputInfo.shape.length;
    const type = getCoordsDataType(rank);
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const dims = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'].slice(0, rank);
    const inputs = dims.map(d => `${d} : i32`).join(', ');
    if (rank < 1) {
        return `
      fn ${funcName}() -> ${typeSnippet(component)} {
        return ${typeSnippet(component)}(${texName}[0]);
      }
    `;
    }
    const shapeStr = `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    let rankStr = `${rank}D`;
    if (rank === 0) {
        rankStr = '1D';
    }
    return `
    fn ${funcName}(${inputs}) -> ${typeSnippet(component)} {
      return ${typeSnippet(component)}(${texName}[getIndexFromCoords${rankStr}(${type}(${dims.join(',')}),
        ${shapeStr})${component === 1 ? '' : ` / ${component}`}]);
    }
   `;
}
function getInputByOutputSnippet(inputInfo, outShape, component, isFlatDispatchLayout) {
    const texName = inputInfo.name;
    const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    const funcName = 'get' + texFuncSnippet + 'ByOutput';
    const inRank = inputInfo.shape.length;
    const outRank = outShape.length;
    const type = getCoordsDataType(outRank);
    // If the inShape equals the outShape and the dispatch layout is flat, we can
    // directly use |gl_GlobalInvocationID.x| as the index and don't need coords
    // conversion between these two shapes.
    if (util.arraysEqual(inputInfo.shape, outShape) && isFlatDispatchLayout) {
        return `
    fn ${funcName}Index(globalIndex : i32) -> ${typeSnippet(component)} {
      return ${typeSnippet(component)}(${texName}[globalIndex]);
    }

    fn ${funcName}Coords(coords : ${type}) -> ${typeSnippet(component)} {
      return ${typeSnippet(component)}(${texName}[${outRank > 1 ? 'getOutputIndexFromCoords(coords)' :
            'coords'}${component === 1 ? '' : ` / ${component}`}]);
    }
    `;
    }
    const broadcastDims = backend_util.getBroadcastDims(inputInfo.shape, outShape);
    const rankDiff = outRank - inRank;
    let coordsSnippet = '';
    if (inRank === 0) {
        return `
    fn ${funcName}Index(globalIndex : i32) -> ${typeSnippet(component)}{
      return get${texFuncSnippet}();
    }

    fn ${funcName}Coords(coords : ${type}) -> ${typeSnippet(component)}{
      return get${texFuncSnippet}();
    }
  `;
    }
    else {
        if (outRank < 2 && broadcastDims.length >= 1) {
            coordsSnippet = 'coords = 0;';
        }
        else {
            coordsSnippet =
                broadcastDims.map(d => `coords.${getCoordsXYZ(d + rankDiff)} = 0;`)
                    .join('\n');
        }
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        if (outRank > 1) {
            const coordsType = getCoordsDataType(inRank);
            const coordsValues = inputInfo.shape.map((s, i) => `coords.${getCoordsXYZ(i + rankDiff)}`)
                .join(', ');
            unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
        }
        else {
            unpackedCoordsSnippet = 'coords';
        }
    }
    const shapeStr = `uniforms.${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape`;
    const rankStr = `${inRank}D`;
    return `
  fn ${funcName}Index(globalIndex : i32) -> ${typeSnippet(component)} {
    var coords = getCoordsFromIndex(globalIndex);
    ${coordsSnippet}
    return ${typeSnippet(component)}(${texName}[getIndexFromCoords${rankStr}(${unpackedCoordsSnippet}, ${shapeStr})${component === 1 ? '' : ` / ${component}`}]);
  }

  fn ${funcName}Coords(coordsIn : ${type}) -> ${typeSnippet(component)} {
    var coords = coordsIn;
    ${coordsSnippet}
    return ${typeSnippet(component)}(${texName}[getIndexFromCoords${rankStr}(${unpackedCoordsSnippet}, ${shapeStr})${component === 1 ? '' : ` / ${component}`}]);
  }
`;
}
function getInputSnippet(inputInfo, outShape, component, isFlatDispatchLayout) {
    let res = getInputAtCoordsSnippet(inputInfo, component);
    const inShape = inputInfo.shape;
    if (inShape.length <= outShape.length) {
        res += getInputByOutputSnippet(inputInfo, outShape, component, isFlatDispatchLayout);
    }
    return res;
}
/**
 * Generates getOutputCoords() function that computes output coordinates
 * from dispatch geometry to reduce arithmetic.
 */
function getOutputCoordsSnippet(outShape, dispatchLayout) {
    const { x, y = [], z = [] } = dispatchLayout;
    const outRank = outShape.length;
    const rank = x.length + y.length + z.length;
    // getOutputCoords is only meaningful when the output rank is same with
    // dispatch layout rank.
    if (rank !== outRank) {
        return '';
    }
    if (x.length === outRank) {
        const dtype = getCoordsDataType(outRank);
        const snippet = `fn getOutputCoords() -> ${dtype}{
    let globalIndex = getGlobalIndex();
    return getCoordsFromIndex(globalIndex);
  }
  `;
        return snippet;
    }
    let gatherDimensionsStr = '';
    const dims = [x, y, z];
    for (let i = 0; i < dims.length; i++) {
        const arr = dims[i];
        if (arr.length === 0) {
            continue;
        }
        if (arr.length === 1) {
            gatherDimensionsStr += `let d${arr[0]} = i32(globalId[${i}]);`;
        }
        else {
            const strides = symbolicallyComputeStrides(arr, 'uniforms.outShape');
            gatherDimensionsStr += `var index${i} = i32(globalId[${i}]);`;
            for (let j = 0; j < strides.length; j++) {
                gatherDimensionsStr += `let d${arr[j]} = index${i} / ${strides[j]};`;
                if (j === strides.length - 1) {
                    gatherDimensionsStr += `let d${arr[j + 1]} = ` +
                        `index${i} - d${arr[j]} * ${strides[j]};`;
                }
                else {
                    gatherDimensionsStr +=
                        `index${i} = index${i} - d${arr[j]} * ${strides[j]};`;
                }
            }
        }
    }
    const dimensions = [];
    for (let i = 0; i < rank; i++) {
        dimensions.push(`d${i}`);
    }
    const dtype = getCoordsDataType(rank);
    let snippet = `fn getOutputCoords() -> ${dtype} {
  ${gatherDimensionsStr}
`;
    if (dimensions.length === 0) {
        snippet += `return ${dtype}(0); }`;
    }
    else {
        snippet += `return ${dtype}(${dimensions.join(',')}); }`;
    }
    return snippet;
}
function getOutputIndexFromCoordsSnippet(outRank) {
    let snippet = '';
    switch (outRank) {
        case 0:
        case 1:
            snippet += `
        fn getOutputIndexFromCoords(coords : i32) -> i32 {
          return coords;
        }
        `;
            break;
        case 2:
            snippet += `
        fn getOutputIndexFromCoords(coords : vec2<i32>) -> i32 {
          return dot(coords, vec2<i32>(uniforms.outShapeStrides, 1));
        }
        `;
            break;
        case 3:
            snippet += `
        fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
          return dot(coords, vec3<i32>(uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, 1));
        }
        `;
            break;
        case 4:
            snippet += `
        fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
          return dot(coords, vec4<i32>(
            uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, uniforms.outShapeStrides.z, 1));
        }
        `;
            break;
        case 5:
            snippet += `
        fn getOutputIndexFromCoords(coords : vec5) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u;
        }
        `;
            break;
        case 6:
            snippet += `
        fn getOutputIndexFromCoords(coords : vec6) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u * uniforms.outShapeStrides.u +
              coords.v;
        }
        `;
            break;
        default:
            util.assert(false, () => `Unsupported ${outRank}D shape`);
            break;
    }
    return snippet;
}
function isFlatDispatch(program) {
    return program.dispatch[1] === 1 && program.dispatch[2] === 1;
}
export function dataTypeToGPUType(type, component = 1) {
    if (type === 'float32') {
        return typeSnippet(component, 'f32');
    }
    else if (type === 'int32' || type === 'bool') {
        return typeSnippet(component, 'i32');
    }
    throw new Error(`type ${type} is not supported.`);
}
function setOutputSnippet(outShape, outBufferType, component) {
    const outRank = outShape.length;
    const gpuType = dataTypeToGPUType(outBufferType, component);
    let snippet = `fn setOutputAtIndex(flatIndex : i32, value : ${typeSnippet(component)}) {
      result[flatIndex] = ${gpuType}(value);
    }

    fn setOutputAtIndexI32(flatIndex : i32, value : ${typeSnippet(component, 'i32')}) {
      result[flatIndex] = ${gpuType}(value);
    }
    `;
    if (outRank >= 2) {
        const dims = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'].slice(0, outRank);
        const type = getCoordsDataType(outRank);
        snippet += `
      fn setOutputAtCoords(${dims.map(d => `${d} : i32`).join(', ')}, value : ${typeSnippet(component)}) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndex(flatIndex${component === 1 ? '' : ` / ${component}`}, value);
      }
      fn setOutputAtCoordsI32(${dims.map(d => `${d} : i32`).join(', ')}, value : ${typeSnippet(component, 'i32')}) {
        let flatIndex = getOutputIndexFromCoords(${type}(${dims.join(', ')}));
        setOutputAtIndexI32(flatIndex${component === 1 ? '' : ` / ${component}`}, value);
      }
    `;
    }
    return snippet;
}
function insertAlignment(uniformShader) {
    // insert alignment when current pattern is vec5 or vec6
    const curInsertRe = /(\w+)\s*:\s*vec(5|6)/g;
    uniformShader = uniformShader.replace(curInsertRe, (match) => {
        return '@align(16) ' + match;
    });
    // insert alignment when previous pattern is vec5 or vec6
    const preInsertRe = /vec(5|6)\s*,\s*(\w+)/g;
    uniformShader = uniformShader.replace(preInsertRe, (_, p1, p2) => {
        return `vec${p1}, @align(16) ${p2}`;
    });
    return uniformShader;
}
function isFlatDispatchLayout(program) {
    if (program.dispatchLayout.hasOwnProperty('y') &&
        program.dispatchLayout.y.length !== 0) {
        return false;
    }
    if (program.dispatchLayout.hasOwnProperty('z') &&
        program.dispatchLayout.z.length !== 0) {
        return false;
    }
    return true;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2ViZ3B1X3Byb2dyYW0uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy93ZWJncHVfcHJvZ3JhbS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUF5QixHQUFHLEVBQW9CLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXZHLE9BQU8sRUFBQywwQkFBMEIsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUV6RCxNQUFNLENBQU4sSUFBWSxZQUdYO0FBSEQsV0FBWSxZQUFZO0lBQ3RCLDZEQUFXLENBQUE7SUFDWCwrQ0FBSSxDQUFBO0FBQ04sQ0FBQyxFQUhXLFlBQVksS0FBWixZQUFZLFFBR3ZCO0FBb0NELE1BQU0sQ0FBQyxNQUFNLGNBQWMsR0FDdkIsQ0FBQyxNQUFpQixFQUFFLE9BQXNCLEVBQUUsVUFBdUIsRUFDbEUsTUFBa0IsRUFBRSxtQkFBNEIsRUFDckIsRUFBRTtJQUM1QixNQUFNLFVBQVUsR0FBRyxFQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsS0FBSyxFQUFDLENBQUM7SUFDOUQsTUFBTSxNQUFNLEdBQUcsVUFBVSxDQUFDLFVBQVUsRUFBRSxVQUFVLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDM0QsTUFBTSxNQUFNLEdBQUcsTUFBTSxDQUFDLGtCQUFrQixDQUNwQyxFQUFDLElBQUksRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLE9BQU8sQ0FBQyxXQUFXLENBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQztJQUVyRCxJQUFJLGlCQUFpQixHQUFHLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxxQkFBcUIsQ0FBVyxDQUFDO0lBQ25FLElBQUksaUJBQWlCLEtBQUssRUFBRSxFQUFFO1FBQzVCLGlCQUFpQixHQUFHLGlCQUFpQixDQUFDLFdBQVcs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LEVBQUU7UUFDL0QsT0FBTyxNQUFNLEVBQUUsZ0JBQWdCLEVBQUUsRUFBRSxDQUFDO0lBQ3RDLENBQUMsQ0FBQyxDQUFDO0lBQ0gsT0FBTyxhQUFhLENBQUM7QUFDdkIsQ0FBQztBQUNELFNBQVMsb0JBQW9CLENBQUMsT0FBc0I7SUFDbEQsSUFBSSxPQUFPLENBQUMsY0FBYyxDQUFDLGNBQWMsQ0FBQyxHQUFHLENBQUM7UUFDMUMsT0FBTyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUN6QyxPQUFPLEtBQUssQ0FBQztLQUNkO0lBQ0QsSUFBSSxPQUFPLENBQUMsY0FBYyxDQUFDLGNBQWMsQ0FBQyxHQUFHLENBQUM7UUFDMUMsT0FBTyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUN6QyxPQUFPLEtBQUssQ0FBQztLQUNkO0lBQ0QsT0FBTyxJQUFJLENBQUM7QUFDZCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgRGF0YVR5cGUsIERhdGFUeXBlTWFwLCBlbnYsIFJhbmssIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7c3ltYm9saWNhbGx5Q29tcHV0ZVN0cmlkZXN9IGZyb20gJy4vc2hhZGVyX3V0aWwnO1xuXG5leHBvcnQgZW51bSBQaXhlbHNPcFR5cGUge1xuICBGUk9NX1BJWEVMUyxcbiAgRFJBV1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdQVVByb2dyYW0ge1xuICAvLyBXaGV0aGVyIHRvIHVzZSBhdG9taWMgYnVpbHQtaW4gZnVuY3Rpb25zLlxuICBhdG9taWM/OiBib29sZWFuO1xuICAvLyBkaXNwYXRjaCBzcGVjaWZpZXMgZ2VvbWV0cnkgb2YgdGhyZWFkIGdyb3VwcyAtIGRlcml2ZWQgZnJvbSBkaXNwYXRjaExheW91dC5cbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgLy8gZGlzcGF0Y2hMYXlvdXQgZW51bWVyYXRlcyBob3cgdGVuc29yIGRpbWVuc2lvbnMgYXJlIGRpc3RyaWJ1dGVkIGFtb25nXG4gIC8vIGRpc3BhdGNoIHgseSx6IGRpbWVuc2lvbnMuXG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW10sIHk/OiBudW1iZXJbXSwgej86IG51bWJlcltdfTtcbiAgLy8gQnkgZGVmYXVsdCwgdGhlIG91dHB1dCBkYXRhIGNvbXBvbmVudCBpcyAxLlxuICBvdXRwdXRDb21wb25lbnQ/OiBudW1iZXI7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgcGl4ZWxzT3BUeXBlPzogUGl4ZWxzT3BUeXBlO1xuICAvLyBUaGUgdW5pcXVlIGtleSB0byBkaXN0aW5ndWlzaCBkaWZmZXJlbnQgc2hhZGVyIHNvdXJjZSBjb2RlLlxuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgLy8gV2hldGhlciB0byB1c2Ugb3V0cHV0IHNpemUgZm9yIGJvdW5kcyBjaGVja2luZy5cbiAgc2l6ZT86IGJvb2xlYW47XG4gIHVuaWZvcm1zPzogc3RyaW5nO1xuICB2YXJpYWJsZU5hbWVzOiBzdHJpbmdbXTtcbiAgLy8gRGVzY3JpYmUgZWFjaCB2YXJpYWJsZSdzIGNvbXBvbmVudCBhbmQgbXVzdCBoYXZlIG9uZS1vbmUgbWFwcGluZyB3aXRoXG4gIC8vIHZhcmlhYmxlTmFtZXMuIElmIG5vdCBzZXQsIGFsbCB2YXJpYWJsZXMgY29tcG9uZW50IHdpbGwgYmUgc2FtZSB3aXRoIG91dHB1dFxuICAvLyBjb21wb25lbnQgbWVtYmVyLlxuICB2YXJpYWJsZUNvbXBvbmVudHM/OiBudW1iZXJbXTtcbiAgLy8gd29ya2dyb3VwU2l6ZS54ICogd29ya2dyb3VwU2l6ZS55ICogd29ya2dyb3VwU2l6ZS56ID0gdGhlIG51bWJlciBvZiB0aHJlYWRzXG4gIC8vIGluIGEgdGhyZWFkIGdyb3VwLiBJbmRpdmlkdWFsIGRpbWVuc2lvbnMgZGV0ZXJtaW5lcyB0aHJlYWQgbGF5b3V0IHdpdGhpblxuICAvLyB0aGUgZ3JvdXAuXG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgLy8gU2l6ZSBvZiByZWdpc3RlciBjYWNoZSBpbiBvbmUgZGltZW5zaW9uIChhc3N1bWVzIHNxdWFyZSBjYWNoZSkuXG4gIC8vIEVhY2ggdGhyZWFkIHdyaXRlcyB0byB3b3JrUGVyVGhyZWFkICogd29ya1BlclRocmVhZCBsb2NhdGlvbnMgaW4gdGhlIG91dHB1dFxuICAvLyBidWZmZXIuXG4gIHdvcmtQZXJUaHJlYWQ/OiBudW1iZXI7XG4gIHBpcGVsaW5lPzogR1BVQ29tcHV0ZVBpcGVsaW5lfFByb21pc2U8R1BVQ29tcHV0ZVBpcGVsaW5lPjtcbiAgZ2V0VXNlckNvZGU6ICgpID0+IHN0cmluZztcbn1cblxuZXhwb3J0IGNvbnN0IGNvbXBpbGVQcm9ncmFtID1cbiAgICAoZGV2aWNlOiBHUFVEZXZpY2UsIHByb2dyYW06IFdlYkdQVVByb2dyYW0sIGlucHV0c0RhdGE6IElucHV0SW5mb1tdLFxuICAgICBvdXRwdXQ6IFRlbnNvckluZm8sIHBhcmFsbGVsQ29tcGlsYXRpb246IGJvb2xlYW4pOiBHUFVDb21wdXRlUGlwZWxpbmV8XG4gICAgUHJvbWlzZTxHUFVDb21wdXRlUGlwZWxpbmU+ID0+IHtcbiAgICAgIGNvbnN0IG91dHB1dERhdGEgPSB7ZHR5cGU6IG91dHB1dC5kdHlwZSwgc2hhcGU6IG91dHB1dC5zaGFwZX07XG4gICAgICBjb25zdCBzb3VyY2UgPSBtYWtlU2hhZGVyKGlucHV0c0RhdGEsIG91dHB1dERhdGEsIHByb2dyYW0pO1xuICAgICAgY29uc3QgbW9kdWxlID0gZGV2aWNlLmNyZWF0ZVNoYWRlck1vZHVsZShcbiAgICAgICAgICB7Y29kZTogc291cmNlLCBsYWJlbDogcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lfSk7XG5cbiAgICAgIGxldCBwcmludFNoYWRlclN0cmluZyA9IGVudigpLmdldCgnV0VCR1BVX1BSSU5UX1NIQURFUicpIGFzIHN0cmluZztcbiAgICAgIGlmIChwcmludFNoYWRlclN0cmluZyAhPT0gJycpIHtcbiAgICAgICAgcHJpbnRTaGFkZXJTdHJpbmcgPSBwcmludFNoYWRlclN0cmluZy50b0xvd2VyQ2FzZSgpO1xuICAgICAgICBjb25zdCBwcmludFNoYWRlckFycmF5ID0gcHJpbnRTaGFkZXJTdHJpbmcuc3BsaXQoJywnKTtcbiAgICAgICAgaWYgKHByaW50U2hhZGVyU3RyaW5nID09PSAnYWxsJyB8fFxuICAgICAgICAgICAgcHJpbnRTaGFkZXJBcnJheS5zb21lKFxuICAgICAgICAgICAgICAgIGl0ZW0gPT4gcHJvZ3JhbS5zaGFkZXJLZXkudG9Mb3dlckNhc2UoKS5pbmNsdWRlcyhpdGVtKSkpIHtcbiAgICAgICAgICBjb25zb2xlLmdyb3VwKHByb2dyYW0uc2hhZGVyS2V5KTtcbiAgICAgICAgICBjb25zb2xlLmRlYnVnKHNvdXJjZSk7XG4gICAgICAgICAgY29uc29sZS5ncm91cEVuZCgpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChwYXJhbGxlbENvbXBpbGF0aW9uKSB7XG4gICAgICAgIHJldHVybiBkZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lQXN5bmMoe1xuICAgICAgICAgIGNvbXB1dGU6IHttb2R1bGUsIGVudHJ5UG9pbnQ6ICdfc3RhcnQnfSxcbiAgICAgICAgICBsYWJlbDogcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lLFxuICAgICAgICAgIGxheW91dDogJ2F1dG8nXG4gICAgICAgIH0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIGRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgICAgIGNvbXB1dGU6IHttb2R1bGUsIGVudHJ5UG9pbnQ6ICdfc3RhcnQnfSxcbiAgICAgICAgICBsYWJlbDogcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lLFxuICAgICAgICAgIGxheW91dDogJ2F1dG8nXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH07XG5cbmV4cG9ydCBjb25zdCB0eXBlU25pcHBldCA9IChjb21wb25lbnQ6IG51bWJlciwgdHlwZSA9ICdmMzInKSA9PiB7XG4gIHN3aXRjaCAoY29tcG9uZW50KSB7XG4gICAgY2FzZSAxOlxuICAgICAgcmV0dXJuIGAke3R5cGV9YDtcbiAgICBjYXNlIDI6XG4gICAgICByZXR1cm4gYHZlYzI8JHt0eXBlfT5gO1xuICAgIGNhc2UgMzpcbiAgICAgIHJldHVybiBgdmVjMzwke3R5cGV9PmA7XG4gICAgY2FzZSA0OlxuICAgICAgcmV0dXJuIGB2ZWM0PCR7dHlwZX0+YDtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGAke2NvbXBvbmVudH0tY29tcG9uZW50ICR7dHlwZX0gaXMgbm90IHN1cHBvcnRlZC5gKTtcbiAgfVxufTtcblxuZXhwb3J0IGZ1bmN0aW9uIGdldENvb3Jkc0RhdGFUeXBlKHJhbms6IG51bWJlcik6IHN0cmluZyB7XG4gIGlmIChyYW5rIDw9IDEpIHtcbiAgICByZXR1cm4gJ2kzMic7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gMikge1xuICAgIHJldHVybiBgdmVjMjxpMzI+YDtcbiAgfSBlbHNlIGlmIChyYW5rID09PSAzKSB7XG4gICAgcmV0dXJuIGB2ZWMzPGkzMj5gO1xuICB9IGVsc2UgaWYgKHJhbmsgPT09IDQpIHtcbiAgICByZXR1cm4gYHZlYzQ8aTMyPmA7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gNSkge1xuICAgIHJldHVybiBgdmVjNWA7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gNikge1xuICAgIHJldHVybiBgdmVjNmA7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgRXJyb3IoYEdQVSBmb3IgcmFuayAke3Jhbmt9IGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldENvb3Jkc1hZWihpbmRleDogbnVtYmVyKTogc3RyaW5nIHtcbiAgaWYgKGluZGV4ID09PSAwKSB7XG4gICAgcmV0dXJuICd4JztcbiAgfSBlbHNlIGlmIChpbmRleCA9PT0gMSkge1xuICAgIHJldHVybiAneSc7XG4gIH0gZWxzZSBpZiAoaW5kZXggPT09IDIpIHtcbiAgICByZXR1cm4gJ3onO1xuICB9IGVsc2UgaWYgKGluZGV4ID09PSAzKSB7XG4gICAgcmV0dXJuICd3JztcbiAgfSBlbHNlIGlmIChpbmRleCA9PT0gNCkge1xuICAgIHJldHVybiAndSc7XG4gIH0gZWxzZSBpZiAoaW5kZXggPT09IDUpIHtcbiAgICByZXR1cm4gJ3YnO1xuICB9IGVsc2Uge1xuICAgIHRocm93IEVycm9yKGBJbmRleCAke2luZGV4fSBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRNYWluSGVhZGVyU3RyaW5nKCk6IHN0cmluZztcbmV4cG9ydCBmdW5jdGlvbiBnZXRNYWluSGVhZGVyU3RyaW5nKGluZGV4OiBzdHJpbmcpOiBzdHJpbmc7XG5leHBvcnQgZnVuY3Rpb24gZ2V0TWFpbkhlYWRlclN0cmluZyguLi5wYXJhbXM6IHN0cmluZ1tdKTogc3RyaW5nIHtcbiAgbGV0IHNuaXBwZXQ6IHN0cmluZztcbiAgc3dpdGNoIChwYXJhbXMubGVuZ3RoKSB7XG4gICAgY2FzZSAwOlxuICAgICAgc25pcHBldCA9IGBcbiAgICAgICAgZm4gbWFpbigpXG4gICAgICBgO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSAxOlxuICAgICAgc25pcHBldCA9IGBcbiAgICAgICAgZm4gbWFpbigke3BhcmFtc1swXX0gOiBpMzIpXG4gICAgICBgO1xuICAgICAgYnJlYWs7XG4gICAgZGVmYXVsdDpcbiAgICAgIHRocm93IEVycm9yKCdVbnJlYWNoYWJsZScpO1xuICB9XG4gIHJldHVybiBzbmlwcGV0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0U3RhcnRIZWFkZXJTdHJpbmcoXG4gICAgdXNlR2xvYmFsSW5kZXg6IGJvb2xlYW4sIHByb2dyYW06IFdlYkdQVVByb2dyYW0pOiBzdHJpbmcge1xuICBsZXQgc25pcHBldDogc3RyaW5nO1xuICBzbmlwcGV0ID0gYFxuICAgICAke2dldFdvcmtncm91cFNpemVTdHJpbmcocHJvZ3JhbSl9XG4gICAgICBmbiBfc3RhcnQoQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZCkgTG9jYWxJZCA6IHZlYzM8dTMyPixcbiAgICAgICAgICAgICAgICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZCkgR2xvYmFsSWQgOiB2ZWMzPHUzMj4sXG4gICAgICAgICAgICAgICAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pbmRleCkgTG9jYWxJbmRleDogdTMyLFxuICAgICAgICAgICAgICAgIEBidWlsdGluKHdvcmtncm91cF9pZCkgV29ya2dyb3VwSWQgOiB2ZWMzPHUzMj4sXG4gICAgICAgICAgICAgICAgQGJ1aWx0aW4obnVtX3dvcmtncm91cHMpIE51bVdvcmtncm91cHMgOiB2ZWMzPHUzMj4pIHtcbiAgICAgICAgbG9jYWxJZCA9IExvY2FsSWQ7XG4gICAgICAgIGxvY2FsSW5kZXggPSBMb2NhbEluZGV4O1xuICAgICAgICBnbG9iYWxJZCA9IEdsb2JhbElkO1xuICAgICAgICBudW1Xb3JrZ3JvdXBzID0gTnVtV29ya2dyb3VwcztcbiAgICAgICAgd29ya2dyb3VwSWQgPSBXb3JrZ3JvdXBJZDtcbiAgICAgICAgJHt1c2VHbG9iYWxJbmRleCA/IGBtYWluKGdldEdsb2JhbEluZGV4KCkpO2AgOiBgbWFpbigpO2B9O1xuICAgICAgfVxuICAgIGA7XG4gIHJldHVybiBzbmlwcGV0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V29ya2dyb3VwU2l6ZVN0cmluZyhwcm9ncmFtOiBXZWJHUFVQcm9ncmFtKTogc3RyaW5nIHtcbiAgcmV0dXJuIGBcbiAgQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKCR7cHJvZ3JhbS53b3JrZ3JvdXBTaXplWzBdfSwgJHtcbiAgICAgIHByb2dyYW0ud29ya2dyb3VwU2l6ZVsxXX0sICR7cHJvZ3JhbS53b3JrZ3JvdXBTaXplWzJdfSlcbmA7XG59XG5cbmZ1bmN0aW9uIG1ha2VTaGFkZXIoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm9bXSwgb3V0cHV0RGF0YToge2R0eXBlOiBEYXRhVHlwZSwgc2hhcGU6IG51bWJlcltdfSxcbiAgICBwcm9ncmFtOiBXZWJHUFVQcm9ncmFtKTogc3RyaW5nIHtcbiAgY29uc3QgcHJlZml4U25pcHBldHM6IHN0cmluZ1tdID0gW107XG4gIGNvbnN0IGZsYXRXb3JrZ3JvdXBTaXplID0gcHJvZ3JhbS53b3JrZ3JvdXBTaXplWzBdICpcbiAgICAgIHByb2dyYW0ud29ya2dyb3VwU2l6ZVsxXSAqIHByb2dyYW0ud29ya2dyb3VwU2l6ZVsyXTtcbiAgcHJvZ3JhbS5vdXRwdXRDb21wb25lbnQgPVxuICAgICAgcHJvZ3JhbS5vdXRwdXRDb21wb25lbnQgPyBwcm9ncmFtLm91dHB1dENvbXBvbmVudCA6IDE7XG4gIHByZWZpeFNuaXBwZXRzLnB1c2goYFxuXG4gICAgICB2YXI8cHJpdmF0ZT4gbG9jYWxJZDogdmVjMzx1MzI+O1xuICAgICAgdmFyPHByaXZhdGU+IGxvY2FsSW5kZXg6IHUzMjtcbiAgICAgIHZhcjxwcml2YXRlPiBnbG9iYWxJZDogdmVjMzx1MzI+O1xuICAgICAgdmFyPHByaXZhdGU+IG51bVdvcmtncm91cHM6IHZlYzM8dTMyPjtcbiAgICAgIHZhcjxwcml2YXRlPiB3b3JrZ3JvdXBJZDogdmVjMzx1MzI+O1xuXG4gICAgICAvLyBPbmx5IHVzZWQgd2hlbiB0aGUgeS96IGRpbWVuc2lvbiBvZiB3b3JrZ3JvdXAgc2l6ZSBpcyAxLlxuICAgICAgZm4gZ2V0R2xvYmFsSW5kZXgoKSAtPiBpMzIge1xuICAgICAgICAke1xuICAgICAgaXNGbGF0RGlzcGF0Y2gocHJvZ3JhbSkgP1xuICAgICAgICAgIGAgIHJldHVybiBpMzIoZ2xvYmFsSWQueCk7YCA6XG4gICAgICAgICAgYCAgcmV0dXJuIGkzMigod29ya2dyb3VwSWQueiAqIG51bVdvcmtncm91cHMueCAqIG51bVdvcmtncm91cHMueSArXG4gICAgICAgICAgICAgICAgd29ya2dyb3VwSWQueSAqIG51bVdvcmtncm91cHMueCArIHdvcmtncm91cElkLngpICogJHtcbiAgICAgICAgICAgICAgZmxhdFdvcmtncm91cFNpemV9dSArXG4gICAgICAgICAgICAgICAgbG9jYWxJbmRleCk7XG4gICAgICAgIGB9XG4gICAgICB9XG4gICAgYCk7XG5cbiAgaWYgKHByb2dyYW0ucGl4ZWxzT3BUeXBlICE9IG51bGwpIHtcbiAgICBjb25zdCBpbm91dFNuaXBwZXQgPSBwcm9ncmFtLnBpeGVsc09wVHlwZSA9PT0gUGl4ZWxzT3BUeXBlLkZST01fUElYRUxTID9cbiAgICAgICAgYEBncm91cCgwKSBAYmluZGluZygwKSB2YXI8c3RvcmFnZSwgcmVhZF93cml0ZT4gcmVzdWx0OiBhcnJheTwke1xuICAgICAgICAgICAgZGF0YVR5cGVUb0dQVVR5cGUob3V0cHV0RGF0YS5kdHlwZSwgcHJvZ3JhbS5vdXRwdXRDb21wb25lbnQpfT47YCA6XG4gICAgICAgIGBAZ3JvdXAoMCkgQGJpbmRpbmcoMSkgdmFyPHN0b3JhZ2UsIHJlYWQ+IGluQnVmIDogYXJyYXk8JHtcbiAgICAgICAgICAgIGRhdGFUeXBlVG9HUFVUeXBlKGlucHV0SW5mb1swXS5kdHlwZSwgcHJvZ3JhbS5vdXRwdXRDb21wb25lbnQpfT47YDtcbiAgICBjb25zdCBvdXRTaGFwZVN0cmlkZXNUeXBlID1cbiAgICAgICAgb3V0cHV0RGF0YS5zaGFwZS5sZW5ndGggPT09IDMgPyAndmVjMjxpMzI+JyA6ICdpMzInO1xuICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYFxuICAgICAgICBzdHJ1Y3QgVW5pZm9ybSB7XG4gICAgICAgICAgb3V0U2hhcGVTdHJpZGVzIDogJHtvdXRTaGFwZVN0cmlkZXNUeXBlfSxcbiAgICAgICAgICBzaXplICAgICAgICAgICAgOiBpMzIsXG4gICAgICAgICAgbnVtQ2hhbm5lbHMgICAgIDogaTMyLFxuICAgICAgICAgIGFscGhhICAgICAgICAgICA6IGYzMixcbiAgICAgICAgfTtcblxuICAgICAgICAke2lub3V0U25pcHBldH1cbiAgICAgICAgQGdyb3VwKDApIEBiaW5kaW5nKDIpIHZhcjx1bmlmb3JtPiB1bmlmb3JtczogVW5pZm9ybTtcbiAgICAgIGApO1xuICAgIGNvbnN0IHVzZUdsb2JhbEluZGV4ID0gaXNGbGF0RGlzcGF0Y2hMYXlvdXQocHJvZ3JhbSk7XG4gICAgcmV0dXJuIFtcbiAgICAgIGNvbW1vblNuaXBwZXQsXG4gICAgICBwcmVmaXhTbmlwcGV0cy5qb2luKCdcXG4nKSxcbiAgICAgIGdldENvb3Jkc0Zyb21JbmRleFNuaXBwZXQob3V0cHV0RGF0YS5zaGFwZSksXG4gICAgICBwcm9ncmFtLmdldFVzZXJDb2RlKCksXG4gICAgICBnZXRTdGFydEhlYWRlclN0cmluZyh1c2VHbG9iYWxJbmRleCwgcHJvZ3JhbSksXG4gICAgXS5qb2luKCdcXG4nKTtcbiAgfVxuXG4gIGxldCBzdHJpZGVzTGVuZ3RoOiBudW1iZXI7XG4gIGxldCBzdHJpZGVzRGF0YVR5cGU6IHN0cmluZztcbiAgbGV0IHVuaWZvcm1EZWNsYXJhdGlvbiA9ICdzdHJ1Y3QgVW5pZm9ybXMgeyBOQU4gOiBmMzIsIElORklOSVRZIDogZjMyLCAnO1xuICBwcm9ncmFtLnZhcmlhYmxlTmFtZXMuZm9yRWFjaCgoeCwgaSkgPT4ge1xuICAgIGNvbnN0IHBlckRhdGFUeXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUoaW5wdXRJbmZvW2ldLnNoYXBlLmxlbmd0aCk7XG4gICAgdW5pZm9ybURlY2xhcmF0aW9uICs9XG4gICAgICAgIGAke3guY2hhckF0KDApLnRvTG93ZXJDYXNlKCkgKyB4LnNsaWNlKDEpfVNoYXBlIDogJHtwZXJEYXRhVHlwZX0sIGA7XG4gICAgc3RyaWRlc0xlbmd0aCA9IGlucHV0SW5mb1tpXS5zaGFwZS5sZW5ndGggLSAxO1xuICAgIHN0cmlkZXNEYXRhVHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKHN0cmlkZXNMZW5ndGgpO1xuICAgIHVuaWZvcm1EZWNsYXJhdGlvbiArPVxuICAgICAgICBgJHt4LmNoYXJBdCgwKS50b0xvd2VyQ2FzZSgpICsgeC5zbGljZSgxKX1TaGFwZVN0cmlkZXM6ICR7XG4gICAgICAgICAgICBzdHJpZGVzRGF0YVR5cGV9LCBgO1xuICB9KTtcbiAgY29uc3Qgb3V0cHV0RGF0YVR5cGUgPSBnZXRDb29yZHNEYXRhVHlwZShvdXRwdXREYXRhLnNoYXBlLmxlbmd0aCk7XG4gIHVuaWZvcm1EZWNsYXJhdGlvbiArPSBgb3V0U2hhcGUgOiAke291dHB1dERhdGFUeXBlfSwgYDtcbiAgc3RyaWRlc0xlbmd0aCA9IG91dHB1dERhdGEuc2hhcGUubGVuZ3RoIC0gMTtcbiAgc3RyaWRlc0RhdGFUeXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUoc3RyaWRlc0xlbmd0aCk7XG4gIHVuaWZvcm1EZWNsYXJhdGlvbiArPSBgXG4gICAgICAgICBvdXRTaGFwZVN0cmlkZXM6ICR7c3RyaWRlc0RhdGFUeXBlfSwgYDtcblxuICBpZiAocHJvZ3JhbS5zaXplKSB7XG4gICAgdW5pZm9ybURlY2xhcmF0aW9uICs9ICdzaXplIDogaTMyLCAnO1xuICB9XG5cbiAgaWYgKHByb2dyYW0udW5pZm9ybXMpIHtcbiAgICB1bmlmb3JtRGVjbGFyYXRpb24gKz0gcHJvZ3JhbS51bmlmb3JtcztcbiAgfVxuICB1bmlmb3JtRGVjbGFyYXRpb24gKz0gJ307JztcbiAgdW5pZm9ybURlY2xhcmF0aW9uID0gaW5zZXJ0QWxpZ25tZW50KHVuaWZvcm1EZWNsYXJhdGlvbik7XG5cbiAgcHJlZml4U25pcHBldHMucHVzaCh1bmlmb3JtRGVjbGFyYXRpb24pO1xuXG4gIC8vIE91dHB1dCBidWZmZXIuXG4gIGlmIChwcm9ncmFtLmF0b21pYykge1xuICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYFxuICAgICAgQGdyb3VwKDApIEBiaW5kaW5nKDApIHZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiByZXN1bHQ6IGFycmF5PGF0b21pYzxpMzI+PjtcbiAgICBgKTtcbiAgfSBlbHNlIHtcbiAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGBcbiAgICAgIEBncm91cCgwKSBAYmluZGluZygwKSB2YXI8c3RvcmFnZSwgcmVhZF93cml0ZT4gcmVzdWx0OiBhcnJheTwke1xuICAgICAgICBkYXRhVHlwZVRvR1BVVHlwZShvdXRwdXREYXRhLmR0eXBlLCBwcm9ncmFtLm91dHB1dENvbXBvbmVudCl9PjtcbiAgICBgKTtcbiAgfVxuICBwcm9ncmFtLnZhcmlhYmxlTmFtZXMuZm9yRWFjaCgoeCwgaSkgPT4ge1xuICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYFxuICAgICAgQGdyb3VwKDApIEBiaW5kaW5nKCR7MSArIGl9KSB2YXI8c3RvcmFnZSwgcmVhZD4gJHt4fTogYXJyYXk8JHtcbiAgICAgICAgcHJvZ3JhbS52YXJpYWJsZUNvbXBvbmVudHMgP1xuICAgICAgICAgICAgZGF0YVR5cGVUb0dQVVR5cGUoXG4gICAgICAgICAgICAgICAgaW5wdXRJbmZvW2ldLmR0eXBlLCBwcm9ncmFtLnZhcmlhYmxlQ29tcG9uZW50c1tpXSkgOlxuICAgICAgICAgICAgZGF0YVR5cGVUb0dQVVR5cGUoaW5wdXRJbmZvW2ldLmR0eXBlLCBwcm9ncmFtLm91dHB1dENvbXBvbmVudCl9PjtcbiAgICAgICAgYCk7XG4gIH0pO1xuXG4gIGlmICh1bmlmb3JtRGVjbGFyYXRpb24gIT09ICcnKSB7XG4gICAgcHJlZml4U25pcHBldHMucHVzaChgXG4gICAgICBAZ3JvdXAoMCkgQGJpbmRpbmcoJHtcbiAgICAgICAgMSArIHByb2dyYW0udmFyaWFibGVOYW1lcy5sZW5ndGh9KSB2YXI8dW5pZm9ybT4gdW5pZm9ybXM6IFVuaWZvcm1zO1xuICAgICAgYCk7XG4gIH1cblxuICBjb25zdCBjb29yZHNTbmlwcGV0ID1cbiAgICAgIGdldE91dHB1dENvb3Jkc1NuaXBwZXQob3V0cHV0RGF0YS5zaGFwZSwgcHJvZ3JhbS5kaXNwYXRjaExheW91dCk7XG5cbiAgY29uc3Qgc291cmNlcyA9IFtcbiAgICBjb21tb25TbmlwcGV0LCBwcmVmaXhTbmlwcGV0cy5qb2luKCdcXG4nKSArIGlzSW5mU25pcHBldCxcbiAgICBnZXRDb29yZHNGcm9tSW5kZXhTbmlwcGV0KG91dHB1dERhdGEuc2hhcGUpLCBjb29yZHNTbmlwcGV0LFxuICAgIGdldE91dHB1dEluZGV4RnJvbUNvb3Jkc1NuaXBwZXQob3V0cHV0RGF0YS5zaGFwZS5sZW5ndGgpXG4gIF07XG4gIGlmICghcHJvZ3JhbS5hdG9taWMpIHtcbiAgICBzb3VyY2VzLnB1c2goc2V0T3V0cHV0U25pcHBldChcbiAgICAgICAgb3V0cHV0RGF0YS5zaGFwZSwgb3V0cHV0RGF0YS5kdHlwZSwgcHJvZ3JhbS5vdXRwdXRDb21wb25lbnQpKTtcbiAgfVxuXG4gIHByb2dyYW0udmFyaWFibGVOYW1lcy5mb3JFYWNoKCh4LCBpKSA9PiB7XG4gICAgc291cmNlcy5wdXNoKGAke2dldENvb3Jkc0Zyb21JbmRleFNuaXBwZXQoaW5wdXRJbmZvW2ldLnNoYXBlLCB4KX1gKTtcbiAgfSk7XG5cbiAgY29uc3QgaW5wdXRTbmlwcGV0ID1cbiAgICAgIGlucHV0SW5mb1xuICAgICAgICAgIC5tYXAoXG4gICAgICAgICAgICAgICh4LCBpKSA9PiBnZXRJbnB1dFNuaXBwZXQoXG4gICAgICAgICAgICAgICAgICB4LCBvdXRwdXREYXRhLnNoYXBlLFxuICAgICAgICAgICAgICAgICAgcHJvZ3JhbS52YXJpYWJsZUNvbXBvbmVudHMgPyBwcm9ncmFtLnZhcmlhYmxlQ29tcG9uZW50c1tpXSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHByb2dyYW0ub3V0cHV0Q29tcG9uZW50LFxuICAgICAgICAgICAgICAgICAgcHJvZ3JhbS5kaXNwYXRjaExheW91dC54Lmxlbmd0aCA9PT0gb3V0cHV0RGF0YS5zaGFwZS5sZW5ndGgpKVxuICAgICAgICAgIC5qb2luKCdcXG4nKTtcbiAgc291cmNlcy5wdXNoKGlucHV0U25pcHBldCk7XG4gIHNvdXJjZXMucHVzaChwcm9ncmFtLmdldFVzZXJDb2RlKCkpO1xuICBjb25zdCB1c2VHbG9iYWxJbmRleCA9IGlzRmxhdERpc3BhdGNoTGF5b3V0KHByb2dyYW0pO1xuICBzb3VyY2VzLnB1c2goZ2V0U3RhcnRIZWFkZXJTdHJpbmcodXNlR2xvYmFsSW5kZXgsIHByb2dyYW0pKTtcbiAgY29uc3Qgc291cmNlID0gc291cmNlcy5qb2luKCdcXG4nKTtcbiAgcmV0dXJuIHNvdXJjZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1ha2VTaGFkZXJLZXk8UiBleHRlbmRzIFJhbms+KFxuICAgIHByb2dyYW06IFdlYkdQVVByb2dyYW0sIGlucHV0c0RhdGE6IElucHV0SW5mb1tdLFxuICAgIG91dHB1dDogVGVuc29ySW5mbyk6IHN0cmluZyB7XG4gIGxldCBrZXkgPSBwcm9ncmFtLnNoYWRlcktleTtcbiAgaWYgKHByb2dyYW0ucGl4ZWxzT3BUeXBlICE9IG51bGwpIHtcbiAgICByZXR1cm4ga2V5O1xuICB9XG5cbiAgY29uc3Qgc2hhcGVzOiBudW1iZXJbXVtdID0gW107XG4gIGNvbnN0IHR5cGVzOiBBcnJheTxrZXlvZiBEYXRhVHlwZU1hcD4gPSBbXTtcbiAgaW5wdXRzRGF0YS5mb3JFYWNoKGVsZW1lbnQgPT4ge1xuICAgIHNoYXBlcy5wdXNoKGVsZW1lbnQuc2hhcGUpO1xuICAgIHR5cGVzLnB1c2goZWxlbWVudC5kdHlwZSk7XG4gIH0pO1xuICBzaGFwZXMucHVzaChvdXRwdXQuc2hhcGUpO1xuICB0eXBlcy5wdXNoKG91dHB1dC5kdHlwZSk7XG5cbiAgY29uc3QgYnJvYWRjYXN0RGltcyA9XG4gICAgICBpbnB1dHNEYXRhLm1hcChkID0+IGJhY2tlbmRfdXRpbC5nZXRCcm9hZGNhc3REaW1zKGQuc2hhcGUsIG91dHB1dC5zaGFwZSkpO1xuICBjb25zdCBpbnB1dFNoYXBlc0VxdWFsc091dFNoYXBlID1cbiAgICAgIGlucHV0c0RhdGEubWFwKGQgPT4gdXRpbC5hcnJheXNFcXVhbChkLnNoYXBlLCBvdXRwdXQuc2hhcGUpKS5qb2luKCdfJyk7XG4gIGNvbnN0IGJyb2FkY2FzdERpbXNLZXkgPSBicm9hZGNhc3REaW1zLm1hcChkID0+IGQuam9pbignXycpKS5qb2luKCc7Jyk7XG5cbiAgY29uc3QgZmxhdERpc3BhdGNoU3RyaW5nID0gaXNGbGF0RGlzcGF0Y2gocHJvZ3JhbSkgPyAnZmxhdERpc3BhdGNoJyA6ICcnO1xuXG4gIGtleSArPSAnXycgKyAocHJvZ3JhbS53b3JrZ3JvdXBTaXplID8gcHJvZ3JhbS53b3JrZ3JvdXBTaXplLmpvaW4oJywnKSA6ICcnKSArXG4gICAgICBzaGFwZXMubWFwKHNoYXBlID0+IHNoYXBlLmxlbmd0aCkuam9pbignLCcpICsgdHlwZXMuam9pbignLCcpICtcbiAgICAgIHByb2dyYW0udmFyaWFibGVOYW1lcy5qb2luKCcsJykgKyBicm9hZGNhc3REaW1zS2V5ICtcbiAgICAgIGlucHV0U2hhcGVzRXF1YWxzT3V0U2hhcGUgKyBmbGF0RGlzcGF0Y2hTdHJpbmc7XG5cbiAgcmV0dXJuIGtleTtcbn1cblxuY29uc3QgY29tbW9uU25pcHBldCA9IGBcbiAgc3RydWN0IHZlYzUge3g6IGkzMiwgeTogaTMyLCB6OiBpMzIsIHc6IGkzMiwgdTogaTMyfTtcbiAgc3RydWN0IHZlYzYge3g6IGkzMiwgeTogaTMyLCB6OiBpMzIsIHc6IGkzMiwgdTogaTMyLCB2OiBpMzJ9O1xuXG4gIC8vIENoZWNrcyB3aGV0aGVyIGNvb3JkaW5hdGVzIGxpZSB3aXRoaW4gdGhlIGJvdW5kcyBvZiB0aGUgc2hhcGUuXG4gIGZuIGNvb3Jkc0luQm91bmRzMkQoY29vcmQgOiB2ZWMyPGkzMj4sIHNoYXBlIDogdmVjMjxpMzI+KSAtPiBib29sIHtcbiAgICByZXR1cm4gYWxsKGNvb3JkID49IHZlYzI8aTMyPigwKSkgJiYgYWxsKGNvb3JkIDwgc2hhcGUpO1xuICB9XG4gIGZuIGNvb3Jkc0luQm91bmRzM0QoY29vcmQgOiB2ZWMzPGkzMj4sIHNoYXBlIDogdmVjMzxpMzI+KSAtPiBib29sIHtcbiAgICByZXR1cm4gYWxsKGNvb3JkID49IHZlYzM8aTMyPigwKSkgJiYgYWxsKGNvb3JkIDwgc2hhcGUpO1xuICB9XG4gIGZuIGNvb3Jkc0luQm91bmRzNEQoY29vcmQgOiB2ZWM0PGkzMj4sIHNoYXBlIDogdmVjNDxpMzI+KSAtPiBib29sIHtcbiAgICByZXR1cm4gYWxsKGNvb3JkID49IHZlYzQ8aTMyPigwKSkgJiYgYWxsKGNvb3JkIDwgc2hhcGUpO1xuICB9XG5cbiAgZm4gZ2V0SW5kZXhGcm9tQ29vcmRzMUQoY29vcmQgOiBpMzIsIHNoYXBlIDogaTMyKSAtPiBpMzIge1xuICAgIHJldHVybiBjb29yZDtcbiAgfVxuICBmbiBnZXRJbmRleEZyb21Db29yZHMyRChjb29yZHMgOiB2ZWMyPGkzMj4sIHNoYXBlIDogdmVjMjxpMzI+KSAtPiBpMzIge1xuICAgIHJldHVybiBkb3QoY29vcmRzLCB2ZWMyPGkzMj4oc2hhcGUueSwgMSkpO1xuICB9XG4gIGZuIGdldEluZGV4RnJvbUNvb3JkczNEKGNvb3JkcyA6IHZlYzM8aTMyPiwgc2hhcGUgOiB2ZWMzPGkzMj4pIC0+IGkzMiB7XG4gICAgcmV0dXJuIGRvdChjb29yZHMsIHZlYzM8aTMyPihzaGFwZS55ICogc2hhcGUueiwgc2hhcGUueiwgMSkpO1xuICB9XG4gIGZuIGdldEluZGV4RnJvbUNvb3JkczREKGNvb3JkcyA6IHZlYzQ8aTMyPiwgc2hhcGUgOiB2ZWM0PGkzMj4pIC0+IGkzMiB7XG4gICAgcmV0dXJuIGRvdChjb29yZHMsIHZlYzQ8aTMyPihcbiAgICAgICAgc2hhcGUueSAqIHNoYXBlLnogKiBzaGFwZS53LCBzaGFwZS56ICogc2hhcGUudywgc2hhcGUudywgMSkpO1xuICB9XG4gIGZuIGdldEluZGV4RnJvbUNvb3JkczVEKGNvb3JkcyA6IHZlYzUsIHNoYXBlIDogdmVjNSkgLT4gaTMyIHtcbiAgICBsZXQgc2hhcGVTdHJpZGVzOiB2ZWM1ID0gdmVjNShzaGFwZS55ICogc2hhcGUueiAqIHNoYXBlLncgKiBzaGFwZS51LCBzaGFwZS56ICogc2hhcGUudyAqIHNoYXBlLnUsIHNoYXBlLncgKiBzaGFwZS51LCBzaGFwZS51LCAxKTtcbiAgICByZXR1cm4gY29vcmRzLngqc2hhcGVTdHJpZGVzLnggKyBjb29yZHMueSpzaGFwZVN0cmlkZXMueSArIGNvb3Jkcy56KnNoYXBlU3RyaWRlcy56ICsgY29vcmRzLncqc2hhcGVTdHJpZGVzLncgKyBjb29yZHMudSpzaGFwZVN0cmlkZXMudTtcbiAgfVxuICBmbiBnZXRJbmRleEZyb21Db29yZHM2RChjb29yZHMgOiB2ZWM2LCBzaGFwZSA6IHZlYzYpIC0+IGkzMiB7XG4gICAgbGV0IHNoYXBlU3RyaWRlczogdmVjNiA9IHZlYzYoc2hhcGUueSAqIHNoYXBlLnogKiBzaGFwZS53ICogc2hhcGUudSAqIHNoYXBlLnYsIHNoYXBlLnogKiBzaGFwZS53ICogc2hhcGUudSAqIHNoYXBlLnYsIHNoYXBlLncgKiBzaGFwZS51ICogc2hhcGUudiwgc2hhcGUudSAqIHNoYXBlLnYsIHNoYXBlLnYsIDEpO1xuICAgIHJldHVybiBjb29yZHMueCpzaGFwZVN0cmlkZXMueCArIGNvb3Jkcy55KnNoYXBlU3RyaWRlcy55ICsgY29vcmRzLnoqc2hhcGVTdHJpZGVzLnogKyBjb29yZHMudypzaGFwZVN0cmlkZXMudyArIGNvb3Jkcy51KnNoYXBlU3RyaWRlcy51ICsgY29vcmRzLnYqc2hhcGVTdHJpZGVzLnY7XG4gIH1cblxuICAvLyBOYU4gZGVmaW5hdGlvbiBpbiBJRUVFIDc1NC0xOTg1IGlzIDpcbiAgLy8gICAtIHNpZ24gPSBlaXRoZXIgMCBvciAxLlxuICAvLyAgIC0gYmlhc2VkIGV4cG9uZW50ID0gYWxsIDEgYml0cy5cbiAgLy8gICAtIGZyYWN0aW9uID0gYW55dGhpbmcgZXhjZXB0IGFsbCAwIGJpdHMgKHNpbmNlIGFsbCAwIGJpdHMgcmVwcmVzZW50cyBpbmZpbml0eSkuXG4gIC8vIGh0dHBzOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL0lFRUVfNzU0LTE5ODUjUmVwcmVzZW50YXRpb25fb2Zfbm9uLW51bWJlcnNcbiAgZm4gaXNuYW4odmFsOiBmMzIpIC0+IGJvb2wge1xuICAgIGxldCBmbG9hdFRvVWludDogdTMyID0gYml0Y2FzdDx1MzI+KHZhbCk7XG4gICAgcmV0dXJuIChmbG9hdFRvVWludCAmIDB4N2ZmZmZmZmZ1KSA+IDB4N2Y4MDAwMDB1O1xuICB9XG4gIGZuIGlzbmFuVmVjNCh2YWwgOiB2ZWM0PGYzMj4pIC0+IHZlYzQ8Ym9vbD4ge1xuICAgIGxldCBmbG9hdFRvVWludDogdmVjNDx1MzI+ID0gYml0Y2FzdDx2ZWM0PHUzMj4+KHZhbCk7XG4gICAgcmV0dXJuIChmbG9hdFRvVWludCAmIHZlYzQ8dTMyPigweDdmZmZmZmZmdSkpID4gdmVjNDx1MzI+KDB4N2Y4MDAwMDB1KTtcbiAgfVxuYDtcblxuY29uc3QgaXNJbmZTbmlwcGV0ID0gYFxuICBmbiBpc2luZih2YWw6IGYzMikgLT4gYm9vbCB7XG4gICAgcmV0dXJuIGFicyh2YWwpID09IHVuaWZvcm1zLklORklOSVRZO1xuICB9XG5gO1xuXG50eXBlIElucHV0SW5mbyA9IHtcbiAgZHR5cGU6IERhdGFUeXBlOyBzaGFwZTogbnVtYmVyW107IG5hbWU6IHN0cmluZztcbn07XG5cbi8qKlxuICogRGVyaXZlcyBsb2dpY2FsIGNvb3JkaW5hdGVzIGZyb20gYSBmbGF0IGluZGV4LiBQZXJmb3JtcyBpbnRlZ2VyIGRpdmlzaW9uXG4gKiB3aXRoIGVhY2ggc3RyaWRlIGFuZCBkZWNyZW1lbnRzIHRoZSBpbmRleCB1bnRpbCB0aGUgaW5kZXggZXF1YWxzIHRoZSBmaW5hbFxuICogZGltZW5zaW9uIGNvb3JkaW5hdGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRDb29yZHNGcm9tSW5kZXhTbmlwcGV0KHNoYXBlOiBudW1iZXJbXSwgbmFtZSA9ICcnKTogc3RyaW5nIHtcbiAgY29uc3QgcmFuayA9IHNoYXBlLmxlbmd0aDtcbiAgY29uc3QgZnVuY05hbWUgPSBuYW1lICE9PSAnJyA/XG4gICAgICBgZ2V0JHtuYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgbmFtZS5zbGljZSgxKX1Db29yZHNGcm9tSW5kZXhgIDpcbiAgICAgICdnZXRDb29yZHNGcm9tSW5kZXgnO1xuICBjb25zdCBzdHJpZGVzTmFtZSA9IG5hbWUgIT09ICcnID9cbiAgICAgIGAke25hbWUuY2hhckF0KDApLnRvTG93ZXJDYXNlKCkgKyBuYW1lLnNsaWNlKDEpfVNoYXBlU3RyaWRlc2AgOlxuICAgICAgYG91dFNoYXBlU3RyaWRlc2A7XG5cbiAgaWYgKHJhbmsgPD0gMSkge1xuICAgIHJldHVybiBgZm4gJHtmdW5jTmFtZX0oaW5kZXggOiBpMzIpIC0+IGkzMiB7IHJldHVybiBpbmRleDsgfWA7XG4gIH1cblxuICBjb25zdCBzdHJpZGVzID0gdXRpbC5jb21wdXRlU3RyaWRlcyhzaGFwZSk7XG4gIGNvbnN0IGR0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUocmFuayk7XG5cbiAgY29uc3QgY29vcmRzOiBzdHJpbmdbXSA9IFtdO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHJhbms7IGkrKykge1xuICAgIGNvb3Jkcy5wdXNoKGBkJHtpfWApO1xuICB9XG5cbiAgaWYgKHN0cmlkZXMubGVuZ3RoID09PSAxKSB7XG4gICAgcmV0dXJuIGAgICAgZm4gJHtmdW5jTmFtZX0oaW5kZXggOiBpMzIpIC0+IHZlYzI8aTMyPiB7XG4gICAgICBsZXQgZDAgPSBpbmRleCAvIHVuaWZvcm1zLiR7XG4gICAgICAgIHN0cmlkZXNOYW1lfTsgbGV0IGQxID0gaW5kZXggLSBkMCAqIHVuaWZvcm1zLiR7c3RyaWRlc05hbWV9O1xuICAgICAgcmV0dXJuIHZlYzI8aTMyPihkMCwgZDEpO1xuICAgIH1gO1xuICB9XG4gIGxldCBzbmlwcGV0O1xuICBzbmlwcGV0ID0gJ3ZhciBpbmRleDIgPSBpbmRleDsnICtcbiAgICAgIHN0cmlkZXNcbiAgICAgICAgICAubWFwKChfLCBpKSA9PiB7XG4gICAgICAgICAgICBjb25zdCBsaW5lMSA9IGBsZXQgJHtjb29yZHNbaV19ID0gaW5kZXgyIC8gdW5pZm9ybXMuJHtcbiAgICAgICAgICAgICAgICBzdHJpZGVzTmFtZX0uJHtnZXRDb29yZHNYWVooaSl9YDtcbiAgICAgICAgICAgIGNvbnN0IGxpbmUyID0gaSA9PT0gc3RyaWRlcy5sZW5ndGggLSAxID9cbiAgICAgICAgICAgICAgICBgbGV0ICR7Y29vcmRzW2kgKyAxXX0gPSBpbmRleDIgLSAke2Nvb3Jkc1tpXX0gKiB1bmlmb3Jtcy4ke1xuICAgICAgICAgICAgICAgICAgICBzdHJpZGVzTmFtZX0uJHtnZXRDb29yZHNYWVooaSl9YCA6XG4gICAgICAgICAgICAgICAgYGluZGV4MiA9IGluZGV4MiAtICR7Y29vcmRzW2ldfSAqIHVuaWZvcm1zLiR7c3RyaWRlc05hbWV9LiR7XG4gICAgICAgICAgICAgICAgICAgIGdldENvb3Jkc1hZWihpKX1gO1xuICAgICAgICAgICAgcmV0dXJuIGAke2xpbmUxfTsgJHtsaW5lMn07YDtcbiAgICAgICAgICB9KVxuICAgICAgICAgIC5qb2luKCcnKTtcblxuICByZXR1cm4gYFxuICAgIGZuICR7ZnVuY05hbWV9KGluZGV4IDogaTMyKSAtPiAke2R0eXBlfSB7XG4gICAgICAke3NuaXBwZXR9XG4gICAgICByZXR1cm4gJHtkdHlwZX0oJHtjb29yZHMuam9pbignLCcpfSk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRJbnB1dEF0Q29vcmRzU25pcHBldChcbiAgICBpbnB1dEluZm86IElucHV0SW5mbywgY29tcG9uZW50OiBudW1iZXIpOiBzdHJpbmcge1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IHJhbmsgPSBpbnB1dEluZm8uc2hhcGUubGVuZ3RoO1xuICBjb25zdCB0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUocmFuayk7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3QgZGltcyA9IFsnZDAnLCAnZDEnLCAnZDInLCAnZDMnLCAnZDQnLCAnZDUnXS5zbGljZSgwLCByYW5rKTtcbiAgY29uc3QgaW5wdXRzID0gZGltcy5tYXAoZCA9PiBgJHtkfSA6IGkzMmApLmpvaW4oJywgJyk7XG5cbiAgaWYgKHJhbmsgPCAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZuICR7ZnVuY05hbWV9KCkgLT4gJHt0eXBlU25pcHBldChjb21wb25lbnQpfSB7XG4gICAgICAgIHJldHVybiAke3R5cGVTbmlwcGV0KGNvbXBvbmVudCl9KCR7dGV4TmFtZX1bMF0pO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCBzaGFwZVN0ciA9XG4gICAgICBgdW5pZm9ybXMuJHt0ZXhOYW1lLmNoYXJBdCgwKS50b0xvd2VyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKX1TaGFwZWA7XG4gIGxldCByYW5rU3RyID0gYCR7cmFua31EYDtcbiAgaWYgKHJhbmsgPT09IDApIHtcbiAgICByYW5rU3RyID0gJzFEJztcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgZm4gJHtmdW5jTmFtZX0oJHtpbnB1dHN9KSAtPiAke3R5cGVTbmlwcGV0KGNvbXBvbmVudCl9IHtcbiAgICAgIHJldHVybiAke3R5cGVTbmlwcGV0KGNvbXBvbmVudCl9KCR7dGV4TmFtZX1bZ2V0SW5kZXhGcm9tQ29vcmRzJHtcbiAgICAgIHJhbmtTdHJ9KCR7dHlwZX0oJHtkaW1zLmpvaW4oJywnKX0pLFxuICAgICAgICAke3NoYXBlU3RyfSkke2NvbXBvbmVudCA9PT0gMSA/ICcnIDogYCAvICR7Y29tcG9uZW50fWB9XSk7XG4gICAgfVxuICAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0SW5wdXRCeU91dHB1dFNuaXBwZXQoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm8sIG91dFNoYXBlOiBudW1iZXJbXSwgY29tcG9uZW50OiBudW1iZXIsXG4gICAgaXNGbGF0RGlzcGF0Y2hMYXlvdXQ6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IHRleEZ1bmNTbmlwcGV0ID0gdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG5cbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleEZ1bmNTbmlwcGV0ICsgJ0J5T3V0cHV0JztcblxuICBjb25zdCBpblJhbmsgPSBpbnB1dEluZm8uc2hhcGUubGVuZ3RoO1xuICBjb25zdCBvdXRSYW5rID0gb3V0U2hhcGUubGVuZ3RoO1xuICBjb25zdCB0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUob3V0UmFuayk7XG5cbiAgLy8gSWYgdGhlIGluU2hhcGUgZXF1YWxzIHRoZSBvdXRTaGFwZSBhbmQgdGhlIGRpc3BhdGNoIGxheW91dCBpcyBmbGF0LCB3ZSBjYW5cbiAgLy8gZGlyZWN0bHkgdXNlIHxnbF9HbG9iYWxJbnZvY2F0aW9uSUQueHwgYXMgdGhlIGluZGV4IGFuZCBkb24ndCBuZWVkIGNvb3Jkc1xuICAvLyBjb252ZXJzaW9uIGJldHdlZW4gdGhlc2UgdHdvIHNoYXBlcy5cbiAgaWYgKHV0aWwuYXJyYXlzRXF1YWwoaW5wdXRJbmZvLnNoYXBlLCBvdXRTaGFwZSkgJiYgaXNGbGF0RGlzcGF0Y2hMYXlvdXQpIHtcbiAgICByZXR1cm4gYFxuICAgIGZuICR7ZnVuY05hbWV9SW5kZXgoZ2xvYmFsSW5kZXggOiBpMzIpIC0+ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0ge1xuICAgICAgcmV0dXJuICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0oJHt0ZXhOYW1lfVtnbG9iYWxJbmRleF0pO1xuICAgIH1cblxuICAgIGZuICR7ZnVuY05hbWV9Q29vcmRzKGNvb3JkcyA6ICR7dHlwZX0pIC0+ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0ge1xuICAgICAgcmV0dXJuICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0oJHt0ZXhOYW1lfVske1xuICAgICAgICBvdXRSYW5rID4gMSA/ICdnZXRPdXRwdXRJbmRleEZyb21Db29yZHMoY29vcmRzKScgOlxuICAgICAgICAgICAgICAgICAgICAgICdjb29yZHMnfSR7Y29tcG9uZW50ID09PSAxID8gJycgOiBgIC8gJHtjb21wb25lbnR9YH1dKTtcbiAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGNvbnN0IGJyb2FkY2FzdERpbXMgPVxuICAgICAgYmFja2VuZF91dGlsLmdldEJyb2FkY2FzdERpbXMoaW5wdXRJbmZvLnNoYXBlLCBvdXRTaGFwZSk7XG4gIGNvbnN0IHJhbmtEaWZmID0gb3V0UmFuayAtIGluUmFuaztcblxuICBsZXQgY29vcmRzU25pcHBldCA9ICcnO1xuXG4gIGlmIChpblJhbmsgPT09IDApIHtcbiAgICByZXR1cm4gYFxuICAgIGZuICR7ZnVuY05hbWV9SW5kZXgoZ2xvYmFsSW5kZXggOiBpMzIpIC0+ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX17XG4gICAgICByZXR1cm4gZ2V0JHt0ZXhGdW5jU25pcHBldH0oKTtcbiAgICB9XG5cbiAgICBmbiAke2Z1bmNOYW1lfUNvb3Jkcyhjb29yZHMgOiAke3R5cGV9KSAtPiAke3R5cGVTbmlwcGV0KGNvbXBvbmVudCl9e1xuICAgICAgcmV0dXJuIGdldCR7dGV4RnVuY1NuaXBwZXR9KCk7XG4gICAgfVxuICBgO1xuICB9IGVsc2Uge1xuICAgIGlmIChvdXRSYW5rIDwgMiAmJiBicm9hZGNhc3REaW1zLmxlbmd0aCA+PSAxKSB7XG4gICAgICBjb29yZHNTbmlwcGV0ID0gJ2Nvb3JkcyA9IDA7JztcbiAgICB9IGVsc2Uge1xuICAgICAgY29vcmRzU25pcHBldCA9XG4gICAgICAgICAgYnJvYWRjYXN0RGltcy5tYXAoZCA9PiBgY29vcmRzLiR7Z2V0Q29vcmRzWFlaKGQgKyByYW5rRGlmZil9ID0gMDtgKVxuICAgICAgICAgICAgICAuam9pbignXFxuJyk7XG4gICAgfVxuICB9XG5cbiAgbGV0IHVucGFja2VkQ29vcmRzU25pcHBldCA9ICcnO1xuICBpZiAob3V0UmFuayA8IDIgJiYgaW5SYW5rID4gMCkge1xuICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9ICdjb29yZHMnO1xuICB9IGVsc2Uge1xuICAgIGlmIChvdXRSYW5rID4gMSkge1xuICAgICAgY29uc3QgY29vcmRzVHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKGluUmFuayk7XG4gICAgICBjb25zdCBjb29yZHNWYWx1ZXMgPVxuICAgICAgICAgIGlucHV0SW5mby5zaGFwZS5tYXAoKHMsIGkpID0+IGBjb29yZHMuJHtnZXRDb29yZHNYWVooaSArIHJhbmtEaWZmKX1gKVxuICAgICAgICAgICAgICAuam9pbignLCAnKTtcbiAgICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9IGAke2Nvb3Jkc1R5cGV9KCR7Y29vcmRzVmFsdWVzfSlgO1xuICAgIH0gZWxzZSB7XG4gICAgICB1bnBhY2tlZENvb3Jkc1NuaXBwZXQgPSAnY29vcmRzJztcbiAgICB9XG4gIH1cblxuICBjb25zdCBzaGFwZVN0ciA9XG4gICAgICBgdW5pZm9ybXMuJHt0ZXhOYW1lLmNoYXJBdCgwKS50b0xvd2VyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKX1TaGFwZWA7XG4gIGNvbnN0IHJhbmtTdHIgPSBgJHtpblJhbmt9RGA7XG5cbiAgcmV0dXJuIGBcbiAgZm4gJHtmdW5jTmFtZX1JbmRleChnbG9iYWxJbmRleCA6IGkzMikgLT4gJHt0eXBlU25pcHBldChjb21wb25lbnQpfSB7XG4gICAgdmFyIGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChnbG9iYWxJbmRleCk7XG4gICAgJHtjb29yZHNTbmlwcGV0fVxuICAgIHJldHVybiAke3R5cGVTbmlwcGV0KGNvbXBvbmVudCl9KCR7dGV4TmFtZX1bZ2V0SW5kZXhGcm9tQ29vcmRzJHtyYW5rU3RyfSgke1xuICAgICAgdW5wYWNrZWRDb29yZHNTbmlwcGV0fSwgJHtzaGFwZVN0cn0pJHtcbiAgICAgIGNvbXBvbmVudCA9PT0gMSA/ICcnIDogYCAvICR7Y29tcG9uZW50fWB9XSk7XG4gIH1cblxuICBmbiAke2Z1bmNOYW1lfUNvb3Jkcyhjb29yZHNJbiA6ICR7dHlwZX0pIC0+ICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0ge1xuICAgIHZhciBjb29yZHMgPSBjb29yZHNJbjtcbiAgICAke2Nvb3Jkc1NuaXBwZXR9XG4gICAgcmV0dXJuICR7dHlwZVNuaXBwZXQoY29tcG9uZW50KX0oJHt0ZXhOYW1lfVtnZXRJbmRleEZyb21Db29yZHMke3JhbmtTdHJ9KCR7XG4gICAgICB1bnBhY2tlZENvb3Jkc1NuaXBwZXR9LCAke3NoYXBlU3RyfSkke1xuICAgICAgY29tcG9uZW50ID09PSAxID8gJycgOiBgIC8gJHtjb21wb25lbnR9YH1dKTtcbiAgfVxuYDtcbn1cblxuZnVuY3Rpb24gZ2V0SW5wdXRTbmlwcGV0KFxuICAgIGlucHV0SW5mbzogSW5wdXRJbmZvLCBvdXRTaGFwZTogbnVtYmVyW10sIGNvbXBvbmVudDogbnVtYmVyLFxuICAgIGlzRmxhdERpc3BhdGNoTGF5b3V0OiBib29sZWFuKTogc3RyaW5nIHtcbiAgbGV0IHJlcyA9IGdldElucHV0QXRDb29yZHNTbmlwcGV0KGlucHV0SW5mbywgY29tcG9uZW50KTtcblxuICBjb25zdCBpblNoYXBlID0gaW5wdXRJbmZvLnNoYXBlO1xuICBpZiAoaW5TaGFwZS5sZW5ndGggPD0gb3V0U2hhcGUubGVuZ3RoKSB7XG4gICAgcmVzICs9IGdldElucHV0QnlPdXRwdXRTbmlwcGV0KFxuICAgICAgICBpbnB1dEluZm8sIG91dFNoYXBlLCBjb21wb25lbnQsIGlzRmxhdERpc3BhdGNoTGF5b3V0KTtcbiAgfVxuXG4gIHJldHVybiByZXM7XG59XG5cbi8qKlxuICogR2VuZXJhdGVzIGdldE91dHB1dENvb3JkcygpIGZ1bmN0aW9uIHRoYXQgY29tcHV0ZXMgb3V0cHV0IGNvb3JkaW5hdGVzXG4gKiBmcm9tIGRpc3BhdGNoIGdlb21ldHJ5IHRvIHJlZHVjZSBhcml0aG1ldGljLlxuICovXG5mdW5jdGlvbiBnZXRPdXRwdXRDb29yZHNTbmlwcGV0KFxuICAgIG91dFNoYXBlOiBudW1iZXJbXSxcbiAgICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5PzogbnVtYmVyW10sIHo/OiBudW1iZXJbXX0pOiBzdHJpbmcge1xuICBjb25zdCB7eCwgeSA9IFtdLCB6ID0gW119ID0gZGlzcGF0Y2hMYXlvdXQ7XG5cbiAgY29uc3Qgb3V0UmFuayA9IG91dFNoYXBlLmxlbmd0aDtcbiAgY29uc3QgcmFuayA9IHgubGVuZ3RoICsgeS5sZW5ndGggKyB6Lmxlbmd0aDtcbiAgLy8gZ2V0T3V0cHV0Q29vcmRzIGlzIG9ubHkgbWVhbmluZ2Z1bCB3aGVuIHRoZSBvdXRwdXQgcmFuayBpcyBzYW1lIHdpdGhcbiAgLy8gZGlzcGF0Y2ggbGF5b3V0IHJhbmsuXG4gIGlmIChyYW5rICE9PSBvdXRSYW5rKSB7XG4gICAgcmV0dXJuICcnO1xuICB9XG5cbiAgaWYgKHgubGVuZ3RoID09PSBvdXRSYW5rKSB7XG4gICAgY29uc3QgZHR5cGUgPSBnZXRDb29yZHNEYXRhVHlwZShvdXRSYW5rKTtcbiAgICBjb25zdCBzbmlwcGV0ID0gYGZuIGdldE91dHB1dENvb3JkcygpIC0+ICR7ZHR5cGV9e1xuICAgIGxldCBnbG9iYWxJbmRleCA9IGdldEdsb2JhbEluZGV4KCk7XG4gICAgcmV0dXJuIGdldENvb3Jkc0Zyb21JbmRleChnbG9iYWxJbmRleCk7XG4gIH1cbiAgYDtcbiAgICByZXR1cm4gc25pcHBldDtcbiAgfVxuXG4gIGxldCBnYXRoZXJEaW1lbnNpb25zU3RyID0gJyc7XG4gIGNvbnN0IGRpbXMgPSBbeCwgeSwgel07XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBkaW1zLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgYXJyID0gZGltc1tpXTtcblxuICAgIGlmIChhcnIubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb250aW51ZTtcbiAgICB9XG5cbiAgICBpZiAoYXJyLmxlbmd0aCA9PT0gMSkge1xuICAgICAgZ2F0aGVyRGltZW5zaW9uc1N0ciArPSBgbGV0IGQke2FyclswXX0gPSBpMzIoZ2xvYmFsSWRbJHtpfV0pO2A7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IHN0cmlkZXMgPSBzeW1ib2xpY2FsbHlDb21wdXRlU3RyaWRlcyhhcnIsICd1bmlmb3Jtcy5vdXRTaGFwZScpO1xuICAgICAgZ2F0aGVyRGltZW5zaW9uc1N0ciArPSBgdmFyIGluZGV4JHtpfSA9IGkzMihnbG9iYWxJZFske2l9XSk7YDtcbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgc3RyaWRlcy5sZW5ndGg7IGorKykge1xuICAgICAgICBnYXRoZXJEaW1lbnNpb25zU3RyICs9IGBsZXQgZCR7YXJyW2pdfSA9IGluZGV4JHtpfSAvICR7c3RyaWRlc1tqXX07YDtcblxuICAgICAgICBpZiAoaiA9PT0gc3RyaWRlcy5sZW5ndGggLSAxKSB7XG4gICAgICAgICAgZ2F0aGVyRGltZW5zaW9uc1N0ciArPSBgbGV0IGQke2FycltqICsgMV19ID0gYCArXG4gICAgICAgICAgICAgIGBpbmRleCR7aX0gLSBkJHthcnJbal19ICogJHtzdHJpZGVzW2pdfTtgO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGdhdGhlckRpbWVuc2lvbnNTdHIgKz1cbiAgICAgICAgICAgICAgYGluZGV4JHtpfSA9IGluZGV4JHtpfSAtIGQke2FycltqXX0gKiAke3N0cmlkZXNbal19O2A7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBjb25zdCBkaW1lbnNpb25zID0gW107XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmFuazsgaSsrKSB7XG4gICAgZGltZW5zaW9ucy5wdXNoKGBkJHtpfWApO1xuICB9XG5cbiAgY29uc3QgZHR5cGUgPSBnZXRDb29yZHNEYXRhVHlwZShyYW5rKTtcbiAgbGV0IHNuaXBwZXQgPSBgZm4gZ2V0T3V0cHV0Q29vcmRzKCkgLT4gJHtkdHlwZX0ge1xuICAke2dhdGhlckRpbWVuc2lvbnNTdHJ9XG5gO1xuICBpZiAoZGltZW5zaW9ucy5sZW5ndGggPT09IDApIHtcbiAgICBzbmlwcGV0ICs9IGByZXR1cm4gJHtkdHlwZX0oMCk7IH1gO1xuICB9IGVsc2Uge1xuICAgIHNuaXBwZXQgKz0gYHJldHVybiAke2R0eXBlfSgke2RpbWVuc2lvbnMuam9pbignLCcpfSk7IH1gO1xuICB9XG5cbiAgcmV0dXJuIHNuaXBwZXQ7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dEluZGV4RnJvbUNvb3Jkc1NuaXBwZXQob3V0UmFuazogbnVtYmVyKSB7XG4gIGxldCBzbmlwcGV0ID0gJyc7XG4gIHN3aXRjaCAob3V0UmFuaykge1xuICAgIGNhc2UgMDpcbiAgICBjYXNlIDE6XG4gICAgICBzbmlwcGV0ICs9IGBcbiAgICAgICAgZm4gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKGNvb3JkcyA6IGkzMikgLT4gaTMyIHtcbiAgICAgICAgICByZXR1cm4gY29vcmRzO1xuICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgICBicmVhaztcbiAgICBjYXNlIDI6XG4gICAgICBzbmlwcGV0ICs9IGBcbiAgICAgICAgZm4gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKGNvb3JkcyA6IHZlYzI8aTMyPikgLT4gaTMyIHtcbiAgICAgICAgICByZXR1cm4gZG90KGNvb3JkcywgdmVjMjxpMzI+KHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcywgMSkpO1xuICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgICBicmVhaztcbiAgICBjYXNlIDM6XG4gICAgICBzbmlwcGV0ICs9IGBcbiAgICAgICAgZm4gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKGNvb3JkcyA6IHZlYzM8aTMyPikgLT4gaTMyIHtcbiAgICAgICAgICByZXR1cm4gZG90KGNvb3JkcywgdmVjMzxpMzI+KHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcy54LCB1bmlmb3Jtcy5vdXRTaGFwZVN0cmlkZXMueSwgMSkpO1xuICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgICBicmVhaztcbiAgICBjYXNlIDQ6XG4gICAgICBzbmlwcGV0ICs9IGBcbiAgICAgICAgZm4gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKGNvb3JkcyA6IHZlYzQ8aTMyPikgLT4gaTMyIHtcbiAgICAgICAgICByZXR1cm4gZG90KGNvb3JkcywgdmVjNDxpMzI+KFxuICAgICAgICAgICAgdW5pZm9ybXMub3V0U2hhcGVTdHJpZGVzLngsIHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcy55LCB1bmlmb3Jtcy5vdXRTaGFwZVN0cmlkZXMueiwgMSkpO1xuICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgICBicmVhaztcbiAgICBjYXNlIDU6XG4gICAgICBzbmlwcGV0ICs9IGBcbiAgICAgICAgZm4gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKGNvb3JkcyA6IHZlYzUpIC0+IGkzMiB7XG4gICAgICAgICAgcmV0dXJuIGNvb3Jkcy54ICogdW5pZm9ybXMub3V0U2hhcGVTdHJpZGVzLnggK1xuICAgICAgICAgICAgICBjb29yZHMueSAqIHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcy55ICtcbiAgICAgICAgICAgICAgY29vcmRzLnogKiB1bmlmb3Jtcy5vdXRTaGFwZVN0cmlkZXMueiArXG4gICAgICAgICAgICAgIGNvb3Jkcy53ICogdW5pZm9ybXMub3V0U2hhcGVTdHJpZGVzLncgK1xuICAgICAgICAgICAgICBjb29yZHMudTtcbiAgICAgICAgfVxuICAgICAgICBgO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSA2OlxuICAgICAgc25pcHBldCArPSBgXG4gICAgICAgIGZuIGdldE91dHB1dEluZGV4RnJvbUNvb3Jkcyhjb29yZHMgOiB2ZWM2KSAtPiBpMzIge1xuICAgICAgICAgIHJldHVybiBjb29yZHMueCAqIHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcy54ICtcbiAgICAgICAgICAgICAgY29vcmRzLnkgKiB1bmlmb3Jtcy5vdXRTaGFwZVN0cmlkZXMueSArXG4gICAgICAgICAgICAgIGNvb3Jkcy56ICogdW5pZm9ybXMub3V0U2hhcGVTdHJpZGVzLnogK1xuICAgICAgICAgICAgICBjb29yZHMudyAqIHVuaWZvcm1zLm91dFNoYXBlU3RyaWRlcy53ICtcbiAgICAgICAgICAgICAgY29vcmRzLnUgKiB1bmlmb3Jtcy5vdXRTaGFwZVN0cmlkZXMudSArXG4gICAgICAgICAgICAgIGNvb3Jkcy52O1xuICAgICAgICB9XG4gICAgICAgIGA7XG4gICAgICBicmVhaztcbiAgICBkZWZhdWx0OlxuICAgICAgdXRpbC5hc3NlcnQoZmFsc2UsICgpID0+IGBVbnN1cHBvcnRlZCAke291dFJhbmt9RCBzaGFwZWApO1xuICAgICAgYnJlYWs7XG4gIH1cbiAgcmV0dXJuIHNuaXBwZXQ7XG59XG5cbmZ1bmN0aW9uIGlzRmxhdERpc3BhdGNoKHByb2dyYW06IFdlYkdQVVByb2dyYW0pOiBib29sZWFuIHtcbiAgcmV0dXJuIHByb2dyYW0uZGlzcGF0Y2hbMV0gPT09IDEgJiYgcHJvZ3JhbS5kaXNwYXRjaFsyXSA9PT0gMTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRhdGFUeXBlVG9HUFVUeXBlKHR5cGU6IERhdGFUeXBlLCBjb21wb25lbnQgPSAxKSB7XG4gIGlmICh0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICByZXR1cm4gdHlwZVNuaXBwZXQoY29tcG9uZW50LCAnZjMyJyk7XG4gIH0gZWxzZSBpZiAodHlwZSA9PT0gJ2ludDMyJyB8fCB0eXBlID09PSAnYm9vbCcpIHtcbiAgICByZXR1cm4gdHlwZVNuaXBwZXQoY29tcG9uZW50LCAnaTMyJyk7XG4gIH1cbiAgdGhyb3cgbmV3IEVycm9yKGB0eXBlICR7dHlwZX0gaXMgbm90IHN1cHBvcnRlZC5gKTtcbn1cblxuZnVuY3Rpb24gc2V0T3V0cHV0U25pcHBldChcbiAgICBvdXRTaGFwZTogbnVtYmVyW10sIG91dEJ1ZmZlclR5cGU6IERhdGFUeXBlLCBjb21wb25lbnQ6IG51bWJlcik6IHN0cmluZyB7XG4gIGNvbnN0IG91dFJhbmsgPSBvdXRTaGFwZS5sZW5ndGg7XG4gIGNvbnN0IGdwdVR5cGUgPSBkYXRhVHlwZVRvR1BVVHlwZShvdXRCdWZmZXJUeXBlLCBjb21wb25lbnQpO1xuICBsZXQgc25pcHBldCA9XG4gICAgICBgZm4gc2V0T3V0cHV0QXRJbmRleChmbGF0SW5kZXggOiBpMzIsIHZhbHVlIDogJHt0eXBlU25pcHBldChjb21wb25lbnQpfSkge1xuICAgICAgcmVzdWx0W2ZsYXRJbmRleF0gPSAke2dwdVR5cGV9KHZhbHVlKTtcbiAgICB9XG5cbiAgICBmbiBzZXRPdXRwdXRBdEluZGV4STMyKGZsYXRJbmRleCA6IGkzMiwgdmFsdWUgOiAke1xuICAgICAgICAgIHR5cGVTbmlwcGV0KGNvbXBvbmVudCwgJ2kzMicpfSkge1xuICAgICAgcmVzdWx0W2ZsYXRJbmRleF0gPSAke2dwdVR5cGV9KHZhbHVlKTtcbiAgICB9XG4gICAgYDtcbiAgaWYgKG91dFJhbmsgPj0gMikge1xuICAgIGNvbnN0IGRpbXMgPSBbJ2QwJywgJ2QxJywgJ2QyJywgJ2QzJywgJ2Q0JywgJ2Q1J10uc2xpY2UoMCwgb3V0UmFuayk7XG4gICAgY29uc3QgdHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKG91dFJhbmspO1xuXG4gICAgc25pcHBldCArPSBgXG4gICAgICBmbiBzZXRPdXRwdXRBdENvb3Jkcygke2RpbXMubWFwKGQgPT4gYCR7ZH0gOiBpMzJgKS5qb2luKCcsICcpfSwgdmFsdWUgOiAke1xuICAgICAgICB0eXBlU25pcHBldChjb21wb25lbnQpfSkge1xuICAgICAgICBsZXQgZmxhdEluZGV4ID0gZ2V0T3V0cHV0SW5kZXhGcm9tQ29vcmRzKCR7dHlwZX0oJHtkaW1zLmpvaW4oJywgJyl9KSk7XG4gICAgICAgIHNldE91dHB1dEF0SW5kZXgoZmxhdEluZGV4JHtcbiAgICAgICAgY29tcG9uZW50ID09PSAxID8gJycgOiBgIC8gJHtjb21wb25lbnR9YH0sIHZhbHVlKTtcbiAgICAgIH1cbiAgICAgIGZuIHNldE91dHB1dEF0Q29vcmRzSTMyKCR7XG4gICAgICAgIGRpbXMubWFwKGQgPT4gYCR7ZH0gOiBpMzJgKS5qb2luKCcsICcpfSwgdmFsdWUgOiAke1xuICAgICAgICB0eXBlU25pcHBldChjb21wb25lbnQsICdpMzInKX0pIHtcbiAgICAgICAgbGV0IGZsYXRJbmRleCA9IGdldE91dHB1dEluZGV4RnJvbUNvb3Jkcygke3R5cGV9KCR7ZGltcy5qb2luKCcsICcpfSkpO1xuICAgICAgICBzZXRPdXRwdXRBdEluZGV4STMyKGZsYXRJbmRleCR7XG4gICAgICAgIGNvbXBvbmVudCA9PT0gMSA/ICcnIDogYCAvICR7Y29tcG9uZW50fWB9LCB2YWx1ZSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIHJldHVybiBzbmlwcGV0O1xufVxuXG5mdW5jdGlvbiBpbnNlcnRBbGlnbm1lbnQodW5pZm9ybVNoYWRlcjogc3RyaW5nKSB7XG4gIC8vIGluc2VydCBhbGlnbm1lbnQgd2hlbiBjdXJyZW50IHBhdHRlcm4gaXMgdmVjNSBvciB2ZWM2XG4gIGNvbnN0IGN1ckluc2VydFJlID0gLyhcXHcrKVxccyo6XFxzKnZlYyg1fDYpL2c7XG4gIHVuaWZvcm1TaGFkZXIgPSB1bmlmb3JtU2hhZGVyLnJlcGxhY2UoY3VySW5zZXJ0UmUsIChtYXRjaCkgPT4ge1xuICAgIHJldHVybiAnQGFsaWduKDE2KSAnICsgbWF0Y2g7XG4gIH0pO1xuXG4gIC8vIGluc2VydCBhbGlnbm1lbnQgd2hlbiBwcmV2aW91cyBwYXR0ZXJuIGlzIHZlYzUgb3IgdmVjNlxuICBjb25zdCBwcmVJbnNlcnRSZSA9IC92ZWMoNXw2KVxccyosXFxzKihcXHcrKS9nO1xuICB1bmlmb3JtU2hhZGVyID0gdW5pZm9ybVNoYWRlci5yZXBsYWNlKHByZUluc2VydFJlLCAoXywgcDEsIHAyKSA9PiB7XG4gICAgcmV0dXJuIGB2ZWMke3AxfSwgQGFsaWduKDE2KSAke3AyfWA7XG4gIH0pO1xuICByZXR1cm4gdW5pZm9ybVNoYWRlcjtcbn1cbmZ1bmN0aW9uIGlzRmxhdERpc3BhdGNoTGF5b3V0KHByb2dyYW06IFdlYkdQVVByb2dyYW0pOiBib29sZWFuIHtcbiAgaWYgKHByb2dyYW0uZGlzcGF0Y2hMYXlvdXQuaGFzT3duUHJvcGVydHkoJ3knKSAmJlxuICAgICAgcHJvZ3JhbS5kaXNwYXRjaExheW91dC55Lmxlbmd0aCAhPT0gMCkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuICBpZiAocHJvZ3JhbS5kaXNwYXRjaExheW91dC5oYXNPd25Qcm9wZXJ0eSgneicpICYmXG4gICAgICBwcm9ncmFtLmRpc3BhdGNoTGF5b3V0LnoubGVuZ3RoICE9PSAwKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiB0cnVlO1xufVxuIl19