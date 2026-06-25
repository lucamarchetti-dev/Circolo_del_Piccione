/**
 * @license
 * Copyright 2024 Google LLC. All Rights Reserved.
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
import * as tf from '@tensorflow/tfjs-core';
import { env, backend_util, util, KernelBackend, DataStorage, engine, buffer, registerBackend, Fill, Reshape, broadcast_util, _FusedMatMul, Identity, Complex, upcastType, tidy, reshape as reshape$1, broadcastTo, TensorBuffer, slice_util, Abs, Acos, Acosh, Add, AddN, Transpose, sumOutType, All, Any, ArgMax, ArgMin, Asin, Asinh, Atan, Atan2, Atanh, Max, Mean, AvgPool, AvgPool3D, AvgPool3DGrad, AvgPoolGrad, BatchMatMul, Slice, BatchToSpaceND, Bincount, BroadcastArgs, NotEqual, Real, Cast, Ceil, ClipByValue, ComplexAbs, Imag, Concat, Conv2D, Conv2DBackpropFilter, Conv2DBackpropInput, Conv3D, Conv3DBackpropFilterV2, Conv3DBackpropInputV2, Cos, Cosh, CropAndResize, Cumprod, Cumsum, DenseBincount, DepthToSpace, DepthwiseConv2dNative, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropInput, Diag, Dilation2D, Dilation2DBackpropFilter, Dilation2DBackpropInput, Draw, Multiply, Sum, Einsum, Elu, EluGrad, Equal, Erf, Exp, ExpandDims, Expm1, FFT, FlipLeftRight, Floor, FloorDiv, FromPixels, FusedBatchNorm, FusedConv2D, FusedDepthwiseConv2D, GatherNd, GatherV2, Greater, GreaterEqual, IFFT, IsFinite, IsInf, IsNan, LeakyRelu, Less, LessEqual, LinSpace, Log, Log1p, LogicalAnd, LogicalNot, LogicalOr, LRN, LRNGrad, Maximum, MaxPool, MaxPool3D, MaxPool3DGrad, MaxPoolGrad, MaxPoolWithArgmax, Min, Minimum, MirrorPad, Mod, Softmax, Multinomial, Neg, NonMaxSuppressionV3, kernel_impls, NonMaxSuppressionV5, OneHot, ZerosLike, OnesLike, Pack, PadV2, Pow, Prelu, Prod, Range, RealDiv, Reciprocal, Relu, Relu6, ResizeBilinear, ResizeBilinearGrad, ResizeNearestNeighbor, ResizeNearestNeighborGrad, Reverse, RotateWithOffset, Round, Rsqrt, ScatterNd, SearchSorted, Select, Selu, Sigmoid, Sign, Sin, Sinh, Softplus, SpaceToBatchND, SparseSegmentMean, SparseSegmentSum, Tile, SparseToDense, SplitV, Sqrt, Square, SquaredDifference, Step, StridedSlice, StringNGrams, Sub, Tan, Tanh, TensorScatterUpdate, TopK, Transform, Unpack, UnsortedSegmentSum, registerKernel } from '@tensorflow/tfjs-core';

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
const ENV = env();
/** The batched dispatching calls size in the device queue. */
ENV.registerFlag('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', () => 15);
/**
 * Whether we forward execution to the CPU backend if tensors are small and
 * reside on the CPU.
 */
ENV.registerFlag('WEBGPU_CPU_FORWARD', () => true);
/**
 * This flag is used to test different types of matmul programs.
 *
 * See MatMulProgramType in webgpu_util.ts for a list of available values.
 */
ENV.registerFlag('WEBGPU_MATMUL_PROGRAM_TYPE', () => -1);
/**
 * Whether to use conv2dTranspose_naive which directly implement the
 * conv2dTranspose logic rather than using a matmul to simulate.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE', () => true);
/**
 * Whether we use low power GPU. Otherwise, a high performance GPU will be
 * requested.
 */
ENV.registerFlag('WEBGPU_USE_LOW_POWER_GPU', () => false);
/**
 * Threshold for input tensor size that determines whether WebGPU backend will
 * delegate computation to CPU.
 *
 * Default value is 1000.
 */
ENV.registerFlag('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD', () => 1000);
/**
 * Whether to use a dummy canvas to make profiling tools like PIX work with
 * TFJS webgpu backend.
 */
ENV.registerFlag('WEBGPU_USE_PROFILE_TOOL', () => false);
/**
 * Whether to use import API.
 */
ENV.registerFlag('WEBGPU_IMPORT_EXTERNAL_TEXTURE', () => true);
/**
 * Whether to use conv2dNaive for debugging.
 */
ENV.registerFlag('WEBGPU_USE_NAIVE_CONV2D_DEBUG', () => false);
/**
 * Threshold to increase dispatched workgroups for matmul. If too few workgroups
 * are dispatched, it means the hardware may be in low occupancy.
 * -1 means it's not set by the user. A default strategy will be applied.
 */
ENV.registerFlag('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL', () => -1);
/**
 * Whether we will run im2col as a separate shader for convolution.
 */
ENV.registerFlag('WEBGPU_CONV_SEPARATE_IM2COL_SHADER', () => false);
/**
 * A string used to match shader key. If any matches, print the related shader.
 * Seperated by comma. 'all' to print all. 'binary' to print binary(add, mul,
 * etc.). 'unary,conv2d' to print both unary and conv2d.
 */
ENV.registerFlag('WEBGPU_PRINT_SHADER', () => '');
/** Experimental flag, whether enter compile only phase. */
ENV.registerFlag('WEBGPU_ENGINE_COMPILE_ONLY', () => false);

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
class AdapterInfo {
    constructor(adapterInfo) {
        if (adapterInfo) {
            this.vendor = adapterInfo.vendor;
            this.architecture = adapterInfo.architecture;
            this.intelGPUGeneration = this.getIntelGPUGeneration();
        }
    }
    getIntelGPUGeneration() {
        if (this.isIntel()) {
            if (this.architecture.startsWith('gen')) {
                return Number(this.architecture.match(/\d+/));
            }
            else if (this.architecture.startsWith('xe')) {
                return 12;
            }
        }
        return 0;
    }
    isIntel() {
        return this.vendor === 'intel';
    }
}

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
class BufferManager {
    constructor(device) {
        this.device = device;
        this.numUsedBuffers = 0;
        this.numFreeBuffers = 0;
        this.freeBuffers = new Map();
        this.usedBuffers = new Map();
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
    acquireBuffer(size, usage, mappedAtCreation = false, reuse = true) {
        let buffer;
        const key = getBufferKey(size, usage);
        if (reuse) {
            if (!this.freeBuffers.has(key)) {
                this.freeBuffers.set(key, []);
            }
            if (this.freeBuffers.get(key).length > 0) {
                buffer = this.freeBuffers.get(key).pop();
                this.numFreeBuffers--;
            }
            else {
                buffer = this.device.createBuffer({ size, usage, mappedAtCreation });
                this.numBytesAllocated += size;
            }
        }
        else {
            buffer = this.device.createBuffer({ size, usage, mappedAtCreation });
            this.numBytesAllocated += size;
        }
        if (!this.usedBuffers.has(key)) {
            this.usedBuffers.set(key, []);
        }
        this.usedBuffers.get(key).push(buffer);
        this.numUsedBuffers++;
        this.numBytesUsed += size;
        return buffer;
    }
    releaseBuffer(buffer, reuse = true) {
        if (this.freeBuffers.size === 0) {
            return;
        }
        const size = buffer.size;
        const usage = buffer.usage;
        const key = getBufferKey(size, usage);
        const bufferArray = this.usedBuffers.get(key);
        const index = bufferArray.indexOf(buffer);
        if (index < 0) {
            throw new Error('Cannot find the buffer in buffer manager');
        }
        bufferArray[index] = bufferArray[bufferArray.length - 1];
        bufferArray.pop();
        this.numUsedBuffers--;
        this.numBytesUsed -= size;
        if (reuse) {
            this.freeBuffers.get(key).push(buffer);
            this.numFreeBuffers++;
        }
        else {
            buffer.destroy();
            this.numBytesAllocated -= size;
        }
    }
    getNumUsedBuffers() {
        return this.numUsedBuffers;
    }
    getNumFreeBuffers() {
        return this.numFreeBuffers;
    }
    dispose() {
        this.freeBuffers.forEach((buffers, key) => {
            buffers.forEach(buffer => {
                buffer.destroy();
            });
        });
        this.usedBuffers.forEach((buffers, key) => {
            buffers.forEach(buffer => {
                buffer.destroy();
            });
        });
        this.freeBuffers = new Map();
        this.usedBuffers = new Map();
        this.numUsedBuffers = 0;
        this.numFreeBuffers = 0;
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
}
function getBufferKey(size, usage) {
    return `${size}_${usage}`;
}

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
class TextureManager {
    constructor(device) {
        this.device = device;
        this.numUsedTextures = 0;
        this.numFreeTextures = 0;
        this.freeTextures = new Map();
        this.usedTextures = new Map();
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
    acquireTexture(width, height, format, usage) {
        const bytesPerElement = getBytesPerElement(format);
        const byteSize = width * height * bytesPerElement;
        const key = getTextureKey(width, height, format, usage);
        if (!this.freeTextures.has(key)) {
            this.freeTextures.set(key, []);
        }
        if (!this.usedTextures.has(key)) {
            this.usedTextures.set(key, []);
        }
        this.numBytesUsed += byteSize;
        this.numUsedTextures++;
        if (this.freeTextures.get(key).length > 0) {
            this.numFreeTextures--;
            const newTexture = this.freeTextures.get(key).shift();
            this.usedTextures.get(key).push(newTexture);
            return newTexture;
        }
        this.numBytesAllocated += byteSize;
        const newTexture = this.device.createTexture({
            size: [width, height],
            format,
            usage,
        });
        this.usedTextures.get(key).push(newTexture);
        return newTexture;
    }
    releaseTexture(texture) {
        if (this.freeTextures.size === 0) {
            return;
        }
        const width = texture.width;
        const height = texture.height;
        const format = texture.format;
        const usage = texture.usage;
        const key = getTextureKey(width, height, format, usage);
        if (!this.freeTextures.has(key)) {
            this.freeTextures.set(key, []);
        }
        this.freeTextures.get(key).push(texture);
        this.numFreeTextures++;
        this.numUsedTextures--;
        const textureList = this.usedTextures.get(key);
        const textureIndex = textureList.indexOf(texture);
        if (textureIndex < 0) {
            throw new Error('Cannot release a texture that was never provided by this ' +
                'texture manager');
        }
        textureList.splice(textureIndex, 1);
        const bytesPerElement = getBytesPerElement(format);
        const byteSize = width * height * bytesPerElement;
        this.numBytesUsed -= byteSize;
    }
    getNumUsedTextures() {
        return this.numUsedTextures;
    }
    getNumFreeTextures() {
        return this.numFreeTextures;
    }
    dispose() {
        this.freeTextures.forEach((textures, key) => {
            textures.forEach(texture => {
                texture.destroy();
            });
        });
        this.usedTextures.forEach((textures, key) => {
            textures.forEach(texture => {
                texture.destroy();
            });
        });
        this.freeTextures = new Map();
        this.usedTextures = new Map();
        this.numUsedTextures = 0;
        this.numFreeTextures = 0;
        this.numBytesUsed = 0;
        this.numBytesAllocated = 0;
    }
}
function getTextureKey(width, height, format, usage) {
    return `${width}_${height}_${format}_${usage}`;
}
function getBytesPerElement(format) {
    if (format === 'rgba8unorm') {
        return 16;
    }
    else {
        throw new Error(`${format} is not supported!`);
    }
}

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
// Generates WGSL that computes strides.
function symbolicallyComputeStrides(indicesArr, variableName) {
    if (Math.max(...indicesArr) > 5) {
        throw new Error('Cannot symbolically compute strides for rank > 6 tensor.');
    }
    const numCoords = indicesArr.length;
    const indicesStr = 'xyzwuv';
    const shape = indicesArr.map(d => `${variableName}.${indicesStr[d]}`);
    const strides = new Array(numCoords - 1);
    strides[numCoords - 2] = shape[numCoords - 1];
    for (let i = numCoords - 3; i >= 0; --i) {
        strides[i] = `(${strides[i + 1]} * ${shape[i + 1]})`;
    }
    return strides;
}
const atomicAddSnippet = (ptr, v, type) => {
    if (type === 'int32') {
        return `atomicAdd(${ptr}, bitcast<i32>(${v}));`;
    }
    else {
        // atomicAdd only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `
          {
            var oldValue = 0;
            loop {
              let newValueF32 = bitcast<f32>(oldValue) + (${v});
              let newValue = bitcast<i32>(newValueF32);
              let res = atomicCompareExchangeWeak(${ptr}, oldValue, newValue);
              if res.exchanged {
                break;
              }
              oldValue = res.old_value;
            }
          }`;
    }
};

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
var PixelsOpType;
(function (PixelsOpType) {
    PixelsOpType[PixelsOpType["FROM_PIXELS"] = 0] = "FROM_PIXELS";
    PixelsOpType[PixelsOpType["DRAW"] = 1] = "DRAW";
})(PixelsOpType || (PixelsOpType = {}));
const compileProgram = (device, program, inputsData, output, parallelCompilation) => {
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
const typeSnippet = (component, type = 'f32') => {
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
function getCoordsDataType(rank) {
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
function getCoordsXYZ(index) {
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
function getMainHeaderString(...params) {
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
function getStartHeaderString(useGlobalIndex, program) {
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
function getWorkgroupSizeString(program) {
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
function makeShaderKey(program, inputsData, output) {
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
function getCoordsFromIndexSnippet(shape, name = '') {
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
function dataTypeToGPUType(type, component = 1) {
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
const arrayProduct = (arr) => {
    let product = 1;
    for (let i = 0; i < arr.length; i++) {
        product *= arr[i];
    }
    return product;
};
function tilesFitEvenlyIntoShape(tileSize, shape) {
    if (tileSize.length !== shape.length) {
        throw new Error(`Cannot compute whether rank ${tileSize.length}` +
            ` tiles fit evenly into rank ${shape.length} shape` +
            ` - ranks must match.`);
    }
    return shape.every((dim, dimIdx) => dim % tileSize[dimIdx] === 0);
}
// Computes dispatch geometry based on layout of output dimensions and
// workgroupSize.
function computeDispatch(layout, outputShape, workgroupSize = [1, 1, 1], elementsPerThread = [1, 1, 1]) {
    const [dispatchX, dispatchY, dispatchZ] = [
        Math.ceil(arrayProduct(layout.x.map(d => outputShape[d])) /
            (workgroupSize[0] * elementsPerThread[0])),
        layout.y ? Math.ceil(arrayProduct(layout.y.map(d => outputShape[d])) /
            (workgroupSize[1] * elementsPerThread[1])) :
            1,
        layout.z ? Math.ceil(arrayProduct(layout.z.map(d => outputShape[d])) /
            (workgroupSize[2] * elementsPerThread[2])) :
            1
    ];
    return [dispatchX, dispatchY, dispatchZ];
}
function computeWorkgroupInfoForMatMul(dimAOuter, dimInner, dimBOuter, transposeA = false) {
    // These are experimental values. Usually, we need to adjust the work group
    // size based on the input shapes to improve the EU occupancy.
    // TODO: WebGPU limits the maximum allowed shared memory size as 16K. To make
    // sure it doesn't exceed this limitations. Temporarily reduce the work group
    // size to [8, 8, 1] and the work per thread size is [4, 4, 1]. But we should
    // revisit it and find the balance between work group size and work per thread
    // size.
    const workgroupSize = [8, 8, 1];
    const elementsPerThread = [4, 4, 1];
    if (!transposeA) {
        if (dimAOuter <= 8) {
            elementsPerThread[1] = 1;
        }
        if (dimInner <= 16 && dimBOuter <= 16) {
            workgroupSize[0] = 4;
        }
    }
    return { workgroupSize, elementsPerThread };
}
function computeWorkgroupSizeForConv2d(layout, outputShape, isVec4 = false) {
    if (isVec4) {
        return [8, 8, 1];
    }
    const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
    const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
    // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
    // These are experimental values. Usually, we need to adjust the work group
    // size based on the output shape. For example, when one dimension is smaller
    // than 4, it will be wasteful if we assign a larger size for this dimension,
    // which results lots of threads doing useless work and reduces parallelism
    // of hardware threads. But it is always a balance between work group size
    // and shared memory. If one dimension is too small, such as 1, shared memory
    // will won't be fully utilized.
    if (dim0 <= 4) {
        return [4, 16, 1];
    }
    if (dim1 <= 4) {
        return [16, 4, 1];
    }
    return [16, 16, 1];
}
function computeWorkPerThreadForConv2d(layout, outputShape, isVec4 = false) {
    if (isVec4) {
        return [4, 4, 1];
    }
    const dim0 = arrayProduct(layout.x.map(d => outputShape[d]));
    const dim1 = arrayProduct(layout.y.map(d => outputShape[d]));
    // TODO(jiajia.qin@intel.com): More fine tune based on outputShape.
    // The following conditions correspond to the values set in
    // computeWorkgroupSizeForConv2d.
    if (dim0 <= 4) {
        return [1, 2, 1];
    }
    if (dim1 <= 4) {
        return [2, 1, 1];
    }
    return [2, 2, 1];
}
function flatDispatchLayout(shape) {
    return { x: shape.map((d, i) => i) };
}
function GPUBytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32' || dtype === 'bool' ||
        dtype === 'string') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
function isWebGPUSupported() {
    return !!(typeof globalThis !== 'undefined' && (globalThis.navigator)
        && (globalThis.navigator.gpu));
}
function assertNotComplex(tensor, opName) {
    if (!Array.isArray(tensor)) {
        tensor = [tensor];
    }
    tensor.forEach(t => {
        if (t != null) {
            util.assert(t.dtype !== 'complex64', () => `${opName} does not support complex64 tensors ` +
                'in the WebGPU backend.');
        }
    });
}
var MatMulProgramType;
(function (MatMulProgramType) {
    MatMulProgramType[MatMulProgramType["MatMulReduceProgram"] = 0] = "MatMulReduceProgram";
    MatMulProgramType[MatMulProgramType["MatMulSplitKProgram"] = 1] = "MatMulSplitKProgram";
    MatMulProgramType[MatMulProgramType["MatMulSmallOutputSizeProgram"] = 2] = "MatMulSmallOutputSizeProgram";
    MatMulProgramType[MatMulProgramType["MatMulPackedProgram"] = 3] = "MatMulPackedProgram";
    MatMulProgramType[MatMulProgramType["MatMulMax"] = 4] = "MatMulMax";
})(MatMulProgramType || (MatMulProgramType = {}));

var webgpu_util = {
    __proto__: null,
    GPUBytesPerElement: GPUBytesPerElement,
    get MatMulProgramType () { return MatMulProgramType; },
    assertNotComplex: assertNotComplex,
    computeDispatch: computeDispatch,
    computeWorkPerThreadForConv2d: computeWorkPerThreadForConv2d,
    computeWorkgroupInfoForMatMul: computeWorkgroupInfoForMatMul,
    computeWorkgroupSizeForConv2d: computeWorkgroupSizeForConv2d,
    flatDispatchLayout: flatDispatchLayout,
    isWebGPUSupported: isWebGPUSupported,
    tilesFitEvenlyIntoShape: tilesFitEvenlyIntoShape
};

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
// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = env().getNumber('WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD');
// Reshape dispatch, not to exceed device limits.
const reshapeDispatch = (device, program) => {
    const MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE = device.limits.maxComputeWorkgroupsPerDimension;
    const layout = program['dispatchLayout'];
    const dispatch = program['dispatch'];
    if (dispatch.every((d) => d <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE)) {
        return dispatch;
    }
    util.assert(dispatch[0] > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE &&
        layout.y === undefined && layout.z === undefined, () => 'Dispatch size exceeds WebGPU limits in Y or Z dimension.');
    let dispatchAverage = Math.ceil(Math.sqrt(dispatch[0]));
    if (dispatchAverage > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE) {
        dispatchAverage = Math.ceil(Math.cbrt(dispatch[0]));
        util.assert(dispatchAverage <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE, () => 'Total dispatch size exceeds WebGPU maximum.');
        return [dispatchAverage, dispatchAverage, dispatchAverage];
    }
    else {
        return [dispatchAverage, dispatchAverage, 1];
    }
};
class WebGPUBackend extends KernelBackend {
    nextDataId() {
        return WebGPUBackend.nextDataId++;
    }
    constructor(device, adapterInfo) {
        super();
        this.commandQueueOwnedIds = new WeakSet();
        this.dispatchCountInPass = 0;
        this.disposed = false;
        this.downloadWaitMs = 0;
        this.tensorDataPendingDisposal = [];
        this.queryResolveBuffer = null;
        this.querySet = null;
        this.querySetCount = 2;
        this.stagingPendingDisposal = [];
        this.uniformPendingDisposal = [];
        this.uploadWaitMs = 0;
        this.hasReadSyncWarned = false;
        this.hasTimestampQueryWarned = false;
        if (!isWebGPUSupported()) {
            throw new Error('WebGPU is not supported on this device');
        }
        this.pipelineCache = {};
        this.device = device;
        this.queue = device.queue;
        this.commandEncoder = null;
        this.computePassEncoder = null;
        this.adapterInfo = new AdapterInfo(adapterInfo);
        this.supportTimestampQuery = this.device.features.has('timestamp-query');
        this.thresholdToIncreaseWorkgroups =
            this.adapterInfo.intelGPUGeneration >= 12 ? 16 : 8;
        this.bufferManager = new BufferManager(this.device);
        this.textureManager = new TextureManager(this.device);
        this.tensorMap = new DataStorage(this, engine());
        // Profiling tools like PIX needs this dummy canvas to
        // trigger capturing a frame.
        if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
            this.dummyCanvas = document.createElement('canvas');
            this.dummyCanvas.width = 1;
            this.dummyCanvas.height = 1;
            this.dummyContext = this.dummyCanvas.getContext('webgpu');
            this.dummyContext.configure({
                device,
                format: 'bgra8unorm',
            });
            document.body.appendChild(this.dummyCanvas);
        }
    }
    floatPrecision() {
        return 32;
    }
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released or delayed in this backend, false if there are still
     * references.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        // No-op if already disposed.
        if (!this.tensorMap.has(dataId)) {
            return true;
        }
        const tensorData = this.tensorMap.get(dataId);
        if (force) {
            tensorData.refCount = 0;
        }
        else {
            tensorData.refCount--;
        }
        if (tensorData.refCount > 0) {
            return false;
        }
        if (tensorData.complexTensorInfos != null) {
            this.disposeData(tensorData.complexTensorInfos.real.dataId);
            this.disposeData(tensorData.complexTensorInfos.imag.dataId);
        }
        if (this.commandQueueOwnedIds.has(dataId)) {
            this.tensorDataPendingDisposal.push(dataId);
            return true;
        }
        this.releaseResource(dataId);
        this.tensorMap.delete(dataId);
        return true;
    }
    memory() {
        return {
            numBytesInGPU: this.bufferManager.numBytesUsed,
            numBytesAllocatedInGPU: this.bufferManager.numBytesAllocated,
            unreliable: false
        };
    }
    releaseResource(dataId) {
        const tensorData = this.tensorMap.get(dataId);
        if (!tensorData || !tensorData.resource) {
            return;
        }
        // If tensor's resource is from external, do not release.
        if (tensorData.external) {
            tensorData.resource = null;
            return;
        }
        if (tensorData.resource instanceof GPUBuffer) {
            this.bufferManager.releaseBuffer(tensorData.resource);
        }
        else if (tensorData.resource instanceof GPUTexture) {
            this.textureManager.releaseTexture(tensorData.resource);
        }
        tensorData.resource = null;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.tensorMap.has(dataId)) {
            const tensorData = this.tensorMap.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    /** Increase refCount of a `TensorData`. */
    incRef(dataId) {
        const tensorData = this.tensorMap.get(dataId);
        tensorData.refCount++;
    }
    /** Decrease refCount of a `TensorData`. */
    decRef(dataId) {
        if (this.tensorMap.has(dataId)) {
            const tensorData = this.tensorMap.get(dataId);
            tensorData.refCount--;
        }
    }
    write(values, shape, dtype) {
        if (dtype === 'complex64' && values != null) {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const dataId = { id: this.nextDataId() };
        this.tensorMap.set(dataId, { dtype, shape, values, refCount: 1 });
        return dataId;
    }
    move(dataId, values, shape, dtype, refCount) {
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        this.tensorMap.set(dataId, { dtype, shape, values, refCount });
    }
    submitQueue() {
        this.queue.submit([this.commandEncoder.finish()]);
        this.commandEncoder = null;
        this.dispatchCountInPass = 0;
        this.commandQueueOwnedIds = new WeakSet();
        this.tensorDataPendingDisposal.forEach(d => {
            this.releaseResource(d);
            this.tensorMap.delete(d);
        });
        this.uniformPendingDisposal.forEach(b => this.bufferManager.releaseBuffer(b));
        this.stagingPendingDisposal.forEach(b => this.bufferManager.releaseBuffer(b, false));
        this.tensorDataPendingDisposal = [];
        this.uniformPendingDisposal = [];
        this.stagingPendingDisposal = [];
    }
    ensureCommandEncoderReady() {
        if (!this.commandEncoder) {
            this.commandEncoder = this.device.createCommandEncoder();
        }
    }
    endComputePassEncoder() {
        if (this.computePassEncoder) {
            this.computePassEncoder.end();
            this.computePassEncoder = null;
        }
    }
    // Check if parallel compilation is done.
    async checkCompileCompletionAsync() {
        let pipelines;
        try {
            pipelines = await Promise.all(Object.values(this.pipelineCache));
        }
        catch (e) {
            // TODO: Add test case to catch this exception.
            throw new Error(e.message);
        }
        Object.keys(this.pipelineCache).map((key, i) => {
            this.pipelineCache[key] = pipelines[i];
        });
    }
    async getBufferData(buffer) {
        if (env().getBool('WEBGPU_ENGINE_COMPILE_ONLY')) {
            console.warn('The data may be invalid since WEBGPU_ENGINE_COMPILE_ONLY is true, this can only be called when WEBGPU_ENGINE_COMPILE_ONLY is false');
            return null;
        }
        const size = buffer.size;
        const stagingBuffer = this.bufferManager.acquireBuffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        this.ensureCommandEncoderReady();
        this.endComputePassEncoder();
        this.commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
        this.submitQueue();
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const values = stagingBuffer.getMappedRange().slice(0);
        stagingBuffer.unmap();
        if (stagingBuffer != null) {
            this.bufferManager.releaseBuffer(stagingBuffer);
        }
        // Need to get texture from swapChain to enable profiling tool
        // to capture a frame
        if (env().getBool('WEBGPU_USE_PROFILE_TOOL')) {
            util.assert(this.dummyContext !== undefined, () => `Fail to get context for profiling tool`);
            this.dummyContext.getCurrentTexture();
        }
        return values;
    }
    convertAndCacheOnCPU(dataId, data) {
        const tensorData = this.tensorMap.get(dataId);
        tensorData.values = data;
        return tensorData.values;
    }
    readSync(dataId) {
        const tensorData = this.tensorMap.get(dataId);
        const { values, complexTensorInfos } = tensorData;
        if (values != null || tensorData.dtype === 'string') {
            return values;
        }
        if (tensorData.dtype === 'complex64') {
            const realValues = this.readSync(complexTensorInfos.real.dataId);
            const imagValues = this.readSync(complexTensorInfos.imag.dataId);
            const complexVals = util.convertBackendValuesAndArrayBuffer(backend_util.mergeRealAndImagArrays(realValues, imagValues).buffer, 'float32');
            this.convertAndCacheOnCPU(dataId, complexVals);
            return complexVals;
        }
        if (!this.hasReadSyncWarned) {
            this.hasReadSyncWarned = true;
            console.warn(`The performance of synchronously reading data from GPU to CPU is ` +
                `poor on the webgpu backend, please use asynchronous APIs instead.`);
        }
        const alphaModes = ['opaque', 'premultiplied'];
        const buffer = tensorData.resource;
        const bufferSize = buffer.size;
        util.assert(bufferSize % 4 === 0, () => 'Because there is 4 bytes for ' +
            'one pixel, buffer size must be multiple of 4.');
        const pixelsSize = bufferSize / 4;
        const valsGPU = new ArrayBuffer(bufferSize);
        // TODO: adjust the reading window size according the `bufferSize`.
        const canvasWidth = 256, canvasHeight = 256;
        const stagingDeviceStorage = alphaModes.map(_ => new OffscreenCanvas(canvasWidth, canvasHeight));
        const stagingHostStorage = new OffscreenCanvas(canvasWidth, canvasHeight);
        this.endComputePassEncoder();
        stagingDeviceStorage
            .map((storage, index) => {
            const context = storage.getContext('webgpu');
            // TODO: use rgba8unorm format when this format is supported on Mac.
            // https://bugs.chromium.org/p/chromium/issues/detail?id=1298618
            context.configure({
                device: this.device,
                format: 'bgra8unorm',
                usage: GPUTextureUsage.COPY_DST,
                alphaMode: alphaModes[index],
            });
            return context.getCurrentTexture();
        })
            .map((texture, index) => {
            const bytesPerRow = canvasWidth * 4;
            const readDataGPUToCPU = (width, height, offset) => {
                this.ensureCommandEncoderReady();
                this.commandEncoder.copyBufferToTexture({
                    buffer,
                    bytesPerRow,
                    offset,
                }, {
                    texture,
                }, {
                    width,
                    height,
                });
                this.submitQueue();
                const context = stagingHostStorage.getContext('2d', {
                    willReadFrequently: true,
                });
                context.clearRect(0, 0, width, height);
                context.drawImage(stagingDeviceStorage[index], 0, 0);
                const stagingValues = context.getImageData(0, 0, width, height).data;
                const alphaMode = alphaModes[index];
                const span = new Uint8ClampedArray(valsGPU, offset, width * height * 4);
                for (let k = 0; k < span.length; k += 4) {
                    if (alphaMode === 'premultiplied') {
                        span[k + 3] = stagingValues[k + 3];
                    }
                    else {
                        const value = stagingValues[k];
                        span[k] = stagingValues[k + 2];
                        span[k + 1] = stagingValues[k + 1];
                        span[k + 2] = value;
                    }
                }
            };
            const fullyReadCount = Math.floor(pixelsSize / (canvasWidth * canvasHeight));
            let width = canvasWidth, height = canvasHeight, offset = 0;
            for (let i = 0; i < fullyReadCount; i++) {
                // Read the buffer data, which fully fill the whole canvas.
                readDataGPUToCPU(width, height, offset);
                offset += canvasWidth * canvasHeight * 4;
            }
            const remainSize = pixelsSize % (canvasWidth * canvasHeight);
            height = Math.floor(remainSize / canvasWidth);
            if (height > 0) {
                // Read the buffer data, which fully fill certain rows of canvas.
                readDataGPUToCPU(width, height, offset);
                offset += height * (canvasWidth * 4);
            }
            width = remainSize % canvasWidth;
            if (width > 0) {
                // Read the buffer data, which not fully fill one row of canvas.
                readDataGPUToCPU(width, 1, offset);
            }
        });
        const vals = util.convertBackendValuesAndArrayBuffer(valsGPU, tensorData.dtype);
        this.convertAndCacheOnCPU(dataId, vals);
        return vals;
    }
    async read(dataId) {
        if (!this.tensorMap.has(dataId)) {
            throw new Error(`Tensor ${dataId} was not registered!`);
        }
        const tensorData = this.tensorMap.get(dataId);
        const { values } = tensorData;
        if (values != null) {
            return values;
        }
        // Download the values from the GPU.
        let vals;
        if (tensorData.dtype === 'complex64') {
            const ps = await Promise.all([
                this.read(tensorData.complexTensorInfos.real.dataId),
                this.read(tensorData.complexTensorInfos.imag.dataId)
            ]);
            const realValues = ps[0];
            const imagValues = ps[1];
            vals = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            const data = await this.getBufferData(tensorData.resource);
            vals = util.convertBackendValuesAndArrayBuffer(data, tensorData.dtype);
        }
        this.convertAndCacheOnCPU(dataId, vals);
        return vals;
    }
    // The source GPUBuffer and destination GPUBuffer have the same size and
    // usage.
    copyBuffer(srcBuffer) {
        const size = srcBuffer.size;
        const usage = srcBuffer.usage;
        const dstBuffer = this.bufferManager.acquireBuffer(size, usage);
        this.ensureCommandEncoderReady();
        this.endComputePassEncoder();
        this.commandEncoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, size);
        this.submitQueue();
        return dstBuffer;
    }
    /**
     * Create a TF.js tensor out of an existing WebGPU buffer.
     */
    createTensorFromGPUData(webGPUData, shape, dtype) {
        let buffer = webGPUData.buffer;
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. `);
        }
        const dataId = { id: this.nextDataId() };
        this.tensorMap.set(dataId, {
            dtype,
            shape,
            values: null,
            refCount: 1,
            external: webGPUData.zeroCopy
        });
        const tensorData = this.tensorMap.get(dataId);
        const size = GPUBytesPerElement(tensorData.dtype) *
            util.sizeFromShape(tensorData.shape);
        if (webGPUData.buffer.size < size) {
            throw new Error(`GPUBuffer size(${webGPUData.buffer.size}) is smaller than tensor size(${size})!`);
        }
        else if ((webGPUData.buffer.usage &
            (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)) !==
            (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)) {
            throw new Error('GPUBuffer.usage should include GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC!');
        }
        // Do buffer copy by default.
        if (webGPUData.zeroCopy !== true) {
            buffer = this.copyBuffer(buffer);
        }
        tensorData.resource = buffer;
        return engine().makeTensorFromDataId(dataId, shape, dtype, this);
    }
    /**
     * Read tensor to a new GPUBuffer.
     * @param dataId The source tensor.
     */
    readToGPU(dataId) {
        const srcTensorData = this.tensorMap.get(dataId);
        const { values, dtype, shape, resource } = srcTensorData;
        if (dtype === 'complex64') {
            throw new Error('Does not support reading buffer for complex64 dtype.');
        }
        if (resource == null) {
            if (values != null) {
                throw new Error('Data is not on GPU but on CPU.');
            }
            else {
                throw new Error('There is no data on GPU or CPU.');
            }
        }
        const srcBuffer = resource;
        const size = srcBuffer.size;
        const usage = srcBuffer.usage;
        const buffer = this.bufferManager.acquireBuffer(size, usage);
        this.ensureCommandEncoderReady();
        this.endComputePassEncoder();
        this.commandEncoder.copyBufferToBuffer(resource, 0, buffer, 0, size);
        this.submitQueue();
        const tensorInfo = this.makeTensorInfo(shape, dtype);
        // Make engine track this tensor, so that we can dispose it later.
        const tensorRef = engine().makeTensorFromTensorInfo(tensorInfo);
        const tensorData = this.tensorMap.get(tensorInfo.dataId);
        tensorData.resource = buffer;
        return { tensorRef, buffer };
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                const strings = data.map(d => util.decodeString(d));
                return buffer(t.shape, t.dtype, strings);
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, data);
    }
    async time(f) {
        if (!this.supportTimestampQuery && !this.hasTimestampQueryWarned) {
            console.warn(`This device doesn't support timestamp-query extension. ` +
                `Start Chrome browser with flag ` +
                `--enable-dawn-features=allow_unsafe_apis to try it again. ` +
                `Otherwise, zero will be shown for the kernel time when profiling ` +
                `mode is enabled.`);
            this.hasTimestampQueryWarned = true;
        }
        const oldActiveTimers = this.activeTimers;
        const newActiveTimers = [];
        let outerMostTime = false;
        if (this.programTimersStack == null) {
            this.programTimersStack = newActiveTimers;
            outerMostTime = true;
        }
        else {
            this.activeTimers.push(newActiveTimers);
        }
        this.activeTimers = newActiveTimers;
        f();
        const flattenedActiveTimerQueries = util.flatten(this.activeTimers.map((d) => d.query))
            .filter(d => d != null);
        const flattenedActiveTimerNames = util.flatten(this.activeTimers.map((d) => d.name))
            .filter(d => d != null);
        this.activeTimers = oldActiveTimers;
        if (outerMostTime) {
            this.programTimersStack = null;
        }
        const res = {
            uploadWaitMs: this.uploadWaitMs,
            downloadWaitMs: this.downloadWaitMs,
            kernelMs: null,
            wallMs: null
        };
        const kernelMs = await Promise.all(flattenedActiveTimerQueries);
        res['kernelMs'] = util.sum(kernelMs);
        res['getExtraProfileInfo'] = () => kernelMs.map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
            .map(d => `${d.name}: ${d.ms}`)
            .join(', ');
        this.uploadWaitMs = 0;
        this.downloadWaitMs = 0;
        return res;
    }
    makeTensorInfo(shape, dtype, values) {
        if (dtype === 'string' && values != null && values.length > 0 &&
            util.isString(values[0])) {
            values = values.map(d => util.encodeString(d));
        }
        const dataId = this.write(values, shape, dtype);
        return { dataId, shape, dtype };
    }
    tensorToBinding(tensor) {
        if (!tensor) {
            return null;
        }
        const tensorData = this.tensorMap.get(tensor.dataId);
        const resource = tensorData.resource;
        if (resource instanceof GPUBuffer) {
            return { buffer: resource };
        }
        if (resource instanceof GPUTexture) {
            return resource.createView();
        }
        // GPUExternalTexture
        return resource;
    }
    uploadToGPU(dataId) {
        const tensorData = this.tensorMap.get(dataId);
        // Already on the GPU.
        if (tensorData.resource != null) {
            return;
        }
        const size = GPUBytesPerElement(tensorData.dtype) *
            util.sizeFromShape(tensorData.shape);
        let buffer;
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST;
        if (tensorData.values) {
            buffer = this.bufferManager.acquireBuffer(size, usage, true);
            if (buffer.mapState === 'unmapped') {
                const stagingBuffer = this.bufferManager.acquireBuffer(size, GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, true, false);
                const arrayBuffer = stagingBuffer.getMappedRange();
                if (tensorData.dtype === 'int32' || tensorData.dtype === 'bool') {
                    new Int32Array(arrayBuffer).set(tensorData.values);
                }
                else {
                    new Float32Array(arrayBuffer).set(tensorData.values);
                }
                stagingBuffer.unmap();
                this.ensureCommandEncoderReady();
                this.endComputePassEncoder();
                this.commandEncoder.copyBufferToBuffer(stagingBuffer, 0, buffer, 0, size);
                this.stagingPendingDisposal.push(stagingBuffer);
            }
            else {
                const arrayBuffer = buffer.getMappedRange();
                if (tensorData.dtype === 'int32' || tensorData.dtype === 'bool') {
                    new Int32Array(arrayBuffer).set(tensorData.values);
                }
                else {
                    new Float32Array(arrayBuffer).set(tensorData.values);
                }
                buffer.unmap();
            }
            // Once uploaded, don't store the values on cpu.
            tensorData.values = null;
        }
        else {
            buffer = this.bufferManager.acquireBuffer(size, usage);
        }
        tensorData.resource = buffer;
    }
    makeUniforms(programUniform) {
        let currentOffset = 0;
        let preLength = 0;
        const offsets = [];
        let maxAlignmentOfField = 1;
        programUniform.forEach((d) => {
            if (d.data.length === 0) {
                d.data = [1];
            }
            // https://www.w3.org/TR/WGSL/#alignof
            let baseAlignment;
            switch (d.data.length) {
                case 1:
                    baseAlignment = 4;
                    break;
                case 2:
                    baseAlignment = 8;
                    break;
                case 3:
                    baseAlignment = 16;
                    break;
                case 4:
                    baseAlignment = 16;
                    break;
                case 5:
                    baseAlignment = 16;
                    break;
                case 6:
                    baseAlignment = 16;
                    break;
                default:
                    util.assert(false, () => `Unsupported ${d.data.length}D shape`);
            }
            if (preLength === 5 || preLength === 6) {
                baseAlignment = 16;
            }
            if (baseAlignment > maxAlignmentOfField) {
                maxAlignmentOfField = baseAlignment;
            }
            currentOffset = Math.ceil(currentOffset / baseAlignment) * baseAlignment;
            preLength = d.data.length;
            offsets.push(currentOffset);
            currentOffset += d.data.length * 4;
        });
        currentOffset =
            Math.ceil(currentOffset / maxAlignmentOfField) * maxAlignmentOfField;
        const arrayBuffer = new ArrayBuffer(currentOffset);
        programUniform.forEach((d, i) => {
            const offset = offsets[i];
            if (d.type === 'int32') {
                new Int32Array(arrayBuffer, offset, d.data.length).set(d.data);
            }
            else if (d.type === 'uint32') {
                new Uint32Array(arrayBuffer, offset, d.data.length).set(d.data);
            }
            else {
                new Float32Array(arrayBuffer, offset, d.data.length).set(d.data);
            }
        });
        const uniformBuffer = this.bufferManager.acquireBuffer(currentOffset, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
        this.queue.writeBuffer(uniformBuffer, 0, arrayBuffer, 0, currentOffset);
        this.uniformPendingDisposal.push(uniformBuffer);
        return { offset: 0, size: currentOffset, buffer: uniformBuffer };
    }
    runWebGPUProgram(program, inputs, outputDtype, programDefinedUniform, output) {
        if (!output) {
            output = this.makeTensorInfo(program.outputShape, outputDtype);
        }
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            this.tensorMap.get(output.dataId).values =
                util.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        this.uploadToGPU(output.dataId);
        program.dispatch = reshapeDispatch(this.device, program);
        const inputsData = inputs.map((input, i) => {
            if (input.dtype === 'complex64') {
                throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                    `dtypes, please separate the program into real and imaginary ` +
                    `parts.`);
            }
            this.uploadToGPU(input.dataId);
            return {
                // Returning dtype from tensorMap because it reflects dtype
                // of underlying buffer, rather than abstract dtype.
                dtype: this.tensorMap.get(input.dataId).dtype,
                shape: input.shape,
                name: program.variableNames[i]
            };
        });
        program.shaderKey =
            makeShaderKey(program, inputsData, output);
        const parallelCompilation = env().getBool('WEBGPU_ENGINE_COMPILE_ONLY');
        if (!(program.shaderKey in this.pipelineCache)) {
            this.pipelineCache[program.shaderKey] = compileProgram(this.device, program, inputsData, output, parallelCompilation);
        }
        program.pipeline = this.pipelineCache[program.shaderKey];
        if (!parallelCompilation) {
            this.recordAndSubmit(program, output, inputs, programDefinedUniform);
        }
        return output;
    }
    recordAndSubmit(program, output, inputs, programDefinedUniform) {
        if (program.pipeline instanceof Promise) {
            throw new Error('Please call checkCompileCompletionAsync to ensure parallel compilation is done!');
        }
        // There are six kinds of uniforms: NAN, INFINITY, shapes, shape strides,
        // program size, program defined uniforms.
        let programUniform = [];
        let bufferShapes = [];
        const uniformsType = 'int32';
        if (program.pixelsOpType == null) {
            programUniform.push({ type: 'float32', data: [NaN] }, { type: 'float32', data: [Infinity] });
            bufferShapes = inputs.concat(output).map(d => d.shape);
            const uniformsType = 'int32';
            bufferShapes.map(d => {
                programUniform.push({ type: uniformsType, data: d });
                const strides = util.computeStrides(d);
                programUniform.push({ type: uniformsType, data: strides });
            });
        }
        else {
            const strides = util.computeStrides(output.shape);
            programUniform.push({ type: uniformsType, data: strides });
        }
        if (program.size) {
            const size = util.sizeFromShape(program.outputShape);
            programUniform.push({
                type: uniformsType,
                data: [program.outputComponent ? size / program.outputComponent : size]
            });
        }
        if (programDefinedUniform) {
            programUniform = [...programUniform, ...programDefinedUniform];
        }
        const bindings = [
            this.tensorToBinding(output), ...inputs.map(t => this.tensorToBinding(t)),
            this.makeUniforms(programUniform)
        ];
        inputs.forEach(input => {
            this.commandQueueOwnedIds.add(input.dataId);
        });
        this.commandQueueOwnedIds.add(output.dataId);
        const bindGroup = this.device.createBindGroup({
            layout: program.pipeline.getBindGroupLayout(0),
            entries: bindings.map((b, i) => ({ binding: i, resource: b })),
        });
        const shouldTimeProgram = this.activeTimers != null;
        this.ensureCommandEncoderReady();
        const computePassDescriptor = {};
        if (shouldTimeProgram && this.supportTimestampQuery) {
            this.endComputePassEncoder();
            if (this.querySet == null) {
                this.querySet = this.device.createQuerySet({
                    type: 'timestamp',
                    count: this.querySetCount,
                });
            }
            computePassDescriptor.timestampWrites = {
                querySet: this.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            };
            this.computePassEncoder =
                this.commandEncoder.beginComputePass(computePassDescriptor);
        }
        else if (!this.computePassEncoder) {
            this.computePassEncoder =
                this.commandEncoder.beginComputePass(computePassDescriptor);
        }
        this.computePassEncoder.setPipeline(program.pipeline);
        this.computePassEncoder.setBindGroup(0, bindGroup);
        this.computePassEncoder.dispatchWorkgroups(program.dispatch[0], program.dispatch[1], program.dispatch[2]);
        this.dispatchCountInPass++;
        if (shouldTimeProgram ||
            env().get('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') <= this.dispatchCountInPass ||
            program.pixelsOpType === PixelsOpType.DRAW) {
            this.endComputePassEncoder();
            if (shouldTimeProgram) {
                this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime() });
            }
            else {
                this.submitQueue();
            }
        }
    }
    async getQueryTime() {
        if (!this.supportTimestampQuery) {
            return 0;
        }
        if (this.queryResolveBuffer == null) {
            this.queryResolveBuffer = this.bufferManager.acquireBuffer(this.querySetCount * 8, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
                GPUBufferUsage.QUERY_RESOLVE);
        }
        this.commandEncoder.resolveQuerySet(this.querySet, 0, this.querySetCount, this.queryResolveBuffer, 0);
        const queryStagingBuffer = this.bufferManager.acquireBuffer(this.querySetCount * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer, 0, queryStagingBuffer, 0, this.querySetCount * 8);
        this.submitQueue();
        await queryStagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = new BigUint64Array(queryStagingBuffer.getMappedRange());
        const time = Number(arrayBuffer[1] - arrayBuffer[0]) / 1000000;
        queryStagingBuffer.unmap();
        this.bufferManager.releaseBuffer(queryStagingBuffer);
        return time;
    }
    shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
        return env().getBool('WEBGPU_CPU_FORWARD') &&
            inputs.every(input => this.tensorMap.get(input.dataId).resource == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
    }
    numDataIds() {
        return this.tensorMap.numDataIds() - this.tensorDataPendingDisposal.length;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        if (this.querySet != null) {
            this.querySet.destroy();
        }
        this.bufferManager.dispose();
        this.textureManager.dispose();
        this.disposed = true;
    }
}
WebGPUBackend.nextDataId = 0;

/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
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
if (isWebGPUSupported()) {
    registerBackend('webgpu', async () => {
        const gpuDescriptor = {
            powerPreference: env().get('WEBGPU_USE_LOW_POWER_GPU') ?
                'low-power' :
                'high-performance'
        };
        const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
        const deviceDescriptor = {};
        const requiredFeatures = [];
        if (adapter.features.has('timestamp-query')) {
            requiredFeatures.push('timestamp-query');
        }
        if (adapter.features.has('bgra8unorm-storage')) {
            requiredFeatures.push(['bgra8unorm-storage']);
        }
        deviceDescriptor.requiredFeatures =
            requiredFeatures;
        const adapterLimits = adapter.limits;
        deviceDescriptor.requiredLimits = {
            'maxComputeWorkgroupStorageSize': adapterLimits.maxComputeWorkgroupStorageSize,
            'maxComputeWorkgroupsPerDimension': adapterLimits.maxComputeWorkgroupsPerDimension,
            'maxStorageBufferBindingSize': adapterLimits.maxStorageBufferBindingSize,
            'maxBufferSize': adapterLimits.maxBufferSize,
            'maxComputeWorkgroupSizeX': adapterLimits.maxComputeWorkgroupSizeX,
            'maxComputeInvocationsPerWorkgroup': adapterLimits.maxComputeInvocationsPerWorkgroup,
        };
        const device = await adapter.requestDevice(deviceDescriptor);
        const adapterInfo = 'info' in adapter
            ? adapter.info
            : 'requestAdapterInfo' in adapter
                // tslint:disable-next-line:no-any
                ? await adapter.requestAdapterInfo()
                : undefined;
        return new WebGPUBackend(device, adapterInfo);
    }, 3 /*priority*/);
}

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
var BinaryOpType;
(function (BinaryOpType) {
    BinaryOpType[BinaryOpType["ADD"] = 0] = "ADD";
    BinaryOpType[BinaryOpType["ATAN2"] = 1] = "ATAN2";
    BinaryOpType[BinaryOpType["COMPLEX_MULTIPLY_IMAG"] = 2] = "COMPLEX_MULTIPLY_IMAG";
    BinaryOpType[BinaryOpType["COMPLEX_MULTIPLY_REAL"] = 3] = "COMPLEX_MULTIPLY_REAL";
    BinaryOpType[BinaryOpType["DIV"] = 4] = "DIV";
    BinaryOpType[BinaryOpType["ELU_DER"] = 5] = "ELU_DER";
    BinaryOpType[BinaryOpType["EQUAL"] = 6] = "EQUAL";
    BinaryOpType[BinaryOpType["FLOOR_DIV"] = 7] = "FLOOR_DIV";
    BinaryOpType[BinaryOpType["GREATER"] = 8] = "GREATER";
    BinaryOpType[BinaryOpType["GREATER_EQUAL"] = 9] = "GREATER_EQUAL";
    BinaryOpType[BinaryOpType["LESS"] = 10] = "LESS";
    BinaryOpType[BinaryOpType["LESS_EQUAL"] = 11] = "LESS_EQUAL";
    BinaryOpType[BinaryOpType["LOGICAL_AND"] = 12] = "LOGICAL_AND";
    BinaryOpType[BinaryOpType["LOGICAL_OR"] = 13] = "LOGICAL_OR";
    BinaryOpType[BinaryOpType["MAX"] = 14] = "MAX";
    BinaryOpType[BinaryOpType["MIN"] = 15] = "MIN";
    BinaryOpType[BinaryOpType["MOD"] = 16] = "MOD";
    BinaryOpType[BinaryOpType["MUL"] = 17] = "MUL";
    BinaryOpType[BinaryOpType["NOT_EQUAL"] = 18] = "NOT_EQUAL";
    BinaryOpType[BinaryOpType["POW"] = 19] = "POW";
    BinaryOpType[BinaryOpType["PRELU"] = 20] = "PRELU";
    BinaryOpType[BinaryOpType["SQUARED_DIFFERENCE"] = 21] = "SQUARED_DIFFERENCE";
    BinaryOpType[BinaryOpType["SUB"] = 22] = "SUB";
})(BinaryOpType || (BinaryOpType = {}));
const ADD = 'let resultTemp = a + b;';
const ATAN2 = 'let resultTemp = atan2(a, b);';
// (Ar + Ai)(Br + Bi) =
// ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
// Yr = ArBr - AB
// Yi = ArBi + AiBr
const COMPLEX_MULTIPLY_REAL = 'let resultTemp = areal * breal - aimag * bimag;';
const COMPLEX_MULTIPLY_IMAG = 'let resultTemp = areal * bimag + aimag * breal;';
const DIV = 'let resultTemp = a / b;';
const ELU_DER = 'let resultTemp = select(a * (b + 1.0), a, b >= b - b);';
const EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a == b);
`;
const FLOOR_DIV = `
  let remainder =
      select(a % b, round(a % b), (round(a) == a) & (round(b) == b));
  let quotient = (a - remainder) / b;
  let resultTemp =
      round(select(quotient, quotient - 1, sign(remainder) == -sign(b)));
`;
const GREATER = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a > b);
`;
const GREATER_EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a >= b);
`;
const LESS = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a < b);
`;
const LESS_EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a <= b);
`;
const LOGICAL_AND = 'return f32(a >= 1.0 && b >= 1.0);';
const LOGICAL_AND_VEC4 = `return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`;
const LOGICAL_OR = 'return f32(a >= 1.0 || b >= 1.0);';
const LOGICAL_OR_VEC4 = `return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`;
const MAX = 'let resultTemp = max(a, b);';
const MIN = 'let resultTemp = min(a, b);';
const MOD = `
  let isNaN = b == 0.;
  var resultTemp = a % b;
  resultTemp = select((resultTemp + b) % b, resultTemp,
      (a < 0. && b < 0.) || (a >= 0. && b > 0.));
`;
const MOD_VEC4 = `
  let isNaN = !vec4<bool>(b);
  var resultTemp = vec4<f32>(a % b);
  if (!((a[0] < 0. && b[0] < 0.) || (a[0] >= 0. && b[0] > 0.))) {
    resultTemp[0] = (resultTemp[0] + b[0]) % b[0];
  }
  if (!((a[1] < 0. && b[1] < 0.) || (a[1] >= 0. && b[1] > 0.))) {
    resultTemp[1] = (resultTemp[1] + b[1]) % b[1];
  }
  if (!((a[2] < 0. && b[2] < 0.) || (a[2] >= 0. && b[2] > 0.))) {
    resultTemp[2] = (resultTemp[2] + b[2]) % b[2];
  }
  if (!((a[3] < 0. && b[3] < 0.) || (a[3] >= 0. && b[3] > 0.))) {
    resultTemp[3] = (resultTemp[3] + b[3]) % b[3];
  }
`;
const MUL = 'let resultTemp = a * b;';
const NOT_EQUAL = `
  var resultTemp = f32(a != b);
  let valueForNaN = 1.0;
`;
const NOT_EQUAL_VEC4 = `
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
`;
const POW = `
  let isNaN = a < 0.0 && floor(b) < b;
  if (b == 0.0) {
    return 1.0;
  }
  var resultTemp = select(sign(a) * pow(abs(a), b), pow(abs(a), b),
      round(abs(b) % 2.0) != 1.0);
`;
const POW_VEC4 = `
  let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = (a < vec4<f32>(0.0)) & (floor(b) < b);
`;
const PRELU = `if (a < 0.0) { return b * a; }  return a;`;
const PRELU_VEC4 = `
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`;
const SQUARED_DIFFERENCE = 'let resultTemp = (a - b) * (a - b);';
const SUB = 'let resultTemp = a - b;';
function getBinaryOpString(type, useVec4) {
    let doOpSnippet;
    // Ops with NaN check
    do {
        switch (type) {
            case BinaryOpType.ATAN2:
                doOpSnippet = ATAN2;
                break;
            case BinaryOpType.MAX:
                doOpSnippet = MAX;
                break;
            case BinaryOpType.MIN:
                doOpSnippet = MIN;
                break;
            case BinaryOpType.MOD:
                doOpSnippet = useVec4 ? MOD_VEC4 : MOD;
                break;
            case BinaryOpType.NOT_EQUAL:
                doOpSnippet = useVec4 ? NOT_EQUAL_VEC4 : NOT_EQUAL;
                break;
            case BinaryOpType.POW:
                doOpSnippet = useVec4 ? POW_VEC4 : POW;
                break;
            default:
                continue;
        }
        let isNaN;
        let dTypeN;
        let boolN;
        if (useVec4) {
            isNaN = 'isnanVec4';
            dTypeN = 'vec4<f32>';
            boolN = 'vec4<bool>';
        }
        else {
            isNaN = 'isnan';
            dTypeN = 'f32';
            boolN = 'bool';
        }
        return `
      let aIsNaN = ${isNaN}(a);
      let aPostLegalization = select(a, ${dTypeN}(42), aIsNaN);
      let bIsNaN = ${isNaN}(b);
      let bPostLegalization = select(b, ${dTypeN}(42), bIsNaN);
      let isNaN = false;
      let valueForNaN = uniforms.NAN;
      {
        let a = aPostLegalization;
        let b = bPostLegalization;
        ${doOpSnippet}
        return select(
            resultTemp, ${dTypeN}(valueForNaN),
            ${boolN}(isNaN) | aIsNaN | bIsNaN);
      }
    `;
    } while (false);
    // Ops without NaN check
    switch (type) {
        case BinaryOpType.ADD:
            doOpSnippet = ADD;
            break;
        case BinaryOpType.COMPLEX_MULTIPLY_IMAG:
            doOpSnippet = COMPLEX_MULTIPLY_IMAG;
            break;
        case BinaryOpType.COMPLEX_MULTIPLY_REAL:
            doOpSnippet = COMPLEX_MULTIPLY_REAL;
            break;
        case BinaryOpType.DIV:
            doOpSnippet = DIV;
            break;
        case BinaryOpType.ELU_DER:
            doOpSnippet = ELU_DER;
            break;
        case BinaryOpType.EQUAL:
            doOpSnippet = EQUAL;
            break;
        case BinaryOpType.FLOOR_DIV:
            doOpSnippet = FLOOR_DIV;
            break;
        case BinaryOpType.GREATER:
            doOpSnippet = GREATER;
            break;
        case BinaryOpType.GREATER_EQUAL:
            doOpSnippet = GREATER_EQUAL;
            break;
        case BinaryOpType.LESS:
            doOpSnippet = LESS;
            break;
        case BinaryOpType.LESS_EQUAL:
            doOpSnippet = LESS_EQUAL;
            break;
        case BinaryOpType.LOGICAL_AND:
            return useVec4 ? LOGICAL_AND_VEC4 : LOGICAL_AND;
        case BinaryOpType.LOGICAL_OR:
            return useVec4 ? LOGICAL_OR_VEC4 : LOGICAL_OR;
        case BinaryOpType.MUL:
            doOpSnippet = MUL;
            break;
        case BinaryOpType.PRELU:
            return useVec4 ? PRELU_VEC4 : PRELU;
        case BinaryOpType.SQUARED_DIFFERENCE:
            doOpSnippet = SQUARED_DIFFERENCE;
            break;
        case BinaryOpType.SUB:
            doOpSnippet = SUB;
            break;
        // throw new Error(`BinaryType ${type} is not implemented!`);
    }
    return `
    ${doOpSnippet}
    return resultTemp;
  `;
}

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
var UnaryOpType;
(function (UnaryOpType) {
    UnaryOpType[UnaryOpType["ABS"] = 0] = "ABS";
    UnaryOpType[UnaryOpType["ACOS"] = 1] = "ACOS";
    UnaryOpType[UnaryOpType["ACOSH"] = 2] = "ACOSH";
    UnaryOpType[UnaryOpType["ASIN"] = 3] = "ASIN";
    UnaryOpType[UnaryOpType["ASINH"] = 4] = "ASINH";
    UnaryOpType[UnaryOpType["ATAN"] = 5] = "ATAN";
    UnaryOpType[UnaryOpType["ATANH"] = 6] = "ATANH";
    UnaryOpType[UnaryOpType["CEIL"] = 7] = "CEIL";
    UnaryOpType[UnaryOpType["COS"] = 8] = "COS";
    UnaryOpType[UnaryOpType["COSH"] = 9] = "COSH";
    UnaryOpType[UnaryOpType["ELU"] = 10] = "ELU";
    UnaryOpType[UnaryOpType["ERF"] = 11] = "ERF";
    UnaryOpType[UnaryOpType["EXP"] = 12] = "EXP";
    UnaryOpType[UnaryOpType["EXPM1"] = 13] = "EXPM1";
    UnaryOpType[UnaryOpType["FLOOR"] = 14] = "FLOOR";
    UnaryOpType[UnaryOpType["IS_FINITE"] = 15] = "IS_FINITE";
    UnaryOpType[UnaryOpType["IS_INF"] = 16] = "IS_INF";
    UnaryOpType[UnaryOpType["IS_NAN"] = 17] = "IS_NAN";
    UnaryOpType[UnaryOpType["LINEAR"] = 18] = "LINEAR";
    UnaryOpType[UnaryOpType["LOG"] = 19] = "LOG";
    UnaryOpType[UnaryOpType["LOG1P"] = 20] = "LOG1P";
    UnaryOpType[UnaryOpType["LOGICAL_NOT"] = 21] = "LOGICAL_NOT";
    UnaryOpType[UnaryOpType["NEG"] = 22] = "NEG";
    UnaryOpType[UnaryOpType["RELU"] = 23] = "RELU";
    UnaryOpType[UnaryOpType["RELU6"] = 24] = "RELU6";
    UnaryOpType[UnaryOpType["LEAKYRELU"] = 25] = "LEAKYRELU";
    UnaryOpType[UnaryOpType["RECIPROCAL"] = 26] = "RECIPROCAL";
    UnaryOpType[UnaryOpType["ROUND"] = 27] = "ROUND";
    UnaryOpType[UnaryOpType["RSQRT"] = 28] = "RSQRT";
    UnaryOpType[UnaryOpType["SELU"] = 29] = "SELU";
    UnaryOpType[UnaryOpType["SIGMOID"] = 30] = "SIGMOID";
    UnaryOpType[UnaryOpType["SIGN"] = 31] = "SIGN";
    UnaryOpType[UnaryOpType["SIN"] = 32] = "SIN";
    UnaryOpType[UnaryOpType["SINH"] = 33] = "SINH";
    UnaryOpType[UnaryOpType["SOFTPLUS"] = 34] = "SOFTPLUS";
    UnaryOpType[UnaryOpType["SQRT"] = 35] = "SQRT";
    UnaryOpType[UnaryOpType["SQUARE"] = 36] = "SQUARE";
    UnaryOpType[UnaryOpType["STEP"] = 37] = "STEP";
    UnaryOpType[UnaryOpType["TAN"] = 38] = "TAN";
    UnaryOpType[UnaryOpType["TANH"] = 39] = "TANH";
    UnaryOpType[UnaryOpType["TO_INT"] = 40] = "TO_INT";
})(UnaryOpType || (UnaryOpType = {}));
const ABS = `return abs(a);`;
const ACOS = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return acos(a);
`;
const ACOSH = `
  if (a < 1.) {
    return uniforms.NAN;
  }
  return acosh(a);
`;
const ASIN = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return asin(a);
`;
const ASINH = `return asinh(a);`;
const ATAN = `
  if (isnan(a)) {
    return uniforms.NAN;
  }
  return atan(a);
`;
const ATANH = `
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  if (a == 1.) {
    return uniforms.INFINITY;
  }
  if (a == -1.) {
    return -uniforms.INFINITY;
  }
  return atanh(a);
`;
const CEIL = `return ceil(a);`;
const COS = `return cos(a);`;
const COSH = `
  let e2x = exp(-a);
  return (e2x + 1.0 / e2x) / 2.0;
`;
const EXPM1 = `return exp(a) - 1.0;`;
const ELU = `if (a >= 0.0) { return a; }  return (exp(a) - 1.0);`;
const ELU_VEC4 = `
  var resFloat = exp(a) - vec4<f32>(1.0);
  if (a.r >= 0.0) {
    resFloat.r = a.r;
  }
  if (a.g >= 0.0) {
    resFloat.g = a.g;
  }
  if (a.b >= 0.0) {
    resFloat.b = a.b;
  }
  if (a.a >= 0.0) {
    resFloat.a = a.a;
  }
  return resFloat;
`;
const ERF = `
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  let p = ${backend_util.ERF_P};
  let a1 = ${backend_util.ERF_A1};
  let a2 = ${backend_util.ERF_A2};
  let a3 = ${backend_util.ERF_A3};
  let a4 = ${backend_util.ERF_A4};
  let a5 = ${backend_util.ERF_A5};

  let sign = sign(a);
  let absA = abs(a);
  let t = 1.0 / (1.0 + p * absA);
  return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absA * absA));
`;
const EXP = `return exp(a);`;
const FLOOR = `return floor(a);`;
const IS_FINITE = `return f32(!isnan(a) && !isinf(a));`;
const IS_INF = `return f32(isinf(a));`;
const IS_NAN = `return f32(isnan(a));`;
const LINEAR = `return a;`;
const LOG = `if (a < 0.0) { return uniforms.NAN; }
  return log(a);`;
const LOG1P = `
  if (isnan(a)) { return a; }
  return log(1.0 + a);
`;
const LOGICAL_NOT = `return f32(!(a >= 1.0));`;
const NEG = `return -a;`;
const LEAKYRELU = `if (a < 0.0) { return uniforms.alpha * a; } return a;`;
const LEAKYRELU_VEC4 = `
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (uniforms.alpha * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`;
const RECIPROCAL = `return 1.0 / a;`;
const RELU = `return select(a, 0.0, a < 0.0);`;
const RELU6 = 'return clamp(a, 0.0, 6.0);';
const RELU6_VEC4 = 'return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));';
const RELU_VEC4 = `
  return select(a, vec4<f32>(0.0), a < vec4<f32>(0.0));
`;
const ROUND = `return round(a);`;
const RSQRT = `return inverseSqrt(a);`;
// Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
// See: https://arxiv.org/abs/1706.02515
const SELU = `
  if (a >= 0.0) {
    return ${backend_util.SELU_SCALE} * a;
  } else {
    return ${backend_util.SELU_SCALEALPHA} * (exp(a) - 1.0);
  }
`;
const SIGMOID = `return 1.0 / (1.0 + exp(-1.0 * a));`;
const SIGN = `return sign(a);`;
const SIN = `return sin(a);`;
const SINH = `
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`;
const SOFTPLUS = `
  let epsilon = 1.1920928955078125e-7;
  let threshold = log(epsilon) + 2.0;

  let too_large = a > -threshold;
  let too_small = a < threshold;
  let exp_a = exp(a);

  if (too_large) {
    return a;
  } else if (too_small) {
    return exp_a;
  } else {
    return log(exp_a + 1.0);
  }
`;
const SQRT = `return sqrt(a);`;
const SQUARE = `return a * a;`;
const STEP = `
  if (isnan(a)) {
    return a;
  }

  return select(uniforms.stepAlpha, 1.0, a > 0.0);
`;
const TAN = `return tan(a);`;
const TANH = `
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`;
const TO_INT = `return f32(i32((a)));`;
function getUnaryOpString(type, useVec4) {
    switch (type) {
        case UnaryOpType.ABS:
            return ABS;
        case UnaryOpType.ACOS:
            return ACOS;
        case UnaryOpType.ACOSH:
            return ACOSH;
        case UnaryOpType.ASIN:
            return ASIN;
        case UnaryOpType.ASINH:
            return ASINH;
        case UnaryOpType.ATAN:
            return ATAN;
        case UnaryOpType.ATANH:
            return ATANH;
        case UnaryOpType.COS:
            return COS;
        case UnaryOpType.COSH:
            return COSH;
        case UnaryOpType.CEIL:
            return CEIL;
        case UnaryOpType.ELU:
            return useVec4 ? ELU_VEC4 : ELU;
        case UnaryOpType.ERF:
            return ERF;
        case UnaryOpType.EXP:
            return EXP;
        case UnaryOpType.EXPM1:
            return EXPM1;
        case UnaryOpType.FLOOR:
            return FLOOR;
        case UnaryOpType.IS_FINITE:
            return IS_FINITE;
        case UnaryOpType.IS_INF:
            return IS_INF;
        case UnaryOpType.IS_NAN:
            return IS_NAN;
        case UnaryOpType.LINEAR:
            return LINEAR;
        case UnaryOpType.LOG:
            return LOG;
        case UnaryOpType.LOG1P:
            return LOG1P;
        case UnaryOpType.LOGICAL_NOT:
            return LOGICAL_NOT;
        case UnaryOpType.NEG:
            return NEG;
        case UnaryOpType.LEAKYRELU:
            return useVec4 ? LEAKYRELU_VEC4 : LEAKYRELU;
        case UnaryOpType.RECIPROCAL:
            return RECIPROCAL;
        case UnaryOpType.RELU:
            return useVec4 ? RELU_VEC4 : RELU;
        case UnaryOpType.RELU6:
            return useVec4 ? RELU6_VEC4 : RELU6;
        case UnaryOpType.ROUND:
            return ROUND;
        case UnaryOpType.RSQRT:
            return RSQRT;
        case UnaryOpType.SELU:
            return SELU;
        case UnaryOpType.SIGMOID:
            return SIGMOID;
        case UnaryOpType.SIGN:
            return SIGN;
        case UnaryOpType.SIN:
            return SIN;
        case UnaryOpType.SINH:
            return SINH;
        case UnaryOpType.SOFTPLUS:
            return SOFTPLUS;
        case UnaryOpType.SQRT:
            return SQRT;
        case UnaryOpType.SQUARE:
            return SQUARE;
        case UnaryOpType.STEP:
            return STEP;
        case UnaryOpType.TAN:
            return TAN;
        case UnaryOpType.TANH:
            return TANH;
        case UnaryOpType.TO_INT:
            return TO_INT;
        default:
            throw new Error(`BinaryType ${type} is not implemented!`);
    }
}

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
function activationFnSnippet(activation, hasPreluActivationWeights = false, packed = false, coordsLength = 3) {
    if (activation === null) {
        return '';
    }
    let activationOpSnippet = '';
    if (activation === 'linear') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LINEAR);
    }
    else if (activation === 'relu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU, packed);
    }
    else if (activation === 'elu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.ELU, packed);
    }
    else if (activation === 'relu6') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU6, packed);
    }
    else if (activation === 'prelu') {
        activationOpSnippet = getBinaryOpString(BinaryOpType.PRELU, packed);
    }
    else if (activation === 'sigmoid') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.SIGMOID, packed);
    }
    else if (activation === 'leakyrelu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LEAKYRELU, packed);
    }
    else {
        throw new Error(`Activation ${activation} has not been implemented for the WebGPU backend.`);
    }
    const elementSize = packed ? 4 : 1;
    const dataType = typeSnippet(elementSize);
    let activationFnSnippet = '';
    if (hasPreluActivationWeights) {
        activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${activationOpSnippet}
      }`;
    }
    else {
        activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
        ${activationOpSnippet}
      }`;
    }
    return activationFnSnippet;
}
function biasActivationSnippet(hasBias, activation) {
    return `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
}

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
function matMulReadFnSource(transposeA, transposeB, fitAOuter = false, fitBOuter = false, fitInner = false, component = 1) {
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
function matMulReadWriteFnSource(hasBias, activation, transposeA, transposeB, fitAOuter = false, fitBOuter = false, fitInner = false, component = 1) {
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
function makeMatMulPackedVec4Source(workPerThread, workgroupSize, transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32, broadcastBatch = false) {
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

  ${getMainHeaderString()} {
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
function makeMatMulPackedSource(workPerThread, workgroupSize, transposeA = false, tileInner = 32, splitK = false, splitedDimInner = 32, sequentialAccessByThreads = false, broadcastBatch = false) {
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

    ${getMainHeaderString()} {
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
function makeVectorMatrixProductSource(workgroupSize, transposeA = false) {
    util.assert(workgroupSize[1] === 1 && workgroupSize[2] === 1, () => `A linear work group size is required. But got ${workgroupSize}.`);
    const tileSize = workgroupSize[0] * 4;
    return `
    var<workgroup> mm_Asub : array<vec4<f32>, ${workgroupSize[0]}>;

    ${getMainHeaderString()} {
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
class MatMulPackedProgram {
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
function makeMatMulReduceSource(workgroupSizeX) {
    return `
    var<workgroup> sumValues : array<f32, ${workgroupSizeX}>;
    ${getMainHeaderString()} {
      let coords = getOutputCoords();
      let batch = coords[0];
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      let row = coords[1];
      let col = coords[2];
      var sum = 0.0;
      let Length = uniforms.dimInner;
      for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
        let dataA = mm_readA(batchA, row, k);
        let dataB = mm_readB(batchB, k, col);
        sum = sum + dataA * dataB;
      }
      sumValues[localId.x] = sum;
      workgroupBarrier();

      for(var currentSize = ${workgroupSizeX / 2}u; currentSize > 1u;
          currentSize = currentSize / 2u) {
        if (localId.x < currentSize)
        {
          sumValues[localId.x] = sumValues[localId.x] + sumValues[localId.x + currentSize];
        }
        workgroupBarrier();
      }

      if (localId.x == 0u) {
        sum = sumValues[0] + sumValues[1];
        mm_write(batch, row, col, sum);
      }
    }
  `;
}
class MatMulReduceProgram {
    constructor(outputShape, transposeA = false, transposeB = false, bias = null, activation = null, preluActivationWeights = null) {
        this.variableNames = ['A', 'B'];
        this.uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.workgroupSize = [256, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [], y: [1, 2], z: [0] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        const addBias = bias != null;
        const hasPreluActivationWeights = preluActivationWeights != null;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.shaderKey =
            `matMulReduce_${this.activation}_${transposeA}_${transposeB}`;
    }
    getUserCode() {
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
      ${matMulReadWriteFnSource(this.addBias, this.activation, this.transposeA, this.transposeB)}
      ${makeMatMulReduceSource(this.workgroupSize[0])}
    `;
        return userCode;
    }
}

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
function makeMatMulSmallOutputSizeSource(workgroupSize) {
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
  ${getMainHeaderString()} {
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
class MatMulSmallOutputSizeProgram {
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
class MatMulSplitKProgram {
    constructor(outputShape, dimInner, transposeA = false, transposeB = false) {
        this.variableNames = ['A', 'B'];
        this.uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.workgroupSize = [8, 8, 1];
        this.atomic = true;
        this.splitedDimInner = 128;
        util.assert(outputShape[0] === 1, () => 'MatMulSplitKProgram only supports batch = 1.');
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [2], y: [1], z: [0, 3] };
        const isVec4 = (transposeA && this.outputShape[1] % 4 === 0 ||
            !transposeA && dimInner % 4 === 0) &&
            this.outputShape[2] % 4 === 0;
        this.elementsPerThread = [4, 4, this.splitedDimInner];
        this.outputComponent = isVec4 ? 4 : 1;
        if (!isVec4) {
            if (this.outputShape[1] < 16) {
                this.elementsPerThread[1] = 1;
            }
            if (this.outputShape[2] < 16) {
                this.elementsPerThread[0] = 1;
            }
        }
        this.dispatch = computeDispatch(this.dispatchLayout, [
            this.outputShape[0], this.outputShape[1], this.outputShape[2],
            dimInner
        ], this.workgroupSize, this.elementsPerThread);
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.shaderKey = `matMulSplitK_${transposeA}_${transposeB}_${this.elementsPerThread}_${this.outputComponent}`;
    }
    getUserCode() {
        const component = this.outputComponent;
        const userCode = `
      ${matMulReadFnSource(false, this.transposeB, false, false, false, component)}
      fn mm_write(batch: i32, row : i32, col : i32, value : ${typeSnippet(component)}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          for (var i = 0; i < ${component}; i = i + 1) {
            ${atomicAddSnippet('&result[flatIndex + i]', `${component > 1 ? 'value[i]' : 'value'}`, 'float32')}
          }
        }
      }
      ${component === 4 ? makeMatMulPackedVec4Source(this.elementsPerThread, this.workgroupSize, this.transposeA, 32, true, this.splitedDimInner) :
            makeMatMulPackedSource(this.elementsPerThread, this.workgroupSize, this.transposeA, 32, true, this.splitedDimInner)}
    `;
        return userCode;
    }
}
class BiasActivationProgram {
    constructor(outputShape, bias = null, activation = null, preluActivationWeights = null) {
        this.uniforms = '';
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.addBias = bias != null;
        this.hasPreluActivationWeights = preluActivationWeights != null;
        this.activation = activation;
        if (this.addBias) {
            this.variableNames.push('bias');
        }
        if (this.hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.shaderKey = `biasActivation_${activation}`;
    }
    getUserCode() {
        return `
    ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${biasActivationSnippet(this.addBias, this.activation)}
        setOutputAtIndex(index, value);
      }
    }
    `;
    }
}

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
class FillProgram {
    constructor(shape) {
        this.variableNames = [];
        this.outputShape = [];
        this.uniforms = 'value : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'fill';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        setOutputAtIndex(index, uniforms.value);
      }
    }
  `;
        return userCode;
    }
}

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
function fill(args) {
    const { backend, attrs } = args;
    const { shape, value } = attrs;
    let { dtype } = attrs;
    dtype = dtype || util.inferDtype(value);
    if (dtype === 'string') {
        // String type should be handled in CPU memory.
        const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
        values.fill(value);
        return backend.makeTensorInfo(shape, dtype, values);
    }
    else {
        const program = new FillProgram(shape);
        const uniformData = [{ type: 'float32', data: [value] }];
        return backend.runWebGPUProgram(program, [], dtype, uniformData);
    }
}
const fillConfig = {
    kernelName: Fill,
    backendName: 'webgpu',
    kernelFunc: fill
};

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
function reshape(args) {
    const { inputs, attrs } = args;
    const { x } = inputs;
    const { shape } = attrs;
    const xSize = util.sizeFromShape(x.shape);
    const $shape = util.inferFromImplicitShape(shape, xSize);
    const $xSize = util.sizeFromShape($shape);
    util.assert(xSize === $xSize, () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
        `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
        `shape must have the same number of elements.`);
    // Backend needs to track refCount for the dataId for reshape op
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: $shape, dtype: x.dtype };
}
const reshapeConfig = {
    kernelName: Reshape,
    backendName: 'webgpu',
    kernelFunc: reshape
};

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
function batchMatMulImpl({ a, b, transposeA, transposeB, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
    const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];
    const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
    const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];
    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);
    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);
    const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(a.shape.slice(0, -2), b.shape.slice(0, -2));
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);
    util.assert(innerShapeA === innerShapeB, () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
        `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
        `${b.shape} and transposeA=${transposeA}` +
        ` and transposeB=${transposeB} must match.`);
    const a3dShape = transposeA ?
        [batchDimA, innerShapeA, outerShapeA] :
        [batchDimA, outerShapeA, innerShapeA];
    const b3dShape = transposeB ?
        [batchDimB, outerShapeB, innerShapeB] :
        [batchDimB, innerShapeB, outerShapeB];
    // The rest of the implementation is designed to operate on rank-3 tensors
    const a3d = reshape({ inputs: { x: a }, backend, attrs: { shape: a3dShape } });
    const b3d = reshape({ inputs: { x: b }, backend, attrs: { shape: b3dShape } });
    const intermediates = [a3d, b3d];
    const batchDim = Math.max(batchDimA, batchDimB);
    const inputs = [a3d, b3d];
    const dimensions = [
        { type: 'int32', data: [outerShapeA] }, { type: 'int32', data: [outerShapeB] },
        { type: 'int32', data: [innerShapeA] }
    ];
    let program;
    let out;
    const outputShape = [batchDim, outerShapeA, outerShapeB];
    let matmulProgramType = env().get('WEBGPU_MATMUL_PROGRAM_TYPE');
    if (matmulProgramType < 0) {
        // Usually increasing workgroups is a good way to gain more performance for
        // few workgroups by tiling 32x32 (default matmul algorithm). Currently,
        // there are three ways to increase workgroups. 1) MatMulReduceProgram,
        // which is used only when the output size is very small (128 for now). 2)
        // MatMulSplitKProgram, increasing workgroups by spliting K. 3)
        // MatMulSmallOutputSizeProgram, increasing workgroups by small tile size.
        // For different devices, the minimum optimal workgroups may be different.
        // So here we set a |thresholdToIncreaseWorkgroups| to indicate whether we
        // need to increase workgroups. And the literal number is an empirical
        // value.
        const thresholdFlagValue = env().getNumber('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL');
        const thresholdToIncreaseWorkgroups = thresholdFlagValue > 0 ?
            thresholdFlagValue :
            backend.thresholdToIncreaseWorkgroups;
        const workgroupsBy32x32 = batchDim * Math.ceil(outerShapeA / 32) * Math.ceil(outerShapeB / 32);
        const hasFewWorkgroups = workgroupsBy32x32 <= thresholdToIncreaseWorkgroups ||
            (outerShapeA <= 8 &&
                workgroupsBy32x32 <= thresholdToIncreaseWorkgroups * 2);
        if (hasFewWorkgroups) {
            if (batchDim * outerShapeA * outerShapeB <= 128) {
                matmulProgramType = MatMulProgramType.MatMulReduceProgram;
            }
            else if (batchDim === 1 && innerShapeB >= 2000) {
                matmulProgramType = MatMulProgramType.MatMulSplitKProgram;
            }
            else {
                matmulProgramType = MatMulProgramType.MatMulSmallOutputSizeProgram;
            }
        }
        else {
            matmulProgramType = MatMulProgramType.MatMulPackedProgram;
        }
    }
    switch (matmulProgramType) {
        case MatMulProgramType.MatMulReduceProgram:
            program = new MatMulReduceProgram(outputShape, transposeA, transposeB, bias, activation, preluActivationWeights);
            break;
        case MatMulProgramType.MatMulSplitKProgram: {
            // The output buffer must be initailzed to zero before using since we
            // use atomicAdd in MatMulSplitKProgram.
            out = fill({ backend, attrs: { shape: outputShape, value: 0, dtype: a.dtype } });
            program = new MatMulSplitKProgram(outputShape, innerShapeB, transposeA, transposeB);
            if (bias || activation) {
                out =
                    backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
                const biasActivationProgram = new BiasActivationProgram(out.shape, bias, activation, preluActivationWeights);
                let uniformData = null;
                const activationInputs = [out];
                if (bias) {
                    activationInputs.push(bias);
                }
                if (preluActivationWeights) {
                    activationInputs.push(preluActivationWeights);
                }
                if (activation === 'leakyrelu') {
                    uniformData = [{ type: 'float32', data: [leakyreluAlpha] }];
                    biasActivationProgram.uniforms += ' alpha : f32,';
                }
                const outActivated = backend.runWebGPUProgram(biasActivationProgram, activationInputs, out.dtype, uniformData);
                intermediates.push(out);
                const outReshaped = reshape({ inputs: { x: outActivated }, backend, attrs: { shape: outShape } });
                intermediates.push(outActivated);
                for (const i of intermediates) {
                    backend.disposeData(i.dataId);
                }
                return outReshaped;
            }
            break;
        }
        case MatMulProgramType.MatMulSmallOutputSizeProgram:
            program = new MatMulSmallOutputSizeProgram(a3dShape, b3dShape, outputShape, transposeA, transposeB, bias, activation, preluActivationWeights);
            break;
        case MatMulProgramType.MatMulPackedProgram:
            // Experiments show that sequential access is more friendly for Intel
            // GPUs.
            const sequentialAccessByThreads = backend.adapterInfo.isIntel();
            program = new MatMulPackedProgram(a3dShape, outputShape, transposeA, transposeB, bias, activation, preluActivationWeights, sequentialAccessByThreads);
            break;
        default:
            throw new Error(`Unsupported MatMulProgramType ${matmulProgramType}.`);
    }
    if (bias) {
        inputs.push(bias);
    }
    if (preluActivationWeights) {
        inputs.push(preluActivationWeights);
    }
    if (activation === 'leakyrelu') {
        dimensions.push({ type: 'float32', data: [leakyreluAlpha] });
        program.uniforms += ' alpha : f32,';
    }
    out = backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
    const outReshaped = reshape({ inputs: { x: out }, backend, attrs: { shape: outShape } });
    intermediates.push(out);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return outReshaped;
}

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
function _fusedMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b, bias, preluActivationWeights } = inputs;
    const { transposeA, transposeB, activation, leakyreluAlpha } = attrs;
    return batchMatMulImpl({
        a,
        b,
        transposeA,
        transposeB,
        backend,
        bias,
        preluActivationWeights,
        leakyreluAlpha,
        activation
    });
}
const _fusedMatMulConfig = {
    kernelName: _FusedMatMul,
    backendName: 'webgpu',
    kernelFunc: _fusedMatMul,
};

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
class BinaryOpComplexProgram {
    constructor(op, aShape, bShape) {
        this.variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `binaryOpComplex_${op}`;
        this.op = op;
    }
    getUserCode() {
        const opStr = getBinaryOpString(this.op, false);
        const userCode = `
      fn binaryOpComplex(
          areal : f32, aimag : f32, breal : f32, bimag : f32) -> f32 {
        ${opStr}
      }

      ${getMainHeaderString('index')} {
        if(index < uniforms.size) {
          let areal = getARealByOutputIndex(index);
          let aimag = getAImagByOutputIndex(index);
          let breal = getBRealByOutputIndex(index);
          let bimag = getBImagByOutputIndex(index);
          setOutputAtIndex(index, binaryOpComplex(areal, aimag, breal, bimag));
        }
      }
    `;
        return userCode;
    }
}

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
class BinaryOpProgram {
    constructor(op, aShape, bShape) {
        this.size = true;
        this.variableNames = ['A', 'B'];
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.op = op;
        this.useSharedMemoryWithA =
            aShape.length <= 1 && bShape.length > 1 && aShape[0] < 128;
        this.useSharedMemoryWithB =
            bShape.length <= 1 && aShape.length > 1 && bShape[0] < 128;
        if (this.useSharedMemoryWithA || this.useSharedMemoryWithB) {
            this.outputComponent = 1;
            this.variableComponents = [1, 1];
            // lastDimensionSize is used as sharedBuf array size, so can not be
            // used as uniform.
            this.lastDimensionSize =
                this.useSharedMemoryWithB ? bShape[0] : aShape[0];
            this.shaderKey = `binary_${op}_${this.lastDimensionSize}`;
            this.type = 'shared';
            // This is an experimental value when using shared memory.
            // Note that the maximum of workgroup X dimension is 256.
            this.workgroupSize = [256, 1, 1];
        }
        else {
            const aDivisibleBy4 = aShape.length > 0 && aShape[aShape.length - 1] % 4 === 0;
            const bDivisibleBy4 = bShape.length > 0 && bShape[bShape.length - 1] % 4 === 0;
            if (aDivisibleBy4 && bDivisibleBy4) {
                this.outputComponent = 4;
                this.variableComponents = [4, 4];
            }
            else if ((aDivisibleBy4 &&
                (util.isScalarShape(bShape) || bShape[bShape.length - 1] === 1)) ||
                (bDivisibleBy4 &&
                    (util.isScalarShape(aShape) || aShape[aShape.length - 1] === 1))) {
                this.outputComponent = 4;
                this.variableComponents = aDivisibleBy4 ? [4, 1] : [1, 4];
            }
            else {
                this.outputComponent = 1;
                this.variableComponents = [1, 1];
            }
            this.type = 'nonshared';
            this.shaderKey = `binary_${op}_${this.variableComponents}`;
            // TODO(jiajia.qin@intel.com): Heuristically select a good work group
            // size.
            this.workgroupSize = [128, 1, 1];
        }
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.outputComponent, 1, 1]);
    }
    getUserCode() {
        let userCode;
        const dType = this.outputComponent === 4 ? 'vec4<f32>' : 'f32';
        const opFnStr = `
    fn binaryOperation(a : ${dType}, b : ${dType}) -> ${dType} {
      ${getBinaryOpString(this.op, this.outputComponent === 4)}
    };
    `;
        if (this.type === 'shared') {
            const sharedIndexSnippet = this.lastDimensionSize > 1 ?
                `coords[${this.outputShape.length - 1}]` :
                '0';
            const accessDataSnippet = this.useSharedMemoryWithB ?
                `let a = getAByOutputIndex(index);
          let b = sharedBuf[${sharedIndexSnippet}];` :
                `let a = sharedBuf[${sharedIndexSnippet}];
          let b = getBByOutputIndex(index);`;
            userCode = `
        ${opFnStr}
        var<workgroup> sharedBuf : array<f32, ${this.lastDimensionSize}>;
        ${getMainHeaderString('index')} {
          // Fill in the shared memory buffer.
          let localIndex = i32(localId.x);
          if(localIndex < ${this.lastDimensionSize}) {
            sharedBuf[localIndex] = f32(${this.useSharedMemoryWithB ? 'B' : 'A'}[localIndex]);
          }
          workgroupBarrier();

          if(index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            ${accessDataSnippet}
            setOutputAtIndex(index, binaryOperation(a, b));
          }
        }
        `;
        }
        else {
            userCode = `
       ${opFnStr}
       ${getMainHeaderString('index')} {
         if (index < uniforms.size) {
           let coords = getCoordsFromIndex(index * ${this.outputComponent});
           let a = ${dType}(getAByOutputCoords(coords));
           let b = ${dType}(getBByOutputCoords(coords));
           setOutputAtIndex(index, binaryOperation(a, b));
         }
       }
       `;
        }
        return userCode;
    }
}

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
function identity(args) {
    const { inputs } = args;
    const { x } = inputs;
    args.backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: x.shape, dtype: x.dtype };
}
const identityConfig = {
    kernelName: Identity,
    backendName: 'webgpu',
    kernelFunc: identity
};

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
/**
 * Complex tensors share data with their real and imaginary components. Complex
 * tensors' reference to the components is tracked by refCount on the individual
 * component. The refCounts are increased by the identity call.
 *
 * When a complex tensor is disposed, it will reduce the refCount on the
 * components by calling disposeData on each.
 */
function complex(args) {
    const { inputs, backend } = args;
    const { real, imag } = inputs;
    const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
    const complex = backend.tensorMap.get(complexInfo.dataId);
    const realTensorInfo = identity({ inputs: { x: real }, backend });
    const imagTensorInfo = identity({ inputs: { x: imag }, backend });
    complex.complexTensorInfos = { real: realTensorInfo, imag: imagTensorInfo };
    return complexInfo;
}
const complexConfig = {
    kernelName: Complex,
    backendName: 'webgpu',
    kernelFunc: complex
};

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
class UnaryOpProgram {
    constructor(outputShape, op, uniforms = '') {
        this.variableNames = ['A'];
        this.size = true;
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
        const workgroupSizeX = 128;
        this.workgroupSize = [workgroupSizeX, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.op = op;
        if (uniforms !== '') {
            this.uniforms = uniforms;
        }
        this.shaderKey = `unary_${op}`;
    }
    getUserCode() {
        return `
      fn unaryOperation(a : f32) -> f32 {
        ${getUnaryOpString(this.op, false)}
      }
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let a = getAByOutputIndex(index);
          setOutputAtIndex(index, unaryOperation(a));
        }
      }
      `;
    }
}

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
/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param opType Op type to create `UnaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function unaryKernelFunc({ opType, cpuKernelImpl, dtype }) {
    return ({ inputs, backend }) => {
        const { x } = inputs;
        const webgpuBackend = backend;
        const $dtype = dtype || x.dtype;
        if (webgpuBackend.shouldExecuteOnCPU([x]) && cpuKernelImpl != null) {
            const xData = webgpuBackend.tensorMap.get(x.dataId);
            const outValues = cpuKernelImpl(xData.values, $dtype);
            return webgpuBackend.makeTensorInfo(x.shape, $dtype, outValues);
        }
        const program = new UnaryOpProgram(x.shape, opType);
        return webgpuBackend.runWebGPUProgram(program, [x], $dtype);
    };
}
/**
 * Template that creates a `KernelFunc` for binary ops.
 * @param opType Op type to create `BinaryOpProgram`.
 * @param cpuKernelImpl Optional. Shared functionality from tfjs-backend-cpu, it
 *     will be involved when necessary.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function binaryKernelFunc({ opType, cpuKernelImpl, supportsComplex = false, dtype }) {
    return ({ inputs, backend }) => {
        const { a, b } = inputs;
        const webgpuBackend = backend;
        if (supportsComplex && a.dtype === 'complex64') {
            const aData = webgpuBackend.tensorMap.get(a.dataId);
            const bData = webgpuBackend.tensorMap.get(b.dataId);
            let real, imag;
            if (opType !== BinaryOpType.MUL) {
                [real, imag] = [
                    [aData.complexTensorInfos.real, bData.complexTensorInfos.real],
                    [aData.complexTensorInfos.imag, bData.complexTensorInfos.imag]
                ].map(complexParts => {
                    const [aPart, bPart] = complexParts;
                    const aHandle = {
                        dataId: aPart.dataId,
                        dtype: aPart.dtype,
                        shape: a.shape
                    };
                    const bHandle = {
                        dataId: bPart.dataId,
                        dtype: bPart.dtype,
                        shape: b.shape
                    };
                    const program = new BinaryOpProgram(opType, a.shape, b.shape);
                    return webgpuBackend.runWebGPUProgram(program, [aHandle, bHandle], upcastType(aPart.dtype, bPart.dtype));
                });
            }
            else {
                const realProgram = new BinaryOpComplexProgram(BinaryOpType.COMPLEX_MULTIPLY_REAL, a.shape, b.shape);
                const imagProgram = new BinaryOpComplexProgram(BinaryOpType.COMPLEX_MULTIPLY_IMAG, a.shape, b.shape);
                const inputs = [
                    {
                        dataId: aData.complexTensorInfos.real.dataId,
                        dtype: aData.complexTensorInfos.real.dtype,
                        shape: a.shape
                    },
                    {
                        dataId: aData.complexTensorInfos.imag.dataId,
                        dtype: aData.complexTensorInfos.imag.dtype,
                        shape: a.shape
                    },
                    {
                        dataId: bData.complexTensorInfos.real.dataId,
                        dtype: bData.complexTensorInfos.real.dtype,
                        shape: b.shape
                    },
                    {
                        dataId: bData.complexTensorInfos.imag.dataId,
                        dtype: bData.complexTensorInfos.imag.dtype,
                        shape: b.shape
                    }
                ];
                real = webgpuBackend.runWebGPUProgram(realProgram, inputs, 'float32');
                imag = webgpuBackend.runWebGPUProgram(imagProgram, inputs, 'float32');
            }
            const complexOutput = complex({ inputs: { real, imag }, backend: webgpuBackend });
            webgpuBackend.disposeData(real.dataId);
            webgpuBackend.disposeData(imag.dataId);
            // TODO: Implement CPU forwarding for complex inputs.
            return complexOutput;
        }
        const $dtype = dtype || upcastType(a.dtype, b.dtype);
        if ((a.dtype === 'string' || b.dtype === 'string' ||
            webgpuBackend.shouldExecuteOnCPU([a, b])) &&
            cpuKernelImpl != null) {
            const aData = webgpuBackend.tensorMap.get(a.dataId).values;
            const bData = webgpuBackend.tensorMap.get(b.dataId).values;
            const decodedAVals = a.dtype === 'string' ?
                // tslint:disable-next-line: no-any
                backend_util.fromUint8ToStringArray(aData) :
                aData;
            const decodedBVals = a.dtype === 'string' ?
                // tslint:disable-next-line: no-any
                backend_util.fromUint8ToStringArray(bData) :
                bData;
            const [outValues, outShape] = cpuKernelImpl(a.shape, b.shape, decodedAVals, decodedBVals, $dtype);
            return webgpuBackend.makeTensorInfo(outShape, $dtype, outValues);
        }
        const program = new BinaryOpProgram(opType, a.shape, b.shape);
        return webgpuBackend.runWebGPUProgram(program, [a, b], $dtype);
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function simpleAbsImpl(vals) {
    const resultValues = new Float32Array(vals.length);
    for (let i = 0; i < vals.length; ++i) {
        resultValues[i] = Math.abs(vals[i]);
    }
    return resultValues;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
/**
 * Template that creates implementation for binary ops. Supports broadcast.
 */
function createSimpleBinaryKernelImpl(op) {
    return (aShape, bShape, aVals, bVals, dtype) => {
        const newShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        const resultRank = newShape.length;
        const resultStrides = util.computeStrides(newShape);
        const resultSize = util.sizeFromShape(newShape);
        const result = util.getTypedArrayFromDType(dtype, resultSize);
        const aRank = aShape.length;
        const bRank = bShape.length;
        const aStrides = util.computeStrides(aShape);
        const bStrides = util.computeStrides(bShape);
        const aBroadcastDims = backend_util.getBroadcastDims(aShape, newShape);
        const bBroadcastDims = backend_util.getBroadcastDims(bShape, newShape);
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (let i = 0; i < result.length; ++i) {
                result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
            }
        }
        else {
            for (let i = 0; i < result.length; ++i) {
                const loc = util.indexToLoc(i, resultRank, resultStrides);
                const aLoc = loc.slice(-aRank);
                aBroadcastDims.forEach(d => aLoc[d] = 0);
                const aIndex = util.locToIndex(aLoc, aRank, aStrides);
                const bLoc = loc.slice(-bRank);
                bBroadcastDims.forEach(d => bLoc[d] = 0);
                const bIndex = util.locToIndex(bLoc, bRank, bStrides);
                result[i] = op(aVals[aIndex], bVals[bIndex]);
            }
        }
        return [result, newShape];
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function castImpl(values, shape, inputType, dtype) {
    if (dtype === 'int32') {
        const resultValues = Int32Array.from(values);
        return [shape, 'int32', resultValues];
    }
    if (dtype === 'bool') {
        // This is essentially the result of notEqual(x, 0). We avoid using
        // kernel notEqual to avoid circular dependency, i.e. binary_utils ->
        // cast -> notEqual -> binary_utils.
        const zero = util.toTypedArray([0], inputType);
        const [resultData, resultShape] = createSimpleBinaryKernelImpl((a, b) => (a !== b) ? 1 : 0)(shape, [], values, zero, 'bool');
        return [resultShape, 'bool', resultData];
    }
    throw new Error(`Error in Cast: failed to cast ${inputType} to ${dtype}`);
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const addImpl = createSimpleBinaryKernelImpl(((a, b) => a + b));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function bincountImpl(xVals, weightsVals, weightsDtype, weightsShape, size) {
    const weightsSize = util.sizeFromShape(weightsShape);
    const outVals = util.makeZerosTypedArray(size, weightsDtype);
    for (let i = 0; i < xVals.length; i++) {
        const value = xVals[i];
        if (value < 0) {
            throw new Error('Input x must be non-negative!');
        }
        if (value >= size) {
            continue;
        }
        if (weightsSize > 0) {
            outVals[value] += weightsVals[i];
        }
        else {
            outVals[value] += 1;
        }
    }
    return outVals;
}
function bincountReduceImpl(xBuf, weightsBuf, size, binaryOutput = false) {
    const numRows = xBuf.shape[0];
    const numCols = xBuf.shape[1];
    const outBuf = buffer([numRows, size], weightsBuf.dtype);
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const value = xBuf.get(i, j);
            if (value < 0) {
                throw new Error('Input x must be non-negative!');
            }
            if (value >= size) {
                continue;
            }
            if (binaryOutput) {
                outBuf.set(1, i, value);
            }
            else {
                if (weightsBuf.size > 0) {
                    outBuf.set(outBuf.get(i, value) + weightsBuf.get(i, j), i, value);
                }
                else {
                    outBuf.set(outBuf.get(i, value) + 1, i, value);
                }
            }
        }
    }
    return outBuf;
}

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
const bitwiseAndImpl = createSimpleBinaryKernelImpl(((a, b) => a & b));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
/**
 * Template that creates implementation for unary op.
 */
function createSimpleUnaryImpl(op) {
    return (values, dtype, attrs) => {
        const newValues = util.getArrayFromDType(dtype, values.length);
        for (let i = 0; i < values.length; ++i) {
            newValues[i] = op(values[i], attrs);
        }
        return newValues;
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const ceilImpl = createSimpleUnaryImpl((xi) => Math.ceil(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function concatImpl$1(inputs, outShape, dtype, simplyConcat) {
    const outVals = util.getArrayFromDType(dtype, util.sizeFromShape(outShape));
    if (simplyConcat && dtype !== 'string') {
        // Use built-in TypedArray.set() method for speed.
        let offset = 0;
        inputs.forEach(input => {
            const size = util.sizeFromShape(input.shape);
            outVals.set(input.vals, offset);
            offset += size;
        });
    }
    else {
        let colOffset = 0;
        inputs.forEach(input => {
            const decodedData = dtype === 'string' ?
                backend_util.fromUint8ToStringArray(input.vals) :
                input.vals;
            let tIdx = 0;
            for (let row = 0; row < input.shape[0]; ++row) {
                const resIdx = row * outShape[1] + colOffset;
                for (let col = 0; col < input.shape[1]; ++col) {
                    outVals[resIdx + col] = decodedData[tIdx++];
                }
            }
            colOffset += input.shape[1];
        });
    }
    return outVals;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const equalImpl = createSimpleBinaryKernelImpl((a, b) => (a === b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const expImpl = createSimpleUnaryImpl((xi) => Math.exp(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const expm1Impl = createSimpleUnaryImpl((xi) => Math.expm1(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const floorImpl = createSimpleUnaryImpl((xi) => Math.floor(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const floorDivImpl = createSimpleBinaryKernelImpl((a, b) => Math.floor(a / b));

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
function gatherNdImpl(indicesData, paramsBuf, dtype, numSlices, sliceRank, sliceSize, strides, paramsShape, paramsSize) {
    const outBuf = buffer([numSlices, sliceSize], dtype);
    for (let i = 0; i < numSlices; i++) {
        const index = [];
        let flattenIndex = 0;
        for (let j = 0; j < sliceRank; j++) {
            const dim = indicesData[i * sliceRank + j];
            flattenIndex += dim * strides[j];
            index.push(dim);
        }
        if (flattenIndex < 0 || flattenIndex >= paramsSize / sliceSize) {
            throw new Error(`Invalid indices: ${index} does not index into ${paramsShape}`);
        }
        for (let k = 0; k < sliceSize; k++) {
            outBuf.values[i * sliceSize + k] =
                paramsBuf.get(...paramsBuf.indexToLoc(flattenIndex * sliceSize + k));
        }
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function gatherV2Impl(xBuf, indicesBuf, flattenOutputShape) {
    const outBuf = buffer(flattenOutputShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const newLoc = outBuf.indexToLoc(i);
        const originalLoc = newLoc.slice();
        const batchIdx = originalLoc[0];
        const indicesIdx = originalLoc[2];
        const indicesIndex = indicesBuf.locToIndex([batchIdx, indicesIdx]);
        originalLoc[2] = indicesBuf.values[indicesIndex];
        const originalIndex = xBuf.locToIndex(originalLoc);
        if (0 <= originalIndex && originalIndex < xBuf.values.length) {
            outBuf.values[i] = xBuf.values[originalIndex];
        } // Else, index is out of bounds, so leave the default zero val in outBuf.
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const greaterImpl = createSimpleBinaryKernelImpl((a, b) => (a > b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const greaterEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a >= b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const lessImpl = createSimpleBinaryKernelImpl((a, b) => (a < b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const lessEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a <= b) ? 1 : 0);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function linSpaceImpl(start, stop, num) {
    const step = (stop - start) / (num - 1);
    const values = util.makeZerosTypedArray(num, 'float32');
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logImpl = createSimpleUnaryImpl((xi) => Math.log(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function maxImpl(aVals, reduceSize, outShape, dtype) {
    const vals = util.getTypedArrayFromDType(dtype, util.sizeFromShape(outShape));
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let max = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (Number.isNaN(value) ||
                value > max) { // comparison with NaN always return false
                max = value;
            }
        }
        vals[i] = max;
    }
    return vals;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const maximumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.max(aValue, bValue)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const minimumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.min(aValue, bValue)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const multiplyImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue * bValue));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function negImpl(xVals, xShape, xDtype) {
    const minusOne = util.createScalarValue(-1, xDtype);
    return multiplyImpl([], xShape, minusOne, xVals, xDtype);
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const notEqualImpl = createSimpleBinaryKernelImpl(((a, b) => (a !== b) ? 1 : 0));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function transposeImpl(xVals, xShape, dtype, perm, newShape) {
    const xRank = xShape.length;
    const xSize = util.sizeFromShape(xShape);
    const xStrides = util.computeStrides(xShape);
    const newStrides = util.computeStrides(newShape);
    const result = util.getTypedArrayFromDType(dtype, util.sizeFromShape(newShape));
    for (let i = 0; i < xSize; ++i) {
        const loc = util.indexToLoc(i, xRank, xStrides);
        // Permute location.
        const newLoc = new Array(loc.length);
        for (let i = 0; i < newLoc.length; i++) {
            newLoc[i] = loc[perm[i]];
        }
        const newIndex = util.locToIndex(newLoc, xRank, newStrides);
        result[newIndex] = xVals[i];
    }
    return result;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function prodImpl(xShape, xDtype, xVals, reductionAxes) {
    const [outShape, reduceShape] = backend_util.computeOutAndReduceShapes(xShape, reductionAxes);
    const outDtype = upcastType(xDtype, 'int32');
    const outVals = util.makeZerosTypedArray(util.sizeFromShape(outShape), outDtype);
    const reduceSize = util.sizeFromShape(reduceShape);
    for (let i = 0; i < outVals.length; ++i) {
        const offset = i * reduceSize;
        let prod = 1;
        for (let j = 0; j < reduceSize; ++j) {
            prod *= xVals[offset + j];
        }
        outVals[i] = prod;
    }
    return { outVals, outShape, outDtype };
}

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
function validateIndices(indices, indicesShape, numParams) {
    indices.forEach((index, i) => {
        if (index < 0 || index >= numParams) {
            const locString = util.indexToLoc(i, indicesShape.length, util.computeStrides(indicesShape))
                .join(',');
            throw new Error(`indices[${locString}] = ${index} is not in [0, ${numParams})`);
        }
    });
}
function validateSplits(paramsNestedSplits, numParamsDenseValues) {
    // Validate
    for (let dim = 0; dim < paramsNestedSplits.length; ++dim) {
        const splits = paramsNestedSplits[dim];
        const lastSplit = (dim === paramsNestedSplits.length - 1) ?
            numParamsDenseValues :
            paramsNestedSplits[dim + 1].length;
        if (splits.length === 0) {
            throw new Error('Ragged splits may not be empty');
        }
        if (splits[0] < 0) {
            throw new Error('Ragged splits must be non-negative');
        }
        if (splits[splits.length - 1] > lastSplit) {
            throw new Error('Ragged splits must not point past values');
        }
        for (let i = 1; i < splits.length; ++i) {
            if (splits[i - 1] > splits[i]) {
                throw new Error('Ragged splits must be sorted in ascending order');
            }
        }
    }
}
// Construct the `splits` output tensors, encoded using a nested vector.
// Also find the slices of values that need to be copied, and store them
// in `valueSlices`.  The total number of values that will be copied (which
// we need for allocating the output values tensor) is stored in `numValues`.
function makeSplits(indices, indicesShape, paramsNestedSplits, numParamsDenseValues) {
    const valueSlices = [];
    let numValues = 0;
    const numSplits = indicesShape.length - 1 + paramsNestedSplits.length;
    const outSplits = new Array(numSplits).fill(null).map(() => [0]);
    validateSplits(paramsNestedSplits, numParamsDenseValues);
    // Add `splits` that come from all but the last dimension of the dense
    // Tensor `indices`.  In particular, for each dimension D, we add a
    // splits tensor whose values are:
    //   range(reduceProd(splits.shape[:D]) + 1) * splits.shape[D+1]
    // E.g., if indices.shape=[2, 3, 4] then we will add splits tensors:
    //   [0, 3, 6]                    # length=2+1, stride=3
    //   [0, 4, 8, 12, 16, 20, 24]    # length=2*3+1, stride=4
    let nrows = 1;
    for (let dim = 0; dim < indicesShape.length - 1; ++dim) {
        nrows *= indicesShape[dim];
        const rowLength = indicesShape[dim + 1];
        for (let i = 1; i < nrows + 1; ++i) {
            outSplits[dim].push(i * rowLength);
        }
    }
    // Add `splits` that come from `paramsNestedSplits`.  Starting with the
    // outermost ragged dimension (i.e., the first `splits` tensor), we work
    // our way in, finding the range of values that should be copied.  As we
    // go, we update the output `splits` for each dimension with the appropriate
    // values.  In particular, the *lengths* of the slices from `param_splits`
    // should be copied to generate corresponding slice lengths in the output
    // splits.  E.g., if we are copying a ragged row with length 4, then we
    // should add a new split point to outSplits that is 4 greater than the
    // previous split point in outSplits.
    for (let i = 0; i < indices.length; ++i) {
        let start = indices[i];
        let limit = indices[i] + 1;
        // Copy splits.
        for (let dim = 0; dim < paramsNestedSplits.length; ++dim) {
            const splits = paramsNestedSplits[dim];
            const outDim = dim + indicesShape.length - 1;
            if (outDim >= 0) {
                const outSplitsOutDim = outSplits[outDim];
                const delta = outSplitsOutDim[outSplitsOutDim.length - 1] - splits[start];
                for (let j = start; j < limit; ++j) {
                    outSplits[outDim].push(splits[j + 1] + delta);
                }
            }
            start = splits[start];
            limit = splits[limit];
        }
        if (limit !== start) {
            valueSlices.push([start, limit]);
            numValues += limit - start;
        }
    }
    return { outSplits, valueSlices, numValues };
}
function getSplits(outSplits) {
    const splitsOut = [];
    for (let i = 0; i < outSplits.length; ++i) {
        const numSplits = outSplits[i].length;
        const splits = util.getArrayFromDType('int32', numSplits);
        splitsOut.push(splits);
        outSplits[i].forEach((value, j) => splits[j] = value);
    }
    return splitsOut;
}
function computeFlatOuterDims(orig, numOutDims) {
    const outDims = orig.slice(0, numOutDims);
    while (outDims.length < numOutDims) {
        outDims.push(1);
    }
    for (let inDim = numOutDims; inDim < orig.length; inDim++) {
        outDims[numOutDims - 1] *= orig[inDim];
    }
    return outDims;
}
// For each slice in `(start, limit)` in `valueSlices`, append
// `paramsDenseValues[start,...,limit] to `values`.  `valueSize` indicates
// the number of scalars contained in each value paramsDenseValues[i].
function writeValueSlices(paramsDenseValues, paramsDenseValuesShape, valueSlices, valueSize, values, valuesShape) {
    const denseM = computeFlatOuterDims(paramsDenseValuesShape, 2)[1];
    const valuesM = computeFlatOuterDims(valuesShape, 2)[1];
    let outPos = 0;
    for (const slice of valueSlices) {
        for (let i = slice[0]; i < slice[1]; ++i) {
            for (let j = 0; j < valueSize; ++j) {
                values[outPos * valuesM + j] = paramsDenseValues[i * denseM + j];
            }
            ++outPos;
        }
    }
}
function getValues(paramsDenseValues, paramsDenseValuesShape, paramsDenseValuesDType, valueSlices, numValues) {
    const valuesShape = paramsDenseValuesShape.slice();
    valuesShape[0] = numValues;
    const valuesOut = util.getArrayFromDType(paramsDenseValuesDType, util.sizeFromShape(valuesShape));
    const numElements = paramsDenseValues.length;
    const valueSize = numElements === 0 ? 0 : (numElements / paramsDenseValuesShape[0]);
    writeValueSlices(paramsDenseValues, paramsDenseValuesShape, valueSlices, valueSize, valuesOut, valuesShape);
    return [valuesOut, valuesShape];
}
function raggedGatherImpl(paramsNestedSplits, paramsNestedSplitsShapes, paramsDenseValues, paramsDenseValuesShape, paramsDenseValuesDType, indices, indicesShape, outputRaggedRank) {
    if (paramsNestedSplits.length === 0) {
        throw new Error('paramsNestedSplits must be non empty');
    }
    if (paramsNestedSplitsShapes[0].length === 0) {
        throw new Error('Split tensors must not be scalars');
    }
    const numParams = paramsNestedSplitsShapes[0][0] - 1;
    validateIndices(indices, indicesShape, numParams);
    if (paramsDenseValuesShape.length === 0) {
        throw new Error('params.rank must be nonzero');
    }
    const numParamsDenseValues = paramsDenseValuesShape[0];
    // Calculate the `splits`, and store the value slices that we need to
    // copy in `valueSlices`.
    const { outSplits, valueSlices, numValues } = makeSplits(indices, indicesShape, paramsNestedSplits, numParamsDenseValues);
    // Write the output tensors.
    const outputNestedSplits = getSplits(outSplits);
    const outputDenseValues = getValues(paramsDenseValues, paramsDenseValuesShape, paramsDenseValuesDType, valueSlices, numValues);
    return [outputNestedSplits, outputDenseValues[0], outputDenseValues[1]];
}

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
const INT32_MAX = 2147483647;
function raggedRangeImpl(starts, startsShape, startsDType, limits, limitsShape, deltas, deltasShape) {
    // Check input tensor shapes.
    if (startsShape.length > 1) {
        throw new Error('starts must be a scalar or vector');
    }
    if (limitsShape.length > 1) {
        throw new Error('limits must be a scalar or vector');
    }
    if (deltasShape.length > 1) {
        throw new Error('deltas must be a scalar or vector');
    }
    // Determine which tensors we need to broadcast.
    const broadcastStarts = startsShape.length === 0;
    const broadcastLimits = limitsShape.length === 0;
    const broadcastDeltas = deltasShape.length === 0;
    // nRows (number of output rows) is the size of the non-broadcast inputs,
    // or 1 if all inputs are scalars.
    const inSizes = [];
    if (!broadcastStarts) {
        inSizes.push(startsShape[0]);
    }
    if (!broadcastLimits) {
        inSizes.push(limitsShape[0]);
    }
    if (!broadcastDeltas) {
        inSizes.push(deltasShape[0]);
    }
    for (let i = 1; i < inSizes.length; ++i) {
        if (inSizes[i] !== inSizes[i - 1]) {
            throw new Error('starts, limits, and deltas must have the same shape');
        }
    }
    const nRows = inSizes.length === 0 ? 1 : inSizes[0];
    // Construct the rtNestedSplits tensor.
    const rtNestedSplits = util.getArrayFromDType('int32', nRows + 1);
    rtNestedSplits[0] = 0;
    for (let row = 0; row < nRows; ++row) {
        const start = broadcastStarts ? starts[0] : starts[row];
        const limit = broadcastLimits ? limits[0] : limits[row];
        const delta = broadcastDeltas ? deltas[0] : deltas[row];
        if (delta === 0) {
            throw new Error('Requires delta != 0');
        }
        let size; // The number of elements in the specified range.
        if (((delta > 0) && (limit < start)) || ((delta < 0) && (limit > start))) {
            size = 0;
        }
        else {
            size = Math.ceil(Math.abs((limit - start) / delta));
            if (size > INT32_MAX) {
                throw new Error(`Requires ((limit - start) / delta) <= ${INT32_MAX}`);
            }
        }
        rtNestedSplits[row + 1] = rtNestedSplits[row] + size;
    }
    const nVals = rtNestedSplits[nRows];
    // Construct the rtDenseValues tensor.
    const rtDenseValues = util.getArrayFromDType(startsDType, nVals);
    let valueIndex = 0;
    for (let row = 0; row < nRows; ++row) {
        const rowSize = rtNestedSplits[row + 1] - rtNestedSplits[row];
        let value = broadcastStarts ? starts[0] : starts[row];
        const delta = broadcastDeltas ? deltas[0] : deltas[row];
        for (let i = 0; i < rowSize; ++i) {
            rtDenseValues[valueIndex++] = value;
            value += delta;
        }
    }
    return [rtNestedSplits, rtDenseValues];
}

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
var RowPartitionType = backend_util.RowPartitionType;
// Based on
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/ragged_tensor_to_tensor_op.cc
class RaggedTensorToTensorOp {
    constructor(shape, shapeShape, values, valuesShape, valuesDType, defaultValue, defaultValueShape, rowPartitionValues, rowPartitionValuesShapes, rowPartitionTypeStrings) {
        this.shape = shape;
        this.shapeShape = shapeShape;
        this.values = values;
        this.valuesShape = valuesShape;
        this.valuesDType = valuesDType;
        this.defaultValue = defaultValue;
        this.defaultValueShape = defaultValueShape;
        this.rowPartitionValues = rowPartitionValues;
        this.rowPartitionValuesShapes = rowPartitionValuesShapes;
        this.rowPartitionTypes =
            backend_util.getRowPartitionTypesHelper(rowPartitionTypeStrings);
        this.raggedRank = backend_util.getRaggedRank(this.rowPartitionTypes);
    }
    getRowPartitionTypeByDimension(dimension) {
        if (this.rowPartitionTypes[0] === RowPartitionType.FIRST_DIM_SIZE) {
            return this.rowPartitionTypes[dimension + 1];
        }
        else {
            return this.rowPartitionTypes[dimension];
        }
    }
    // Returns the relationship between dimension and dimension + 1.
    getRowPartitionTensor(dimension) {
        if (this.rowPartitionTypes[0] === RowPartitionType.FIRST_DIM_SIZE) {
            return this.rowPartitionValues[dimension + 1];
        }
        else {
            return this.rowPartitionValues[dimension];
        }
    }
    getMaxWidth(dimension) {
        const rowPartitionTensor = this.getRowPartitionTensor(dimension - 1);
        switch (this.getRowPartitionTypeByDimension(dimension - 1)) {
            case RowPartitionType.VALUE_ROWIDS:
                return RaggedTensorToTensorOp.getMaxWidthValueRowID(rowPartitionTensor);
            case RowPartitionType.ROW_SPLITS:
                return RaggedTensorToTensorOp.getMaxWidthRowSplit(rowPartitionTensor);
            default:
                throw new Error(`Cannot handle partition type ${RowPartitionType[this.getRowPartitionTypeByDimension(dimension - 1)]}`);
        }
    }
    static getMaxWidthRowSplit(rowSplit) {
        const tensorLength = rowSplit.length;
        if (tensorLength === 0 || tensorLength === 1) {
            return 0;
        }
        let maxWidth = 0;
        for (let i = 0; i < tensorLength - 1; ++i) {
            const currentWidth = rowSplit[i + 1] - rowSplit[i];
            if (currentWidth > maxWidth) {
                maxWidth = currentWidth;
            }
        }
        return maxWidth;
    }
    static getMaxWidthValueRowID(valueRowIds) {
        const indexLength = valueRowIds.length;
        if (indexLength === 0) {
            return 0;
        }
        let firstEqualIndex = 0;
        let firstEqualIndexValue = valueRowIds[0];
        let maxWidth = 0;
        for (let i = 1; i < indexLength; ++i) {
            const value = valueRowIds[i];
            if (value !== firstEqualIndexValue) {
                firstEqualIndexValue = value;
                maxWidth = Math.max(i - firstEqualIndex, maxWidth);
                firstEqualIndex = i;
            }
        }
        return Math.max(indexLength - firstEqualIndex, maxWidth);
    }
    tensorShapeFromTensor(t, tShape, isPartial = true) {
        if (tShape.length === 0) {
            if (t[0] === -1) {
                return [];
            }
            throw new Error(`The only valid scalar shape tensor is the fully unknown shape specified as -1.`);
        }
        // MakePartialShape/MakeShapeHelper.
        return makeShape(t, isPartial);
    }
    calculateOutputSize(firstDim) {
        const valueShape = this.valuesShape;
        const defaultValueShape = this.defaultValueShape;
        backend_util.validateDefaultValueShape(defaultValueShape, valueShape);
        const shape = this.tensorShapeFromTensor(this.shape, this.shapeShape);
        const outputShape = backend_util.combineRaggedTensorToTensorShapes(this.raggedRank, shape, valueShape);
        const result = outputShape;
        if (result[0] < 0) {
            result[0] = firstDim;
        }
        for (let i = 1; i <= this.raggedRank; ++i) {
            if (result[i] < 0) {
                result[i] = this.getMaxWidth(i);
            }
        }
        return result;
    }
    /**
     * The outputIndex represents the index in the output tensor
     * where the first element of a particular dimension would be written.
     * If it is -1, it indicates that the index is out of scope.
     * Example, given firstDimension = 10, firstDimensionOutput = 6,
     * and outputIndexMultiplier = 100:
     * result = [0 100 200 300 400 500 -1 -1 -1 -1]
     * If firstDimensionOutput = 11 instead, then:
     * result = [0 100 200 300 400 500 600 700 800 900]
     */
    calculateFirstParentOutputIndex(firstDimension, outputIndexMultiplier, firstDimensionOutput) {
        const minDimension = Math.min(firstDimension, firstDimensionOutput);
        const result = [];
        let currentOutputIndex = 0;
        for (let i = 0; i < minDimension; ++i, currentOutputIndex += outputIndexMultiplier) {
            result.push(currentOutputIndex);
        }
        for (let i = minDimension; i < firstDimension; ++i) {
            result.push(-1);
        }
        util.assert(result.length === firstDimension, () => 'Final length of result must be equal to firstDimension.');
        return result;
    }
    calculateOutputIndexRowSplit(rowSplit, parentOutputIndex, outputIndexMultiplier, outputSize) {
        const rowSplitSize = rowSplit.length;
        const result = [];
        for (let i = 0; i < rowSplitSize - 1; ++i) {
            const rowLength = rowSplit[i + 1] - rowSplit[i];
            let realLength = Math.min(outputSize, rowLength);
            let parentOutputIndexCurrent = parentOutputIndex[i];
            if (parentOutputIndexCurrent === -1) {
                realLength = 0;
            }
            for (let j = 0; j < realLength; ++j) {
                result.push(parentOutputIndexCurrent);
                parentOutputIndexCurrent += outputIndexMultiplier;
            }
            for (let j = 0; j < rowLength - realLength; ++j) {
                result.push(-1);
            }
        }
        if (rowSplitSize > 0 && result.length !== rowSplit[rowSplitSize - 1]) {
            throw new Error('Invalid row split size.');
        }
        return result;
    }
    // Calculate the output index of the first element of a list.
    // The parentOutputIndex is the same computation for the previous list.
    // -1 indicates an element or list that is out of range.
    // The outputIndexMultiplier is the number of output indices one moves
    // forward for each column.
    // E.g., given:
    // valueRowIds:[0 1 2 2 2 3 5 5 6]
    // parentOutputIndex:[1000 1100 2000 2100 -1 3000 4000]
    // outputIndexMultiplier: 10
    // outputSize: 2
    // You get:
    // result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
    // result[0] = parentOutputIndex[valueRowIds[0]]
    // result[1] = parentOutputIndex[valueRowIds[1]]
    // result[2] = parentOutputIndex[valueRowIds[2]]
    // result[3] = parentOutputIndex[valueRowIds[2] + 10]
    // result[4] = -1 because it is the third element the size is 2.
    // result[5] = parentOutputIndex[valueRowIds[3]]
    // result[6] = -1 because parentOutputIndex[valueRowIds[6]] == -1
    // result[7] = -1 because parentOutputIndex[valueRowIds[6]] == -1
    // result[8] = parentOutputIndex[valueRowIds[7]]
    calculateOutputIndexValueRowID(valueRowIds, parentOutputIndex, outputIndexMultiplier, outputSize) {
        const indexSize = valueRowIds.length;
        const result = [];
        if (indexSize === 0) {
            return [];
        }
        let currentOutputColumn = 0;
        let currentValueRowId = valueRowIds[0];
        if (currentValueRowId >= parentOutputIndex.length) {
            throw new Error(`Got currentValueRowId=${currentValueRowId}, which is not less than ${parentOutputIndex.length}`);
        }
        let currentOutputIndex = parentOutputIndex[currentValueRowId];
        result.push(currentOutputIndex);
        for (let i = 1; i < indexSize; ++i) {
            const nextValueRowId = valueRowIds[i];
            if (nextValueRowId === currentValueRowId) {
                if (currentOutputIndex >= 0) {
                    ++currentOutputColumn;
                    if (currentOutputColumn < outputSize) {
                        currentOutputIndex += outputIndexMultiplier;
                    }
                    else {
                        currentOutputIndex = -1;
                    }
                }
            }
            else {
                currentOutputColumn = 0;
                currentValueRowId = nextValueRowId;
                if (nextValueRowId >= parentOutputIndex.length) {
                    throw new Error(`Got nextValueRowId=${nextValueRowId} which is not less than ${parentOutputIndex.length}`);
                }
                currentOutputIndex = parentOutputIndex[nextValueRowId];
            }
            result.push(currentOutputIndex);
        }
        if (result.length !== valueRowIds.length) {
            throw new Error('Invalid row ids.');
        }
        return result;
    }
    calculateOutputIndex(dimension, parentOutputIndex, outputIndexMultiplier, outputSize) {
        const rowPartitionTensor = this.getRowPartitionTensor(dimension);
        const partitionType = this.getRowPartitionTypeByDimension(dimension);
        switch (partitionType) {
            case RowPartitionType.VALUE_ROWIDS:
                return this.calculateOutputIndexValueRowID(rowPartitionTensor, parentOutputIndex, outputIndexMultiplier, outputSize);
            case RowPartitionType.ROW_SPLITS:
                if (rowPartitionTensor.length - 1 > parentOutputIndex.length) {
                    throw new Error(`Row partition size is greater than output size: ${rowPartitionTensor.length - 1} > ${parentOutputIndex.length}`);
                }
                return this.calculateOutputIndexRowSplit(rowPartitionTensor, parentOutputIndex, outputIndexMultiplier, outputSize);
            default:
                throw new Error(`Unsupported partition type: ${RowPartitionType[partitionType]}`);
        }
    }
    getFirstDimensionSize() {
        const firstPartitionTensor = this.rowPartitionValues[0];
        if (this.rowPartitionTypes.length === 0) {
            throw new Error('No row_partition_types given.');
        }
        const firstPartitionType = this.rowPartitionTypes[0];
        switch (firstPartitionType) {
            case RowPartitionType.FIRST_DIM_SIZE:
                return firstPartitionTensor[0];
            case RowPartitionType.VALUE_ROWIDS:
                throw new Error('Cannot handle VALUE_ROWIDS in first dimension.');
            case RowPartitionType.ROW_SPLITS:
                return this.rowPartitionValuesShapes[0][0] - 1;
            default:
                throw new Error(`Cannot handle type ${RowPartitionType[firstPartitionType]}`);
        }
    }
    compute() {
        const firstPartitionTensor = this.rowPartitionValues[0];
        if (firstPartitionTensor.length <= 0) {
            throw new Error('Invalid first partition input. ' +
                'Tensor requires at least one element.');
        }
        const firstDimension = this.getFirstDimensionSize();
        const outputSize = this.calculateOutputSize(firstDimension);
        const multiplier = new Array(this.raggedRank + 1);
        multiplier[multiplier.length - 1] = 1;
        for (let i = multiplier.length - 2; i >= 0; --i) {
            multiplier[i] = multiplier[i + 1] * outputSize[i + 1];
        }
        // Full size of the tensor.
        const outputShape = makeShape(outputSize, false);
        const outputTensor = util.getArrayFromDType(this.valuesDType, util.sizeFromShape(outputShape));
        const fullSize = multiplier[0] * outputSize[0];
        if (fullSize > 0) {
            let outputIndex = this.calculateFirstParentOutputIndex(firstDimension, multiplier[0], outputSize[0]);
            for (let i = 1; i <= this.raggedRank; ++i) {
                const newOutputIndex = this.calculateOutputIndex(i - 1, outputIndex, multiplier[i], outputSize[i]);
                outputIndex = newOutputIndex;
            }
            this.setOutput(this.raggedRank, outputIndex, outputTensor, outputShape);
        }
        return [outputShape, outputTensor];
    }
    setOutput(raggedRank, outputIndex, outputTensor, outputShape) {
        if (outputTensor.length === 0) {
            return;
        }
        const valuesBase = this.values;
        const outputBase = outputTensor;
        let elementShape = outputShape.slice();
        elementShape = elementShape.slice(raggedRank + 1);
        const valueElementSize = util.sizeFromShape(elementShape);
        const outputIndexSize = outputIndex.length;
        // Broadcast the default value to value_element_size.  (We can skip this
        // if defaultValueTensor.size == 1, since we use fill when that's true.)
        let defaultValue = this.defaultValue;
        if (defaultValue.length !== valueElementSize && defaultValue.length !== 1) {
            const srcShape = this.defaultValueShape;
            tidy(() => {
                const defaultValueTensor = reshape$1(defaultValue, srcShape);
                const bCastDefault = broadcastTo(defaultValueTensor, elementShape);
                defaultValue = bCastDefault.dataSync();
            });
        }
        // Loop through the outputIndex array, finding contiguous regions that
        // should be copied.  Once we find the end of a contiguous region, copy it
        // and add any necessary padding (with defaultValue).
        let srcStart = 0; // Start of contiguous region (in values)
        let dstStart = 0; // Destination for contiguous region (in output)
        let dstEnd = 0; // Destination for contiguous region (in output)
        for (let srcI = 0; srcI <= outputIndexSize; ++srcI) {
            // dstI is the destination where the value at srcI should be copied.
            let dstI = srcI < outputIndexSize ? outputIndex[srcI] : -1;
            // If we're still in a contiguous region, then update dstEnd go to the
            // next srcI.
            if (dstI === dstEnd) {
                ++dstEnd;
                continue;
            }
            // We found the end of contiguous region.  This can be because we found
            // a gap (dstI > dstEnd), or a source value that shouldn't be copied
            // because it's out-of-bounds (dstI == -1), or the end of the tensor
            // (dstI === -1).
            if (dstStart < dstEnd) {
                // Copy the contiguous region.
                const src = valuesBase.subarray(srcStart * valueElementSize);
                const dst = outputBase.subarray(dstStart * valueElementSize);
                const nVals = (dstEnd - dstStart) * valueElementSize;
                copyArray(dst, src, nVals);
            }
            // Add any necessary padding (w/ defaultValue).
            if (srcI >= outputIndexSize) {
                // We reached the end of values: pad to the end of output.
                const outputSize = outputTensor.length;
                dstI = Math.floor(outputSize / valueElementSize);
            }
            if (dstI > dstEnd) {
                if (this.defaultValue.length === 1) {
                    outputBase
                        .subarray(dstEnd * valueElementSize, dstI * valueElementSize)
                        .fill(this.defaultValue[0]);
                    dstEnd = dstI;
                }
                else {
                    while (dstI > dstEnd) {
                        const dst = outputBase.slice(dstEnd * valueElementSize);
                        copyArray(dst, defaultValue, valueElementSize);
                        ++dstEnd;
                    }
                }
            }
            // Update indices.
            if (dstI < 0) {
                // srcI should be skipped -- leave it out of the contiguous region.
                srcStart = srcI + 1;
                dstStart = dstEnd;
            }
            else {
                // srcI should be copied -- include it in the contiguous region.
                srcStart = srcI;
                dstStart = dstEnd;
                dstEnd = dstStart + 1;
            }
        }
    }
}
function copyArray(dst, src, size) {
    for (let i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}
function makeShape(shape, isPartial) {
    const out = [];
    for (let dim of shape) {
        if (dim < 0) {
            if (!isPartial) {
                throw new Error(`Dimension ${dim} must be >= 0`);
            }
            if (dim < -1) {
                throw new Error(`Dimension ${dim} must be >= -1`);
            }
            dim = -1;
        }
        out.push(dim);
    }
    return out;
}
function raggedTensorToTensorImpl(shape, shapesShape, values, valuesShape, valuesDType, defaultValue, defaultValueShape, rowPartitionValues, rowPartitionValuesShapes, rowPartitionTypes) {
    return new RaggedTensorToTensorOp(shape, shapesShape, values, valuesShape, valuesDType, defaultValue, defaultValueShape, rowPartitionValues, rowPartitionValuesShapes, rowPartitionTypes)
        .compute();
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function rangeImpl(start, stop, step, dtype) {
    const sameStartStop = start === stop;
    const increasingRangeNegativeStep = start < stop && step < 0;
    const decreasingRangePositiveStep = stop < start && step > 1;
    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
        return util.makeZerosTypedArray(0, dtype);
    }
    const numElements = Math.abs(Math.ceil((stop - start) / step));
    const values = util.makeZerosTypedArray(numElements, dtype);
    if (stop < start && step === 1) {
        // Auto adjust the step's sign if it hasn't been set
        // (or was set to 1)
        step = -1;
    }
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const rsqrtImpl = createSimpleUnaryImpl((xi) => 1 / Math.sqrt(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function scatterImpl(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices) {
    const flattenShape = [outputSize / sliceSize, sliceSize];
    const indicesData = indices.values;
    const updatesData = updates.values;
    if (outputSize === 0) {
        return buffer(shape, updates.dtype);
    }
    const outBuf = (defaultValue instanceof TensorBuffer) ?
        defaultValue :
        buffer(flattenShape, updates.dtype);
    if (typeof defaultValue === 'string') {
        outBuf.values.fill(defaultValue);
    }
    else if (typeof defaultValue === 'number') {
        outBuf.values.fill(defaultValue);
    }
    else if (typeof defaultValue === 'boolean') {
        outBuf.values.fill(+defaultValue);
    }
    for (let i = 0; i < numUpdates; i++) {
        const index = [];
        let flattenIndex = 0;
        for (let j = 0; j < sliceRank; j++) {
            const dim = indicesData[i * sliceRank + j];
            index.push(dim);
            flattenIndex += dim * strides[j];
        }
        if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
            throw new Error(`Invalid indices: ${index} does not index into ${shape}`);
        }
        for (let k = 0; k < sliceSize; k++) {
            if (sumDupeIndices) {
                outBuf.values[flattenIndex * sliceSize + k] +=
                    updatesData[i * sliceSize + k];
            }
            else {
                outBuf.values[flattenIndex * sliceSize + k] = updates.rank === 0 ?
                    updatesData[0] :
                    updatesData[i * sliceSize + k];
            }
        }
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sigmoidImpl = createSimpleUnaryImpl((xi) => 1 / (1 + Math.exp(-xi)));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function sliceImpl(vals, begin, size, shape, dtype) {
    const isContinous = slice_util.isSliceContinous(shape, begin, size);
    const length = util.sizeFromShape(size);
    const xStrides = util.computeStrides(shape);
    if (isContinous) {
        const flatOffset = slice_util.computeFlatOffset(begin, xStrides);
        if (dtype === 'string') {
            return vals.slice(flatOffset, flatOffset + length);
        }
        return vals.subarray(flatOffset, flatOffset + length);
    }
    const decodedData = dtype === 'string' ?
        backend_util.fromUint8ToStringArray(vals) :
        vals;
    const inBuf = buffer(shape, dtype, decodedData);
    const outBuf = buffer(size, dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const outLoc = outBuf.indexToLoc(i);
        const inLoc = outLoc.map((idx, j) => idx + begin[j]);
        outBuf.set(inBuf.get(...inLoc), ...outLoc);
    }
    if (dtype === 'string') {
        return backend_util.fromStringArrayToUint8(outBuf.values);
    }
    return outBuf.values;
}

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
function sparseFillEmptyRowsImpl(indices, indicesShape, indicesDType, values, valuesDType, denseShape, defaultValue) {
    const indicesCount = indicesShape[0];
    const denseRows = denseShape[0];
    const emptyRowIndicator = new Array(denseRows);
    const reverseIndexMap = new Array(indicesCount);
    const rank = indicesShape[1];
    if (denseRows === 0) {
        if (indicesCount !== 0) {
            throw new Error(backend_util.getSparseFillEmptyRowsIndicesDenseShapeMismatch(indicesCount));
        }
        const outputIndices = util.getArrayFromDType(indicesDType, 0);
        const outputValues = util.getArrayFromDType(valuesDType, 0);
        return [
            outputIndices, [0, rank], outputValues, emptyRowIndicator, reverseIndexMap
        ];
    }
    let rowsAreOrdered = true;
    let lastIndicesRow = 0;
    const csrOffset = new Array(denseRows).fill(0);
    for (let i = 0; i < indicesCount; ++i) {
        // indices is a 2d tensor with shape of [N, rank]
        const row = indices[i * rank];
        if (row < 0) {
            throw new Error(backend_util.getSparseFillEmptyRowsNegativeIndexErrorMessage(i, row));
        }
        if (row >= denseRows) {
            throw new Error(backend_util.getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(i, row, denseRows));
        }
        ++csrOffset[row];
        rowsAreOrdered = rowsAreOrdered && (row >= lastIndicesRow);
        lastIndicesRow = row;
    }
    let allRowsFull = true;
    for (let row = 0; row < denseRows; ++row) {
        // csrOffset here describes the number of elements in this dense row
        const rowEmpty = (csrOffset[row] === 0);
        emptyRowIndicator[row] = rowEmpty;
        allRowsFull = allRowsFull && !rowEmpty;
        // In filled version, each row has at least one element.
        csrOffset[row] = Math.max(csrOffset[row], 1);
        // Update csrOffset to represent the number of elements up to and
        // including denseRows + 1:
        //  csrOffset[0] == #{elements of row 0}
        //  csrOffset[1] == #{elements of row 1} + #{elements of row 0}
        //  ..
        //  csrOffset[i] == starting index for elements in row i + 1.
        if (row > 0) {
            csrOffset[row] += csrOffset[row - 1];
        }
    }
    if (allRowsFull && rowsAreOrdered) {
        const outputIndices = indices;
        const outputValues = values;
        for (let i = 0; i < indicesCount; ++i) {
            reverseIndexMap[i] = i;
        }
        return [
            outputIndices, [indicesCount, rank], outputValues, emptyRowIndicator,
            reverseIndexMap
        ];
    }
    else {
        const fullIndicesCount = csrOffset[denseRows - 1];
        const outputIndices = util.getArrayFromDType(indicesDType, fullIndicesCount * rank);
        const outputValues = util.getArrayFromDType(valuesDType, fullIndicesCount);
        const filledCount = new Array(denseRows).fill(0);
        // Fill in values for rows that are not missing
        for (let i = 0; i < indicesCount; ++i) {
            // indices is a 2d tensor with shape of [N, rank]
            const row = indices[i * rank];
            const offset = filledCount[row];
            const outputI = ((row === 0) ? 0 : csrOffset[row - 1]) + offset;
            filledCount[row]++; // Increment the filled count for this row.
            for (let j = 0; j < rank; ++j) {
                // indices and outputIndices are 2d tensors with shape of [N, rank]
                outputIndices[outputI * rank + j] = indices[i * rank + j];
            }
            outputValues[outputI] = values[i];
            // We'll need this reverse index map to backprop correctly.
            reverseIndexMap[i] = outputI;
        }
        // Fill in values for rows that are missing
        for (let row = 0; row < denseRows; ++row) {
            const rowCount = filledCount[row];
            if (rowCount === 0) { // We haven't filled this row
                const startingIndex = (row === 0) ? 0 : csrOffset[row - 1];
                // Remaining index values were set to zero already.
                // Just need to set the row index in the right location.
                // outputIndices is a 2d tensor with shape of [N, rank]
                outputIndices[startingIndex * rank + 0] = row;
                for (let col = 1; col < rank; ++col) {
                    outputIndices[startingIndex * rank + col] = 0;
                }
                outputValues[startingIndex] = defaultValue;
            }
        }
        return [
            outputIndices, [fullIndicesCount, rank], outputValues, emptyRowIndicator,
            reverseIndexMap
        ];
    }
}

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
function sparseReshapeImpl(inputIndices, inputIndicesShape, inputDType, inputShape, targetShape) {
    const denseSize = util.sizeFromShape(inputShape);
    const nnz = inputIndicesShape[0];
    const outputRank = targetShape.length;
    // Compute the output shape. Determine product of specified dimensions, and
    // find the index of the unspecified one.
    const outputShape = [];
    let product = 1;
    let unknownIndex = -1;
    for (let d = 0; d < outputRank; ++d) {
        const size = targetShape[d];
        if (size === -1) {
            if (unknownIndex !== -1) {
                throw new Error(backend_util
                    .getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(unknownIndex, d));
            }
            unknownIndex = d;
            outputShape.push(1);
        }
        else {
            if (size < 0) {
                throw new Error(backend_util.getSparseReshapeNegativeOutputDimErrorMessage(d, size));
            }
            product *= size;
            outputShape.push(size);
        }
    }
    if (unknownIndex !== -1) {
        if (product <= 0) {
            throw new Error(backend_util.getSparseReshapeEmptyTensorZeroOutputDimErrorMessage());
        }
        const missing = Math.trunc(denseSize / product);
        if (product * missing !== denseSize) {
            throw new Error(backend_util.getSparseReshapeInputOutputMultipleErrorMessage(inputShape, outputShape));
        }
        outputShape[unknownIndex] = missing;
    }
    const outputSize = util.sizeFromShape(outputShape);
    if (outputSize !== denseSize) {
        throw new Error(backend_util.getSparseReshapeInputOutputMismatchErrorMessage(inputShape, outputShape));
    }
    const inputRank = inputShape.length;
    const inputStrides = [];
    if (inputRank > 0) {
        inputStrides[inputRank - 1] = 1;
        for (let d = inputRank - 2; d >= 0; --d) {
            inputStrides[d] = inputStrides[d + 1] * inputShape[d + 1];
        }
    }
    const outputStrides = [];
    if (outputRank > 0) {
        outputStrides[outputRank - 1] = 1;
        for (let d = outputRank - 2; d >= 0; --d) {
            outputStrides[d] = outputStrides[d + 1] * outputShape[d + 1];
        }
    }
    const newIndices = util.getArrayFromDType(inputDType, nnz * outputRank);
    for (let i = 0; i < nnz; ++i) {
        let id = 0;
        for (let j = 0; j < inputRank; ++j) {
            // inputIndices is a 2d tensor with shape of [nnz, inputRank]
            id += inputIndices[i * inputRank + j] * inputStrides[j];
        }
        for (let j = 0; j < outputRank; ++j) {
            // newIndices is a 2d tensor with shape of [nnz, outputRank]
            newIndices[i * outputRank + j] = Math.trunc(id / outputStrides[j]);
            id %= outputStrides[j];
        }
    }
    return [newIndices, [nnz, outputRank], outputShape];
}

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
function sparseSegmentReductionImpl(input, inputShape, inputDType, indices, segmentIds, isMean = false, defaultValue = 0) {
    const numIndices = indices.length;
    // Flatten the array to two dimensions
    const inputFlat = [inputShape[0], input.length / inputShape[0]];
    const numCol = inputFlat[1];
    // Note that the current implementation assumes that segmentIds values are
    // sorted.
    const lastSegmentIdPlusOne = numIndices > 0 ? segmentIds[numIndices - 1] + 1 : 0;
    const outputRows = lastSegmentIdPlusOne;
    if (outputRows < 0) {
        throw new Error(backend_util.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());
    }
    const outputShape = inputShape.slice();
    outputShape[0] = outputRows;
    const outputLength = outputShape.reduce((product, value) => product * value, 1);
    // Output array is initialized with the value 0 by default.
    const output = util.getArrayFromDType(inputDType, outputLength);
    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    if (numIndices === 0) {
        if (outputRows > 0) {
            output.fill(defaultValue);
        }
        return [output, outputShape];
    }
    if (outputRows <= 0) {
        throw new Error(backend_util.getSparseSegmentReductionNegativeSegmentIdsErrorMessage());
    }
    let start = 0, end = 1;
    // Index from which the output is not initialized.
    let uninitializedIndex = 0;
    let outIndex = segmentIds[start];
    while (true) {
        // We initialize nextIndex to 0 to avoid may be uninitialized warning
        let nextIndex = 0;
        if (end < numIndices) {
            nextIndex = segmentIds[end];
            if (outIndex === nextIndex) {
                ++end;
                continue;
            }
            // We have a new segment here.  Verify that the segment ids are growing.
            if (outIndex >= nextIndex) {
                throw new Error(backend_util
                    .getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage());
            }
        }
        if (outIndex < 0 || outIndex >= outputRows) {
            throw new Error(backend_util.getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage(outIndex, outputRows));
        }
        // If there is a gap between two indices, we need to set that gap to the
        // default value.
        if (outIndex > uninitializedIndex) {
            output.fill(defaultValue, uninitializedIndex * numCol, outIndex * numCol);
        }
        for (let i = start; i < end; ++i) {
            const index = indices[i];
            if (index < 0 || index >= inputFlat[0]) {
                throw new Error(backend_util.getSparseSegmentReductionIndicesOutOfRangeErrorMessage(i, indices[i], inputFlat[0]));
            }
            for (let j = 0; j < numCol; j++) {
                output[outIndex * numCol + j] += input[index * numCol + j];
            }
        }
        if (isMean) {
            for (let j = 0; j < numCol; j++) {
                output[outIndex * numCol + j] /= end - start;
            }
        }
        start = end;
        ++end;
        uninitializedIndex = outIndex + 1;
        outIndex = nextIndex;
        if (end > numIndices) {
            break;
        }
    }
    // Fill the gap at the end with the default value.
    if (uninitializedIndex < outputRows) {
        output.fill(defaultValue, uninitializedIndex * numCol, outputRows * numCol);
    }
    return [output, outputShape];
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sqrtImpl = createSimpleUnaryImpl((xi) => Math.sqrt(xi));

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const squaredDifferenceImpl = createSimpleBinaryKernelImpl(((a, b) => {
    const diff = a - b;
    return diff * diff;
}));

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
const staticRegexReplaceImpl = createSimpleUnaryImpl((x, attrs) => {
    const { pattern, replaceGlobal, rewrite } = attrs;
    // TODO(mattSoulanille): Don't create a regex each time.
    return x.replace(new RegExp(pattern, replaceGlobal ? 'g' : ''), rewrite);
});

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function stridedSliceImpl(outShape, xBuf, strides, begin) {
    const outBuf = buffer(outShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; i++) {
        const loc = outBuf.indexToLoc(i);
        const newLoc = new Array(loc.length);
        for (let j = 0; j < newLoc.length; j++) {
            newLoc[j] = loc[j] * strides[j] + begin[j];
        }
        outBuf.set(xBuf.get(...newLoc), ...loc);
    }
    return outBuf;
}

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
/**
 * The StringNGramsOp class creates ngrams from ragged string data.
 * The constructor contains all attributes related to the operation such as
 * padding widths and strings, and the compute function can be used to
 * compute the ngrams for different ragged tensor inputs.
 */
class StringNGramsOp {
    constructor(separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences) {
        this.separator = util.encodeString(separator);
        this.nGramWidths = nGramWidths;
        this.leftPad = util.encodeString(leftPad);
        this.rightPad = util.encodeString(rightPad);
        this.padWidth = padWidth;
        this.preserveShort = preserveShortSequences;
    }
    getPadWidth(nGramWidth) {
        // Ngrams can be padded with either a fixed pad width or a dynamic pad
        // width depending on the 'padWidth' arg, but in no case should the padding
        // ever be wider than 'nGramWidth' - 1.
        return Math.min(this.padWidth < 0 ? nGramWidth - 1 : this.padWidth, nGramWidth - 1);
    }
    getNumNGrams(length, nGramWidth) {
        const padWidth = this.getPadWidth(nGramWidth);
        return Math.max(0, ((length + 2 * padWidth) - nGramWidth) + 1);
    }
    createNGrams(data, splitIndex, output, outputStartIndex, numNGrams, nGramWidth) {
        for (let nGramIndex = 0; nGramIndex < numNGrams; ++nGramIndex) {
            const padWidth = this.getPadWidth(nGramWidth);
            const leftPadding = Math.max(0, padWidth - nGramIndex);
            const rightPadding = Math.max(0, padWidth - (numNGrams - (nGramIndex + 1)));
            const numTokens = nGramWidth - (leftPadding + rightPadding);
            const dataStartIndex = splitIndex + (leftPadding > 0 ? 0 : nGramIndex - padWidth);
            // Calculate the total expected size of the nGram so we can reserve the
            // correct amount of space in the string.
            let nGramSize = 0;
            // Size of the left padding.
            nGramSize += leftPadding * this.leftPad.length;
            // Size of the tokens.
            for (let n = 0; n < numTokens; ++n) {
                nGramSize += data[dataStartIndex + n].length;
            }
            // Size of the right padding.
            nGramSize += rightPadding * this.rightPad.length;
            // Size of the separators.
            const numSeparators = leftPadding + rightPadding + numTokens - 1;
            nGramSize += numSeparators * this.separator.length;
            // Build the nGram.
            output[outputStartIndex + nGramIndex] = new Uint8Array(nGramSize);
            const nGram = output[outputStartIndex + nGramIndex];
            let nextNGramIndex = 0;
            const appendToNGram = (str) => str.forEach((value) => nGram[nextNGramIndex++] = value);
            for (let n = 0; n < leftPadding; ++n) {
                appendToNGram(this.leftPad);
                appendToNGram(this.separator);
            }
            // Only output first numTokens - 1 pairs of data and separator
            for (let n = 0; n < numTokens - 1; ++n) {
                appendToNGram(data[dataStartIndex + n]);
                appendToNGram(this.separator);
            }
            // Handle case when there are no tokens or no right padding as these
            // can result in consecutive separators.
            if (numTokens > 0) {
                // If we have tokens, then output last and then pair each separator
                // with the right padding that follows, to ensure nGram ends either with
                // the token or with the right pad.
                appendToNGram(data[dataStartIndex + numTokens - 1]);
                for (let n = 0; n < rightPadding; ++n) {
                    appendToNGram(this.separator);
                    appendToNGram(this.rightPad);
                }
            }
            else {
                // If we don't have tokens, then the last item inserted into the nGram
                // has been the separator from the left padding loop above. Hence,
                // output right pad and separator and make sure to finish with a
                // padding, not a separator.
                for (let n = 0; n < rightPadding - 1; ++n) {
                    appendToNGram(this.rightPad);
                    appendToNGram(this.separator);
                }
                appendToNGram(this.rightPad);
            }
        }
    }
    // Data and splits together form the definition of the ragged tensor,
    // where data is 1 dimensional and contains the values of the tensor
    // and splits denotes the indices at which each row starts.
    compute(data, splits) {
        // Validate that the splits are valid indices into data, only if there are
        // splits specified.
        const inputDataSize = data.length;
        const splitsSize = splits.length;
        if (splitsSize > 0) {
            let prevSplit = splits[0];
            if (prevSplit !== 0) {
                throw new Error(`First split value must be 0, got ${prevSplit}`);
            }
            for (let i = 1; i < splitsSize; ++i) {
                let validSplits = splits[i] >= prevSplit;
                validSplits = validSplits && (splits[i] <= inputDataSize);
                if (!validSplits) {
                    throw new Error(`Invalid split value ${splits[i]}, must be in [${prevSplit}, ${inputDataSize}]`);
                }
                prevSplit = splits[i];
            }
            if (prevSplit !== inputDataSize) {
                throw new Error(`Last split value must be data size. Expected ${inputDataSize}, got ${prevSplit}`);
            }
        }
        const numBatchItems = splitsSize - 1;
        const nGramsSplits = util.getArrayFromDType('int32', splitsSize);
        // If there is no data or size, return an empty ragged tensor.
        if (inputDataSize === 0 || splitsSize === 0) {
            const empty = new Array(inputDataSize);
            for (let i = 0; i <= numBatchItems; ++i) {
                nGramsSplits[i] = 0;
            }
            return [empty, nGramsSplits];
        }
        nGramsSplits[0] = 0;
        for (let i = 1; i <= numBatchItems; ++i) {
            const length = splits[i] - splits[i - 1];
            let numNGrams = 0;
            this.nGramWidths.forEach((nGramWidth) => {
                numNGrams += this.getNumNGrams(length, nGramWidth);
            });
            if (this.preserveShort && length > 0 && numNGrams === 0) {
                numNGrams = 1;
            }
            nGramsSplits[i] = nGramsSplits[i - 1] + numNGrams;
        }
        const nGrams = new Array(nGramsSplits[numBatchItems]);
        for (let i = 0; i < numBatchItems; ++i) {
            const splitIndex = splits[i];
            let outputStartIdx = nGramsSplits[i];
            this.nGramWidths.forEach((nGramWidth) => {
                const length = splits[i + 1] - splits[i];
                const numNGrams = this.getNumNGrams(length, nGramWidth);
                this.createNGrams(data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
                outputStartIdx += numNGrams;
            });
            // If we're preserving short sequences, check to see if no sequence was
            // generated by comparing the current output start idx to the original
            // one (nGramSplitsdata). If no ngrams were generated, then they will
            // be equal (since we increment outputStartIdx by numNGrams every
            // time we create a set of ngrams.)
            if (this.preserveShort && outputStartIdx === nGramsSplits[i]) {
                const dataLength = splits[i + 1] - splits[i];
                // One legitimate reason to not have any ngrams when this.preserveShort
                // is true is if the sequence itself is empty. In that case, move on.
                if (dataLength === 0) {
                    continue;
                }
                // We don't have to worry about dynamic padding sizes here: if padding
                // was dynamic, every sequence would have had sufficient padding to
                // generate at least one nGram.
                const nGramWidth = dataLength + 2 * this.padWidth;
                const numNGrams = 1;
                this.createNGrams(data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
            }
        }
        return [nGrams, nGramsSplits];
    }
}
function stringNGramsImpl(data, dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences) {
    return new StringNGramsOp(separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences)
        .compute(data, dataSplits);
}

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
function split(str, delimiters, skipEmpty, result) {
    if (!str.length) {
        return;
    }
    // When the delimiter is empty, the input is split into individual characters.
    if (delimiters.length === 0) {
        for (let i = 0; i < str.length; ++i) {
            result.push(str.subarray(i, i + 1));
        }
        return;
    }
    // When there is one delimiter, the input is split only at that delimiter.
    if (delimiters.length === 1) {
        const delimiter = delimiters[0];
        let f = str.indexOf(delimiter);
        while (f !== -1) {
            const token = str.subarray(0, f);
            if (!skipEmpty || token.length !== 0) {
                result.push(token);
            }
            str = str.subarray(f + 1);
            f = str.indexOf(delimiter);
        }
        if (!skipEmpty || str.length !== 0) {
            result.push(str);
        }
        return;
    }
    // When there are multiple delimiters, the input is split at every instance
    // one of the delimiters appears.
    let tokenStart = 0;
    for (let i = 0; i < str.length + 1; i++) {
        if ((i === str.length) || (delimiters.indexOf(str[i]) !== -1)) {
            const token = str.subarray(tokenStart, i);
            if (!skipEmpty || token.length !== 0) {
                result.push(token);
            }
            tokenStart = i + 1;
        }
    }
}
function stringSplitImpl(input, delimiter, skipEmpty) {
    const batchSize = input.length;
    // Empty delimiter means split the input character by character.
    const tokens = [];
    let outputSize = 0;
    let maxNumEntries = 0;
    const numIndices = new Array(batchSize);
    for (let i = 0; i < batchSize; ++i) {
        const prevTokensLength = tokens.length;
        split(input[i], delimiter, skipEmpty, tokens);
        const nEntries = tokens.length - prevTokensLength;
        numIndices[i] = nEntries;
        outputSize += nEntries;
        maxNumEntries = Math.max(maxNumEntries, nEntries);
    }
    const indices = util.getArrayFromDType('int32', outputSize * 2);
    const values = new Array(outputSize);
    const shape = [batchSize, maxNumEntries];
    let c = 0;
    for (let i = 0; i < batchSize; ++i) {
        for (let j = 0; j < numIndices[i]; ++j) {
            // indices is a 2d tensor with shape of [outputSize, 2]
            indices[c * 2] = i;
            indices[c * 2 + 1] = j;
            values[c] = tokens[c];
            ++c;
        }
    }
    return [indices, values, shape];
}

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
function stringToHashBucketFastImpl(input, numBuckets) {
    const output = util.getArrayFromDType('int32', input.length);
    for (let i = 0; i < input.length; ++i) {
        output[i] =
            util.fingerPrint64(input[i]).modulo(numBuckets).getLowBitsUnsigned();
    }
    return output;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const subImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue - bValue));

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
/**
 * An implementation of the tile kernel shared between webgl and cpu for string
 * tensors only.
 */
function tileImpl(xBuf, reps) {
    const newShape = new Array(xBuf.rank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = xBuf.shape[i] * reps[i];
    }
    const result = buffer(newShape, xBuf.dtype);
    for (let i = 0; i < result.values.length; ++i) {
        const newLoc = result.indexToLoc(i);
        const originalLoc = new Array(xBuf.rank);
        for (let j = 0; j < originalLoc.length; j++) {
            originalLoc[j] = newLoc[j] % xBuf.shape[j];
        }
        const originalIndex = xBuf.locToIndex(originalLoc);
        result.values[i] = xBuf.values[originalIndex];
    }
    return result;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const comparePair = (a, b) => {
    const valueDiff = b.value - a.value;
    return valueDiff === 0 ? a.index - b.index : valueDiff;
};
/**
 * Partitions array where all elements smaller than the (k+1) smallest element
 * are found to the left of it, and all larger to the right of it.
 * Based on the Floyd-Rivest Algorithm, ref:
 * https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
 * @param array: Array to partition
 * @param left: Left index for the interval
 * @param right: Right index for the interval
 * @param k: Desired index value, where array[k] is the (k+1)th smallest element
 *           when left = 0
 */
function select$1(array, k, left = 0, right = array.length - 1) {
    while (right > left) {
        // Use select recursively to sample a smaller set of size s
        // the arbitrary constants 600 and 0.5 are used in the original
        // version to minimize execution time.
        if (right - left > 600) {
            const n = right - left + 1;
            const i = k - left + 1;
            const z = Math.log(n);
            const s = 0.5 * Math.exp(2 * z / 3);
            const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Math.sign(i - n / 2);
            const newLeft = Math.max(left, Math.floor(k - i * s / n + sd));
            const newRight = Math.min(right, Math.floor(k + (n - i) * s / n + sd));
            select$1(array, k, newLeft, newRight);
        }
        // partition the elements between left and right around t
        const t = array[k];
        let i = left;
        let j = right;
        util.swap(array, left, k);
        if (comparePair(array[right], t) > 0) {
            util.swap(array, left, right);
        }
        while (i < j) {
            util.swap(array, i, j);
            i++;
            j--;
            while (comparePair(array[i], t) < 0) {
                i = i + 1;
            }
            while (comparePair(array[j], t) > 0) {
                j = j - 1;
            }
        }
        if (comparePair(array[left], t) === 0) {
            util.swap(array, left, j);
        }
        else {
            j = j + 1;
            util.swap(array, j, right);
        }
        // Adjust left and right towards the boundaries of the subset
        // containing the (k - left + 1)th smallest element.
        if (j <= k) {
            left = j + 1;
        }
        if (k <= j) {
            right = j - 1;
        }
    }
}
function topKImpl(x, xShape, xDtype, k, sorted) {
    // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
    const lastDim = xShape[xShape.length - 1];
    const [batch, size] = [x.length / lastDim, lastDim];
    const allTopKVals = util.getTypedArrayFromDType(xDtype, batch * k);
    const allTopKIndices = util.getTypedArrayFromDType('int32', batch * k);
    for (let b = 0; b < batch; b++) {
        const offset = b * size;
        const vals = x.subarray(offset, offset + size);
        let valAndInd = new Array(vals.length);
        vals.forEach((value, index) => valAndInd[index] = { value, index });
        if (k < valAndInd.length) {
            select$1(valAndInd, k);
            valAndInd = valAndInd.slice(0, k);
        }
        if (sorted) {
            valAndInd.sort(comparePair);
        }
        const outOffset = b * k;
        const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
        const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
        for (let i = 0; i < k; i++) {
            topKVals[i] = valAndInd[i].value;
            topKIndices[i] = valAndInd[i].index;
        }
    }
    // Reshape back to the original input shape, except that the last
    // dimension is k.
    const outputShape = xShape.slice();
    outputShape[outputShape.length - 1] = k;
    return [
        buffer(outputShape, xDtype, allTopKVals),
        buffer(outputShape, 'int32', allTopKIndices)
    ];
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function uniqueImpl(values, axis, shape, dtype) {
    // Normalize and validate axis.
    const $axis = util.parseAxisParam(axis, shape)[0];
    // Calculate the new shape that is suitable for extracting data along the
    // given axis.
    //
    // The rank is 3.
    // The size of the 1st dimension is the size of all the axes < the given axis.
    // The size of the 2nd dimension is the same as the size of the given axis.
    // The size of the 3rd dimension is the size of all the axes > the given axis.
    //
    // For example, for a 4D tensor with shape=[2, 3, 5, 4] and axis=2, the
    // newShape would be: [2*3, 5, 4].
    //
    // Note that this is not the final output shape. This will be the shape for an
    // intermediate TensorBuffer (see inputBuffer below) to allow us to extract
    // values along the given axis. To demonstrate how it works, consider the
    // following example:
    //
    // Input: a 3D tensor, with shape [1, 2, 3]
    // [
    //   [
    //      [1,2,3],
    //      [4,5,6]
    //   ]
    // ]
    // Axis: 2 (the last axis).
    // Along axis 2, we expect to extract 3 tensors: [1,4], [2,5], [3,6].
    //
    // For this example, newShape would be: [2, 3, 1], where 2 is calculated from
    // 1*2. The re-shaped data would look like:
    //
    // [
    //   [
    //     [1], [2], [3]
    //   ],
    //   [
    //     [4], [5], [6]
    //   ]
    // ]
    //
    // Then, we can construct a 3-level nested loop by the following dimension
    // order to extract the values along the axis (dimension1):
    // i: dimension1       // 0,1,2 (newShape[1])
    //   m: dimension0     // 0,1   (newShape[0])
    //     n: dimension2   // 0     (newShape[2])
    //
    //                       m, i, n
    //                      ---------
    // Iteration 0: data at [0, 0, 0] => "1"
    // Iteration 1: data at [1, 0, 0] => "4"
    // We got [1,4].
    // Iteration 2: data at [0, 1, 0] => "2"
    // Iteration 3: data at [1, 1, 0] => "5"
    // We got [2,5].
    // Iteration 4: data at [0, 2, 0] => "3"
    // Iteration 5: data at [1, 2, 0] => "6"
    // We got [3,6].
    const newShape = [1, shape[0], 1];
    for (let i = 0; i < $axis; i++) {
        newShape[0] *= shape[i];
    }
    newShape[1] = shape[$axis];
    for (let i = $axis + 1; i < shape.length; i++) {
        newShape[2] *= shape[i];
    }
    // A map from unique elements (their string representations) to their values
    // in "indices" (below).
    const uniqueElements = new Map();
    // The indices of each unique element in the original tensor along the given
    // axis. It is 1D and has the same size as the given axis.
    const indices = new Int32Array(shape[$axis]);
    // Create a buffer so we can easily extract value at a given location.
    const inputBuffer = new TensorBuffer(newShape, dtype, values);
    // The indices along the given axis that have unique elements. This is a
    // de-duped version of "indices" above.
    const uniqueIndices = [];
    const is1DTensor = newShape[0] === 1 && newShape[2] === 1;
    for (let i = 0; i < shape[$axis]; i++) {
        // Extract values along the axis.
        let element;
        if (is1DTensor) {
            // Fast path for 1D tensor input.
            element = values[i].toString();
        }
        else {
            const axisValues = [];
            for (let m = 0; m < newShape[0]; m++) {
                for (let n = 0; n < newShape[2]; n++) {
                    axisValues.push(inputBuffer.get(m, i, n));
                }
            }
            element = axisValues.join(',');
        }
        // Dedup and update various indices.
        const existingIndex = uniqueElements.get(element);
        if (existingIndex != null) {
            indices[i] = existingIndex;
        }
        else {
            const uniqueIndex = uniqueElements.size;
            uniqueElements.set(element, uniqueIndex);
            indices[i] = uniqueIndex;
            uniqueIndices.push(i);
        }
    }
    // Now we know where each of the unique elements are located along the axis
    // (uniqueIndices). Extract them from input buffer and store them in the
    // output buffer.
    const outputTmpShape = newShape.slice();
    outputTmpShape[1] = uniqueElements.size;
    const outputBuffer = new TensorBuffer(outputTmpShape, dtype);
    uniqueIndices.forEach((uniqueElementIndex, i) => {
        for (let m = 0; m < newShape[0]; m++) {
            for (let n = 0; n < newShape[2]; n++) {
                outputBuffer.set(inputBuffer.get(m, uniqueElementIndex, n), m, i, n);
            }
        }
    });
    // The output shape can be calculated from the input shape with the size of
    // the given axis replaced by the number of unique elements along that axis.
    const outputShape = shape.slice();
    outputShape[$axis] = outputTmpShape[1];
    return {
        outputValues: outputBuffer.values,
        outputShape,
        indices,
    };
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

var shared = {
    __proto__: null,
    addImpl: addImpl,
    bincountImpl: bincountImpl,
    bincountReduceImpl: bincountReduceImpl,
    bitwiseAndImpl: bitwiseAndImpl,
    castImpl: castImpl,
    ceilImpl: ceilImpl,
    concatImpl: concatImpl$1,
    equalImpl: equalImpl,
    expImpl: expImpl,
    expm1Impl: expm1Impl,
    floorDivImpl: floorDivImpl,
    floorImpl: floorImpl,
    gatherNdImpl: gatherNdImpl,
    gatherV2Impl: gatherV2Impl,
    greaterEqualImpl: greaterEqualImpl,
    greaterImpl: greaterImpl,
    lessEqualImpl: lessEqualImpl,
    lessImpl: lessImpl,
    linSpaceImpl: linSpaceImpl,
    logImpl: logImpl,
    maxImpl: maxImpl,
    maximumImpl: maximumImpl,
    minimumImpl: minimumImpl,
    multiplyImpl: multiplyImpl,
    negImpl: negImpl,
    notEqualImpl: notEqualImpl,
    prodImpl: prodImpl,
    raggedGatherImpl: raggedGatherImpl,
    raggedRangeImpl: raggedRangeImpl,
    raggedTensorToTensorImpl: raggedTensorToTensorImpl,
    rangeImpl: rangeImpl,
    rsqrtImpl: rsqrtImpl,
    scatterImpl: scatterImpl,
    sigmoidImpl: sigmoidImpl,
    simpleAbsImpl: simpleAbsImpl,
    sliceImpl: sliceImpl,
    sparseFillEmptyRowsImpl: sparseFillEmptyRowsImpl,
    sparseReshapeImpl: sparseReshapeImpl,
    sparseSegmentReductionImpl: sparseSegmentReductionImpl,
    sqrtImpl: sqrtImpl,
    squaredDifferenceImpl: squaredDifferenceImpl,
    staticRegexReplaceImpl: staticRegexReplaceImpl,
    stridedSliceImpl: stridedSliceImpl,
    stringNGramsImpl: stringNGramsImpl,
    stringSplitImpl: stringSplitImpl,
    stringToHashBucketFastImpl: stringToHashBucketFastImpl,
    subImpl: subImpl,
    tileImpl: tileImpl,
    topKImpl: topKImpl,
    transposeImpl: transposeImpl,
    uniqueImpl: uniqueImpl
};

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
const { addImpl: addImplCPU, castImpl: castImplCPU, ceilImpl: ceilImplCPU, concatImpl: concatImplCPU, equalImpl: equalImplCPU, expImpl: expImplCPU, expm1Impl: expm1ImplCPU, floorImpl: floorImplCPU, floorDivImpl: floorDivImplCPU, gatherNdImpl: gatherNdImplCPU, gatherV2Impl: gatherV2ImplCPU, greaterEqualImpl: greaterEqualImplCPU, greaterImpl: greaterImplCPU, lessEqualImpl: lessEqualImplCPU, lessImpl: lessImplCPU, logImpl: logImplCPU, maxImpl: maxImplCPU, maximumImpl: maximumImplCPU, minimumImpl: minimumImplCPU, multiplyImpl: multiplyImplCPU, negImpl: negImplCPU, notEqualImpl: notEqualImplCPU, prodImpl: prodImplCPU, rangeImpl: rangeImplCPU, rsqrtImpl: rsqrtImplCPU, scatterImpl: scatterImplCPU, simpleAbsImpl: simpleAbsImplCPU, sliceImpl: sliceImplCPU, stridedSliceImpl: stridedSliceImplCPU, stringNGramsImpl: stringNGramsImplCPU, subImpl: subImplCPU, tileImpl: tileImplCPU, topKImpl: topKImplCPU, transposeImpl: transposeImplCPU, uniqueImpl: uniqueImplCPU, } = shared;

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
const abs = unaryKernelFunc({ opType: UnaryOpType.ABS, cpuKernelImpl: simpleAbsImplCPU });
const absConfig = {
    kernelName: Abs,
    backendName: 'webgpu',
    kernelFunc: abs
};

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
const acos = unaryKernelFunc({ opType: UnaryOpType.ACOS });
const acosConfig = {
    kernelName: Acos,
    backendName: 'webgpu',
    kernelFunc: acos
};

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
const acosh = unaryKernelFunc({ opType: UnaryOpType.ACOSH });
const acoshConfig = {
    kernelName: Acosh,
    backendName: 'webgpu',
    kernelFunc: acosh
};

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
const addKernelFunc = binaryKernelFunc({ opType: BinaryOpType.ADD, cpuKernelImpl: addImplCPU, supportsComplex: true });
const addConfig = {
    kernelName: Add,
    backendName: 'webgpu',
    kernelFunc: addKernelFunc
};

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
class AddNPackedProgram {
    constructor(shapes) {
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shapes[0];
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'addN';
    }
    getUserCode() {
        const snippets = [];
        // Get target elements from every input tensor.
        this.variableNames.forEach(variable => {
            snippets.push(`let v${variable} = get${variable}ByOutputCoords(coords);`);
        });
        // Calculate the sum of all elements.
        const operation = this.variableNames
            .map(variable => {
            return `v${variable}`;
        })
            .join(' + ');
        const userCode = `
      ${getMainHeaderString('index')} {
        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutputAtIndex(flatIndex, ${operation});
          }
        }
      }
    `;
        return userCode;
    }
}

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
function addN(args) {
    const { inputs, backend } = args;
    const tensors = inputs;
    if (tensors.length === 1) {
        return identity({ inputs: { x: tensors[0] }, backend });
    }
    const dtype = tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
    const shapes = tensors.map(t => t.shape);
    const program = new AddNPackedProgram(shapes);
    return backend.runWebGPUProgram(program, tensors, dtype);
}
const addNConfig = {
    kernelName: AddN,
    backendName: 'webgpu',
    kernelFunc: addN
};

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
class TransposeSharedProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        // Note that the maximum number of workgroup invocations by webgpu is 256.
        this.workgroupSize = [16, 16, 1];
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [0], y: [1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [1, 1, 1]);
        this.shaderKey = 'transposeShared';
    }
    getUserCode() {
        util.assert(this.workgroupSize[0] === this.workgroupSize[1], () => `Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`);
        const tileSize = this.workgroupSize[0];
        const userCode = `
      var<workgroup> tile : array<array<f32, ${this.workgroupSize[0] + 1}>, ${this.workgroupSize[0]}>;
      ${getMainHeaderString()} {
        var x = i32(workgroupId.x) * ${tileSize} + i32(localId.x);
        var y = i32(workgroupId.y) * ${tileSize} + i32(localId.y);
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] = f32(A[y * width + x]);
        }
        workgroupBarrier();

        x = i32(workgroupId.y) * ${tileSize} + i32(localId.x);
        y = i32(workgroupId.x) * ${tileSize} + i32(localId.y);
        if (x < height && y < width) {
          setOutputAtIndex((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `;
        return userCode;
    }
}

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
class TransposeProgram {
    constructor(aShape, newDim) {
        this.variableNames = ['A'];
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.newDim = newDim;
        this.shaderKey = `transpose_${newDim}`;
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      ${getMainHeaderString('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${this.outputShape.length}D(
              ${dtype}(${switched}), uniforms.aShape)]);
          }
        }
      }
    `;
        return userCode;
    }
}
function getSwitchedCoords(newDim) {
    const rank = newDim.length;
    if (rank > 6) {
        throw Error(`Transpose for rank ${rank} is not yet supported`);
    }
    const switchedCoords = new Array(rank);
    for (let i = 0; i < newDim.length; i++) {
        switchedCoords[newDim[i]] = `coords.${getCoordsXYZ(i)}`;
    }
    return switchedCoords.join();
}

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
function transpose(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { perm } = attrs;
    const webgpuBackend = backend;
    const xRank = x.shape.length;
    const newShape = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = x.shape[perm[i]];
    }
    if (backend.shouldExecuteOnCPU([x])) {
        const xData = webgpuBackend.tensorMap.get(x.dataId);
        const values = xData.values;
        const outValues = transposeImplCPU(values, x.shape, x.dtype, perm, newShape);
        return backend.makeTensorInfo(newShape, x.dtype, outValues);
    }
    if (x.shape.length === 2 && util.arraysEqual(perm, [1, 0])) {
        const program = new TransposeSharedProgram(x.shape, perm);
        return webgpuBackend.runWebGPUProgram(program, [x], x.dtype);
    }
    const program = new TransposeProgram(x.shape, perm);
    return webgpuBackend.runWebGPUProgram(program, [x], x.dtype);
}
const transposeConfig = {
    kernelName: Transpose,
    backendName: 'webgpu',
    kernelFunc: transpose
};

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
class ReduceProgram {
    constructor(reduceInfo, reduceType, maxComputeWorkgroupSizeX) {
        this.variableNames = ['x'];
        this.uniforms = 'reduceSize : i32,';
        this.size = true;
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        const [outputShape,] = backend_util.computeOutAndReduceShapes(this.inputShape, [1]);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        // If reduceSize |reduceInfo.inSize| is very large, the I/O accessing will
        // become the bottleneck. Increasing workgroupSize can reduce the times of
        // accessing global memory. The threshold value is just to make sure the
        // reduceSize is large enough for a bigger workgroupSize.
        if (reduceInfo.inSize >= 32768 && maxComputeWorkgroupSizeX >= 512) {
            this.workgroupSize = [512, 1, 1];
        }
        else if (reduceInfo.inSize >= 4096) {
            this.workgroupSize = [256, 1, 1];
        }
        else {
            this.workgroupSize = [64, 1, 1];
        }
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        // A work group only outputs a data, so we transfer [1, 1, 1] to compute
        // dispatch size.
        this.dispatch =
            computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);
        this.reduceType = reduceType;
        this.shaderKey = `reduce_${reduceType}`;
    }
    getUserCode() {
        let reduceOp = ``;
        let initValue = '0.0';
        const workgroupSizeX = this.workgroupSize[0];
        if (this.reduceType === 'min' || this.reduceType === 'max') {
            reduceOp = `
         if (isnan(candidate)) {
          bestValue = uniforms.NAN;
         } else if (!isnan(bestValue) && candidate ${this.reduceType === 'min' ? '<' : '>'} bestValue)
           {  bestValue = candidate; }`;
            initValue = 'f32(x[offset])';
        }
        else if (this.reduceType === 'sum' || this.reduceType === 'mean') {
            reduceOp = ' bestValue = bestValue + candidate; ';
        }
        else if (this.reduceType === 'prod') {
            reduceOp = ' bestValue = bestValue * candidate; ';
            initValue = '1.0';
        }
        else if (this.reduceType === 'all') {
            reduceOp = ' bestValue = f32(bestValue >= 1.0 && candidate >= 1.0); ';
            initValue = '1.0';
        }
        else if (this.reduceType === 'any') {
            reduceOp = ' bestValue = f32(bestValue >= 1.0 || candidate >= 1.0); ';
            initValue = '0.0';
        }
        const outputSnippet = this.reduceType === 'mean' ?
            // tslint:disable-next-line:max-line-length
            `setOutputAtIndex(outputIndex, bestValue / f32(uniforms.reduceSize));` :
            `setOutputAtIndex(outputIndex, bestValue);`;
        const sharedMemorySnippet = `
         var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
       `;
        const userCode = `
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }

       ${sharedMemorySnippet}
       fn getOffset(outputIndex : i32) -> i32 {
         let outputCoords = getCoordsFromIndex(outputIndex);
         let offset = ${this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * uniforms.reduceSize;
          return offset;
       }
       ${getMainHeaderString('index')} {
         let outputIndex = index / ${workgroupSizeX};
         let offset = getOffset(outputIndex);
         var bestValue = ${initValue};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(u32(Length), ${workgroupSizeX}u);
         for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
             k = k + ${workgroupSizeX}) {
           let candidate = f32(x[offset + k]);
           ${reduceOp}
         }
         xBestValues[localId.x] = bestValue;
         workgroupBarrier();

         var reduceSize = min(u32(Length), ${workgroupSizeX}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            ${reduceOp}
            xBestValues[localId.x] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (localId.x == 0u && outputIndex < uniforms.size) {
          ${outputSnippet}
        }
       }
     `;
        return userCode;
    }
}

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
const RETURN_TYPES = {
    'mean': 'float32',
    'all': 'bool',
    'any': 'bool',
};
function reduce(x, axis, keepDims, reduceType, backend) {
    const xRank = x.shape.length;
    const toDispose = [];
    const origAxes = util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
    let input = x;
    if (permutedAxes != null) {
        input = transpose({ inputs: { x }, attrs: { perm: permutedAxes }, backend });
        axes = backend_util.getInnerMostAxes(axes.length, xRank);
        toDispose.push(input);
    }
    backend_util.assertAxesAreInnerMostDims(reduceType, axes, xRank);
    const [reduceOutShape, reduceShape] = backend_util.computeOutAndReduceShapes(input.shape, axes);
    let resOutShape = reduceOutShape;
    if (keepDims) {
        // rather than reshape at the end, set the target shape here.
        resOutShape = backend_util.expandShapeToKeepDim(reduceOutShape, origAxes);
    }
    let res;
    if ((reduceType === 'max' || reduceType === 'prod') &&
        backend.shouldExecuteOnCPU([input])) {
        const xVals = backend.tensorMap.get(input.dataId).values;
        switch (reduceType) {
            case 'max':
                const outValues = maxImplCPU(xVals, util.sizeFromShape(reduceShape), resOutShape, x.dtype);
                res = backend.makeTensorInfo(resOutShape, x.dtype, outValues);
                break;
            case 'prod':
                const { outVals, outShape, outDtype } = prodImplCPU(input.shape, input.dtype, xVals, axes);
                res = backend.makeTensorInfo(outShape, outDtype, outVals);
                break;
            default:
                throw new Error(`${reduceType} CPU implementation is not yet supported.`);
        }
    }
    else {
        const inSize = util.sizeFromShape(reduceShape);
        const xSize = util.sizeFromShape(input.shape);
        const batchSize = xSize / inSize;
        const reduceInfo = { windowSize: inSize, inSize, batchSize, outSize: 1 };
        const dtype = RETURN_TYPES[reduceType] || sumOutType(x.dtype);
        const uniformData = [
            { type: 'int32', data: [inSize] },
        ];
        const program = new ReduceProgram(reduceInfo, reduceType, backend.device.limits.maxComputeWorkgroupSizeX);
        const reduced = backend.runWebGPUProgram(program, [input], dtype, uniformData);
        toDispose.push(reduced);
        res = reshape({ inputs: { x: reduced }, attrs: { shape: resOutShape }, backend });
    }
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return res;
}

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
function all(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { keepDims, axis } = attrs;
    return reduce(x, axis, keepDims, 'all', backend);
}
const allConfig = {
    kernelName: All,
    backendName: 'webgpu',
    kernelFunc: all
};

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
function any(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { keepDims, axis } = attrs;
    return reduce(x, axis, keepDims, 'any', backend);
}
const anyConfig = {
    kernelName: Any,
    backendName: 'webgpu',
    kernelFunc: any
};

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
class ArgMinMaxProgram {
    constructor(inputShape, axis, reduceType) {
        this.workgroupSize = [64, 1, 1];
        this.variableNames = ['x'];
        this.uniforms = 'infinityValue : f32,';
        this.size = true;
        const axes = [axis];
        this.op = reduceType === 'min' ? '<' : '>';
        // |outShape| is the shape with the removed axis
        const [outputShape, reduceShape] = backend_util.computeOutAndReduceShapes(inputShape, axes);
        this.outputShape = outputShape.length === 0 ? [1] : outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        // The shared algorithm is mainly used for large reduce size. It fully
        // utilizes the threads in one workgroup to do the reduction. However,
        // when the reduce size is very small, it's better to use the plain
        // algorithm to reduce the number of workgroups to speedup. The threthold
        // can be further tuned.
        if (util.sizeFromShape(reduceShape) < 32) {
            this.type = 'plain';
            this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        }
        else {
            this.type = 'shared';
            // A work group only outputs a data, so we transfer [1, 1, 1] to compute
            // dispatch size.
            this.dispatch =
                computeDispatch(this.dispatchLayout, this.outputShape, [1, 1, 1]);
        }
        this.inputShape = inputShape;
        this.shaderKey = `argMinMax_${this.op}_${this.type}`;
    }
    getUserCode() {
        const workgroupSizeX = this.workgroupSize[0];
        const getInputShapeLastDim = () => {
            if (this.inputShape.length === 1) {
                return 'uniforms.xShape';
            }
            else {
                return `uniforms.xShape.${getCoordsXYZ(this.inputShape.length - 1)}`;
            }
        };
        const splitOutputCoords = () => {
            let snippet = '';
            if (this.outputShape.length === 1) {
                if (this.inputShape.length !== 1) {
                    snippet += 'outputCoords,';
                }
            }
            else {
                for (let i = 0; i < this.outputShape.length; i++) {
                    snippet += `outputCoords.${getCoordsXYZ(i)},`;
                }
            }
            return snippet;
        };
        if (this.type === 'shared') {
            const sharedMemorySnippet = `
      var<workgroup> xBestIndices : array<i32, ${workgroupSizeX}>;
      var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
    `;
            const userCode = `
      fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
      }

      ${sharedMemorySnippet}

      ${getMainHeaderString('index')} {
        let outputIndex = index / ${workgroupSizeX};
        let reduceLength = ${getInputShapeLastDim()};

        var bestIndex = i32(localId.x);
        var bestValue = uniforms.infinityValue;
        let outputCoords = getCoordsFromIndex(outputIndex);
        for (var k = i32(localId.x); k < reduceLength && outputIndex < uniforms.size;
            k = k + ${workgroupSizeX}) {
          let candidate = getX(${splitOutputCoords()} k);
          if (!isnan(candidate) && candidate ${this.op} bestValue) {
            bestValue = candidate;
            bestIndex = k;
          }
        }
        xBestValues[localId.x] = bestValue;
        xBestIndices[localId.x] = bestIndex;
        workgroupBarrier();

        var reduceSize = min(u32(reduceLength), ${workgroupSizeX}u);
        for (var currentSize = reduceSize / 2u; reduceSize > 1u;
            currentSize = reduceSize / 2u) {
          let interval = DIV_CEIL(reduceSize, 2u);
          if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              xBestValues[localId.x] = bestValue;
              xBestIndices[localId.x] = xBestIndices[localId.x + interval];
            }
          }
          reduceSize = interval;
          workgroupBarrier();
        }

        if (localId.x == 0u && outputIndex < uniforms.size) {
          setOutputAtIndexI32(outputIndex, xBestIndices[localId.x]);
        }
      }
    `;
            return userCode;
        }
        else {
            const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let outputCoords = getCoordsFromIndex(index);
          var bestIndex = 0;
          var bestValue = getX(${splitOutputCoords()} 0);
          let reduceLength = ${getInputShapeLastDim()};
          for (var i = 1; i < reduceLength; i++) {
            let candidate = getX(${splitOutputCoords()} i);
            if (candidate ${this.op} bestValue) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
          setOutputAtIndexI32(index, bestIndex);
        }
      }
      `;
            return userCode;
        }
    }
}

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
function argMax(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    let axes = util.parseAxisParam(axis, x.shape);
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('argMax', [axes[0]], $x.shape.length);
    const program = new ArgMinMaxProgram($x.shape, axes[0], 'max');
    const uniformData = [{ type: 'float32', data: [Number.NEGATIVE_INFINITY] }];
    const out = backend.runWebGPUProgram(program, [$x], 'int32', uniformData);
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return out;
}
const argMaxConfig = {
    kernelName: ArgMax,
    backendName: 'webgpu',
    kernelFunc: argMax
};

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
function argMin(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    let axes = util.parseAxisParam(axis, x.shape);
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = backend_util.getInnerMostAxes(axes.length, $x.shape.length);
    }
    backend_util.assertAxesAreInnerMostDims('argMin', [axes[0]], $x.shape.length);
    const program = new ArgMinMaxProgram($x.shape, axes[0], 'min');
    const uniformData = [{ type: 'float32', data: [Number.POSITIVE_INFINITY] }];
    const out = backend.runWebGPUProgram(program, [$x], 'int32', uniformData);
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return out;
}
const argMinConfig = {
    kernelName: ArgMin,
    backendName: 'webgpu',
    kernelFunc: argMin
};

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
const asin = unaryKernelFunc({ opType: UnaryOpType.ASIN });
const asinConfig = {
    kernelName: Asin,
    backendName: 'webgpu',
    kernelFunc: asin
};

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
const asinh = unaryKernelFunc({ opType: UnaryOpType.ASINH });
const asinhConfig = {
    kernelName: Asinh,
    backendName: 'webgpu',
    kernelFunc: asinh
};

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
const atan = unaryKernelFunc({ opType: UnaryOpType.ATAN });
const atanConfig = {
    kernelName: Atan,
    backendName: 'webgpu',
    kernelFunc: atan
};

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
const atan2 = binaryKernelFunc({ opType: BinaryOpType.ATAN2 });
const atan2Config = {
    kernelName: Atan2,
    backendName: 'webgpu',
    kernelFunc: atan2
};

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
const atanh = unaryKernelFunc({ opType: UnaryOpType.ATANH });
const atanhConfig = {
    kernelName: Atanh,
    backendName: 'webgpu',
    kernelFunc: atanh
};

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
class PoolWithFilterSizeEqualsOneProgram {
    constructor(convInfo) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec2<i32>,`;
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'poolWithFilterSizeEqualsOne';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];

          let xRCCorner = coords.yz * uniforms.strides;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutputAtIndex(index, value);
        }
      }
    `;
        return userCode;
    }
}

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
class Pool2DProgram {
    constructor(convInfo, poolType, computePositions = false, flattenPositions = false, includeBatchIndex = false) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,`;
        // TODO(jiajia.qin@intel.com): Dynamically choose different workgroupSize for
        // different output shapes.
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.poolType = poolType;
        this.computePositions = computePositions;
        this.flattenPositions = flattenPositions;
        this.includeBatchIndex = includeBatchIndex;
        this.shaderKey = `pool2D_${poolType}_${computePositions}_${flattenPositions}_${includeBatchIndex}`;
    }
    getUserCode() {
        let updateSnippet;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue = resultValue + value; count = count + 1.0;`;
        }
        else if (this.computePositions) {
            const positionStr = this.flattenPositions ?
                (this.includeBatchIndex ?
                    `((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d` :
                    `(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d`) :
                `wR * uniforms.filterDims.y + wC`;
            updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
        }
        else {
            updateSnippet = `resultValue = max(value, resultValue);`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / max(count, 1.0)`;
        }
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.strides - uniforms.pads;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          ${this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilations.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilations.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, d);
              ${updateSnippet}
            }
          }

          ${this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
            `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
        return userCode;
    }
}
class Pool3DProgram {
    constructor(convInfo, poolType, computePositions = false, flattenPositions = false, includeBatchIndex = false) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, convDims : vec3<i32>, filterDims : vec3<i32>,`;
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.poolType = poolType;
        this.computePositions = computePositions;
        this.flattenPositions = flattenPositions;
        this.includeBatchIndex = includeBatchIndex;
        this.shaderKey = `pool3D_${poolType}_${computePositions}_${flattenPositions}_${includeBatchIndex}`;
    }
    getUserCode() {
        let updateSnippet;
        if (this.poolType === 'avg') {
            updateSnippet = `resultValue += value; count += 1.0;`;
        }
        else if (this.computePositions) {
            const positionStr = this.flattenPositions ?
                (this.includeBatchIndex ?
                    `(((batch * uniforms.xShape.y + xD) * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch` :
                    `((xD * uniforms.xShape.z + xR) * uniforms.xShape.w + xC) * uniforms.xShape.u + ch`) :
                `wD * uniforms.filterDims.y * uniforms.filterDims.y + wR * uniforms.filterDims.z + wC`;
            updateSnippet = `let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${positionStr};
      }`;
        }
        else {
            updateSnippet = `resultValue = max(value, resultValue);`;
        }
        let returnValue = `resultValue`;
        if (this.poolType === 'avg') {
            returnValue = `resultValue / max(count, 1.0)`;
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords.x;
          let ch = coords.u;

          let xCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
          let xDCorner = xCorner.x;
          let xRCorner = xCorner.y;
          let xCCorner = xCorner.z;

          ${this.computePositions ?
            `var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;` :
            `var resultValue = ${this.poolType === 'avg' ? '0.0' : '-1.0 / pow(10.0, -20.0)'};`}

          var count = 0.0;
          for (var wD = 0; wD < uniforms.filterDims.x; wD++) {
            let xD = xDCorner + wD;
            if (xD < 0 || xD >= uniforms.convDims.x) {
              continue;
            }

            for (var wR = 0; wR < uniforms.filterDims.y; wR++) {
              let xR = xRCorner + wR;
              if (xR < 0 || xR >= uniforms.convDims.y) {
                continue;
              }

              for (var wC = 0; wC < uniforms.filterDims.z; wC++) {
                let xC = xCCorner + wC;
                if (xC < 0 || xC >= uniforms.convDims.z) {
                  continue;
                }

                let value = getX(batch, xD, xR, xC, ch);
                ${updateSnippet}
              }
            }
          }

          ${this.computePositions ? `setOutputAtIndexI32(index, maxPosition);` :
            `setOutputAtIndex(index, ${returnValue});`}
        }
      }
    `;
        return userCode;
    }
}

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
function max(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { reductionIndices, keepDims } = attrs;
    return reduce(x, reductionIndices, keepDims, 'max', backend);
}
const maxConfig = {
    kernelName: Max,
    backendName: 'webgpu',
    kernelFunc: max
};

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
function mean(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { keepDims, axis } = attrs;
    return reduce(x, axis, keepDims, 'mean', backend);
}
const meanConfig = {
    kernelName: Mean,
    backendName: 'webgpu',
    kernelFunc: mean
};

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
function poolImpl(x, convInfo, poolType, backend) {
    if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
        util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
        return identity({ inputs: { x }, backend });
    }
    if (convInfo.filterWidth === convInfo.inWidth &&
        convInfo.filterHeight === convInfo.inHeight && convInfo.batchSize === 1 &&
        convInfo.padInfo.type === 'VALID') {
        const length = x.shape.length;
        const reshapeX = reshape({
            inputs: { x },
            backend,
            attrs: {
                shape: [
                    x.shape[length - 3] * x.shape[length - 2] /* height * width */,
                    x.shape[length - 1] /* channel */
                ]
            }
        });
        let reduceX;
        if (poolType === 'avg') {
            reduceX = mean({ inputs: { x: reshapeX }, backend, attrs: { axis: 0, keepDims: false } });
        }
        else {
            util.assert(poolType === 'max', () => `Invalid pool type ${poolType}`);
            reduceX = max({
                inputs: { x: reshapeX },
                backend,
                attrs: { reductionIndices: 0, keepDims: false }
            });
        }
        const result = reshape({ inputs: { x: reduceX }, backend, attrs: { shape: convInfo.outShape } });
        backend.disposeData(reshapeX.dataId);
        backend.disposeData(reduceX.dataId);
        return result;
    }
    let program;
    const dimensions = [{ type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }];
    if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
        program = new PoolWithFilterSizeEqualsOneProgram(convInfo);
    }
    else {
        if (poolType === 'avg') {
            program = new Pool2DProgram(convInfo, 'avg');
        }
        else {
            util.assert(poolType === 'max', () => `Invalid pool type ${poolType}`);
            program = new Pool2DProgram(convInfo, 'max');
        }
        dimensions.push({ type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] }, {
            type: 'int32',
            data: [convInfo.dilationHeight, convInfo.dilationWidth]
        }, { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        });
    }
    return backend.runWebGPUProgram(program, [x], x.dtype, dimensions);
}

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
function avgPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    return poolImpl(x, convInfo, 'avg', backend);
}
const avgPoolConfig = {
    kernelName: AvgPool,
    backendName: 'webgpu',
    kernelFunc: avgPool
};

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
function avgPool3D(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dataFormat, dimRoundingMode } = attrs;
    const dilations = [1, 1, 1];
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode, dataFormat);
    const avgPoolProgram = new Pool3DProgram(convInfo, 'avg');
    const dimensions = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left]
        },
        {
            type: 'int32',
            data: [convInfo.inDepth, convInfo.inHeight, convInfo.inWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        }
    ];
    return backend.runWebGPUProgram(avgPoolProgram, [x], x.dtype, dimensions);
}
const avgPool3DConfig = {
    kernelName: AvgPool3D,
    backendName: 'webgpu',
    kernelFunc: avgPool3D
};

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
class AvgPool2DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, avgMultiplier : f32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `avgPool2DBackprop`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d = coords[3];

        let dyRCCorner = vec2<i32>(coords.yz) - uniforms.pads;
        let dyRCorner = dyRCCorner.x;
        let dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR = wR + uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC = wC + uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);

            dotProd = dotProd + dyValue * uniforms.avgMultiplier;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
class AvgPool3DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
       outDepth : i32, outHeight : i32, outWidth : i32, avgMultiplier : f32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `avgPool3DBackprop`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let ch = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyDCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        for (var wD = 0; wD < uniforms.filterDims[0]; wD++) {
          let dyD = f32(dyDCorner + wD) / f32(uniforms.strides[0]);

          if (dyD < 0.0 || dyD >= f32(uniforms.outDepth) || fract(dyD) > 0.0) {
            continue;
          }
          let idyD = i32(dyD);

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let dyValue = getDy(batch, idyD, idyR, idyC, ch);
              dotProd += dyValue * uniforms.avgMultiplier;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}

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
function avgPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const program = new AvgPool3DBackpropProgram(convInfo);
    const avgMultiplier = 1 / (convInfo.filterDepth * convInfo.filterHeight * convInfo.filterWidth);
    const uniformData = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth - 1 - convInfo.padInfo.front,
                convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
                convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
            ]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        },
        { type: 'int32', data: [convInfo.outDepth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'float32', data: [avgMultiplier] }
    ];
    return backend.runWebGPUProgram(program, [dy], x.dtype, uniformData);
}
const avgPool3DGradConfig = {
    kernelName: AvgPool3DGrad,
    backendName: 'webgpu',
    kernelFunc: avgPool3DGrad
};

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
function avgPoolGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    assertNotComplex([dy, input], 'avgPoolGrad');
    const { filterSize, strides, pad } = attrs;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad);
    const program = new AvgPool2DBackpropProgram(convInfo);
    const avgMultiplier = 1 / (convInfo.filterHeight * convInfo.filterWidth);
    const uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [
                convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
                convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
            ]
        },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'float32', data: [avgMultiplier] }
    ];
    return backend.runWebGPUProgram(program, [dy], x.dtype, uniformData);
}
const avgPoolGradConfig = {
    kernelName: AvgPoolGrad,
    backendName: 'webgpu',
    kernelFunc: avgPoolGrad
};

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
function batchMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b } = inputs;
    const { transposeA, transposeB } = attrs;
    return batchMatMulImpl({ a, b, transposeA, transposeB, backend });
}
const batchMatMulConfig = {
    kernelName: BatchMatMul,
    backendName: 'webgpu',
    kernelFunc: batchMatMul,
};

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
class SliceProgram {
    constructor(start, destSize) {
        this.variableNames = ['source'];
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = destSize;
        this.rank = destSize.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.start = start;
        this.uniforms = `start : ${getCoordsDataType(start.length)}, `;
        this.shaderKey = 'slice';
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.rank);
        const sourceCoords = getCoords$1(this.rank);
        let coordSum;
        if (this.start.length === 1) {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc = uniforms.start + coords;`;
            });
        }
        else {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = uniforms.start.${getCoordsXYZ(i)} + coords.${coords[i]};`;
            });
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          var sourceLoc : ${dtype};
          let coords = getCoordsFromIndex(index);
          ${coordSum.join('\n')}
          setOutputAtIndex(index, getSource(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
const coords = ['x', 'y', 'z', 'w', 'u', 'v'];
function getCoords$1(rank) {
    if (rank === 1) {
        return 'sourceLoc';
    }
    else if (rank <= 6) {
        return coords.slice(0, rank).map(coord => `sourceLoc.${coord}`).join(',');
    }
    else {
        throw Error(`Slicing for rank ${rank} is not yet supported`);
    }
}

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
function slice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, size } = attrs;
    const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
    slice_util.assertParamsValid(x, $begin, $size);
    if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string') {
        const xTensorData = backend.tensorMap.get(x.dataId);
        const outValues = sliceImplCPU(xTensorData.values, $begin, $size, x.shape, x.dtype);
        return backend.makeTensorInfo($size, x.dtype, outValues);
    }
    if (util.sizeFromShape($size) === 0) {
        return backend.makeTensorInfo($size, x.dtype, []);
    }
    // TODO(xing.xu): Add shadow slice support.
    const program = new SliceProgram($begin, $size);
    const uniformData = [{ type: 'int32', data: $begin }];
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const sliceConfig = {
    kernelName: Slice,
    backendName: 'webgpu',
    kernelFunc: slice
};

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
const batchToSpaceND = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, crops } = attrs;
    util.assert(x.shape.length <= 4, () => 'batchToSpaceND for rank > 4 with a WebGPU backend not ' +
        'implemented yet');
    const prod = blockShape.reduce((a, b) => a * b);
    const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
    const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted = backend_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords = backend_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize = backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
    const toDispose = [];
    const reshapedIntermediate = reshape({ inputs: { x }, backend, attrs: { shape: reshaped } });
    const transposedIntermediate = transpose({ inputs: { x: reshapedIntermediate }, backend, attrs: { perm: permuted } });
    const reshapedIntermediate2 = reshape({
        inputs: { x: transposedIntermediate },
        backend,
        attrs: { shape: reshapedPermuted }
    });
    const sliced = slice({
        inputs: { x: reshapedIntermediate2 },
        backend,
        attrs: { begin: sliceBeginCoords, size: sliceSize }
    });
    toDispose.push(reshapedIntermediate);
    toDispose.push(transposedIntermediate);
    toDispose.push(reshapedIntermediate2);
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return sliced;
};
const batchToSpaceNDConfig = {
    kernelName: BatchToSpaceND,
    backendName: 'webgpu',
    kernelFunc: batchToSpaceND
};

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
const writeSnippet = `
  fn bincount_write(index: i32, value: f32) {
    ${atomicAddSnippet('&result[index]', 'value', 'float32')}
  }
`;
const binaryWriteSnippet = `
  fn bincount_write(index: i32, value: f32) {
    atomicStore(&result[index], bitcast<i32>(value));
  }
`;
class BincountProgram {
    constructor(shape, hasWeights, binaryOutput = false) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'binCountSize : i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.hasWeights = true;
        this.binaryOutput = false;
        this.outputShape = shape;
        this.rank = shape.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.binaryOutput = binaryOutput;
        if (binaryOutput) {
            this.atomic = false;
        }
        this.hasWeights = hasWeights;
        if (this.hasWeights) {
            this.variableNames.push('w');
        }
        this.shaderKey =
            `bincount_${this.hasWeights}_${this.binaryOutput}_${this.rank}`;
    }
    getUserCode() {
        const userCode = `
    ${this.binaryOutput ? binaryWriteSnippet : writeSnippet}
  ${getMainHeaderString('index')} {
    ${this.rank === 1 ?
            `if (index < uniforms.xShape) {
      let indexVal = i32(getX(index));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput ? 1. :
                (this.hasWeights ? 'getW(index)' : '1.')};
        bincount_write(indexVal, value);
      }
    }` :
            `let coord = getCoordsFromIndex(index);
    if (coordsInBounds2D(coord, uniforms.xShape)) {
      let indexVal = i32(getX(coord[0], coord[1]));
      if (indexVal < uniforms.binCountSize) {
        let value = ${this.binaryOutput ?
                1. :
                (this.hasWeights ? 'getW(coord[0], coord[1])' : '1.')};
        bincount_write(coord.x * uniforms.binCountSize + indexVal, value);
      }
    }`}
  }
  `;
        return userCode;
    }
}

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
function bincount(args) {
    const { inputs, backend, attrs } = args;
    const { x, weights } = inputs;
    const { size } = attrs;
    const xSize = util.sizeFromShape(x.shape);
    const weightsSize = util.sizeFromShape(weights.shape);
    const hasWeights = weightsSize > 0;
    const outputSize = [size];
    const dtype = weights.dtype;
    const output = fill({ backend, attrs: { shape: outputSize, value: 0, dtype } });
    const program = new BincountProgram([xSize], hasWeights);
    const uniformData = [{ type: 'int32', data: [size] }];
    const bincountInputs = hasWeights ? [x, weights] : [x];
    const res = backend.runWebGPUProgram(program, bincountInputs, dtype, uniformData, output);
    return res;
}
const bincountConfig = {
    kernelName: Bincount,
    backendName: 'webgpu',
    kernelFunc: bincount
};

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
class BroadcastArgsProgram {
    constructor(shape) {
        this.outputShape = [];
        this.variableNames = ['s0', 's1'];
        this.uniforms = 's0Size : i32, s1Size : i32, ';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [shape];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'broadcastArgs';
    }
    getUserCode() {
        const userCode = `
  ${getMainHeaderString('index')} {
    if (index < uniforms.size) {
      var s0 = 1.0;
      var s1 = 1.0;
      let indexS0 = index - uniforms.size + uniforms.s0Size;
      let indexS1 = index - uniforms.size + uniforms.s1Size;
      if (indexS0 >= 0) {
        s0 = getS0(indexS0);
      }
      if (indexS1 >= 0) {
        s1 = getS1(indexS1);
      }

      if (s0 == 1.0) {
        setOutputAtIndex(index, s1);
      } else if (s1 == 1.0) {
        setOutputAtIndex(index, s0);
      } else if (s0 != s1) {
        setOutputAtIndex(index, uniforms.NAN);
      } else {
        setOutputAtIndex(index, s0);
      }
    }
  }
  `;
        return userCode;
    }
}

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
function broadcastArgs(args) {
    const { inputs, backend } = args;
    const { s0, s1 } = inputs;
    if (backend.shouldExecuteOnCPU([s0, s1])) {
        const s0TensorInfo = backend.tensorMap.get(s0.dataId);
        const s1TensorInfo = backend.tensorMap.get(s1.dataId);
        const s0Vals = s0TensorInfo.values;
        const s1Vals = s1TensorInfo.values;
        const broadcastShape = backend_util.assertAndGetBroadcastShape(Array.from(s0Vals), Array.from(s1Vals));
        return backend.makeTensorInfo([broadcastShape.length], 'int32', Int32Array.from(broadcastShape));
    }
    const s0Size = util.sizeFromShape(s0.shape);
    const s1Size = util.sizeFromShape(s1.shape);
    const outputSize = Math.max(s0Size, s1Size);
    const program = new BroadcastArgsProgram(outputSize);
    const uniformData = [{ type: 'int32', data: [s0Size] }, { type: 'int32', data: [s1Size] }];
    return backend.runWebGPUProgram(program, [s0, s1], 'int32', uniformData);
}
const broadcastArgsConfig = {
    kernelName: BroadcastArgs,
    backendName: 'webgpu',
    kernelFunc: broadcastArgs
};

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
const notEqual = binaryKernelFunc({
    opType: BinaryOpType.NOT_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: notEqualImplCPU
});
const notEqualConfig = {
    kernelName: NotEqual,
    backendName: 'webgpu',
    kernelFunc: notEqual
};

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
function real(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputData = backend.tensorMap.get(input.dataId);
    return identity({ inputs: { x: inputData.complexTensorInfos.real }, backend });
}
const realConfig = {
    kernelName: Real,
    backendName: 'webgpu',
    kernelFunc: real
};

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
function int(input, backend) {
    const program = new UnaryOpProgram(input.shape, UnaryOpType.TO_INT);
    const output = backend.runWebGPUProgram(program, [input], 'int32');
    return { dataId: output.dataId, shape: output.shape, dtype: output.dtype };
}

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
function cast(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dtype } = attrs;
    // Casting to complex64.
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return identity({ inputs: { x }, backend });
        }
        // TODO: Import kernel function once zeros is modularized.
        const zerosTensor = tf.zeros(x.shape);
        const floatX = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
        const result = complex({ inputs: { real: floatX, imag: zerosTensor }, backend });
        zerosTensor.dispose();
        backend.disposeData(floatX.dataId);
        return result;
    }
    // Casting from complex64
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const result = cast({ inputs: { x: realPart }, backend, attrs: { dtype } });
        backend.disposeData(realPart.dataId);
        return result;
    }
    if (!util.hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        const result = identity({ inputs: { x }, backend });
        return { dataId: result.dataId, shape: result.shape, dtype };
    }
    if (backend.shouldExecuteOnCPU([x])) {
        const values = backend.tensorMap.get(x.dataId).values;
        const [resultShape, resultType, resultData] = castImplCPU(values, x.shape, x.dtype, dtype);
        return backend.makeTensorInfo(resultShape, resultType, resultData);
    }
    if (dtype === 'int32') {
        return int(x, backend);
    }
    if (dtype === 'bool') {
        const zerosTensorInfo = backend.makeTensorInfo([], 'bool', util.getTypedArrayFromDType('bool', 1));
        const binaryInputs = { a: x, b: zerosTensorInfo };
        const result = notEqual({ inputs: binaryInputs, backend });
        backend.disposeData(zerosTensorInfo.dataId);
        return result;
    }
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}
const castConfig = {
    kernelName: Cast,
    backendName: 'webgpu',
    kernelFunc: cast
};

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
const ceil = unaryKernelFunc({ opType: UnaryOpType.CEIL, cpuKernelImpl: ceilImplCPU });
const ceilConfig = {
    kernelName: Ceil,
    backendName: 'webgpu',
    kernelFunc: ceil
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
class ClipVec4Program {
    constructor(outputShape) {
        this.variableNames = ['A'];
        this.uniforms = 'minVal : f32, maxVal : f32,';
        this.workPerThread = 4;
        this.workgroupSize = [64, 1, 1];
        this.outputComponent = 4;
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'clipVec4';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if(index < uniforms.size) {
          let value = getAByOutputIndex(index);
          var clampedValue = clamp(
              value, vec4<f32>(uniforms.minVal), vec4<f32>(uniforms.maxVal));
          clampedValue = select(clampedValue, value, isnanVec4(value));
          setOutputAtIndex(index, clampedValue);
        }
      }
    `;
        return userCode;
    }
}

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
class ClipProgram {
    constructor(outputShape) {
        this.variableNames = ['A'];
        this.uniforms = 'minVal : f32, maxVal : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'clip';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if(index < uniforms.size) {
          let value = getAByOutputIndex(index);
          if (isnan(value)) {
            setOutputAtIndex(index, value);
            return;
          }
          setOutputAtIndex(index, clamp(value, uniforms.minVal, uniforms.maxVal));
        }
      }
    `;
        return userCode;
    }
}

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
function clipByValue(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { clipValueMin, clipValueMax } = attrs;
    let program;
    const uniformData = [
        { type: 'float32', data: [clipValueMin] },
        { type: 'float32', data: [clipValueMax] }
    ];
    if (util.sizeFromShape(x.shape) % 4 === 0) {
        program = new ClipVec4Program(x.shape);
    }
    else {
        program = new ClipProgram(x.shape);
    }
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const clipByValueConfig = {
    kernelName: ClipByValue,
    backendName: 'webgpu',
    kernelFunc: clipByValue
};

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
class ComplexAbsProgram {
    constructor(shape) {
        this.outputShape = [];
        this.variableNames = ['real', 'imag'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'complexAbs';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let re = abs(getRealByOutputIndex(index));
        let im = abs(getImagByOutputIndex(index));
        let mx = max(re, im);

        // The length function in wgsl may be not underflow-safe on some GPUs.
        // So the safe solution is to ensure underflow-safety in all cases.
        setOutputAtIndex(index, select(mx * length(vec2<f32>(1, min(re, im)/mx)), 0.0, mx == 0.0));
      }
    }
  `;
        return userCode;
    }
}

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
// Returns a TensorInfo with the complex shape and the dataId of the
// underlying part. We need to do this because a reshaped complex tensor is
// not reflected in its parts.
function makeComplexComponentTensorInfo(complexTensor, complexPart) {
    return {
        dataId: complexPart.dataId,
        dtype: complexPart.dtype,
        shape: complexTensor.shape
    };
}
function complexAbs(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    const xData = backend.tensorMap.get(x.dataId);
    const program = new ComplexAbsProgram(x.shape);
    const programInputs = [
        makeComplexComponentTensorInfo(x, xData.complexTensorInfos.real),
        makeComplexComponentTensorInfo(x, xData.complexTensorInfos.imag),
    ];
    return backend.runWebGPUProgram(program, programInputs, programInputs[0].dtype);
}
const complexAbsConfig = {
    kernelName: ComplexAbs,
    backendName: 'webgpu',
    kernelFunc: complexAbs
};

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
class ConcatProgram {
    constructor(shapes) {
        this.uniforms = '';
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape =
            backend_util.computeOutShape(shapes, 1 /* axis */);
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.offsetLength = shapes.length - 1;
        for (let i = 0; i < this.offsetLength; i++) {
            this.uniforms += `offset${i} : i32,`;
        }
        this.shaderKey = 'concat';
    }
    getUserCode() {
        const snippets = [];
        if (this.offsetLength > 0) {
            snippets.push(`if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }`);
            for (let i = 1; i < this.offsetLength; i++) {
                snippets.push(`else if (yC < uniforms.offset${[i]}){ ` +
                    `setOutputAtCoords(coords.x, coords.y, getT${i}(yR, yC - uniforms.offset${i - 1})); }`);
            }
            const lastIndex = this.offsetLength;
            const lastShiftIndex = this.offsetLength - 1;
            snippets.push(`else { setOutputAtCoords(coords.x, coords.y, getT${lastIndex}(yR, yC - uniforms.offset${lastShiftIndex})); }`);
        }
        else {
            snippets.push(`setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));`);
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
        return userCode;
    }
}

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
function imag(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputData = backend.tensorMap.get(input.dataId);
    return identity({ inputs: { x: inputData.complexTensorInfos.imag }, backend });
}
const imagConfig = {
    kernelName: Imag,
    backendName: 'webgpu',
    kernelFunc: imag
};

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
function concatImpl(inputs, axis, backend) {
    const dtype = inputs[0].dtype;
    if (dtype === 'complex64') {
        const reals = inputs.map((t) => real({ inputs: { input: t }, backend }));
        const imags = inputs.map((t) => imag({ inputs: { input: t }, backend }));
        const realConcated = concatImpl(reals, axis, backend);
        const imagConcated = concatImpl(imags, axis, backend);
        const result = complex({ inputs: { real: realConcated, imag: imagConcated }, backend });
        reals.forEach(r => backend.disposeData(r.dataId));
        imags.forEach(i => backend.disposeData(i.dataId));
        backend.disposeData(realConcated.dataId);
        backend.disposeData(imagConcated.dataId);
        return result;
    }
    let runOnCpu = backend.shouldExecuteOnCPU(inputs);
    // Run on cpu if dtype is string. For string, the backend represents it
    // as Uint8Array[], where each Uint8Array is a character. Given that the
    // computation is only on the outer array, uploading the whole data onto
    // gpu is wasteful. Also, currently webgpu doesn't have a design to
    // upload and retrieve Uint8Array[] between cpu and gpu. Therefore, we
    // just run the kernel on cpu if dtype is string.
    if (dtype === 'string') {
        runOnCpu = true;
    }
    if (runOnCpu) {
        // Any concat of n-dimensional tensors across any axis can be reduced to
        // a concatenation of two-dimensional tensors across the axis 1 by first
        // partitioning the axes of the original tensors into those less than the
        // axis to be concatenated and the rest. Then reshape the tensors
        // into a two-dimensional tensor by collapsing these two sets of axes and
        // concatenate the resulting matrices across the axis 1, finally reshaping
        // the result to have the proper shape.
        const tensors2D = inputs.map(t => {
            const innerSize = util.sizeFromShape(t.shape.slice(axis));
            const shape = [-1, innerSize];
            return reshape({ inputs: { x: t }, backend, attrs: { shape } });
        });
        const inputsValShapes = tensors2D.map(t => {
            return { vals: backend.readSync(t.dataId), shape: t.shape };
        });
        // Concats 2d tensors along axis=1.
        const outShape = backend_util.computeOutShape(tensors2D.map(t => t.shape), 1 /* axis */);
        const simplyConcat = tensors2D[0].shape[0] === 1;
        const outVals = concatImplCPU(inputsValShapes, outShape, dtype, simplyConcat);
        const finalOutShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
        const outInfo = backend.makeTensorInfo(finalOutShape, dtype, outVals);
        tensors2D.forEach(t => backend.disposeData(t.dataId));
        return outInfo;
    }
    // There is a storage buffer limitation in compute stage, one for output so
    // the maximum for input is limits.maxStorageBuffersPerShaderStage - 1
    const maxInputNum = backend.device.limits.maxStorageBuffersPerShaderStage - 1;
    if (inputs.length > maxInputNum) {
        const reducedInputs = [];
        for (let i = 0; i < inputs.length; i += maxInputNum) {
            const subArray = inputs.slice(i, i + maxInputNum);
            reducedInputs.push(concatImpl(subArray, axis, backend));
        }
        const result = concatImpl(reducedInputs, axis, backend);
        for (const i of reducedInputs) {
            backend.disposeData(i.dataId);
        }
        return result;
    }
    const { tensors2D, outShape } = computeTensors2D(inputs, axis, backend);
    const shapes = (tensors2D).map(t => t.shape);
    const program = new ConcatProgram(shapes);
    const uniformData = [];
    const offsets = new Array(shapes.length - 1);
    if (offsets.length > 0) {
        offsets[0] = shapes[0][1];
        uniformData.push({ type: 'int32', data: [offsets[0]] });
        for (let i = 1; i < offsets.length; i++) {
            offsets[i] = offsets[i - 1] + shapes[i][1];
            uniformData.push({ type: 'int32', data: [offsets[i]] });
        }
    }
    const res = backend.runWebGPUProgram(program, tensors2D, tensors2D[0].dtype, uniformData);
    tensors2D.forEach(r => backend.disposeData(r.dataId));
    const reshapedResult = reshape({ inputs: { x: res }, backend, attrs: { shape: outShape } });
    backend.disposeData(res.dataId);
    return reshapedResult;
}
function computeTensors2D(inputs, axis, backend) {
    const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), axis);
    const tensors2D = inputs.map(t => reshape({
        inputs: { x: t },
        backend,
        attrs: {
            shape: [
                util.sizeFromShape(t.shape.slice(0, axis)),
                util.sizeFromShape(t.shape.slice(axis))
            ]
        }
    }));
    return { tensors2D, outShape };
}

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
function concat(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];
    const shapes = inputs.map(t => t.shape);
    backend_util.assertParamsConsistent(shapes, $axis);
    const outShape = backend_util.computeOutShape(inputs.map(t => t.shape), $axis);
    if (util.sizeFromShape(outShape) === 0) {
        return backend.makeTensorInfo(outShape, inputs[0].dtype, []);
    }
    // Keep only non-empty tensors (ignore tensors with 0 in their shape).
    const $inputs = inputs.filter(t => util.sizeFromShape(t.shape) > 0);
    if ($inputs.length === 1) {
        return identity({ inputs: { x: $inputs[0] }, backend });
    }
    return concatImpl($inputs, $axis, backend);
}
const concatConfig = {
    kernelName: Concat,
    backendName: 'webgpu',
    kernelFunc: concat
};

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
function conv2dCommonSnippet(isChannelsLast, fitAOuter, fitBOuter, fitInner, addBias = false, activation = null, hasPreluActivationWeights = false, innerElementSizeX = 4, innerElementSizeW = 4, innerElementSize = 4) {
    const getXSnippet = (innerElementSize) => {
        switch (innerElementSize) {
            case 1:
                return 'resData = f32(x[xIndex]);';
            case 3:
                return 'resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);';
            case 4:
                return 'resData = vec4<f32>(x[xIndex / 4]);';
            default:
                throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
    };
    const getWSnippet = (innerElementSize) => {
        switch (innerElementSize) {
            case 1:
                return 'return f32(W[row * uniforms.wShape[3] + col]);';
            case 4:
                return 'return vec4<f32>(W[(row * uniforms.wShape[3] + col) / 4]);';
            default:
                throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
        }
    };
    const coordASnippet = isChannelsLast ? `
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      ` :
        `
      let coord = vec4<i32>(batch, xCh, xRow, xCol);
      `;
    const coordResSnippet = isChannelsLast ? `
      let coords = vec4<i32>(
        batch,
        row / outWidth,
        row % outWidth,
        col);
      ` :
        `
      let coords = vec4<i32>(
        batch,
        row,
        col / outWidth,
        col % outWidth);
      `;
    const xHight = isChannelsLast ? 'uniforms.xShape[1]' : 'uniforms.xShape[2]';
    const xWidth = isChannelsLast ? 'uniforms.xShape[2]' : 'uniforms.xShape[3]';
    const row = isChannelsLast ? 'row' : 'col';
    const col = isChannelsLast ? 'col' : 'row';
    const readXSnippet = `
      let inChannels = uniforms.wShape[2];
      let outWidth = ${isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
      let outRow = ${row} / outWidth;
      let outCol = ${row} % outWidth;

      let WRow = ${col} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${col} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
      let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
      let xCh = ${col} % inChannels;
      var resData = ${typeSnippet(innerElementSizeX)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${xHight} && xCol >= 0 && xCol < ${xWidth}) {
        ${coordASnippet}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${getXSnippet(innerElementSizeX)}
      }
      return resData;`;
    const sampleX = isChannelsLast ? (fitAOuter && fitInner ? `
      ${readXSnippet}` :
        `
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`) :
        (fitInner && fitBOuter ? `
      ${readXSnippet}` :
            `
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`);
    const sampleW = `${getWSnippet(innerElementSizeW)}`;
    const resType = typeSnippet(innerElementSize);
    const aType = isChannelsLast ? typeSnippet(innerElementSizeX) :
        typeSnippet(innerElementSizeW);
    const bType = isChannelsLast ? typeSnippet(innerElementSizeW) :
        typeSnippet(innerElementSizeX);
    const userCode = `
      ${activationFnSnippet(activation, hasPreluActivationWeights, innerElementSize === 4, 4)}
      fn mm_readA(batch: i32, row : i32, col : i32) -> ${aType} {
        ${isChannelsLast ? sampleX : sampleW}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> ${bType} {
        ${isChannelsLast ? sampleW : sampleX}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : ${resType}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        var value = valueIn;
        let outWidth = ${isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
        ${coordResSnippet}
        ${biasActivationSnippet(addBias, activation)}
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }`;
    return userCode;
}
class Conv2DMMProgram {
    constructor(convInfo, dimAOuter, dimBOuter, dimInner, addBias = false, activation = null, hasPreluActivationWeights = false, sequentialAccessByThreads = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = `filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
        this.outputShape = convInfo.outShape;
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        this.isVec4 =
            (((convInfo.inChannels % 4 === 0 || convInfo.inChannels % 3 === 0) &&
                this.isChannelsLast) ||
                (convInfo.outWidth % 4 === 0 && !this.isChannelsLast)) &&
                convInfo.outChannels % 4 === 0;
        this.dispatchLayout = this.isChannelsLast ? { x: [3], y: [1, 2], z: [0] } :
            { x: [2, 3], y: [1], z: [0] };
        this.workgroupSize = computeWorkgroupSizeForConv2d(this.dispatchLayout, this.outputShape, this.isVec4);
        this.elementsPerThread = computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape, this.isVec4);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, this.elementsPerThread);
        if (this.isVec4) {
            this.outputComponent = 4;
            if (this.isChannelsLast && convInfo.inChannels % 4 !== 0) {
                this.innerElementSize = 3;
                this.variableComponents = [1, 4];
            }
            else {
                this.innerElementSize = 4;
                this.variableComponents = [4, 4];
            }
            if (addBias) {
                this.variableNames.push('bias');
                this.variableComponents.push(4);
            }
            if (hasPreluActivationWeights) {
                this.variableNames.push('preluActivationWeights');
                this.variableComponents.push(4);
            }
        }
        else {
            this.innerElementSize = this.elementsPerThread[0];
            if (addBias) {
                this.variableNames.push('bias');
            }
            if (hasPreluActivationWeights) {
                this.variableNames.push('preluActivationWeights');
            }
        }
        this.sequentialAccessByThreads = sequentialAccessByThreads;
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        this.tileAOuter = this.workgroupSize[1] * this.elementsPerThread[1];
        this.tileBOuter = this.workgroupSize[0] * this.elementsPerThread[0];
        this.tileInner = Math.max(this.workgroupSize[0] * this.innerElementSize, this.workgroupSize[1]);
        this.fitAOuter = dimAOuter % this.tileAOuter === 0;
        this.fitBOuter = dimBOuter % this.tileBOuter === 0;
        this.fitInner = dimInner % this.tileInner === 0;
        this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.innerElementSize}_${this.isChannelsLast}_${this.sequentialAccessByThreads}`;
    }
    getUserCode() {
        const matMulSource = this.isVec4 ?
            makeMatMulPackedVec4Source(this.elementsPerThread, this.workgroupSize, !this.isChannelsLast, this.tileInner) :
            makeMatMulPackedSource(this.elementsPerThread, this.workgroupSize, !this.isChannelsLast, this.tileInner, false, null, this.sequentialAccessByThreads);
        const elementsSize = this.isVec4 ? [this.innerElementSize, 4, 4] : [1, 1, 1];
        const userCode = `
    ${conv2dCommonSnippet(this.isChannelsLast, this.fitAOuter, this.fitBOuter, this.fitInner, this.addBias, this.activation, this.hasPreluActivationWeights, elementsSize[0], elementsSize[1], elementsSize[2])}
    ${matMulSource}
  `;
        return userCode;
    }
}

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
class Conv2DNaiveProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivationWeights = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>,';
        this.workgroupSize = [4, 4, 8];
        this.outputShape = convInfo.outShape;
        this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
        this.dispatchLayout = this.isChannelsLast ? { x: [2], y: [1], z: [0, 3] } :
            { x: [3], y: [2], z: [0, 1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivationWeights = hasPreluActivationWeights;
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivationWeights) {
            this.variableNames.push('preluActivationWeights');
        }
        this.shaderKey = `conv2dnaive_${this.activation}_${this.isChannelsLast}`;
    }
    getUserCode() {
        const userCode = `
       ${activationFnSnippet(this.activation, this.hasPreluActivationWeights, false, 4)}
       fn readInp(batch : i32, row : i32, col : i32, chan : i32) -> f32{
         let coords = vec4<i32>(batch, row, col, chan);
         if (coordsInBounds4D(coords, uniforms.xShape)) {
           return  getX(batch, row, col, chan);
         } else {
          return 0.0;
         }
       }
       fn readFilt(row : i32, col : i32, xChannel : i32, outChannel : i32) -> f32{
         let coords = vec4<i32>(row, col, xChannel, outChannel);
         if(coordsInBounds4D(coords, uniforms.wShape)) {
           return getW(row, col, xChannel, outChannel);
          } else {
            return 0.0;
          }
       }
       fn writeResult(batch : i32, row : i32, col : i32, chan : i32, valueIn : f32) {
         let coords = ${this.isChannelsLast ? `vec4<i32>(batch, row, col, chan);` :
            `vec4<i32>(batch, chan, row, col);`}
         if (coordsInBounds4D(coords, uniforms.outShape)) {
           var value = valueIn;
           ${biasActivationSnippet(this.addBias, this.activation)}
           setOutputAtCoords(coords.x, coords.y, coords.z, coords.w, value);
         }
       }
       ${getMainHeaderString('index')} {
         let coords = getOutputCoords();
         let batch = coords[0];
         let outChannel = ${this.isChannelsLast ? `coords[3];` : `coords[1];`}
         let outRow = ${this.isChannelsLast ? `coords[1];` : `coords[2];`}
         let outCol = ${this.isChannelsLast ? `coords[2];` : `coords[3];`}
         var acc : f32 = 0.0;
         for (var row = 0; row < uniforms.filterDims[0]; row = row + 1) {
           for (var col = 0; col < uniforms.filterDims[1]; col = col + 1) {
             let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * row - uniforms.pads[0];
             let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * col - uniforms.pads[1];
             for (var xChannel = 0; xChannel < ${this.isChannelsLast ? `uniforms.xShape[3];` :
            `uniforms.xShape[1];`} xChannel = xChannel + 1) {
               ${this.isChannelsLast ? `let v = readInp(batch, xRow, xCol, xChannel);` :
            `let v = readInp(batch, xChannel, xRow, xCol);`}
               let f = readFilt(row, col, xChannel, outChannel);
               acc = acc + v * f;
             }
           }
         }
         writeResult(batch, outRow, outCol, outChannel, acc);
       }
     `;
        return userCode;
    }
}

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
class Im2ColProgram {
    constructor(outputShape, isChannelsLast) {
        this.variableNames = ['x'];
        this.uniforms = `pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.isChannelsLast = isChannelsLast;
        this.shaderKey = `im2col_${this.isChannelsLast}`;
    }
    getUserCode() {
        const rowDim = this.isChannelsLast ? 1 : 2;
        const colDim = this.isChannelsLast ? 2 : 3;
        const row = this.isChannelsLast ? 'coords[1]' : 'coords[2]';
        const col = this.isChannelsLast ? 'coords[2]' : 'coords[1]';
        const getXSnippet = this.isChannelsLast ? 'getX(batch, xRow, xCol, ch)' :
            'getX(batch, ch, xRow, xCol)';
        const userCode = `
    ${getMainHeaderString('index')} {
      let coords = getCoordsFromIndex(index);
      if(index < uniforms.size) {
        let batch = coords[0];
        let row = ${row};
        let col = ${col};
        let offsetY = (row / uniforms.outWidth) * uniforms.strides[0] - uniforms.pads[0];
        let xRow = offsetY + uniforms.dilations[0] * (col / uniforms.itemsPerBlockRow);
        var value = 0.0;
        if(xRow < uniforms.xShape[${rowDim}] && xRow >= 0) {
          let offsetX = (row % uniforms.outWidth) * uniforms.strides[1] -
              uniforms.pads[1];
          let xCol = offsetX + uniforms.dilations[1] * ((col %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
          let ch = col % uniforms.inChannels;
          if(xCol < uniforms.xShape[${colDim}] && xCol >= 0) {
            value = ${getXSnippet};
          }
        }
        setOutputAtIndex(index, value);
      }
    }
   `;
        return userCode;
    }
}

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
// conv2dByMatMul fuses height and width into one dimension to compute
// batchMatMul, so bias and activation weights are also supposed to fuse the two
// dimensions into one.
//
// This function computes the target shape for fusing height and width
// dimensions. Returning null means the shape is already compatible.
function getShapeForBatchMatMul(shape, isChannelsLast) {
    const length = shape.length;
    if (length >= 3) {
        return isChannelsLast ?
            [
                ...shape.slice(0, -3) /* batch */,
                shape[length - 3] * shape[length - 2] /* height * width */,
                shape[length - 1] /* channel */
            ] :
            [
                ...shape.slice(0, -3) /* batch */, shape[length - 3] /* channel */,
                shape[length - 2] * shape[length - 1] /* height * width */
            ];
    }
    else if (!isChannelsLast && length === 1 && shape[0] > 1) {
        return [shape[0], 1];
    }
    else {
        return null;
    }
}
// For 1x1 kernels that iterate through every point in the input, convolution
// can be expressed as matrix multiplication (without need for memory
// remapping).
function conv2dByMatMul({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const transposeA = isChannelsLast ? false : true;
    const transposeB = false;
    const sameSize = isChannelsLast &&
        convInfo.filterHeight === convInfo.inHeight &&
        convInfo.filterWidth === convInfo.inWidth &&
        convInfo.padInfo.type === 'VALID';
    const intermediates = [];
    let xReshaped;
    let filterReshaped;
    if (sameSize) {
        const sharedDim = convInfo.inHeight * convInfo.inWidth * convInfo.inChannels;
        xReshaped = reshape({
            inputs: { x },
            backend,
            attrs: { shape: [1, convInfo.batchSize, sharedDim] }
        });
        filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, sharedDim, convInfo.outChannels] }
        });
    }
    else {
        xReshaped = reshape({
            inputs: { x },
            backend,
            attrs: {
                shape: isChannelsLast ?
                    [
                        convInfo.batchSize, convInfo.inHeight * convInfo.inWidth,
                        convInfo.inChannels
                    ] :
                    [
                        convInfo.batchSize, convInfo.inChannels,
                        convInfo.inHeight * convInfo.inWidth
                    ]
            }
        });
        filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
        });
    }
    intermediates.push(xReshaped);
    intermediates.push(filterReshaped);
    if (preluActivationWeights != null) {
        const targetShape = getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
        if (targetShape != null) {
            preluActivationWeights = reshape({
                inputs: { x: preluActivationWeights },
                backend,
                attrs: { shape: targetShape }
            });
            intermediates.push(preluActivationWeights);
        }
    }
    if (bias != null) {
        const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
        if (targetShape != null) {
            bias = reshape({ inputs: { x: bias }, backend, attrs: { shape: targetShape } });
            intermediates.push(bias);
        }
    }
    const result = batchMatMulImpl({
        a: isChannelsLast ? xReshaped : filterReshaped,
        b: isChannelsLast ? filterReshaped : xReshaped,
        transposeA,
        transposeB,
        backend,
        bias,
        activation,
        preluActivationWeights,
        leakyreluAlpha
    });
    const out = reshape({ inputs: { x: result }, backend, attrs: { shape: convInfo.outShape } });
    intermediates.push(result);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return out;
}
// Implements the im2col algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
function conv2dWithIm2Col({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Rearranges conv2d input so each block to be convolved over forms the
    // row of a new matrix with shape [outHeight * outWidth,
    // filterWidth * filterHeight * inChannels]. The filter is also rearranged so
    // each output channel forms a col of a new matrix with shape [
    // filterWidth * filterHeight * inChannels, outChannels]. The convolution is
    // then computed by multiplying these matrices and reshaping the result.
    const { filterWidth, filterHeight, inChannels, strideWidth, strideHeight, padInfo, outWidth, outHeight, dilationWidth, dilationHeight, dataFormat } = convInfo;
    const isChannelsLast = dataFormat === 'channelsLast';
    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = isChannelsLast ? [convInfo.batchSize, numCols, sharedDim] :
        [convInfo.batchSize, sharedDim, numCols];
    const im2ColProgram = new Im2ColProgram(x2ColShape, isChannelsLast);
    const dimensions = [
        { type: 'int32', data: [padInfo.top, padInfo.left] },
        { type: 'int32', data: [strideHeight, strideWidth] },
        { type: 'int32', data: [dilationHeight, dilationWidth] },
        { type: 'int32', data: [outWidth] },
        { type: 'int32', data: [inChannels * filterWidth] },
        { type: 'int32', data: [inChannels] }
    ];
    const x2Col = backend.runWebGPUProgram(im2ColProgram, [x], x.dtype, dimensions);
    const intermediates = [];
    intermediates.push(x2Col);
    const filterReshaped = reshape({ inputs: { x: filter }, backend, attrs: { shape: [1, sharedDim, -1] } });
    intermediates.push(filterReshaped);
    if (preluActivationWeights != null) {
        const targetShape = getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
        if (targetShape != null) {
            preluActivationWeights = reshape({
                inputs: { x: preluActivationWeights },
                backend,
                attrs: { shape: targetShape }
            });
            intermediates.push(preluActivationWeights);
        }
    }
    if (bias != null) {
        const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
        if (targetShape != null) {
            bias = reshape({ inputs: { x: bias }, backend, attrs: { shape: targetShape } });
            intermediates.push(bias);
        }
    }
    const transposeA = isChannelsLast ? false : true;
    const transposeB = false;
    const result = batchMatMulImpl({
        a: isChannelsLast ? x2Col : filterReshaped,
        b: isChannelsLast ? filterReshaped : x2Col,
        transposeA,
        transposeB,
        backend,
        bias,
        activation,
        preluActivationWeights,
        leakyreluAlpha
    });
    const out = reshape({ inputs: { x: result }, backend, attrs: { shape: convInfo.outShape } });
    intermediates.push(result);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return out;
}
function conv2DImpl({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const sameSize = isChannelsLast &&
        convInfo.filterHeight === convInfo.inHeight &&
        convInfo.filterWidth === convInfo.inWidth &&
        convInfo.padInfo.type === 'VALID';
    const useNaiveConv2d = env().getBool('WEBGPU_USE_NAIVE_CONV2D_DEBUG');
    if (!useNaiveConv2d &&
        (sameSize ||
            (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
                convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
                convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
                (convInfo.padInfo.type === 'SAME' ||
                    convInfo.padInfo.type === 'VALID')))) {
        return conv2dByMatMul({
            x,
            filter,
            convInfo,
            backend,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
    }
    const thresholdFlagValue = env().getNumber('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL');
    const thresholdToIncreaseWorkgroups = thresholdFlagValue > -1 ?
        thresholdFlagValue :
        backend.thresholdToIncreaseWorkgroups;
    const workgroupsBy32x32 = convInfo.batchSize *
        Math.ceil((convInfo.outHeight * convInfo.outWidth) / 32) *
        Math.ceil(convInfo.outChannels / 32);
    if (env().getBool('WEBGPU_CONV_SEPARATE_IM2COL_SHADER') ||
        workgroupsBy32x32 <= thresholdToIncreaseWorkgroups) {
        return conv2dWithIm2Col({
            x,
            filter,
            convInfo,
            backend,
            bias,
            preluActivationWeights,
            leakyreluAlpha,
            activation
        });
    }
    let program;
    const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [...padInfo] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }
    ];
    if (useNaiveConv2d) {
        program = new Conv2DNaiveProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
    }
    else {
        const dimAOuter = isChannelsLast ? convInfo.outHeight * convInfo.outWidth :
            convInfo.outChannels;
        const dimBOuter = isChannelsLast ? convInfo.outChannels :
            convInfo.outHeight * convInfo.outWidth;
        const dimInner = convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
        dimensions.push({ type: 'int32', data: [dimAOuter] }, { type: 'int32', data: [dimBOuter] }, { type: 'int32', data: [dimInner] });
        // Experiments show that sequential access is more friendly for Intel GPUs.
        const sequentialAccessByThreads = backend.adapterInfo.isIntel();
        program = new Conv2DMMProgram(convInfo, dimAOuter, dimBOuter, dimInner, hasBias, activation, hasPreluActivationWeights, sequentialAccessByThreads);
    }
    const intermediates = [];
    const inputVar = [x, filter];
    if (hasBias) {
        if (!isChannelsLast && bias.shape.length === 1) {
            bias = reshape({ inputs: { x: bias }, backend, attrs: { shape: [bias.shape[0], 1, 1] } });
            intermediates.push(bias);
        }
        inputVar.push(bias);
    }
    if (hasPreluActivationWeights) {
        if (!isChannelsLast && preluActivationWeights.shape.length === 1) {
            preluActivationWeights = reshape({
                inputs: { x: preluActivationWeights },
                backend,
                attrs: { shape: [preluActivationWeights.shape[0], 1, 1] }
            });
            intermediates.push(preluActivationWeights);
        }
        inputVar.push(preluActivationWeights);
    }
    if (activation === 'leakyrelu') {
        dimensions.push({ type: 'float32', data: [leakyreluAlpha] });
        program.uniforms += ' alpha : f32,';
    }
    const out = backend.runWebGPUProgram(program, inputVar, x.dtype, dimensions);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return out;
}

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
function conv2d(args) {
    const { inputs, attrs, backend } = args;
    const { x, filter } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    return conv2DImpl({ x, filter, convInfo, backend });
}
const conv2DConfig = {
    kernelName: Conv2D,
    backendName: 'webgpu',
    kernelFunc: conv2d
};

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
class Conv2DDerInputProgram {
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
    ${getMainHeaderString()} {
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
    ${getMainHeaderString('index')} {
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
class Conv2DDerFilterProgram {
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
    ${getMainHeaderString('index')} {
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
class Conv3DDerFilterProgram {
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
    ${getMainHeaderString('index')} {
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
class Conv3DDerInputProgram {
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
    ${getMainHeaderString('index')} {
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
function conv2DBackpropFilter(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, pad, dataFormat, dimRoundingMode, filterShape } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filterShape, strides, 1 /* dilations */, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    const program = new Conv2DDerFilterProgram(convInfo);
    const uniformData = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.batchSize] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'int32', data: [convInfo.inHeight] },
        { type: 'int32', data: [convInfo.inWidth] }
    ];
    return backend.runWebGPUProgram(program, [x, dy], x.dtype, uniformData);
}
const conv2DBackpropFilterConfig = {
    kernelName: Conv2DBackpropFilter,
    backendName: 'webgpu',
    kernelFunc: conv2DBackpropFilter
};

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
class Conv2DDerInputMMProgram {
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
function conv2DBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { inputShape, strides, pad, dataFormat, dimRoundingMode } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(inputShape, filter.shape, strides, 1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);
    const dimensions = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        {
            type: 'int32',
            data: [
                convInfo.filterHeight - 1 - convInfo.padInfo.top,
                convInfo.filterWidth - 1 - convInfo.padInfo.left
            ]
        },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        {
            type: 'int32',
            data: [
                convInfo.batchSize, convInfo.outHeight, convInfo.outWidth,
                convInfo.outChannels
            ]
        },
    ];
    let program;
    // TODO: Experiment when to use Conv2DDerInputMMProgram algorithm.
    if (env().getBool('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE') ||
        convInfo.dataFormat !== 'channelsLast') {
        program = new Conv2DDerInputProgram(convInfo);
    }
    else {
        program = new Conv2DDerInputMMProgram(convInfo);
        const dimAOuter = convInfo.inHeight * convInfo.inWidth;
        const dimBOuter = convInfo.inChannels;
        const dimInner = convInfo.filterHeight * convInfo.filterWidth * convInfo.outChannels;
        dimensions.push({ type: 'uint32', data: [dimAOuter] }, { type: 'uint32', data: [dimBOuter] }, { type: 'uint32', data: [dimInner] });
    }
    return backend.runWebGPUProgram(program, [dy, filter], 'float32', dimensions);
}
const conv2DBackpropInputConfig = {
    kernelName: Conv2DBackpropInput,
    backendName: 'webgpu',
    kernelFunc: conv2DBackpropInput,
};

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
class Conv3DNaiveProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'W'];
        this.uniforms = 'filterDims: vec3<i32>, pads: vec3<i32>, strides: vec3<i32>, dilations: vec3<i32>,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `conv3dnaive`;
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords.x;
        let d2 = coords.u;

        let xFRCCorner = vec3<i32>(coords.y, coords.z, coords.w) * uniforms.strides - uniforms.pads;
        let xFCorner = xFRCCorner.x;
        let xRCorner = xFRCCorner.y;
        let xCCorner = xFRCCorner.z;

        let inputDepthNearestVec4 = (uniforms.xShape.u / 4) * 4;
        let inputDepthVec4Remainder = uniforms.xShape.u % 4;

        var dotProd = 0.0;
        for (var wF = 0; wF < uniforms.filterDims[0]; wF++) {
          let xF = xFCorner + wF * uniforms.dilations[0];
          if (xF < 0 || xF >= uniforms.xShape.y) {
            continue;
          }

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let xR = xRCorner + wR * uniforms.dilations[1];
            if (xR < 0 || xR >= uniforms.xShape.z) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let xC = xCCorner + wC * uniforms.dilations[2];
              if (xC < 0 || xC >= uniforms.xShape.w) {
                continue;
              }

              for (var d1 = 0; d1 < inputDepthNearestVec4; d1 += 4) {
                let xValues = vec4<f32>(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                let wValues = vec4<f32>(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (inputDepthVec4Remainder == 1) {
                dotProd += getX(batch, xF, xR, xC, inputDepthNearestVec4) *
                  getW(wF, wR, wC, inputDepthNearestVec4, d2);
              } else if (inputDepthVec4Remainder == 2) {
                let xValues = vec2<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1)
                );
                let wValues = vec2<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (inputDepthVec4Remainder == 3) {
                let xValues = vec3<f32>(
                  getX(batch, xF, xR, xC, inputDepthNearestVec4),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 1),
                  getX(batch, xF, xR, xC, inputDepthNearestVec4 + 2)
                );
                let wValues = vec3<f32>(
                  getW(wF, wR, wC, inputDepthNearestVec4, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 1, d2),
                  getW(wF, wR, wC, inputDepthNearestVec4 + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }`;
        return userCode;
    }
}

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
function conv3D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations } = attrs;
    const convInfo = backend_util.computeConv3DInfo(x.shape, filter.shape, strides, dilations, pad);
    const padInfo = [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left];
    const dimensions = [
        {
            type: 'int32',
            data: [convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth]
        },
        { type: 'int32', data: [...padInfo] }, {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth
            ]
        }
    ];
    const program = new Conv3DNaiveProgram(convInfo);
    const dtype = upcastType(x.dtype, filter.dtype);
    return backend.runWebGPUProgram(program, [x, filter], dtype, dimensions);
}
const conv3DConfig = {
    kernelName: Conv3D,
    backendName: 'webgpu',
    kernelFunc: conv3D,
};

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
function conv3DBackpropFilterV2(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, pad, filterShape } = attrs;
    const convInfo = backend_util.computeConv3DInfo(x.shape, filterShape, strides, 1 /* dilations */, pad);
    const program = new Conv3DDerFilterProgram(convInfo);
    const uniformData = [
        {
            type: 'int32',
            data: [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left]
        },
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        { type: 'int32', data: [convInfo.batchSize] },
        { type: 'int32', data: [convInfo.outDepth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'int32', data: [convInfo.inDepth] },
        { type: 'int32', data: [convInfo.inHeight] },
        { type: 'int32', data: [convInfo.inWidth] }
    ];
    return backend.runWebGPUProgram(program, [x, dy], dy.dtype, uniformData);
}
const conv3DBackpropFilterV2Config = {
    kernelName: Conv3DBackpropFilterV2,
    backendName: 'webgpu',
    kernelFunc: conv3DBackpropFilterV2
};

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
function conv3DBackpropInputV2(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { strides, pad, inputShape } = attrs;
    const convInfo = backend_util.computeConv3DInfo(inputShape, filter.shape, strides, 1 /* dilations */, pad);
    const program = new Conv3DDerInputProgram(convInfo);
    const uniformData = [
        {
            type: 'int32',
            data: [convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.filterDepth - 1 - convInfo.padInfo.front,
                convInfo.filterHeight - 1 - convInfo.padInfo.top,
                convInfo.filterWidth - 1 - convInfo.padInfo.left
            ]
        },
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        { type: 'int32', data: [convInfo.outDepth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'int32', data: [convInfo.outChannels] }
    ];
    return backend.runWebGPUProgram(program, [dy, filter], dy.dtype, uniformData);
}
const conv3DBackpropInputV2Config = {
    kernelName: Conv3DBackpropInputV2,
    backendName: 'webgpu',
    kernelFunc: conv3DBackpropInputV2,
};

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
const cos = unaryKernelFunc({ opType: UnaryOpType.COS });
const cosConfig = {
    kernelName: Cos,
    backendName: 'webgpu',
    kernelFunc: cos
};

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
const cosh = unaryKernelFunc({ opType: UnaryOpType.COSH });
const coshConfig = {
    kernelName: Cosh,
    backendName: 'webgpu',
    kernelFunc: cosh
};

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
class CropAndResizeProgram {
    constructor(channnel, boxShape, cropSize, method) {
        this.variableNames = ['Image', 'Boxes', 'BoxInd'];
        this.uniforms = 'extrapolationValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const [numBoxes,] = boxShape;
        this.outputShape = [numBoxes, cropSize[0], cropSize[1], channnel];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.methodId = method === 'bilinear' ? 1 : 0;
        this.cropHeightBiggerThan1 = this.outputShape[1] > 1;
        this.cropWidthBiggerThan1 = this.outputShape[2] > 1;
        this.shaderKey = `cropAndResize_${this.methodId}_${this.cropHeightBiggerThan1}_${this.cropWidthBiggerThan1}`;
    }
    getUserCode() {
        const [inputHeightFloat, inputWidthFloat] = [`f32(uniforms.imageShape[1] - 1)`, `f32(uniforms.imageShape[2] - 1)`];
        const [heightRatio, heightScale, inY] = this.cropHeightBiggerThan1 ?
            [
                `(${inputHeightFloat} / f32(uniforms.outShape[1] - 1))`,
                '(y2-y1) * height_ratio',
                `y1*${inputHeightFloat} + f32(y)*(height_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (y1+y2) * ${inputHeightFloat}`,
            ];
        const [widthRatio, widthScale, inX] = this.cropWidthBiggerThan1 ?
            [
                `(${inputWidthFloat} / f32(uniforms.outShape[2] - 1))`,
                '(x2-x1) * width_ratio',
                `x1*${inputWidthFloat} + f32(x)*(width_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (x1+x2) * ${inputWidthFloat}`,
            ];
        // Reference implementation
        // tslint:disable-next-line:max-line-length
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let height_ratio = f32(${heightRatio});
        let width_ratio = f32(${widthRatio});
        let b = coords[0];
        let y = coords[1];
        let x = coords[2];
        let d = coords[3];
        // get box vals
        let y1 = getBoxes(b, 0);
        let x1 = getBoxes(b, 1);
        let y2 = getBoxes(b, 2);
        let x2 = getBoxes(b, 3);
        // get image in batch index
        let bInd = i32(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= uniforms.outShape[0]) {
          return;
        }
        let height_scale = ${heightScale};
        let width_scale = ${widthScale};
        let in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let sourceFracIndexCR = vec2<f32>(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          let sourceFloorCR = vec2<i32>(sourceFracIndexCR);
          let sourceCeilCR = vec2<i32>(ceil(sourceFracIndexCR));
          let topLeft = getImage(bInd, sourceFloorCR.y, sourceFloorCR.x, d);
          let bottomLeft = getImage(bInd, sourceCeilCR.y, sourceFloorCR.x, d);
          let topRight = getImage(bInd, sourceFloorCR.y, sourceCeilCR.x, d);
          let bottomRight = getImage(bInd, sourceCeilCR.y, sourceCeilCR.x, d);
          let fracCR = sourceFracIndexCR - vec2<f32>(sourceFloorCR);
          let top = topLeft + (topRight - topLeft) * fracCR.x;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          let newValue = top + (bottom - top) * fracCR.y;
          setOutputAtIndex(index, newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          let sourceNearestCR = vec2<i32>(floor(
            sourceFracIndexCR + vec2<f32>(0.5,0.5)));
          let newValue = getImage(
            bInd, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutputAtIndex(index, newValue);
        }
      }
    }
    `;
        return userCode;
    }
}

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
const cropAndResize = (args) => {
    const { inputs, backend, attrs } = args;
    const { image, boxes, boxInd } = inputs;
    const { cropSize, method, extrapolationValue } = attrs;
    const program = new CropAndResizeProgram(image.shape[3], boxes.shape, cropSize, method);
    const uniformData = [{ type: 'float32', data: [extrapolationValue] }];
    return backend.runWebGPUProgram(program, [image, boxes, boxInd], 'float32', uniformData);
};
const cropAndResizeConfig = {
    kernelName: CropAndResize,
    backendName: 'webgpu',
    kernelFunc: cropAndResize
};

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
var CumOpType;
(function (CumOpType) {
    CumOpType["Prod"] = "*";
    CumOpType["Sum"] = "+";
})(CumOpType || (CumOpType = {}));
class CumProgram {
    constructor(op, shape, exclusive, reverse) {
        this.variableNames = ['x'];
        // pow(i32, i32) is not supported, use pow(f32, f32) instead.
        this.uniforms = 'index : f32,';
        this.size = true;
        this.workgroupSize = [128, 1, 1];
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.exclusive = exclusive;
        this.reverse = reverse;
        this.op = op;
        this.shaderKey = `cum_${this.op}_${this.exclusive}_${this.reverse}`;
    }
    getUserCode() {
        const rank = this.outputShape.length;
        const initVal = this.op === CumOpType.Prod ? '1.0' : '0.0';
        const val = this.exclusive ? initVal :
            `getX(${getCoords(rank, 'coords', this.op)})`;
        const length = this.outputShape[this.outputShape.length - 1];
        let condition = '';
        let idxString = '';
        // When exclusive is set, the cum op becomes roll op that copies the
        // value from the previous index based on the direction specified by the
        // reverse flag.
        if (this.exclusive) {
            condition = this.reverse ? `end != ${length - 1}` : 'end != 0';
            idxString = this.reverse ? 'end + 1' : 'end - 1';
        }
        else {
            condition = this.reverse ? `end + pow2 < ${length}` : 'end >= pow2';
            idxString = (this.reverse ? 'end + pow2' : 'end - pow2');
        }
        return `
      ${getMainHeaderString('index')} {
       if (index < uniforms.size) {
         var coords = getCoordsFromIndex(index);

         let end = ${getFinalCoord(rank, 'coords', this.op)};
         var val = ${val};
         let pow2 = i32(pow(2.0, uniforms.index));
         if (${condition}) {
           let idx = ${idxString};
           ${getFinalCoord(rank, 'coords', this.op)} = idx;
           val ${this.op}= getX(${getCoords(rank, 'coords', this.op)});
         }
         setOutputAtIndex(index, val);
       }
      }
    `;
    }
}
function getCoords(rank, name, op) {
    if (rank === 1) {
        return `${name}`;
    }
    else if (rank === 2) {
        return `${name}.x, ${name}.y`;
    }
    else if (rank === 3) {
        return `${name}.x, ${name}.y, ${name}.z`;
    }
    else if (rank === 4) {
        return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
    }
    else {
        throw Error(`Cumulative ${op} for rank ${rank} is not yet supported`);
    }
}
function getFinalCoord(rank, name, op) {
    if (rank === 1) {
        return `${name}`;
    }
    else if (rank === 2) {
        return `${name}.y`;
    }
    else if (rank === 3) {
        return `${name}.z`;
    }
    else if (rank === 4) {
        return `${name}.w`;
    }
    else {
        throw Error(`Cumulative ${op} for rank ${rank} is not yet supported`);
    }
}

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
function cumImpl(op, x, backend, axis, exclusive, reverse) {
    const xRank = x.shape.length;
    const permutation = backend_util.getAxesPermutation([axis], xRank);
    let permutedX = x;
    if (permutation != null) {
        permutedX = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
    }
    const permutedAxis = backend_util.getInnerMostAxes(1, xRank)[0];
    if (permutedAxis !== xRank - 1) {
        throw new Error(`WebGPU cumprod shader expects an inner-most axis=${x.shape.length - 1} ` +
            `but got axis=${axis}`);
    }
    const size = permutedX.shape[permutedAxis];
    let result = identity({ inputs: { x: permutedX }, backend });
    // Use cum parallel algorithm, inspired by:
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    // Note: although the algorithm is called sum, it works for any associtative
    // operator with an identity.
    for (let i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
        const program = new CumProgram(op, permutedX.shape, false, reverse);
        const prevResult = result;
        const uniformData = [{ type: 'float32', data: [i] }];
        result =
            backend.runWebGPUProgram(program, [result], result.dtype, uniformData);
        backend.disposeData(prevResult.dataId);
    }
    // For exclusive cum, shift the end result in the direction of product or sum
    // and add 1 for product or 0 for sum to the front index.
    if (exclusive) {
        const program = new CumProgram(op, permutedX.shape, exclusive, reverse);
        const prevResult = result;
        const uniformData = [{ type: 'float32', data: [0] }];
        result =
            backend.runWebGPUProgram(program, [result], result.dtype, uniformData);
        backend.disposeData(prevResult.dataId);
    }
    if (permutation != null) {
        const reversePermutation = backend_util.getUndoAxesPermutation(permutation);
        const reverseTransposedResult = transpose({ inputs: { x: result }, backend, attrs: { perm: reversePermutation } });
        backend.disposeData(result.dataId);
        backend.disposeData(permutedX.dataId);
        return reverseTransposedResult;
    }
    return result;
}

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
function cumprod(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, exclusive, reverse } = attrs;
    return cumImpl(CumOpType.Prod, x, backend, axis, exclusive, reverse);
}
const cumprodConfig = {
    kernelName: Cumprod,
    backendName: 'webgpu',
    kernelFunc: cumprod
};

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
function cumsum(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, exclusive, reverse } = attrs;
    return cumImpl(CumOpType.Sum, x, backend, axis, exclusive, reverse);
}
const cumsumConfig = {
    kernelName: Cumsum,
    backendName: 'webgpu',
    kernelFunc: cumsum
};

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
function denseBincount(args) {
    const { inputs, backend, attrs } = args;
    const { x, weights } = inputs;
    const { size, binaryOutput } = attrs;
    const xRankOne = x.shape.length === 1;
    const weightsSize = util.sizeFromShape(weights.shape);
    const hasWeights = weightsSize > 0;
    const dtype = weights.dtype;
    const xSize = xRankOne ? [x.shape[0]] : [x.shape[0], x.shape[1]];
    const outputSize = xRankOne ? [size] : [x.shape[0], size];
    const output = fill({ backend, attrs: { shape: outputSize, value: 0, dtype } });
    const program = new BincountProgram(xSize, hasWeights, binaryOutput);
    const uniformData = [{ type: 'int32', data: [size] }];
    const bincountInputs = hasWeights ? [x, weights] : [x];
    const res = backend.runWebGPUProgram(program, bincountInputs, dtype, uniformData, output);
    return res;
}
const denseBincountConfig = {
    kernelName: DenseBincount,
    backendName: 'webgpu',
    kernelFunc: denseBincount
};

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
class DepthToSpaceProgram {
    constructor(outputShape, dataFormat) {
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.uniforms = 'blockSize : i32,';
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthToSpace_${dataFormat}`;
        this.dataFormat = dataFormat;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let h = ${this.getHeightCoordString()};
          let w = ${this.getWidthCoordString()};
          let d = ${this.getDepthCoordString()};

          let in_h = h / uniforms.blockSize;
          let offset_h = h % uniforms.blockSize;
          let in_w = w / uniforms.blockSize;
          let offset_w = w % uniforms.blockSize;
          let offset_d = (offset_h * uniforms.blockSize + offset_w) *
            ${this.getOutputDepthSize()};
          let in_d = d + offset_d;

          let rlt = ${this.getInputSamplingString()};
          setOutputAtIndex(index, rlt);
        }
      }`;
        return userCode;
    }
    getHeightCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[1]`;
        }
        else {
            return `coords[2]`;
        }
    }
    getWidthCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[2]`;
        }
        else {
            return `coords[3]`;
        }
    }
    getDepthCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[3]`;
        }
        else {
            return `coords[1]`;
        }
    }
    getOutputDepthSize() {
        if (this.dataFormat === 'NHWC') {
            return `uniforms.outShape[3]`;
        }
        else {
            return `uniforms.outShape[1]`;
        }
    }
    getInputSamplingString() {
        if (this.dataFormat === 'NHWC') {
            return `getX(b, in_h, in_w, in_d)`;
        }
        else {
            return `getX(b, in_d, in_h, in_w)`;
        }
    }
}

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
function depthToSpace(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockSize, dataFormat } = attrs;
    const batchSize = x.shape[0];
    const inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
    const inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
    const inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];
    const outputHeight = inputHeight * blockSize;
    const outputWidth = inputWidth * blockSize;
    const outputDepth = inputDepth / (blockSize * blockSize);
    const outputShape = (dataFormat === 'NHWC') ?
        [batchSize, outputHeight, outputWidth, outputDepth] :
        [batchSize, outputDepth, outputHeight, outputWidth];
    const uniformData = [
        { type: 'int32', data: [blockSize] },
    ];
    const program = new DepthToSpaceProgram(outputShape, dataFormat);
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const depthToSpaceConfig = {
    kernelName: DepthToSpace,
    backendName: 'webgpu',
    kernelFunc: depthToSpace
};

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
class DepthwiseConv2DNCHWSharedProgram {
    constructor(outputShape, filterHeight, filterWidth, addBias = false, activation = null, hasPreluActivation = false) {
        this.variableNames = ['x', 'W'];
        this.uniforms = `pads : vec2<i32>, inDims : vec2<i32>,`;
        this.workgroupSize = [16, 16, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = { x: [3], y: [2], z: [0, 1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        this.addBias = addBias;
        this.activation = activation;
        this.hasPreluActivation = hasPreluActivation;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        this.shaderKey = `depthwiseNCHW_${this.activation}_${this.filterHeight}_${this.filterWidth}`;
    }
    getUserCode() {
        const filterSize = this.filterWidth * this.filterHeight;
        const flatWorkgroupSize = this.workgroupSize[0] * this.workgroupSize[1] * this.workgroupSize[2];
        const tileAHeight = this.workgroupSize[1] + this.filterHeight - 1;
        const tileAWidth = this.workgroupSize[0] + this.filterWidth - 1;
        const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, false, 4)}

      var<workgroup> mm_Asub : array<array<f32, ${tileAWidth}>, ${tileAHeight}>;
      var<workgroup> mm_Bsub : array<array<f32, ${this.filterWidth}>, ${this.filterHeight}>;
      fn readX(batch : i32, channel : i32, row : i32, col : i32) -> f32 {
        var value = 0.0;
        if (row >=0 && row < uniforms.inDims[0] && col >=0 && col < uniforms.inDims[1])
        {
          value = getX(batch, channel, row, col);
        }
        return value;
      }

      ${getMainHeaderString()} {
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.zw) - uniforms.pads;
        let channelMul = uniforms.wShape[3];
        let d1 = coords[1] / channelMul;
        let q = coords[1] % channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;

        let localRow = i32(localId.y);
        let localCol = i32(localId.x);

        // Load one tile of X into local memory.
        for (var inputRow = localRow; inputRow < ${tileAHeight}; inputRow = inputRow + ${this.workgroupSize[1]}) {
          for (var inputCol = localCol; inputCol < ${tileAWidth}; inputCol = inputCol + ${this.workgroupSize[0]}) {
            let rowOffset = inputRow - localRow;
            let colOffset = inputCol - localCol;
            mm_Asub[inputRow][inputCol] = readX(batch, d1, inputRowStart + rowOffset, inputColStart + colOffset);
          }
        }

        // Load one tile of W into local memory.
        var wIndex = i32(localIndex);
        ${filterSize < flatWorkgroupSize ?
            `if (wIndex < ${filterSize})` :
            `for(; wIndex < ${filterSize}; wIndex = wIndex + ${flatWorkgroupSize})`}

        {
          let wRow = wIndex / ${this.filterWidth};
          let wCol = wIndex % ${this.filterWidth};
          mm_Bsub[wRow][wCol] = getW(wRow, wCol, d1, q);
        }

        workgroupBarrier();

        var value = 0.0;
        for (var wR = 0; wR < ${this.filterHeight}; wR = wR + 1) {
          for (var wC = 0; wC < ${this.filterWidth}; wC = wC + 1) {
            let xVal = mm_Asub[localRow + wR][localCol + wC];
            let wVal = mm_Bsub[wR][wC];
            value = fma(xVal, wVal, value);
          }
        }
        ${biasActivationSnippet(this.addBias, this.activation)}
        if (coordsInBounds4D(coords, uniforms.outShape)) {
          setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }
    `;
        return userCode;
    }
}

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
class DepthwiseConv2DVec4Program {
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

      ${getMainHeaderString('index')} {
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
class DepthwiseConv2DProgram {
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

      ${getMainHeaderString('index')} {
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
function depthwiseConv2dNative(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */, $dataFormat);
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] },
    ];
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    let program;
    if (!isChannelsLast && convInfo.inHeight > 16 && convInfo.inWidth > 16 &&
        convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
        convInfo.dilationWidth === 1 && convInfo.dilationHeight === 1 &&
        convInfo.inChannels === convInfo.outChannels) {
        program = new DepthwiseConv2DNCHWSharedProgram(convInfo.outShape, convInfo.filterHeight, convInfo.filterWidth);
    }
    else if (isChannelsLast && convInfo.outHeight > 4 && convInfo.outWidth > 4 &&
        convInfo.strideWidth <= 2 &&
        convInfo.inChannels === convInfo.outChannels &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.inChannels % 4 === 0) {
        program = new DepthwiseConv2DVec4Program(convInfo);
        dimensions.push({ type: 'int32', data: [program.virtualWidth] });
    }
    else {
        program = new DepthwiseConv2DProgram(convInfo);
        dimensions.push({ type: 'int32', data: [convInfo.filterHeight] }, { type: 'int32', data: [convInfo.filterWidth] }, { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [convInfo.dilationHeight, convInfo.dilationWidth]
        });
    }
    return backend.runWebGPUProgram(program, [x, filter], x.dtype, dimensions);
}
const depthwiseConv2dNativeConfig = {
    kernelName: DepthwiseConv2dNative,
    backendName: 'webgpu',
    kernelFunc: depthwiseConv2dNative,
};

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
class DepthwiseConv2DDerFilterProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'dy'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>, outHeight : i32,
      outWidth : i32, inHeight : i32, inWidth : i32, batchSize : i32, channelMul : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.filterShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthwise_conv2d_backprop_filter`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let wR = coords[0];
        let wC = coords[1];
        let d1 = coords[2];
        let dm = coords[3];
        let d2 = d1 * uniforms.channelMul + dm;

        var dotProd = 0.0;
        for (var b = 0; b < uniforms.batchSize; b++) {
          for (var yR = 0; yR < uniforms.outHeight; yR++) {
            let xR = wR + yR * uniforms.strides[0] - uniforms.pads[0];

            if (xR < 0 || xR >= uniforms.inHeight) {
              continue;
            }

            for (var yC = 0; yC < uniforms.outWidth; yC++) {
              let xC = wC + yC * uniforms.strides[1] - uniforms.pads[1];

              if (xC < 0 || xC >= uniforms.inWidth) {
                continue;
              }

              let dyValue = getDy(b, yR, yC, d2);
              let xValue = getX(b, xR, xC, d1);
              dotProd += xValue * dyValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
class DepthwiseConv2DDerInputProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'W'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32, channelMul : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthwise_conv2d_backprop_input`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d1 = coords[3];
        let dyCorner = coords.yz - uniforms.pads;
        let dyRCorner = dyCorner.x;
        let dyCCorner = dyCorner.y;

        var dotProd = 0.0;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }

          let idyR = i32(dyR);
          let wRPerm = uniforms.filterDims[0] - 1 - wR;

          for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }

            let idyC = i32(dyC);
            let wCPerm = uniforms.filterDims[1] - 1 - wC;

            for (var dm = 0; dm < uniforms.channelMul; dm++) {
              let d2 = d1 * uniforms.channelMul + dm;
              let xValue = getDy(batch, idyR, idyC, d2);
              let wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}

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
function depthwiseConv2dNativeBackpropFilter(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, dilations, pad, dimRoundingMode, filterShape } = attrs;
    const convInfo = backend_util.computeConv2DInfo(x.shape, filterShape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    const program = new DepthwiseConv2DDerFilterProgram(convInfo);
    const uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'int32', data: [convInfo.inHeight] },
        { type: 'int32', data: [convInfo.inWidth] },
        { type: 'int32', data: [convInfo.batchSize] },
        { type: 'int32', data: [convInfo.outChannels / convInfo.inChannels] }
    ];
    return backend.runWebGPUProgram(program, [x, dy], 'float32', uniformData);
}
const depthwiseConv2dNativeBackpropFilterConfig = {
    kernelName: DepthwiseConv2dNativeBackpropFilter,
    backendName: 'webgpu',
    kernelFunc: depthwiseConv2dNativeBackpropFilter
};

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
function depthwiseConv2dNativeBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { strides, dilations, pad, dimRoundingMode, inputShape } = attrs;
    const convInfo = backend_util.computeConv2DInfo(inputShape, filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    const program = new DepthwiseConv2DDerInputProgram(convInfo);
    const uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [
                convInfo.filterHeight - 1 - convInfo.padInfo.top,
                convInfo.filterWidth - 1 - convInfo.padInfo.left
            ]
        },
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] },
        { type: 'int32', data: [convInfo.outChannels / convInfo.inChannels] }
    ];
    return backend.runWebGPUProgram(program, [dy, filter], dy.dtype, uniformData);
}
const depthwiseConv2dNativeBackpropInputConfig = {
    kernelName: DepthwiseConv2dNativeBackpropInput,
    backendName: 'webgpu',
    kernelFunc: depthwiseConv2dNativeBackpropInput
};

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
class DiagProgram {
    constructor(size) {
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [size, size];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'diag';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let value = select(0.0, getX(coords[0]), coords[0] == coords[1]);
          setOutputAtIndex(index, value);
        }
      }
    `;
        return userCode;
    }
}

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
function diag(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    const outShape = [...x.shape, ...x.shape];
    const xSize = util.sizeFromShape(x.shape);
    const flat = reshape({ inputs: { x }, backend, attrs: { shape: [xSize] } });
    const program = new DiagProgram(xSize);
    const res = backend.runWebGPUProgram(program, [flat], flat.dtype);
    const out = reshape({ inputs: { x: res }, backend, attrs: { shape: outShape } });
    backend.disposeData(flat.dataId);
    backend.disposeData(res.dataId);
    return out;
}
const diagConfig = {
    kernelName: Diag,
    backendName: 'webgpu',
    kernelFunc: diag
};

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
class Dilation2DProgram {
    constructor(convInfo) {
        this.variableNames = ['x', 'w'];
        this.uniforms = 'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'dilation2d';
    }
    getUserCode() {
        const userCode = `
       ${getMainHeaderString('index')} {
         if (index < uniforms.size) {
           let neg_infinity = -3.4e38;
           let coords = getOutputCoords();
           let batch = coords.x;
           let d1 = coords.w;
           let outTopLeftCorner = coords.yz * uniforms.strides - uniforms.pads;
           let hBeg = outTopLeftCorner.x;
           let wBeg = outTopLeftCorner.y;

           var curVal = neg_infinity;
           for (var h = 0; h < uniforms.filterDims[0]; h = h + 1) {
             let hIn = hBeg + h * uniforms.dilations[0];

             if (hIn >= 0 && hIn < uniforms.xShape[1]) {
               for (var w = 0; w < uniforms.filterDims[1]; w = w + 1) {
                 let wIn = wBeg + w * uniforms.dilations[1];

                 if (wIn >= 0 && wIn < uniforms.xShape[2]) {
                   let val = getX(batch, hIn, wIn, d1) + getW(h, w, d1);
                   if (val > curVal) {
                     curVal = val;
                   }
                 }
               }
             }
           }

           setOutputAtIndex(index, curVal);
         }
       }
     `;
        return userCode;
    }
}

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
function dilation2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations } = attrs;
    const convInfo = backend_util.computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
    const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
    const uniformData = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [...padInfo] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }
    ];
    const program = new Dilation2DProgram(convInfo);
    const out = backend.runWebGPUProgram(program, [x, filter], x.dtype, uniformData);
    return out;
}
const dilation2DConfig = {
    kernelName: Dilation2D,
    backendName: 'webgpu',
    kernelFunc: dilation2D
};

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
class Dilation2DBackpropInputProgram {
    constructor(convInfo, outputDtype) {
        this.variableNames = ['x', 'w', 'dy'];
        this.uniforms = 'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(convInfo.outShape);
        this.dispatch = computeDispatch(this.dispatchLayout, convInfo.outShape, this.workgroupSize);
        if (outputDtype !== 'float32' && outputDtype !== 'int32') {
            throw new Error(`Dilation2DBackpropInput only supports float32 and int32
          types, does not support ${outputDtype} type.`);
        }
        this.type = outputDtype;
        this.shaderKey = 'dilation2DBackpropInput';
    }
    getUserCode() {
        // This implementation follows the TF c++ cuda implementation:
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
        const userCode = `
       ${getMainHeaderString('index')} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var xRMax = 0;
           var xCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     xRMax = xR;
                     xCMax = xC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.xShape[3] *
               (xCMax + uniforms.xShape[2] * (xRMax + uniforms.xShape[1] * b));
           let value = getDy(b, r, c, d);
           ${atomicAddSnippet('&result[flatIndexIn]', 'value', this.type)}
         }
       }
     `;
        return userCode;
    }
}
class Dilation2DBackpropFilterProgram {
    constructor(convInfo, shape, outputDtype) {
        this.variableNames = ['x', 'w', 'dy'];
        this.uniforms = 'filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>, dySize: i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = convInfo.filterShape;
        this.dispatchLayout = flatDispatchLayout(convInfo.outShape);
        this.dispatch = computeDispatch(this.dispatchLayout, convInfo.outShape, this.workgroupSize);
        if (outputDtype !== 'float32' && outputDtype !== 'int32') {
            throw new Error(`Dilation2DBackpropFilter only supports float32 and int32
          types, does not support ${outputDtype} type.`);
        }
        this.type = outputDtype;
        this.shaderKey = 'dilation2DBackpropFilter';
    }
    getUserCode() {
        // This implementation follows the TF c++ cuda implementation:
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/dilation_ops_gpu.cu.cc
        const userCode = `
       ${getMainHeaderString('index')} {
         if (index < uniforms.dySize) {
           let coords = getDyCoordsFromIndex(index);
           let b = coords[0];
           let r = coords[1];
           let c = coords[2];
           let d = coords[3];

           let dyCorner = vec2<i32>(r, c) * uniforms.strides - uniforms.pads;
           var curVal = -3.4e38;  // neg_infinity
           var wRMax = 0;
           var wCMax = 0;

           // In the case of multiple argmax branches, we only back-propagate
           // along the last branch, i.e., the one with largest value of
           // 'wR * uniforms.filterDims[1] + wC', similarly to the max-pooling
           // backward routines.
           for (var wR = 0; wR < uniforms.filterDims[0]; wR++) {
             let xR = dyCorner.x + wR * uniforms.dilations[0];

             if (xR >= 0 && xR < uniforms.xShape[1]) {
               for (var wC = 0; wC < uniforms.filterDims[1]; wC++) {
                 let xC = dyCorner.y + wC * uniforms.dilations[1];

                 if (xC >= 0 && xC < uniforms.xShape[2]) {
                   let val = getX(b, xR, xC, d) + getW(wR, wC, d);
                   if (val > curVal) {
                     curVal = val;
                     wRMax = wR;
                     wCMax = wC;
                   }
                 }
               }
             }
           }

           let flatIndexIn = d + uniforms.wShape[2] * (wCMax + wRMax * uniforms.wShape[1]);
           let value = getDy(b, r, c, d);
           ${atomicAddSnippet('&result[flatIndexIn]', 'value', this.type)}
         }
       }
     `;
        return userCode;
    }
}

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
function dilation2DBackpropFilter(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, dy } = inputs;
    const { strides, pad, dilations } = attrs;
    const convInfo = backend_util.computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
    const dtype = filter.dtype;
    const program = new Dilation2DBackpropFilterProgram(convInfo, filter.shape, dtype);
    const uniformData = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [util.sizeFromShape(convInfo.outShape)] }
    ];
    const output = fill({ backend, attrs: { shape: filter.shape, value: 0, dtype } });
    return backend.runWebGPUProgram(program, [x, filter, dy], dtype, uniformData, output);
}
const dilation2DBackpropFilterConfig = {
    kernelName: Dilation2DBackpropFilter,
    backendName: 'webgpu',
    kernelFunc: dilation2DBackpropFilter
};

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
function dilation2DBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, dy } = inputs;
    const { strides, pad, dilations } = attrs;
    const convInfo = backend_util.computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
    const dtype = x.dtype;
    const program = new Dilation2DBackpropInputProgram(convInfo, dtype);
    const uniformData = [
        { type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth] },
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [util.sizeFromShape(convInfo.outShape)] }
    ];
    const output = fill({ backend, attrs: { shape: convInfo.inShape, value: 0, dtype } });
    return backend.runWebGPUProgram(program, [x, filter, dy], dtype, uniformData, output);
}
const dilation2DBackpropInputConfig = {
    kernelName: Dilation2DBackpropInput,
    backendName: 'webgpu',
    kernelFunc: dilation2DBackpropInput
};

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
class DrawProgram {
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
       ${getMainHeaderString('index')} {
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

/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
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
function draw(args) {
    const { inputs, backend, attrs } = args;
    const { image } = inputs;
    const { canvas, options } = attrs;
    const [height, width] = image.shape.slice(0, 2);
    const { imageOptions } = options || {};
    const alpha = (imageOptions === null || imageOptions === void 0 ? void 0 : imageOptions.alpha) || 1;
    //  'rgba8unorm' should work on macOS according to
    //  https://bugs.chromium.org/p/chromium/issues/detail?id=1298618. But
    //  failed on macOS/M2. So use 'bgra8unorm' first when available.
    const format = backend.device.features.has('bgra8unorm-storage') ?
        'bgra8unorm' :
        'rgba8unorm';
    const outShape = [height, width];
    const program = new DrawProgram(outShape, image.dtype, format);
    canvas.width = width;
    canvas.height = height;
    const backendName = 'webgpu';
    let gpuContext = canvas.getContext(backendName);
    let canvasWebGPU;
    if (!gpuContext) {
        canvasWebGPU = new OffscreenCanvas(width, height);
        gpuContext = canvasWebGPU.getContext(backendName);
    }
    const numChannels = image.shape.length === 3 ? image.shape[2] : 1;
    gpuContext.configure({
        device: backend.device,
        format,
        usage: GPUTextureUsage.STORAGE_BINDING,
        alphaMode: 'premultiplied'
    });
    const outputDtype = 'int32';
    const output = backend.makeTensorInfo(outShape, outputDtype);
    const info = backend.tensorMap.get(output.dataId);
    info.resource = gpuContext.getCurrentTexture();
    info.external = true;
    const uniformData = [{ type: 'uint32', data: [numChannels] }, { type: 'float32', data: [alpha] }];
    backend.runWebGPUProgram(program, [image], outputDtype, uniformData, output);
    if (canvasWebGPU) {
        const canvas2dContext = canvas.getContext('2d');
        if (!canvas2dContext) {
            throw new Error(`Please make sure this canvas has only been used for 2d or webgpu context!`);
        }
        canvas2dContext.drawImage(canvasWebGPU, 0, 0);
    }
    backend.disposeData(output.dataId);
    return image;
}
const drawConfig = {
    kernelName: Draw,
    backendName: 'webgpu',
    kernelFunc: draw
};

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
const multiplyKernelFunc = binaryKernelFunc({
    opType: BinaryOpType.MUL,
    cpuKernelImpl: multiplyImplCPU,
    supportsComplex: true
});
const multiplyConfig = {
    kernelName: Multiply,
    backendName: 'webgpu',
    kernelFunc: multiplyKernelFunc
};

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
function sum(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'sum', backend);
}
const sumConfig = {
    kernelName: Sum,
    backendName: 'webgpu',
    kernelFunc: sum
};

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
function einsum(args) {
    const { inputs, backend, attrs } = args;
    const { equation } = attrs;
    const tensors = inputs;
    const { allDims, summedDims, idDims } = backend_util.decodeEinsumEquation(equation, tensors.length);
    backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
    const { path, steps } = backend_util.getEinsumComputePath(summedDims, idDims);
    const nSteps = steps.length;
    let out = null;
    let numDimsRemaining = allDims.length;
    const tensorsToDispose = [];
    for (let i = 0; i < nSteps; ++i) {
        for (const idTerm of steps[i]) {
            const { permutationIndices: perm, expandDims: dimsToExpand } = backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]);
            let x;
            if (backend_util.isIdentityPermutation(perm)) {
                x = tensors[idTerm];
            }
            else {
                x = transpose({ inputs: { x: tensors[idTerm] }, backend, attrs: { perm } });
                tensorsToDispose.push(x);
            }
            const targetShape = x.shape.slice();
            for (let k = 0; k < dimsToExpand.length; ++k) {
                targetShape.splice(dimsToExpand[k], 0, 1);
            }
            if (!util.arraysEqual(x.shape, targetShape)) {
                x = reshape({ inputs: { x }, backend, attrs: { shape: targetShape } });
                tensorsToDispose.push(x);
            }
            if (out === null) {
                out = x;
            }
            else {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                out =
                    multiplyKernelFunc({ inputs: { a: x, b: out }, backend });
                tensorsToDispose.push(out);
            }
        }
        if (i < nSteps - 1) {
            if (path[i] >= 0) {
                out = sum({
                    inputs: { x: out },
                    backend,
                    attrs: {
                        axis: path[i] - (allDims.length - numDimsRemaining),
                        keepDims: false
                    }
                });
                tensorsToDispose.push(out);
            }
            numDimsRemaining--;
        }
    }
    // Clean up intermediate tensors.
    for (const tensorInfo of tensorsToDispose) {
        if (tensorInfo === out) {
            continue;
        }
        backend.disposeData(tensorInfo.dataId);
    }
    return out;
}
const einsumConfig = {
    kernelName: Einsum,
    backendName: 'webgpu',
    kernelFunc: einsum
};

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
const elu = unaryKernelFunc({ opType: UnaryOpType.ELU });
const eluConfig = {
    kernelName: Elu,
    backendName: 'webgpu',
    kernelFunc: elu
};

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
const eluGrad = (args) => {
    const { inputs, backend } = args;
    const { dy, y } = inputs;
    const program = new BinaryOpProgram(BinaryOpType.ELU_DER, dy.shape, y.shape);
    return backend.runWebGPUProgram(program, [dy, y], dy.dtype);
};
const eluGradConfig = {
    kernelName: EluGrad,
    backendName: 'webgpu',
    kernelFunc: eluGrad
};

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
const equal = binaryKernelFunc({ opType: BinaryOpType.EQUAL, dtype: 'bool', cpuKernelImpl: equalImplCPU });
const equalConfig = {
    kernelName: Equal,
    backendName: 'webgpu',
    kernelFunc: equal
};

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
const erf = unaryKernelFunc({ opType: UnaryOpType.ERF });
const erfConfig = {
    kernelName: Erf,
    backendName: 'webgpu',
    kernelFunc: erf
};

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
const exp = unaryKernelFunc({
    opType: UnaryOpType.EXP,
    cpuKernelImpl: expImplCPU,
    dtype: 'float32',
});
const expConfig = {
    kernelName: Exp,
    backendName: 'webgpu',
    kernelFunc: exp
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function expandDims(args) {
    const { inputs, attrs, backend } = args;
    const { dim } = attrs;
    const { input } = inputs;
    const inputRank = input.shape.length;
    const newShape = input.shape.slice();
    let $dim = dim;
    if (dim < 0) {
        // Negative value is counted from the tail of rank.
        util.assert(-(inputRank + 1) <= dim, () => `Axis must be in the interval [${-(inputRank + 1)}, ${inputRank}]`);
        $dim = inputRank + dim + 1;
    }
    newShape.splice($dim, 0, 1);
    return reshape({ inputs: { x: input }, backend, attrs: { shape: newShape } });
}
const expandDimsConfig = {
    kernelName: ExpandDims,
    backendName: 'webgpu',
    kernelFunc: expandDims,
};

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
const expm1 = unaryKernelFunc({ opType: UnaryOpType.EXPM1, cpuKernelImpl: expm1ImplCPU });
const expm1Config = {
    kernelName: Expm1,
    backendName: 'webgpu',
    kernelFunc: expm1
};

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
class FFTProgram {
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

    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        setOutputAtIndex(index, mulMatDFT(coords[0], coords[1]));
      }
    }
  `;
        return userCode;
    }
}

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
function fftImpl(x, inverse, backend) {
    const xData = backend.tensorMap.get(x.dataId);
    const inputSize = util.sizeFromShape(x.shape);
    // Collapse all outer dimensions to a single batch dimension.
    const innerDimensionSize = x.shape[x.shape.length - 1];
    const batch = inputSize / innerDimensionSize;
    const toDispose = [];
    const input2D = reshape({ inputs: { x }, backend, attrs: { shape: [batch, innerDimensionSize] } });
    toDispose.push(input2D);
    const xShape = input2D.shape;
    const realProgram = new FFTProgram('real', xShape);
    const imagProgram = new FFTProgram('imag', xShape);
    const inputs = [
        {
            dataId: xData.complexTensorInfos.real.dataId,
            dtype: xData.complexTensorInfos.real.dtype,
            shape: xShape
        },
        {
            dataId: xData.complexTensorInfos.imag.dataId,
            dtype: xData.complexTensorInfos.imag.dtype,
            shape: xShape
        }
    ];
    const exponentMultiplier = inverse ? 2.0 * Math.PI : -2.0 * Math.PI;
    const denominator = inverse ? xShape[1] : 1.0;
    const uniformData = [
        { type: 'float32', data: [exponentMultiplier] },
        { type: 'float32', data: [denominator] }
    ];
    const realPart = backend.runWebGPUProgram(realProgram, inputs, 'float32', uniformData);
    toDispose.push(realPart);
    const imagPart = backend.runWebGPUProgram(imagProgram, inputs, 'float32', uniformData);
    toDispose.push(imagPart);
    const complexOutput = complex({ inputs: { real: realPart, imag: imagPart }, backend });
    toDispose.push(complexOutput);
    const complexOutputReshaped = reshape({ inputs: { x: complexOutput }, backend, attrs: { shape: x.shape } });
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return complexOutputReshaped;
}

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
function fft(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    return fftImpl(input, false /* inverse */, backend);
}
const fftConfig = {
    kernelName: FFT,
    backendName: 'webgpu',
    kernelFunc: fft
};

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
class FlipLeftRightProgram {
    constructor(imageShape) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = imageShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'flipLeftRight';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let coordX = uniforms.xShape[2] - coords[2] - 1;
          let outputValue = getX(coords[0], coords[1], coordX, coords[3]);
          setOutputAtIndex(index, outputValue);
        }
      }
    `;
        return userCode;
    }
}

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
const flipLeftRightConfig = {
    kernelName: FlipLeftRight,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, backend }) => {
        const { image } = inputs;
        const webgpuBackend = backend;
        const program = new FlipLeftRightProgram(image.shape);
        const output = webgpuBackend.runWebGPUProgram(program, [image], image.dtype);
        return output;
    }
};

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
const floor = unaryKernelFunc({ opType: UnaryOpType.FLOOR, cpuKernelImpl: floorImplCPU });
const floorConfig = {
    kernelName: Floor,
    backendName: 'webgpu',
    kernelFunc: floor
};

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
const floorDiv = binaryKernelFunc({
    opType: BinaryOpType.FLOOR_DIV,
    cpuKernelImpl: floorDivImplCPU,
    dtype: 'int32'
});
const floorDivConfig = {
    kernelName: FloorDiv,
    backendName: 'webgpu',
    kernelFunc: floorDiv
};

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
class FromPixelsProgram {
    constructor(outputShape, numChannels, importVideo = false) {
        this.pixelsOpType = PixelsOpType.FROM_PIXELS;
        this.outputShape = [0];
        this.variableNames = [];
        this.workgroupSize = [256, 1, 1]; // The empirical value.
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [numChannels, 1, 1]);
        this.importVideo = importVideo;
        this.shaderKey = `fromPixels_${this.importVideo}`;
    }
    getUserCode() {
        const textureLoad = this.importVideo ?
            'textureLoad(src, vec2<i32>(coords.yx));' :
            'textureLoad(src, vec2<i32>(coords.yx), 0)';
        const textureType = this.importVideo ? 'texture_external' : 'texture_2d<f32>';
        return `
      @binding(1) @group(0) var src: ${textureType};
      ${getMainHeaderString('index')} {
        let flatIndex = index * uniforms.numChannels;
        if (flatIndex < uniforms.size) {
          let coords = getCoordsFromIndex(flatIndex);
          let values = ${textureLoad};
          for (var i = 0; i < uniforms.numChannels; i = i + 1) {
            result[flatIndex + i] = i32(floor(255.0 * values[i]));
          }
        }
      }
  `;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
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
const fromPixelsConfig = {
    kernelName: FromPixels,
    backendName: 'webgpu',
    kernelFunc: fromPixels,
};
let fromPixels2DContext;
let willReadFrequently = env().getBool('CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU');
function fromPixels(args) {
    const { inputs, backend, attrs } = args;
    let { pixels } = inputs;
    const { numChannels } = attrs;
    if (pixels == null) {
        throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
    }
    const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement;
    const isImage = typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement;
    const isCanvas = (typeof (HTMLCanvasElement) !== 'undefined' &&
        pixels instanceof HTMLCanvasElement) ||
        (typeof (OffscreenCanvas) !== 'undefined' &&
            pixels instanceof OffscreenCanvas);
    const isImageBitmap = typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap;
    const [width, height] = isVideo ?
        [
            pixels.videoWidth,
            pixels.videoHeight
        ] :
        [pixels.width, pixels.height];
    const outputShape = [height, width, numChannels];
    const importVideo = env().getBool('WEBGPU_IMPORT_EXTERNAL_TEXTURE') && isVideo;
    const isVideoOrImage = isVideo || isImage;
    if (isImageBitmap || isCanvas || isVideoOrImage) {
        let resource;
        if (importVideo) {
            resource = backend.device.importExternalTexture({ source: pixels });
        }
        else {
            if (isVideoOrImage) {
                const newWillReadFrequently = env().getBool('CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU');
                if (fromPixels2DContext == null ||
                    newWillReadFrequently !== willReadFrequently) {
                    willReadFrequently = newWillReadFrequently;
                    fromPixels2DContext = document.createElement('canvas').getContext('2d', { willReadFrequently });
                }
                fromPixels2DContext.canvas.width = width;
                fromPixels2DContext.canvas.height = height;
                fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
                pixels = fromPixels2DContext.canvas;
            }
            const usage = GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
            const format = 'rgba8unorm';
            const texture = backend.textureManager.acquireTexture(outputShape[1], outputShape[0], format, usage);
            backend.queue.copyExternalImageToTexture({ source: pixels }, { texture }, [outputShape[1], outputShape[0]]);
            resource = texture;
        }
        const size = util.sizeFromShape(outputShape);
        const strides = util.computeStrides(outputShape);
        const program = new FromPixelsProgram(outputShape, numChannels, importVideo);
        const uniformData = [
            { type: 'uint32', data: [size] }, { type: 'uint32', data: [numChannels] },
            { type: 'uint32', data: [...strides] }
        ];
        const input = backend.makeTensorInfo([height, width], 'int32');
        const info = backend.tensorMap.get(input.dataId);
        info.resource = resource;
        const result = backend.runWebGPUProgram(program, [input], 'int32', uniformData);
        backend.disposeData(input.dataId);
        return result;
    }
    // TODO: Encoding should happen on GPU once we no longer have to download
    // image data to the CPU.
    const imageData = pixels.data;
    let pixelArray = imageData;
    if (numChannels != null && numChannels !== 4) {
        pixelArray = new Uint8Array(pixels.width * pixels.height * numChannels);
        const dataLength = imageData.length;
        let j = 0;
        for (let i = 0; i < dataLength; i++) {
            if (i % 4 < numChannels) {
                pixelArray[j++] = imageData[i];
            }
        }
    }
    const output = backend.makeTensorInfo(outputShape, 'int32', new Int32Array(pixelArray));
    backend.uploadToGPU(output.dataId);
    return output;
}

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
class BatchNormProgram {
    constructor(xShape, meanShape, varianceShape, offsetShape, scaleShape) {
        this.uniforms = 'varianceEpsilon : f32,';
        // This is an experimental value.
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        this.variableNames = ['x', 'mean', 'variance'];
        backend_util.assertAndGetBroadcastShape(xShape, meanShape);
        backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        if (offsetShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
            this.variableNames.push('offset');
        }
        if (scaleShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
            this.variableNames.push('scale');
        }
        this.offsetShape = offsetShape;
        this.scaleShape = scaleShape;
        this.shaderKey = 'batchNorm';
    }
    getUserCode() {
        let offsetSnippet = '0.0';
        if (this.offsetShape != null) {
            offsetSnippet = 'getOffsetByOutputIndex(index)';
        }
        let scaleSnippet = '1.0';
        if (this.scaleShape != null) {
            scaleSnippet = 'getScaleByOutputIndex(index)';
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size)
        {
          let xValue = getXByOutputIndex(index);
          let meanValue = getMeanByOutputIndex(index);
          let varianValue = getVarianceByOutputIndex(index);
          let offsetValue = ${offsetSnippet};
          let scaleValue = ${scaleSnippet};
          let inv = scaleValue * inverseSqrt(varianValue + f32(uniforms.varianceEpsilon));
          setOutputAtIndex(index,dot(vec3<f32>(xValue, -meanValue, offsetValue), vec3<f32>(inv, inv, 1.0)));
        }
      }
  `;
        return userCode;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const fusedBatchNormConfig = {
    kernelName: FusedBatchNorm,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x, scale, offset, mean, variance } = inputs;
        const { varianceEpsilon } = attrs;
        const webGPUBackend = backend;
        const batchNormInputs = [x, mean, variance];
        let offsetShape = null;
        if (offset != null) {
            offsetShape = offset.shape;
            batchNormInputs.push(offset);
        }
        let scaleShape = null;
        if (scale != null) {
            scaleShape = scale.shape;
            batchNormInputs.push(scale);
        }
        const program = new BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape);
        const uniformData = [{ type: 'float32', data: [varianceEpsilon] }];
        return webGPUBackend.runWebGPUProgram(program, batchNormInputs, x.dtype, uniformData);
    }
};

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
function fusedConv2d(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    return conv2DImpl({
        x,
        filter,
        convInfo,
        backend,
        bias,
        preluActivationWeights,
        leakyreluAlpha,
        activation
    });
}
const fusedConv2DConfig = {
    kernelName: FusedConv2D,
    backendName: 'webgpu',
    kernelFunc: fusedConv2d,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function fusedDepthwiseConv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    util.assert(backend_util.eitherStridesOrDilationsAreOne(strides, $dilations), () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
        `1. Got strides ${strides} and dilations '${$dilations}'`);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
    const programInputs = [x, filter];
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasBias) {
        programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
        programInputs.push(preluActivationWeights);
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] },
    ];
    let program;
    if (convInfo.outHeight > 4 && convInfo.outWidth > 4 &&
        convInfo.strideWidth <= 2 &&
        convInfo.inChannels === convInfo.outChannels &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.inChannels % 4 === 0) {
        program = new DepthwiseConv2DVec4Program(convInfo, hasBias, activation, hasPreluActivationWeights);
        dimensions.push({ type: 'int32', data: [program.virtualWidth] });
    }
    else {
        program = new DepthwiseConv2DProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
        dimensions.push({ type: 'int32', data: [convInfo.filterHeight] }, { type: 'int32', data: [convInfo.filterWidth] }, { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [convInfo.dilationHeight, convInfo.dilationWidth]
        });
    }
    if (activation === 'leakyrelu') {
        dimensions.push({ type: 'float32', data: [leakyreluAlpha] });
        program.uniforms += ' alpha : f32,';
    }
    const result = backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);
    return result;
}
const fusedDepthwiseConv2DConfig = {
    kernelName: FusedDepthwiseConv2D,
    backendName: 'webgpu',
    kernelFunc: fusedDepthwiseConv2D,
};

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
class GatherNDProgram {
    constructor(sliceDim, shape) {
        this.variableNames = ['A', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `gathernd_${sliceDim}`;
        this.sliceDim = sliceDim;
        this.uniforms = `sliceDim : i32, strides : ${getCoordsDataType(sliceDim)},`;
    }
    getUserCode() {
        let strideString;
        if (this.sliceDim > 1) {
            strideString = 'uniforms.strides[j]';
        }
        else {
            strideString = 'uniforms.strides';
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          var flattenIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexTemp = i32(round(getIndices(coords[0], j)));
            let strideNum = ${strideString};
            flattenIndex = flattenIndex + indexTemp * strideNum;
          }

          setOutputAtIndex(index, getA(flattenIndex, coords[1]));
        }
      }
      `;
        return userCode;
    }
}

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
function gatherNd(args) {
    const { inputs, backend } = args;
    const { params, indices } = inputs;
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];
    const paramsSize = util.sizeFromShape(params.shape);
    const [resultShape, numSlices, sliceSize, strides] = backend_util.prepareAndValidate(params, indices);
    const flattenIndices = reshape({ inputs: { x: indices }, backend, attrs: { shape: [numSlices, sliceRank] } });
    const flattenX = reshape({
        inputs: { x: params },
        backend,
        attrs: { shape: [(util.sizeFromShape(params.shape) / sliceSize), sliceSize] }
    });
    if (backend.shouldExecuteOnCPU([params, indices]) ||
        params.dtype === 'string') {
        const indicesData = backend.readSync(indices.dataId);
        const paramsBuf = backend.bufferSync(params);
        const outValue = gatherNdImplCPU(indicesData, paramsBuf, params.dtype, numSlices, sliceRank, sliceSize, strides, params.shape, paramsSize);
        return backend.makeTensorInfo(resultShape, params.dtype, outValue.values);
    }
    const program = new GatherNDProgram(sliceRank, [numSlices, sliceSize]);
    const uniformData = [{ type: 'int32', data: [sliceRank] }, { type: 'int32', data: strides }];
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndices], flattenX.dtype, uniformData);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: resultShape } });
    backend.disposeData(flattenIndices.dataId);
    backend.disposeData(flattenX.dataId);
    backend.disposeData(res.dataId);
    return reshaped;
}
const gatherNdConfig = {
    kernelName: GatherNd,
    backendName: 'webgpu',
    kernelFunc: gatherNd
};

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
class GatherProgram {
    constructor(aShape, outputShape) {
        this.variableNames = ['A', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = aShape.slice();
        this.aShape = aShape;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `gather`;
    }
    getUserCode() {
        const sourceCoords = getSourceCoords$1(this.aShape);
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let indexZ = i32(getIndices(resRC.x, resRC.z));
          let inBounds = select(0.0, 1.0, indexZ >= 0 && indexZ < uniforms.aShape[2]);
          setOutputAtIndex(index, inBounds * getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
// The input and output are always flattened into rank 4 tensors.
function getSourceCoords$1(aShape) {
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < aShape.length; i++) {
        if (i === 2) {
            sourceCoords.push('indexZ');
        }
        else {
            sourceCoords.push(`${currentCoords[i]}`);
        }
    }
    return sourceCoords.join();
}

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
function gatherV2(args) {
    const { inputs, backend, attrs } = args;
    const { x, indices } = inputs;
    const { axis, batchDims } = attrs;
    // Unlike WebGL, WebGPU won't check if index is out of bound by calling
    // backend.readSync() function in debug mode.
    const parsedAxis = util.parseAxisParam(axis, x.shape)[0];
    const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(x, indices, parsedAxis, batchDims);
    const indicesSize = util.sizeFromShape(indices.shape);
    const toDispose = [];
    const flattenX = reshape({
        inputs: { x },
        backend,
        attrs: {
            shape: [
                shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
                shapeInfo.sliceSize
            ]
        }
    });
    const flattenIndex = reshape({
        inputs: { x: indices },
        backend,
        attrs: { shape: [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize] }
    });
    toDispose.push(flattenX);
    toDispose.push(flattenIndex);
    const flattenOutputShape = [
        shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
        shapeInfo.sliceSize
    ];
    if (backend.shouldExecuteOnCPU([x, indices])) {
        const indicesTensorData = backend.tensorMap.get(flattenIndex.dataId);
        const indicesValues = indicesTensorData.values;
        const indicesBuffer = buffer(flattenIndex.shape, flattenIndex.dtype, indicesValues);
        const flattenXTensorData = backend.tensorMap.get(flattenX.dataId);
        const xValues = flattenXTensorData.values;
        const xBuffer = buffer(flattenX.shape, flattenX.dtype, xValues);
        const outBuf = gatherV2ImplCPU(xBuffer, indicesBuffer, flattenOutputShape);
        toDispose.forEach(t => backend.disposeData(t.dataId));
        return backend.makeTensorInfo(shapeInfo.outputShape, outBuf.dtype, outBuf.values);
    }
    const program = new GatherProgram(flattenX.shape, flattenOutputShape);
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndex], flattenX.dtype);
    toDispose.push(res);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: shapeInfo.outputShape } });
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return reshaped;
}
const gatherV2Config = {
    kernelName: GatherV2,
    backendName: 'webgpu',
    kernelFunc: gatherV2
};

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
const greater = binaryKernelFunc({
    opType: BinaryOpType.GREATER,
    cpuKernelImpl: greaterImplCPU,
    dtype: 'bool',
});
const greaterConfig = {
    kernelName: Greater,
    backendName: 'webgpu',
    kernelFunc: greater
};

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
const greaterEqual = binaryKernelFunc({
    opType: BinaryOpType.GREATER_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: greaterEqualImplCPU
});
const greaterEqualConfig = {
    kernelName: GreaterEqual,
    backendName: 'webgpu',
    kernelFunc: greaterEqual
};

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
function ifft(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    return fftImpl(input, true /* inverse */, backend);
}
const ifftConfig = {
    kernelName: IFFT,
    backendName: 'webgpu',
    kernelFunc: ifft
};

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
const isFinite = unaryKernelFunc({ opType: UnaryOpType.IS_FINITE, dtype: 'bool' });
const isFiniteConfig = {
    kernelName: IsFinite,
    backendName: 'webgpu',
    kernelFunc: isFinite
};

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
const isInf = unaryKernelFunc({ opType: UnaryOpType.IS_INF, dtype: 'bool' });
const isInfConfig = {
    kernelName: IsInf,
    backendName: 'webgpu',
    kernelFunc: isInf
};

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
const isNaN = unaryKernelFunc({ opType: UnaryOpType.IS_NAN, dtype: 'bool' });
const isNaNConfig = {
    kernelName: IsNan,
    backendName: 'webgpu',
    kernelFunc: isNaN
};

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
function leakyRelu(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { alpha } = attrs;
    const uniformData = [{ type: 'float32', data: [alpha] }];
    const program = new UnaryOpProgram(x.shape, UnaryOpType.LEAKYRELU, 'alpha : f32,');
    return backend.runWebGPUProgram(program, [x], 'float32', uniformData);
}
const leakyReluConfig = {
    kernelName: LeakyRelu,
    backendName: 'webgpu',
    kernelFunc: leakyRelu
};

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
const less = binaryKernelFunc({ opType: BinaryOpType.LESS, dtype: 'bool', cpuKernelImpl: lessImplCPU });
const lessConfig = {
    kernelName: Less,
    backendName: 'webgpu',
    kernelFunc: less
};

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
const lessEqual = binaryKernelFunc({
    opType: BinaryOpType.LESS_EQUAL,
    dtype: 'bool',
    cpuKernelImpl: lessEqualImplCPU
});
const lessEqualConfig = {
    kernelName: LessEqual,
    backendName: 'webgpu',
    kernelFunc: lessEqual
};

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
class LinSpaceProgram {
    constructor(shape) {
        this.variableNames = [];
        this.outputShape = [];
        this.uniforms = 'start : f32, step : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [shape];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'linSpace';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          setOutputAtIndex(index, uniforms.start + f32(index) * uniforms.step);
        }
      }
    `;
        return userCode;
    }
}

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
function linSpace(args) {
    const { backend, attrs } = args;
    const { start, stop, num } = attrs;
    const step = (stop - start) / (num - 1);
    const program = new LinSpaceProgram(num);
    const uniformData = [{ type: 'float32', data: [start] }, { type: 'float32', data: [step] }];
    return backend.runWebGPUProgram(program, [], 'float32', uniformData);
}
const linSpaceConfig = {
    kernelName: LinSpace,
    backendName: 'webgpu',
    kernelFunc: linSpace
};

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
const log = unaryKernelFunc({ opType: UnaryOpType.LOG, cpuKernelImpl: logImplCPU });
const logConfig = {
    kernelName: Log,
    backendName: 'webgpu',
    kernelFunc: log
};

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
const log1p = unaryKernelFunc({ opType: UnaryOpType.LOG1P });
const log1pConfig = {
    kernelName: Log1p,
    backendName: 'webgpu',
    kernelFunc: log1p
};

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
const logicalAnd = binaryKernelFunc({ opType: BinaryOpType.LOGICAL_AND, dtype: 'bool' });
const logicalAndConfig = {
    kernelName: LogicalAnd,
    backendName: 'webgpu',
    kernelFunc: logicalAnd
};

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
const logicalNot = unaryKernelFunc({ opType: UnaryOpType.LOGICAL_NOT });
const logicalNotConfig = {
    kernelName: LogicalNot,
    backendName: 'webgpu',
    kernelFunc: logicalNot
};

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
const logicalOr = binaryKernelFunc({ opType: BinaryOpType.LOGICAL_OR });
const logicalOrConfig = {
    kernelName: LogicalOr,
    backendName: 'webgpu',
    kernelFunc: logicalOr
};

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
const powOperatorSnippet = `
  var powValue = 0.0;
  let basis = uniforms.bias + uniforms.alpha * sum;
  if (uniforms.beta == 0.5) {
    powValue = inverseSqrt(basis);
  } else if (uniforms.beta == 1.0) {
    powValue = 1.0 / basis;
  } else {
    powValue = exp(log(basis) * (-uniforms.beta));
  }
`;
class LRNProgram {
    constructor(xShape) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'lrn';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let b = coords[0];
        let r = coords[1];
        let c = coords[2];
        let d = coords[3];

        let x = getX(b, r, c, d);
        var sum = 0.0;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let idx = d + i;
          if (idx >= 0 && idx < uniforms.xShape[3]) {
            let z = getX(b, r, c, idx);
            sum = sum + z * z;
          }
        }
        ${powOperatorSnippet}

        setOutputAtIndex(index, x * powValue);
      }
    }
  `;
        return userCode;
    }
}
class LRNSharedProgram {
    constructor(xShape, radius) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.uniforms = 'radius : i32, bias : f32, alpha : f32, beta : f32,';
        this.workgroupSize = [256, 1, 1];
        this.maxAllowRadius = 16;
        util.assert(radius <= this.maxAllowRadius, () => `Radius must be less than or equal to ${this.maxAllowRadius}, current radius is ${radius}`);
        this.outputShape = xShape;
        // The reason why not using this.workgroupSize[0] + 2 * maxAllowRadius here
        // is to make sure that there is only one time global memory load access for
        // each thread.
        this.elementsPerWorkgroup = this.workgroupSize[0] - 2 * this.maxAllowRadius;
        this.dispatchLayout = { x: [3], y: [2], z: [0, 1] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, [
            this.elementsPerWorkgroup, this.workgroupSize[1], this.workgroupSize[2]
        ]);
        this.shaderKey = 'lrn_shared';
    }
    getUserCode() {
        const userCode = `
    var <workgroup>lrnSub: array<f32, ${this.workgroupSize[0]}>;
    const elementsPerWorkgroup = ${this.elementsPerWorkgroup};
    const maxAllowRadius = ${this.maxAllowRadius};

    ${getMainHeaderString()} {
      let localDepth = i32(localId.x);
      let workgroupDepth = i32(workgroupId.x) * elementsPerWorkgroup;
      let xDepth = workgroupDepth + localDepth - maxAllowRadius;
      let b = i32(globalId.z) / uniforms.xShape[1];
      let r = i32(globalId.z) - b * uniforms.xShape[1];
      let c = i32(globalId.y);
      let d = workgroupDepth + localDepth;

      var x = 0.0;
      if (xDepth >= 0 && xDepth < uniforms.xShape[3]) {
        x = getX(b, r, c, xDepth);
      }
      lrnSub[localDepth] = x;
      workgroupBarrier();

      if (localDepth < elementsPerWorkgroup && d < uniforms.outShape[3]) {
        var sum = 0.0;
        let index = localDepth + maxAllowRadius;
        for (var i = -uniforms.radius; i <= uniforms.radius; i = i + 1) {
          let z = lrnSub[index + i];
          sum = sum + z * z;
        }
        ${powOperatorSnippet}

        setOutputAtCoords(b, r, c, d, lrnSub[index] * powValue);
      }
    } `;
        return userCode;
    }
}

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
function lrn(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    // When the adjacent channels is less than or equal to 16, which could cover
    // most cases, we use shared memory version to get better performance.
    // The theoretical adjacent channels may be very large, but the shared memory
    // size of hardware is limited, so we use the naive version when the adjacent
    // channels is large.
    let program;
    if (depthRadius > 16) {
        program = new LRNProgram(x.shape);
    }
    else {
        program = new LRNSharedProgram(x.shape, depthRadius);
    }
    const uniformData = [
        { type: 'int32', data: [depthRadius] }, { type: 'float32', data: [bias] },
        { type: 'float32', data: [alpha] }, { type: 'float32', data: [beta] }
    ];
    const res = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
    return res;
}
const lrnConfig = {
    kernelName: LRN,
    backendName: 'webgpu',
    kernelFunc: lrn
};

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
class LRNGradProgram {
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
    ${getMainHeaderString('index')} {
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
function lrnGrad(args) {
    const { inputs, backend, attrs } = args;
    const { x, y, dy } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    const program = new LRNGradProgram(x.shape);
    const uniformData = [
        { type: 'int32', data: [depthRadius] }, { type: 'float32', data: [bias] },
        { type: 'float32', data: [alpha] }, { type: 'float32', data: [beta] }
    ];
    const res = backend.runWebGPUProgram(program, [x, y, dy], x.dtype, uniformData);
    return res;
}
const lrnGradConfig = {
    kernelName: LRNGrad,
    backendName: 'webgpu',
    kernelFunc: lrnGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const maximum = binaryKernelFunc({
    opType: BinaryOpType.MAX,
    cpuKernelImpl: maximumImplCPU,
});
const maximumConfig = {
    kernelName: Maximum,
    backendName: 'webgpu',
    kernelFunc: maximum
};

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
function maxPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    return poolImpl(x, convInfo, 'max', backend);
}
const maxPoolConfig = {
    kernelName: MaxPool,
    backendName: 'webgpu',
    kernelFunc: maxPool
};

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
function maxPool3d(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dataFormat, dimRoundingMode } = attrs;
    const dilations = [1, 1, 1];
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode, dataFormat);
    const maxPoolProgram = new Pool3DProgram(convInfo, 'max');
    const dimensions = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left]
        },
        {
            type: 'int32',
            data: [convInfo.inDepth, convInfo.inHeight, convInfo.inWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        }
    ];
    return backend.runWebGPUProgram(maxPoolProgram, [x], x.dtype, dimensions);
}
const maxPool3DConfig = {
    kernelName: MaxPool3D,
    backendName: 'webgpu',
    kernelFunc: maxPool3d
};

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
class MaxPool2DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'maxPos'];
        this.uniforms = `strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, filterDims : vec2<i32>,
       outHeight : i32, outWidth : i32`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'maxPool2DBackprop';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords[0];
        let d = coords[3];

        let dyRCCorner = vec2<i32>(coords.yz) - uniforms.pads;
        let dyRCorner = dyRCCorner.x;
        let dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] - 1;
        for (var wR = 0; wR < uniforms.filterDims[0]; wR += uniforms.dilations[0]) {
          let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[0]);

          if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
            continue;
          }
          let idyR = i32(dyR);

          for (var wC = 0; wC < uniforms.filterDims[1]; wC += uniforms.dilations[1]) {
            let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[1]);

            if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
              continue;
            }
            let idyC = i32(dyC);

            let dyValue = getDy(batch, idyR, idyC, d);
            let maxPosValue = lastIndex - i32(getMaxPos(batch, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            let curPosValue = wR * uniforms.filterDims[1] + wC;
            let mask = select(0.0, 1.0, maxPosValue == curPosValue);
            dotProd += dyValue * mask;
          }
        }
        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}
class MaxPool3DBackpropProgram {
    constructor(convInfo) {
        this.variableNames = ['dy', 'maxPos'];
        this.uniforms = `strides : vec3<i32>, pads : vec3<i32>, filterDims : vec3<i32>,
      outDepth : i32, outHeight : i32, outWidth : i32`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = convInfo.inShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'maxPool3DBackprop';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let batch = coords.x;
        let ch = coords.u;

        let dyCorner = vec3<i32>(coords.y, coords.z, coords.w) - uniforms.pads;
        let dyDCorner = dyCorner.x;
        let dyRCorner = dyCorner.y;
        let dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;
        let lastIndex = uniforms.filterDims[0] * uniforms.filterDims[1] * uniforms.filterDims[2] - 1;

        for (var wD = 0; wD < uniforms.filterDims[0]; wD++) {
          let dyD = f32(dyDCorner + wD) / f32(uniforms.strides[0]);

          if (dyD < 0.0 || dyD >= f32(uniforms.outDepth) || fract(dyD) > 0.0) {
            continue;
          }
          let idyD = i32(dyD);

          for (var wR = 0; wR < uniforms.filterDims[1]; wR++) {
            let dyR = f32(dyRCorner + wR) / f32(uniforms.strides[1]);

            if (dyR < 0.0 || dyR >= f32(uniforms.outHeight) || fract(dyR) > 0.0) {
              continue;
            }
            let idyR = i32(dyR);

            for (var wC = 0; wC < uniforms.filterDims[2]; wC++) {
              let dyC = f32(dyCCorner + wC) / f32(uniforms.strides[2]);

              if (dyC < 0.0 || dyC >= f32(uniforms.outWidth) || fract(dyC) > 0.0) {
                continue;
              }
              let idyC = i32(dyC);

              let dyValue = getDy(batch, idyD, idyR, idyC, ch);
              let maxPosValue = lastIndex - i32(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              let curPosValue = wD * uniforms.filterDims[1] * uniforms.filterDims[2] + wR * uniforms.filterDims[2] + wC;
              let mask = select(0.0, 1.0, maxPosValue == curPosValue);
              dotProd += dyValue * mask;
            }
          }
        }

        setOutputAtIndex(index, dotProd);
      }
    }
    `;
        return userCode;
    }
}

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
function maxPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = [1, 1, 1];
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    const maxPool3dPositionsProgram = new Pool3DProgram(convInfo, 'max', true /* get positions */);
    let uniformData = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left]
        },
        {
            type: 'int32',
            data: [convInfo.inDepth, convInfo.inHeight, convInfo.inWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        }
    ];
    const maxPool3dPositions = backend.runWebGPUProgram(maxPool3dPositionsProgram, [x], 'int32', uniformData);
    const maxPool3dBackpropProgram = new MaxPool3DBackpropProgram(convInfo);
    uniformData = [
        {
            type: 'int32',
            data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth - 1 - convInfo.padInfo.front,
                convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
                convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
            ]
        },
        {
            type: 'int32',
            data: [
                convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
                convInfo.effectiveFilterWidth
            ]
        },
        { type: 'int32', data: [convInfo.outDepth] },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] }
    ];
    const result = backend.runWebGPUProgram(maxPool3dBackpropProgram, [dy, maxPool3dPositions], x.dtype, uniformData);
    backend.disposeData(maxPool3dPositions.dataId);
    return result;
}
const maxPool3DGradConfig = {
    kernelName: MaxPool3DGrad,
    backendName: 'webgpu',
    kernelFunc: maxPool3DGrad
};

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
function maxPoolGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input, output } = inputs;
    const x = input;
    assertNotComplex([input, output], 'maxPoolGrad');
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const maxPoolPositionsProgram = new Pool2DProgram(convInfo, 'max', true);
    let uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        }
    ];
    const maxPoolPositions = backend.runWebGPUProgram(maxPoolPositionsProgram, [x], 'int32', uniformData);
    const maxPoolBackpropProgram = new MaxPool2DBackpropProgram(convInfo);
    uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [
                convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
                convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
            ]
        },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        },
        { type: 'int32', data: [convInfo.outHeight] },
        { type: 'int32', data: [convInfo.outWidth] }
    ];
    const result = backend.runWebGPUProgram(maxPoolBackpropProgram, [dy, maxPoolPositions], x.dtype, uniformData);
    backend.disposeData(maxPoolPositions.dataId);
    return result;
}
const maxPoolGradConfig = {
    kernelName: MaxPoolGrad,
    backendName: 'webgpu',
    kernelFunc: maxPoolGrad
};

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
function maxPoolWithArgmax(args) {
    const { inputs, backend, attrs } = args;
    const { filterSize, strides, pad, includeBatchInIndex } = attrs;
    const { x } = inputs;
    util.assert(x.shape.length === 4, () => `Error in maxPool: input must be rank 4 but got rank ${x.shape.length}.`);
    const dilations = [1, 1];
    util.assert(backend_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = backend_util.computePool2DInfo(x.shape, filterSize, strides, dilations, pad);
    const uniformData = [
        { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] },
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] }, {
            type: 'int32',
            data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        }
    ];
    let program = new Pool2DProgram(convInfo, 'max', false);
    const poolOutput = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
    program = new Pool2DProgram(convInfo, 'max', true, true, includeBatchInIndex);
    const indexOutput = backend.runWebGPUProgram(program, [x], 'int32', uniformData);
    return [poolOutput, indexOutput];
}
const maxPoolWithArgmaxConfig = {
    kernelName: MaxPoolWithArgmax,
    backendName: 'webgpu',
    kernelFunc: maxPoolWithArgmax
};

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
function min(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'min', backend);
}
const minConfig = {
    kernelName: Min,
    backendName: 'webgpu',
    kernelFunc: min
};

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
const minimum = binaryKernelFunc({
    opType: BinaryOpType.MIN,
    cpuKernelImpl: minimumImplCPU,
});
const minimumConfig = {
    kernelName: Minimum,
    backendName: 'webgpu',
    kernelFunc: minimum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
class MirrorPadProgram {
    constructor(xShape, paddings, mode) {
        this.uniforms = '';
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.xShape = xShape;
        paddings.map((_, i) => {
            this.uniforms += ` pad${i} : vec2<i32>,`;
        });
        this.offset = mode === 'reflect' ? 0 : 1;
        this.shaderKey = `mirrorPad_${mode}`;
    }
    getUserCode() {
        const rank = this.xShape.length;
        // The length of paddings are same with the rank of the input tensor.
        const start = this.xShape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
        const end = this.xShape
            .map((_, i) => `uniforms.pad${i}[0] + uniforms.xShape${rank > 1 ? `[${i}]` : ''}`)
            .join(',');
        const shaderStart = rank === 1 ? 'start' : 'start[i]';
        const shaderEnd = rank === 1 ? 'end' : 'end[i]';
        const shaderOutC = rank === 1 ? 'outC' : 'outC[i]';
        const dtype = getCoordsDataType(rank);
        const unpackedCoords = rank > 1 ?
            ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
            'coords';
        return `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let start = ${dtype}(${start});
          let end = ${dtype}(${end});
          var outC = getCoordsFromIndex(index);
          for (var i = 0; i < ${rank}; i = i + 1) {
            if (${shaderOutC} < ${shaderStart}) {
              ${shaderOutC} = ${shaderStart} * 2 - ${shaderOutC} - ${this.offset};
            } else if(${shaderOutC} >= ${shaderEnd}) {
              ${shaderOutC} = (${shaderEnd} - 1) * 2 - ${shaderOutC} + ${this.offset};
            }
          }
          let coords = outC - start;
          setOutputAtIndex(index, getX(${unpackedCoords}));
        }
      }
    `;
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const mirrorPadConfig = {
    kernelName: MirrorPad,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        const { paddings, mode } = attrs;
        const webGPUBackend = backend;
        const uniformData = paddings.map(p => {
            return { type: 'int32', data: [p[0], p[1]] };
        });
        const program = new MirrorPadProgram(x.shape, paddings, mode);
        const output = webGPUBackend.runWebGPUProgram(program, [x], x.dtype, uniformData);
        return output;
    }
};

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
const mod = binaryKernelFunc({ opType: BinaryOpType.MOD });
const modConfig = {
    kernelName: Mod,
    backendName: 'webgpu',
    kernelFunc: mod
};

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
class MultinomialProgram {
    constructor(batchSize, numSamples) {
        this.variableNames = ['probs'];
        this.outputShape = [];
        this.uniforms = 'seed : f32, numOutcomes: i32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [batchSize, numSamples];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'multinomial';
    }
    getUserCode() {
        const userCode = `
    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    fn random (seed : f32, resultUV : vec2<f32>) -> f32 {
      let HASHSCALE1 = 443.8975;
      let p = resultUV * seed;
      var p3  = fract(vec3<f32>(p.xyx) * HASHSCALE1);
      p3 = p3 + dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords[0];

        let resUV = vec2<f32>(f32(coords[1]) / f32(uniforms.outShape[1]),
            f32(coords[0]) / f32(uniforms.outShape[0]));
        let r = random(uniforms.seed, resUV);
        var cdf = 0.0;
        for (var i = 0; i < uniforms.numOutcomes - 1; i = i + 1) {
          cdf = cdf + getProbs(batch, i);

          if (r < cdf) {
            setOutputAtIndexI32(index, i);
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutputAtIndexI32(index, uniforms.numOutcomes - 1);
      }
    }
  `;
        return userCode;
    }
}

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
class SoftmaxProgram {
    constructor(outputShape) {
        this.variableNames = ['logits'];
        this.outputShape = outputShape; // [rows, cols]
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [this.outputShape[0], 1, 1];
        if (this.outputShape[1] >= 4096) {
            this.workgroupSize = [256, 1, 1];
        }
        else {
            this.workgroupSize = [64, 1, 1];
        }
        this.shaderKey = 'softmax';
    }
    getUserCode() {
        const userCode = `
    var<workgroup> buf : array<f32, ${this.workgroupSize[0]}>;
    var<workgroup> rowMaxShared : f32;
    var<workgroup> rowSumShared : f32;
    const blockSize = ${this.workgroupSize[0]};
    ${getMainHeaderString('index')} {
      let row = index / blockSize;
      let tid = i32(localId.x);
      let cols = uniforms.outShape[1];

      var threadMax = -3.402823e+38f;
      for (var col = tid; col < cols; col += blockSize) {
        let value = getLogits(row, col);
        threadMax = max(threadMax, value);
      }
      if (tid < cols) {
        buf[tid] = threadMax;
      }
      workgroupBarrier();

      var reduceSize = min(cols, blockSize);
      for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
        reduceSize = currSize + (reduceSize & 1);
        if (tid < currSize) {
          buf[tid] = max(buf[tid], buf[tid + reduceSize]);
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowMaxShared = buf[0];
      }
      workgroupBarrier();

      var threadSum = 0.0;
      for (var col = tid; col < cols; col += blockSize) {
        let subExp = exp(getLogits(row, col) - rowMaxShared);
        threadSum += subExp;
      }
      buf[tid] = threadSum;
      workgroupBarrier();

      for (var currSize = blockSize >> 1;  currSize > 0; currSize = currSize >> 1) {
        if (tid < currSize) {
          buf[tid] = buf[tid] + buf[tid + currSize];
        }
        workgroupBarrier();
      }

      if (tid == 0) {
        rowSumShared = buf[0];
      }
      workgroupBarrier();

      for (var col = tid; col < cols; col += blockSize) {
        let value = exp(getLogits(row, col) - rowMaxShared) / rowSumShared;
        setOutputAtCoords(row, col, value);
      }
  }
    `;
        return userCode;
    }
}

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
function softmax(args) {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { dim } = attrs;
    const logitsReshaped = reshape({
        inputs: { x: logits },
        backend,
        attrs: {
            shape: [
                util.sizeFromShape(logits.shape) / logits.shape[dim], logits.shape[dim]
            ]
        }
    });
    const program = new SoftmaxProgram(logitsReshaped.shape);
    const res = backend.runWebGPUProgram(program, [logitsReshaped], logits.dtype);
    const resReshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: logits.shape } });
    backend.disposeData(logitsReshaped.dataId);
    backend.disposeData(res.dataId);
    return resReshaped;
}
const softmaxConfig = {
    kernelName: Softmax,
    backendName: 'webgpu',
    kernelFunc: softmax
};

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
function multinomial(args) {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { numSamples, seed, normalized } = attrs;
    const probs = normalized ?
        logits :
        softmax({ inputs: { logits }, backend, attrs: { dim: logits.shape.length - 1 } });
    const batchSize = probs.shape[0];
    const numOutcomes = probs.shape[1];
    const program = new MultinomialProgram(batchSize, numSamples);
    const uniformData = [{ type: 'float32', data: [seed] }, { type: 'int32', data: [numOutcomes] }];
    const res = backend.runWebGPUProgram(program, [probs], 'int32', uniformData);
    if (!normalized) {
        backend.disposeData(probs.dataId);
    }
    return res;
}
const multinomialConfig = {
    kernelName: Multinomial,
    backendName: 'webgpu',
    kernelFunc: multinomial
};

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
// This doesn't use unaryKernelFunc because negImplCPU is not of type
// SimpleUnaryKernelImplCPU.
function neg(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (backend.shouldExecuteOnCPU([x])) {
        const xData = backend.tensorMap.get(x.dataId);
        const [outValues, newShape] = negImplCPU(xData.values, x.shape, x.dtype);
        return backend.makeTensorInfo(newShape, x.dtype, outValues);
    }
    const program = new UnaryOpProgram(x.shape, UnaryOpType.NEG);
    return backend.runWebGPUProgram(program, [x], x.dtype);
}
const negConfig = {
    kernelName: Neg,
    backendName: 'webgpu',
    kernelFunc: neg
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function nonMaxSuppressionV3(args) {
    console.warn('tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold } = attrs;
    const boxesVals = backend.readSync(boxes.dataId);
    const scoresVals = backend.readSync(scores.dataId);
    const { selectedIndices } = kernel_impls.nonMaxSuppressionV3Impl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    return backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices));
}
const nonMaxSuppressionV3Config = {
    kernelName: NonMaxSuppressionV3,
    backendName: 'webgpu',
    kernelFunc: nonMaxSuppressionV3
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
function nonMaxSuppressionV5(args) {
    console.warn('tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
    const boxesVals = backend.readSync(boxes.dataId);
    const scoresVals = backend.readSync(scores.dataId);
    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;
    const { selectedIndices, selectedScores } = kernel_impls.nonMaxSuppressionV5Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
    return [
        backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
        backend.makeTensorInfo([selectedScores.length], 'float32', new Float32Array(selectedScores))
    ];
}
const nonMaxSuppressionV5Config = {
    kernelName: NonMaxSuppressionV5,
    backendName: 'webgpu',
    kernelFunc: nonMaxSuppressionV5
};

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
class OneHotProgram {
    constructor(numIndices, depth) {
        this.variableNames = ['x'];
        this.uniforms = 'onValue : f32, offValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [numIndices, depth];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'onehot';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          setOutputAtIndex(index, mix(uniforms.offValue, uniforms.onValue,
                                      f32(i32(round(getX(coords.x))) == coords.y)));
        }
      }
    `;
        return userCode;
    }
}

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
function oneHot(args) {
    const { inputs, backend, attrs } = args;
    const { indices } = inputs;
    const { dtype, depth, onValue, offValue } = attrs;
    const indicesSize = util.sizeFromShape(indices.shape);
    const program = new OneHotProgram(indicesSize, depth);
    const reshaped = reshape({ inputs: { x: indices }, backend, attrs: { shape: [indicesSize] } });
    const uniformData = [{ type: 'float32', data: [onValue] }, { type: 'float32', data: [offValue] }];
    const result = backend.runWebGPUProgram(program, [reshaped], dtype, uniformData);
    backend.disposeData(reshaped.dataId);
    const outShape = [...indices.shape, depth];
    const out = reshape({ inputs: { x: result }, backend, attrs: { shape: outShape } });
    backend.disposeData(result.dataId);
    return out;
}
const oneHotConfig = {
    kernelName: OneHot,
    backendName: 'webgpu',
    kernelFunc: oneHot
};

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
function zerosLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = zerosLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeData(realPart.dataId);
        backend.disposeData(r.dataId);
        backend.disposeData(imagPart.dataId);
        backend.disposeData(i.dataId);
        return result;
    }
    else {
        return fill({
            attrs: {
                shape: x.shape,
                dtype: x.dtype,
                value: x.dtype === 'string' ? '' : 0
            },
            backend
        });
    }
}
const zerosLikeConfig = {
    kernelName: ZerosLike,
    backendName: 'webgpu',
    kernelFunc: zerosLike
};

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
function onesLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'string') {
        throw new Error('onesLike is not supported under string dtype');
    }
    else if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = onesLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeData(realPart.dataId);
        backend.disposeData(r.dataId);
        backend.disposeData(imagPart.dataId);
        backend.disposeData(i.dataId);
        return result;
    }
    else {
        return fill({ attrs: { shape: x.shape, dtype: x.dtype, value: 1 }, backend });
    }
}
const onesLikeConfig = {
    kernelName: OnesLike,
    backendName: 'webgpu',
    kernelFunc: onesLike
};

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
function pack(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    if (inputs.length === 1) {
        return expandDims({ inputs: { input: inputs[0] }, backend, attrs: { dim: axis } });
    }
    const shape = inputs[0].shape;
    const dtype = inputs[0].dtype;
    inputs.forEach(t => {
        util.assertShapesMatch(shape, t.shape, 'All tensors passed to stack must have matching shapes');
        util.assert(dtype === t.dtype, () => 'All tensors passed to stack must have matching dtypes');
    });
    const intermediateTensorInfos = [];
    const expandedTensors = inputs.map(t => {
        const expandedT = expandDims({ inputs: { input: t }, backend, attrs: { dim: axis } });
        intermediateTensorInfos.push(expandedT);
        return expandedT;
    });
    const result = concat({ inputs: expandedTensors, backend, attrs: { axis } });
    intermediateTensorInfos.forEach(t => backend.disposeData(t.dataId));
    return result;
}
const packConfig = {
    kernelName: Pack,
    backendName: 'webgpu',
    kernelFunc: pack
};

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
function padCommon(shape, fillZero = false) {
    const rank = shape.length;
    const type = getCoordsDataType(rank);
    const start = shape.map((_, i) => `uniforms.pad${i}[0]`).join(',');
    const end = shape
        .map((_, i) => `uniforms.pad${i}[0] + uniforms.xShape${rank > 1 ? `[${i}]` : ''}`)
        .join(',');
    const startValue = rank > 1 ? `${type}(${start})` : `${start}`;
    const endValue = rank > 1 ? `${type}(${end})` : `${end}`;
    const leftPadCondition = rank > 1 ? `any(paddedCoords < start)` : `paddedCoords < start`;
    const rightPadCondition = rank > 1 ? `any(paddedCoords >= end)` : `paddedCoords >= end`;
    const unpackedCoords = rank > 1 ?
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank) :
        'coords';
    return `
        let start = ${startValue};
        let end = ${endValue};
        if (${leftPadCondition} || ${rightPadCondition}) {
          setOutputAtIndex(index, ${fillZero ? 0.0 : 'uniforms.constantValue'});
        } else {
          let coords = paddedCoords - start;
          setOutputAtIndex(index, getX(${unpackedCoords}));
        }
  `;
}
class PadProgram {
    constructor(xShape, paddings) {
        this.variableNames = ['x'];
        this.uniforms = 'constantValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = paddings.map((p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        paddings.map((_, i) => {
            this.uniforms += ` pad${i} : vec2<i32>,`;
        });
        this.xShape = xShape;
        this.shaderKey = 'pad';
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let paddedCoords = getCoordsFromIndex(index);
          ${padCommon(this.xShape)}
        }
      }
    `;
        return userCode;
    }
}

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
const padV2 = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { paddings, constantValue } = attrs;
    if (paddings.every(p => util.arraysEqual(p, [0, 0]))) {
        return identity({ inputs: { x }, backend });
    }
    if (util.sizeFromShape(x.shape) === 0) {
        // Short-circuit the computation, since x doesn't have value, only
        // the shape is used to compute output shape to pad.
        const outputShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
        return fill({
            backend,
            attrs: { shape: outputShape, value: constantValue, dtype: x.dtype }
        });
    }
    const uniformData = [{ type: 'float32', data: [constantValue] }];
    paddings.map(p => uniformData.push({ type: 'int32', data: [p[0], p[1]] }));
    const program = new PadProgram(x.shape, paddings);
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
};
const padV2Config = {
    kernelName: PadV2,
    backendName: 'webgpu',
    kernelFunc: padV2
};

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
const pow = binaryKernelFunc({
    opType: BinaryOpType.POW,
});
const powConfig = {
    kernelName: Pow,
    backendName: 'webgpu',
    kernelFunc: pow
};

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
function prelu(args) {
    const { inputs, backend } = args;
    const { x, alpha } = inputs;
    const program = new BinaryOpProgram(BinaryOpType.PRELU, x.shape, alpha.shape);
    return backend.runWebGPUProgram(program, [x, alpha], 'float32');
}
const preluConfig = {
    kernelName: Prelu,
    backendName: 'webgpu',
    kernelFunc: prelu
};

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
function prod(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    return reduce(x, axis, keepDims, 'prod', backend);
}
const prodConfig = {
    kernelName: Prod,
    backendName: 'webgpu',
    kernelFunc: prod
};

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
const range = (args) => {
    const { backend, attrs } = args;
    const { start, stop, step, dtype } = attrs;
    const values = rangeImplCPU(start, stop, step, dtype);
    return backend.makeTensorInfo([values.length], dtype, values);
};
const rangeConfig = {
    kernelName: Range,
    backendName: 'webgpu',
    kernelFunc: range
};

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
const realDiv = binaryKernelFunc({ opType: BinaryOpType.DIV });
const realDivConfig = {
    kernelName: RealDiv,
    backendName: 'webgpu',
    kernelFunc: realDiv
};

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
const reciprocal = unaryKernelFunc({ opType: UnaryOpType.RECIPROCAL });
const reciprocalConfig = {
    kernelName: Reciprocal,
    backendName: 'webgpu',
    kernelFunc: reciprocal
};

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
const relu = unaryKernelFunc({ opType: UnaryOpType.RELU });
const reluConfig = {
    kernelName: Relu,
    backendName: 'webgpu',
    kernelFunc: relu
};

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
const relu6 = unaryKernelFunc({ opType: UnaryOpType.RELU6 });
const relu6Config = {
    kernelName: Relu6,
    backendName: 'webgpu',
    kernelFunc: relu6
};

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
class ResizeBilinearProgram {
    constructor(inputShape, newHeight, newWidth) {
        this.variableNames = ['x'];
        this.uniforms = 'adjustHeightWidth : vec2<f32>, halfPixelCenters : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `resizeBilinear`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            f32(uniforms.xShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.xShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveOutSize = vec2<f32>(
            f32(uniforms.outShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.outShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC =
            (vec2<f32>(rc) + vec2<f32>(uniforms.halfPixelCenters)) *
            effectiveInputOverOutputRatioRC - vec2<f32>(uniforms.halfPixelCenters);

          // Compute the four integer indices.
          let sourceFloorRC = vec2<i32>(sourceFracIndexRC);
          let sourceCeilRC = vec2<i32>(
            min(vec2<f32>(uniforms.xShape.yz) - vec2<f32>(1.0), ceil(sourceFracIndexRC)));

          let topLeft = getX(b, sourceFloorRC.x, sourceFloorRC.y, d);
          let bottomLeft = getX(b, sourceCeilRC.x, sourceFloorRC.y, d);
          let topRight = getX(b, sourceFloorRC.x, sourceCeilRC.y, d);
          let bottomRight = getX(b, sourceCeilRC.x, sourceCeilRC.y, d);

          let fracRC = sourceFracIndexRC - vec2<f32>(sourceFloorRC);

          let top = topLeft + (topRight - topLeft) * fracRC.y;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
          let newValue = top + (bottom - top) * fracRC.x;

          setOutputAtIndex(index, newValue);
        }
      }
    `;
        return userCode;
    }
}

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
function resizeBilinear(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, size, halfPixelCenters } = attrs;
    const [newHeight, newWidth] = size;
    const adjustHeight = alignCorners && newHeight > 1 ? 1.0 : 0.0;
    const adjustWidth = alignCorners && newWidth > 1 ? 1.0 : 0.0;
    const halfPixelCentersValue = halfPixelCenters ? 0.5 : 0.0;
    const uniformData = [
        { type: 'float32', data: [adjustHeight, adjustWidth] },
        { type: 'float32', data: [halfPixelCentersValue] }
    ];
    const program = new ResizeBilinearProgram(images.shape, newHeight, newWidth);
    return backend.runWebGPUProgram(program, [images], 'float32', uniformData);
}
const resizeBilinearConfig = {
    kernelName: ResizeBilinear,
    backendName: 'webgpu',
    kernelFunc: resizeBilinear
};

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
class ResizeBilinearBackpropProgram {
    constructor(inputShape, alignCorners) {
        this.variableNames = ['dy'];
        this.uniforms = `effectiveXSize : vec2<i32>, effectiveYSize : vec2<i32>, heightScale : f32, widthScale : f32,
       invHeightScale : f32, invWidthScale : f32, winHeight : i32, winWidth : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = inputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.alignCorners = alignCorners;
        this.shaderKey = `resizeBilinearBackprop_${alignCorners}`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let b = coords[0];
          let d = coords[3];
          let r = coords[1];
          let c = coords[2];

          var accumulator = 0.0;

          // Compute bounds for where in dy we will look
          let startRLerp = floor(f32(r) * uniforms.invHeightScale);
          let startDyR = i32(startRLerp - f32(uniforms.winHeight / 2));

          let startCLerp = floor(f32(c) * uniforms.invWidthScale);
          let startDyC = i32(startCLerp - f32(uniforms.winWidth / 2));

          // Loop over dy
          for (var dyROffset = 0; dyROffset < uniforms.winHeight; dyROffset++) {
            let dyR = startDyR + dyROffset;

            // Guard against the window exceeding the bounds of dy
            if (dyR < 0 || dyR >= uniforms.dyShape[1]) {
              continue;
            }

            for (var dyCOffset = 0; dyCOffset < uniforms.winWidth; dyCOffset++) {
              let dyC = startDyC + dyCOffset;

              // Guard against the window exceeding the bounds of dy
              if (dyC < 0 || dyC >= uniforms.dyShape[2]) {
                continue;
              }

              let dxR = f32(dyR) * uniforms.heightScale;
              let topDxRIndex = i32(floor(dxR));
              let bottomDxRIndex = i32(min(ceil(dxR), f32(uniforms.outShape[1] - 1)));
              let dxRLerp = dxR - f32(topDxRIndex);
              let inverseDxRLerp = 1.0 - dxRLerp;

              let dxC = f32(dyC) * uniforms.widthScale;
              let leftDxCIndex = i32(floor(dxC));
              let rightDxCIndex = i32(min(ceil(dxC), f32(uniforms.outShape[2] - 1)));
              let dxCLerp = dxC - f32(leftDxCIndex);
              let inverseDxCLerp = 1.0 - dxCLerp;

              if (r == topDxRIndex && c == leftDxCIndex) {
                // topLeft
                accumulator +=
                  getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
              }

              if (r == topDxRIndex && c == rightDxCIndex) {
                // topRight
                accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
              }

              if (r == bottomDxRIndex && c == leftDxCIndex) {
                // bottomLeft
                accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
              }

              if (r == bottomDxRIndex && c == rightDxCIndex) {
                // bottomRight
                accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
              }
            }
          }
          // End loop over dy

          setOutputAtIndex(index, accumulator);
        }
      }
    `;
        return userCode;
    }
}

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
function resizeBilinearGrad(args) {
    const { inputs, backend, attrs } = args;
    const { images, dy } = inputs;
    const { alignCorners } = attrs;
    const [, xHeight, xWidth,] = images.shape;
    const [, yHeight, yWidth] = dy.shape;
    const effectiveXSize = [
        (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
        (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    ];
    const effectiveYSize = [
        (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
        (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
    ];
    const heightScale = effectiveXSize[0] / effectiveYSize[0];
    const widthScale = effectiveXSize[1] / effectiveYSize[1];
    const invHeightScale = 1 / heightScale;
    const invWidthScale = 1 / widthScale;
    // This defines the size of the window of values around a particular
    // index in dy that we want to search for contributions to dx.
    const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
    const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
    const program = new ResizeBilinearBackpropProgram(images.shape, alignCorners);
    const uniformData = [
        { type: 'int32', data: effectiveXSize },
        { type: 'int32', data: effectiveYSize },
        { type: 'float32', data: [heightScale] },
        { type: 'float32', data: [widthScale] },
        { type: 'float32', data: [invHeightScale] },
        { type: 'float32', data: [invWidthScale] },
        { type: 'int32', data: [winHeight] }, { type: 'int32', data: [winWidth] }
    ];
    return backend.runWebGPUProgram(program, [dy], dy.dtype, uniformData);
}
const resizeBilinearGradConfig = {
    kernelName: ResizeBilinearGrad,
    backendName: 'webgpu',
    kernelFunc: resizeBilinearGrad
};

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
class ResizeNearestNeighborProgram {
    constructor(inputShape, newHeight, newWidth, halfPixelCenters) {
        this.variableNames = ['x'];
        this.uniforms = 'adjustHeightWidth : vec2<f32>, roundBase : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [inputShape[0], newHeight, newWidth, inputShape[3]];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.halfPixelCenters = halfPixelCenters;
        this.shaderKey = `resizeNearest_${halfPixelCenters}`;
    }
    getUserCode() {
        let sourceFracIndexRC;
        if (this.halfPixelCenters) {
            sourceFracIndexRC =
                `max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC` +
                    `, vec2<f32>(0.0))`;
        }
        else {
            sourceFracIndexRC = `vec2<f32>(rc) * effectiveInputOverOutputRatioRC`;
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            f32(uniforms.xShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.xShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveOutSize = vec2<f32>(
            f32(uniforms.outShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.outShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = ${sourceFracIndexRC};

          // Compute the coordinators of nearest neighbor point.
          let inputShapeRC = vec2<f32>(f32(uniforms.xShape.y), f32(uniforms.xShape.z));
          let sourceNearestRC = vec2<i32>(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + uniforms.roundBase)));
          let newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutputAtIndex(index, newValue);
        }
      }
    `;
        return userCode;
    }
}

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
function resizeNearestNeighbor(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    const [newHeight, newWidth] = size;
    const adjustHeight = alignCorners && newHeight > 1 ? 1.0 : 0.0;
    const adjustWidth = alignCorners && newWidth > 1 ? 1.0 : 0.0;
    // When align corners is false, we rounds the value with floor.
    const roundBase = alignCorners ? 0.5 : 0.0;
    const uniformData = [
        { type: 'float32', data: [adjustHeight, adjustWidth] },
        { type: 'float32', data: [roundBase] }
    ];
    const program = new ResizeNearestNeighborProgram(images.shape, newHeight, newWidth, halfPixelCenters);
    return backend.runWebGPUProgram(program, [images], images.dtype, uniformData);
}
const resizeNearestNeighborConfig = {
    kernelName: ResizeNearestNeighbor,
    backendName: 'webgpu',
    kernelFunc: resizeNearestNeighbor
};

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
class ResizeNearestNeigborBackpropProgram {
    constructor(inputShape, alignCorners) {
        this.variableNames = ['dy'];
        this.uniforms = `effectiveXSize : vec2<i32>, effectiveYSize : vec2<i32>, invHeightScale : f32, invWidthScale : f32,
       winHeight : i32, winWidth : i32,`;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = inputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.alignCorners = alignCorners;
        this.shaderKey = `resizeNearestNeigborBackprop_${alignCorners}`;
    }
    getUserCode() {
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getOutputCoords();
          let b = coords[0];
          let d = coords[3];
          let r = coords[1];
          let c = coords[2];

          var accumulator = 0.0;

          // Compute bounds for where in dy we will look
          let startRLerp = floor(f32(r) * uniforms.invHeightScale);
          let startDyR = i32(floor(startRLerp - f32(uniforms.winHeight / 2)));

          let startCLerp = floor(f32(c) * uniforms.invWidthScale);
          let startDyC = i32(floor(startCLerp - f32(uniforms.winWidth / 2)));

          // Loop over dy
          for (var dyROffset = 0; dyROffset < uniforms.winHeight; dyROffset++) {
            let dyR = startDyR + dyROffset;

            // Guard against the window exceeding the bounds of dy
            if (dyR < 0 || dyR >= uniforms.dyShape[1]) {
              continue;
            }

            for (var dyCOffset = 0; dyCOffset < uniforms.winWidth; dyCOffset++) {
              let dyC = startDyC + dyCOffset;

              // Guard against the window exceeding the bounds of dy
              if (dyC < 0 || dyC >= uniforms.dyShape[2]) {
                continue;
              }

              let sourceFracRow = f32(uniforms.effectiveXSize[0]) *
                  (f32(dyR) / f32(uniforms.effectiveYSize[0]));

              let sourceFracCol = f32(uniforms.effectiveXSize[1]) *
                  (f32(dyC) / f32(uniforms.effectiveYSize[1]));

              let sourceNearestRow =
                  i32(min(f32(uniforms.outShape[1] - 1),
                  ${this.alignCorners ? 'floor(sourceFracRow + 0.5)' :
            'floor(sourceFracRow)'}));

              let sourceNearestCol =
                  i32(min(f32(uniforms.outShape[2] - 1),
                  ${this.alignCorners ? 'floor(sourceFracCol + 0.5)' :
            'floor(sourceFracCol)'}));

              if (r == sourceNearestRow && c == sourceNearestCol) {
                accumulator += getDy(b, dyR, dyC, d);
              }
            }
          }
          // End loop over dy

          setOutputAtIndex(index, accumulator);
        }
      }
    `;
        return userCode;
    }
}

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
function resizeNearestNeighborGrad(args) {
    const { inputs, backend, attrs } = args;
    const { images, dy } = inputs;
    const { alignCorners } = attrs;
    const [, xHeight, xWidth] = images.shape;
    const [, yHeight, yWidth] = dy.shape;
    const effectiveXSize = [
        (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
        (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    ];
    const effectiveYSize = [
        (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
        (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
    ];
    const heightScale = effectiveXSize[0] / effectiveYSize[0];
    const widthScale = effectiveXSize[1] / effectiveYSize[1];
    const invHeightScale = 1 / heightScale;
    const invWidthScale = 1 / widthScale;
    // This defines the size of the window of values around a particular
    // index in dy that we want to search for contributions to dx.
    const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
    const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
    const program = new ResizeNearestNeigborBackpropProgram(images.shape, alignCorners);
    const uniformData = [
        { type: 'int32', data: effectiveXSize },
        { type: 'int32', data: effectiveYSize },
        { type: 'float32', data: [invHeightScale] },
        { type: 'float32', data: [invWidthScale] },
        { type: 'int32', data: [winHeight] }, { type: 'int32', data: [winWidth] }
    ];
    return backend.runWebGPUProgram(program, [dy], dy.dtype, uniformData);
}
const resizeNearestNeighborGradConfig = {
    kernelName: ResizeNearestNeighborGrad,
    backendName: 'webgpu',
    kernelFunc: resizeNearestNeighborGrad
};

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
class ReverseProgram {
    constructor(xShape) {
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = ` axis : vec4<i32>,`;
        this.shaderKey = 'reverse';
    }
    getUserCode() {
        const reverseCoordsSnippet = `
      // Using uniform variables as judging conditions, so the function has
      // coherent execution within all threads.
      fn getReverseCoords(coords : vec4<i32>) -> vec4<i32> {
        var reverseCoords = coords;
        if (uniforms.axis[0] == 1) {
          reverseCoords[0] = uniforms.xShape[0] - coords[0] - 1;
        }
        if (uniforms.axis[1] == 1) {
          reverseCoords[1] = uniforms.xShape[1] - coords[1] - 1;
        }
        if (uniforms.axis[2] == 1) {
          reverseCoords[2] = uniforms.xShape[2] - coords[2] - 1;
        }
        if (uniforms.axis[3] == 1) {
          reverseCoords[3] = uniforms.xShape[3] - coords[3] - 1;
        }

        return reverseCoords;
      }
    `;
        const userCode = `
      ${reverseCoordsSnippet}
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let reverseCoords = getReverseCoords(coords);
          setOutputAtIndex(index, getX(reverseCoords[0],
              reverseCoords[1], reverseCoords[2], reverseCoords[3]));
        }
      }
    `;
        return userCode;
    }
}

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
function reverse(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dims } = attrs;
    const xRank = x.shape.length;
    if (xRank === 0) {
        return identity({ inputs: { x }, backend });
    }
    const xShape = x.shape;
    const xShape4D = [1, 1, 1, 1];
    xShape.forEach((d, i) => {
        const index = i + 4 - xRank;
        xShape4D[index] = d;
    });
    const axes = util.parseAxisParam(dims, x.shape);
    const dims4D = [0, 0, 0, 0];
    axes.forEach(ax => {
        const index = ax + 4 - xRank;
        dims4D[index] = 1;
    });
    const uniformData = [{ type: 'int32', data: dims4D }];
    const xReshaped = reshape({ inputs: { x }, backend, attrs: { shape: xShape4D } });
    const program = new ReverseProgram(xShape4D);
    const values = backend.runWebGPUProgram(program, [xReshaped], xReshaped.dtype, uniformData);
    backend.disposeData(xReshaped.dataId);
    const result = reshape({ inputs: { x: values }, backend, attrs: { shape: xShape } });
    backend.disposeData(values.dataId);
    return result;
}
const reverseConfig = {
    kernelName: Reverse,
    backendName: 'webgpu',
    kernelFunc: reverse
};

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
class RotateProgram {
    constructor(imageShape, fillValue) {
        this.outputShape = [];
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = imageShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = `centerX : f32, centerY : f32, sinRadians : f32,
          cosRadians : f32,`;
        this.shaderKey = 'rotate';
        this.outputShape = imageShape;
        if (typeof fillValue === 'number') {
            this.uniforms += ` fillValue : f32,`;
            this.fillSnippet = `var outputValue = uniforms.fillValue;`;
            this.shaderKey += '_float';
        }
        else {
            this.uniforms += ` fillValue : vec3<f32>,`;
            this.fillSnippet = `var outputValue = uniforms.fillValue[coords[3]];`;
            this.shaderKey += '_vec3';
        }
    }
    getUserCode() {
        const userCode = `
        ${getMainHeaderString('index')} {
          if (index < uniforms.size) {
            let coords = getCoordsFromIndex(index);
            let coordXFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.cosRadians - (f32(coords[1]) - uniforms.centerY) *
                uniforms.sinRadians;
            let coordYFloat = (f32(coords[2]) - uniforms.centerX) *
                uniforms.sinRadians + (f32(coords[1]) - uniforms.centerY) *
                uniforms.cosRadians;
            let coordX = i32(round(coordXFloat + uniforms.centerX));
            let coordY = i32(round(coordYFloat + uniforms.centerY));
            ${this.fillSnippet}
            if(coordX >= 0 && coordX < uniforms.xShape[2] && coordY >= 0 &&
                coordY < uniforms.xShape[1]) {
              outputValue = getX(coords[0], coordY, coordX, coords[3]);
            }
            setOutputAtIndex(index, outputValue);
          }
        }
      `;
        return userCode;
    }
}

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
const rotateWithOffsetConfig = {
    kernelName: RotateWithOffset,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { image } = inputs;
        const { radians, fillValue, center } = attrs;
        const webgpuBackend = backend;
        const program = new RotateProgram(image.shape, fillValue);
        const [centerX, centerY] = backend_util.getImageCenter(center, image.shape[1], image.shape[2]);
        const uniformData = [
            { type: 'float32', data: [centerX] },
            { type: 'float32', data: [centerY] },
            { type: 'float32', data: [Math.sin(radians)] },
            { type: 'float32', data: [Math.cos(radians)] }
        ];
        if (typeof fillValue === 'number') {
            uniformData.push({ type: 'float32', data: [Number.parseFloat(fillValue.toFixed(2))] });
        }
        else {
            uniformData.push({ type: 'float32', data: fillValue });
        }
        const output = webgpuBackend.runWebGPUProgram(program, [image], image.dtype, uniformData);
        return output;
    }
};

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
const round = unaryKernelFunc({ opType: UnaryOpType.ROUND });
const roundConfig = {
    kernelName: Round,
    backendName: 'webgpu',
    kernelFunc: round
};

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
const rsqrt = unaryKernelFunc({ opType: UnaryOpType.RSQRT, cpuKernelImpl: rsqrtImplCPU });
const rsqrtConfig = {
    kernelName: Rsqrt,
    backendName: 'webgpu',
    kernelFunc: rsqrt
};

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
class ScatterProgram {
    constructor(flattenXShape, sliceDim, indicesRank, updatesRank, strides, shape, outputDtype, sumDupeIndices = true) {
        this.variableNames = ['updates', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = shape;
        this.type = outputDtype;
        this.sumDupeIndices = sumDupeIndices;
        this.dispatchLayout = flatDispatchLayout(flattenXShape);
        // Dispatching based on |updates| shape instead of output shape.
        this.dispatch =
            computeDispatch(this.dispatchLayout, flattenXShape, this.workgroupSize);
        this.sliceDimGreaterThanOne = sliceDim > 1;
        this.shaderKey =
            `scatter_${indicesRank}_${updatesRank}_${this.sliceDimGreaterThanOne}_${outputDtype}_${sumDupeIndices}_${strides.length}`;
        const stridesType = getCoordsDataType(strides.length);
        this.uniforms =
            `sliceDim : i32, strides: ${stridesType}, updatesSize: i32,`;
        this.updatesRank = updatesRank;
        this.indicesRank = indicesRank;
    }
    getUserCode() {
        let indicesString = '';
        if (this.indicesRank === 1) {
            indicesString = 'coords[0]';
        }
        else if (this.indicesRank === 2) {
            indicesString = 'coords[0], j';
        }
        const indicesSnippet = `getIndices(${indicesString})`;
        const strideString = this.sliceDimGreaterThanOne ? 'uniforms.strides[j]' :
            'uniforms.strides';
        let outCoordsString = '';
        let getUpdatesCoordsFromFlatIndex = '';
        if (this.dispatchLayout.x.length === 1) {
            outCoordsString = 'flattenedIndex';
            getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> i32 {
        return index;
      }
      `;
        }
        else if (this.dispatchLayout.x.length === 2) {
            outCoordsString = 'vec2<i32>(flattenedIndex, coords[1])';
            getUpdatesCoordsFromFlatIndex = `
      fn getUpdatesCoordsFromFlatIndex(index : i32) -> vec2<i32> {
        // N.B. |updates| could be a scalar tensor, conceptually representing a
        // 2D tensor with all values equal to that. By design, its size must be
        // the same as |outShape[1]| in one dimension, and |indicesShape[0]|
        // gives the other.
        let sliceSize = uniforms.outShape[1];
        let d0 = index / sliceSize;
        let d1 = index - d0 * sliceSize;
        return vec2<i32>(d0, d1);
      }
      `;
        }
        const updatesString = Array.from({ length: this.updatesRank }, (_, idx) => `coords[${idx}]`);
        const updatesSnippet = `getUpdates(${updatesString.join(', ')})`;
        const userCode = `
    ${getUpdatesCoordsFromFlatIndex}
      ${getMainHeaderString('index')} {
        if (index < uniforms.updatesSize) {
          let coords = getUpdatesCoordsFromFlatIndex(index);
          var flattenedIndex = 0;
          for (var j = 0; j < uniforms.sliceDim; j = j + 1) {
            let indexInside = i32(round(${indicesSnippet}));
            flattenedIndex = flattenedIndex + indexInside * ${strideString};
          }
          let updateValue =
              ${dataTypeToGPUType(this.type)}(${updatesSnippet});
          let flatIndex = getOutputIndexFromCoords(${outCoordsString});

          ${this.sumDupeIndices ?
            atomicAddSnippet('&result[flatIndex]', 'updateValue', this.type) :
            `atomicStore(&result[flatIndex], bitcast<i32>(updateValue));`}
        }
      }`;
        return userCode;
    }
}

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
function scatterNd(args) {
    const { inputs, backend, attrs } = args;
    const { indices, updates } = inputs;
    const { shape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(updates, indices, shape);
    const flattenShape = [outputSize / sliceSize, sliceSize];
    if (outputSize === 0) {
        return backend.makeTensorInfo(shape, indices.dtype);
    }
    const flattenIndices = reshape({ inputs: { x: indices }, backend, attrs: { shape: [numUpdates, sliceRank] } });
    const flattenX = reshape({ inputs: { x: updates }, backend, attrs: { shape: [numUpdates, sliceSize] } });
    const type = flattenX.dtype;
    const output = fill({ backend, attrs: { shape: flattenShape, value: 0, dtype: type } });
    const size = util.sizeFromShape(flattenX.shape);
    const uniformData = [
        { type: 'int32', data: [sliceRank] }, { type: 'int32', data: strides },
        { type: 'int32', data: [size] }
    ];
    const program = new ScatterProgram(flattenX.shape, sliceRank, flattenIndices.shape.length, flattenX.shape.length, strides, flattenShape, type);
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndices], type, uniformData, output);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape } });
    backend.disposeData(flattenIndices.dataId);
    backend.disposeData(flattenX.dataId);
    backend.disposeData(res.dataId);
    return reshaped;
}
const scatterNdConfig = {
    kernelName: ScatterNd,
    backendName: 'webgpu',
    kernelFunc: scatterNd
};

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
class SearchSortedProgram {
    constructor(outputShape, side) {
        this.outputShape = [];
        this.variableNames = ['sortedSequence', 'values'];
        this.uniforms = 'numInputs : i32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.side = side;
        this.shaderKey = `search_sorted_${side}`;
    }
    getUserCode() {
        const boundComparator = this.side === 'left' ? '<' : '<=';
        const userCode = `
      fn findBound(batch: i32, value: f32) -> i32 {
        var left = i32(0);
        var right = uniforms.numInputs;
        while (left < right) {
          var mid = (left + right) / 2;
          if (getSortedSequence(batch, mid) ${boundComparator} value) {
            left = mid + 1;
          } else {
            right = mid;
          }
        }
        return right;
      }

      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let value = getValuesByOutputIndex(index);
          setOutputAtIndexI32(index, findBound(coords[0], value));
        }
      }
    `;
        return userCode;
    }
}

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
function searchSorted(args) {
    const { inputs, backend, attrs } = args;
    const { sortedSequence, values } = inputs;
    const { side } = attrs;
    const program = new SearchSortedProgram([values.shape[0], values.shape[1]], side);
    const uniformData = [{ type: 'int32', data: [sortedSequence.shape[1]] }];
    return backend.runWebGPUProgram(program, [sortedSequence, values], 'int32', uniformData);
}
const searchSortedConfig = {
    kernelName: SearchSorted,
    backendName: 'webgpu',
    kernelFunc: searchSorted,
};

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
class SelectProgram {
    constructor(cRank, shape, rank) {
        this.variableNames = ['c', 'a', 'b'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.cRank = cRank;
        this.rank = rank;
        this.shaderKey = 'select';
    }
    getUserCode() {
        // TODO(WGSL): below code can be merged with getUserCode.
        let cCoords;
        let abCoords;
        if (this.rank > 4) {
            throw Error(`Where for rank ${this.rank} is not yet supported`);
        }
        if (this.rank === 1) {
            abCoords = `resRC`;
            cCoords = `resRC`;
        }
        else {
            const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
            const cCoordVars = [];
            const abCoordVars = [];
            for (let i = 0; i < this.outputShape.length; i++) {
                abCoordVars.push(`${currentCoords[i]}`);
                if (i < this.cRank) {
                    cCoordVars.push(`${currentCoords[i]}`);
                }
            }
            cCoords = cCoordVars.join();
            abCoords = abCoordVars.join();
        }
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutputAtIndex(index, getA(${abCoords}));
          } else {
            setOutputAtIndex(index, getB(${abCoords}));
          }
        }
      }
    `;
        return userCode;
    }
}

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
function select(args) {
    const { inputs, backend } = args;
    const { condition, t, e } = inputs;
    const program = new SelectProgram(condition.shape.length, t.shape, t.shape.length);
    return backend.runWebGPUProgram(program, [condition, t, e], upcastType(t.dtype, e.dtype));
}
const selectConfig = {
    kernelName: Select,
    backendName: 'webgpu',
    kernelFunc: select
};

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
const selu = unaryKernelFunc({ opType: UnaryOpType.SELU });
const seluConfig = {
    kernelName: Selu,
    backendName: 'webgpu',
    kernelFunc: selu
};

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
const sigmoid = unaryKernelFunc({ opType: UnaryOpType.SIGMOID });
const sigmoidConfig = {
    kernelName: Sigmoid,
    backendName: 'webgpu',
    kernelFunc: sigmoid,
};

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
const sign = unaryKernelFunc({ opType: UnaryOpType.SIGN });
const signConfig = {
    kernelName: Sign,
    backendName: 'webgpu',
    kernelFunc: sign
};

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
const sin = unaryKernelFunc({ opType: UnaryOpType.SIN });
const sinConfig = {
    kernelName: Sin,
    backendName: 'webgpu',
    kernelFunc: sin
};

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
const sinh = unaryKernelFunc({ opType: UnaryOpType.SINH });
const sinhConfig = {
    kernelName: Sinh,
    backendName: 'webgpu',
    kernelFunc: sinh
};

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
const softplus = unaryKernelFunc({ opType: UnaryOpType.SOFTPLUS });
const softplusConfig = {
    kernelName: Softplus,
    backendName: 'webgpu',
    kernelFunc: softplus
};

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
class SpaceToBatchNDProgram {
    constructor(xShape, paddedXShape, paddings, reshapedPaddedXShape, newDim, paddedXShapeStridesShapeLength) {
        this.variableNames = ['x'];
        this.outputShape = [];
        this.uniforms = '';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(reshapedPaddedXShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = reshapedPaddedXShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.newDim = newDim;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.xShape = xShape;
        this.paddedXShape = paddedXShape;
        this.uniforms += `reshapedPaddedXShape : ${getCoordsDataType(reshapedPaddedXShape.length)}, paddedXShapeStrides : ${getCoordsDataType(paddedXShapeStridesShapeLength)}, `;
        paddings.map((_, i) => {
            this.uniforms += ` pad${i} : vec2<i32>,`;
        });
        this.shaderKey = `spaceToBatchND_${newDim}`;
    }
    getUserCode() {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);
        const userCode = `
      ${getCoordsFromIndexSnippet(this.paddedXShape, 'PaddedX')}
      ${getMainHeaderString('index')} {
        if(index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let switchedIndex = getIndexFromCoords${this.outputShape.length}D(${dtype}(${switched}), uniforms.reshapedPaddedXShape);
          let paddedCoords = getPaddedXCoordsFromIndex(switchedIndex);
          ${padCommon(this.xShape, true)}
        }
      }
    `;
        return userCode;
    }
}

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
const spaceToBatchND = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, paddings } = attrs;
    util.assert(x.shape.length <= 4, () => 'spaceToBatchND for rank > 4 with a WebGPU backend not ' +
        'implemented yet');
    const prod = blockShape.reduce((a, b) => a * b);
    const completePaddings = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
        completePaddings.push([0, 0]);
    }
    const paddedXShape = completePaddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    const reshapedPaddedShape = backend_util.getReshaped(paddedXShape, blockShape, prod, false);
    const permutedReshapedPaddedPermutation = backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
    const flattenShape = backend_util.getReshapedPermuted(paddedXShape, blockShape, prod, false);
    const paddedXShapeStrides = util.computeStrides(paddedXShape);
    const program = new SpaceToBatchNDProgram(x.shape, paddedXShape, completePaddings, reshapedPaddedShape, permutedReshapedPaddedPermutation, paddedXShapeStrides.length);
    const uniformData = [
        { type: 'int32', data: reshapedPaddedShape },
        { type: 'int32', data: paddedXShapeStrides }
    ];
    completePaddings.map(p => uniformData.push({ type: 'int32', data: [p[0], p[1]] }));
    const paddedXT = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
    const result = reshape({ inputs: { x: paddedXT }, backend, attrs: { shape: flattenShape } });
    backend.disposeData(paddedXT.dataId);
    return result;
};
const spaceToBatchNDConfig = {
    kernelName: SpaceToBatchND,
    backendName: 'webgpu',
    kernelFunc: spaceToBatchND
};

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
class SparseSegmentSumProgram {
    constructor(outShape, sparseSize, outputDtype) {
        this.variableNames = ['input', 'indices', 'segmentIds'];
        this.outputShape = [];
        this.uniforms = 'segmentSize : i32, sparseSize : i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = outShape;
        this.type = outputDtype;
        this.dispatchLayout = flatDispatchLayout([sparseSize]);
        this.dispatch =
            computeDispatch(this.dispatchLayout, [sparseSize], this.workgroupSize);
        this.shaderKey = 'sparseSegmentSum';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.sparseSize) {
        let indexInSegmentIds = index / uniforms.segmentSize;
        let indexInSegment = index % uniforms.segmentSize;
        let indexInInput = indices[indexInSegmentIds];
        let segmentId = segmentIds[indexInSegmentIds];

        let value = input[indexInInput * uniforms.segmentSize + indexInSegment];
        let outIndex = segmentId * uniforms.segmentSize + indexInSegment;
        ${atomicAddSnippet('&result[outIndex]', 'value', this.type)}
      }
    }
  `;
        return userCode;
    }
}
class SparseSegmentIdCountProgram {
    constructor(outShape, segmentIdsShape) {
        this.variableNames = ['segmentIds'];
        this.outputShape = [];
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = [outShape];
        this.dispatchLayout = flatDispatchLayout(segmentIdsShape);
        this.dispatch = computeDispatch(this.dispatchLayout, segmentIdsShape, this.workgroupSize);
        this.shaderKey = 'sparseSegmentIdCountProgram';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.segmentIdsShape) {
        let segmentId = segmentIds[index];
        ${atomicAddSnippet('&result[segmentId]', '1', 'int32')}
      }
    }
  `;
        return userCode;
    }
}
class SparseSegmentMeanProgram {
    constructor(outShape, outputDtype) {
        this.variableNames = ['segmentSum', 'sameSegmentIdCount'];
        this.outputShape = [];
        this.uniforms = 'segmentSize : i32';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outShape;
        this.type = outputDtype;
        this.dispatchLayout = flatDispatchLayout(outShape);
        this.dispatch =
            computeDispatch(this.dispatchLayout, outShape, this.workgroupSize);
        this.shaderKey = 'sparseSegmentMean';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let segmentId = index / uniforms.segmentSize;
        let count = sameSegmentIdCount[segmentId];
        if (count != 0) {
          ${this.type === 'float32' ?
            'setOutputAtIndex(index, segmentSum[index] / f32(count));' :
            'setOutputAtIndexI32(index, segmentSum[index] / count);'}
        }
      }
    }
  `;
        return userCode;
    }
}

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
function sparseSegmentReduce(input, indices, segmentIds, isSum = false, backend) {
    const inputSize = util.sizeFromShape(input.shape);
    const segmentSize = inputSize / input.shape[0];
    const dtype = input.dtype;
    // Note that the current implementation assumes that segmentIds values are
    // sorted.
    const numIndices = util.sizeFromShape(indices.shape);
    const $segmentIds = backend.readSync(segmentIds.dataId);
    const lastSegmentIdPlusOne = numIndices > 0 ? $segmentIds[numIndices - 1] + 1 : 0;
    const outputRows = lastSegmentIdPlusOne;
    let program;
    const outputShape = input.shape.slice();
    outputShape[0] = outputRows;
    const sparseSize = numIndices * segmentSize;
    const sparseSegmentSum = fill({ backend, attrs: { shape: outputShape, value: 0, dtype } });
    program = new SparseSegmentSumProgram(outputShape, sparseSize, dtype);
    let uniformData = [
        { type: 'int32', data: [segmentSize] }, { type: 'int32', data: [sparseSize] }
    ];
    const $sparseSegmentSum = backend.runWebGPUProgram(program, [input, indices, segmentIds], dtype, uniformData, sparseSegmentSum);
    if (isSum) {
        return $sparseSegmentSum;
    }
    const sparseSegmentIdCount = fill({ backend, attrs: { shape: [outputRows], value: 0, dtype: 'int32' } });
    program = new SparseSegmentIdCountProgram(outputRows, segmentIds.shape);
    const $sparseSegmentIdCount = backend.runWebGPUProgram(program, [segmentIds], 'int32', null, sparseSegmentIdCount);
    const sparseSegmentMean = fill({ backend, attrs: { shape: outputShape, value: 0, dtype } });
    program = new SparseSegmentMeanProgram(outputShape, dtype);
    uniformData = [{ type: 'int32', data: [segmentSize] }];
    const $sparseSegmentMean = backend.runWebGPUProgram(program, [$sparseSegmentSum, $sparseSegmentIdCount], dtype, uniformData, sparseSegmentMean);
    backend.disposeData($sparseSegmentSum.dataId);
    backend.disposeData($sparseSegmentIdCount.dataId);
    return $sparseSegmentMean;
}

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
function sparseSegmentMean(args) {
    const { inputs, backend } = args;
    const { data, indices, segmentIds } = inputs;
    return sparseSegmentReduce(data, indices, segmentIds, false, backend);
}
const sparseSegmentMeanConfig = {
    kernelName: SparseSegmentMean,
    backendName: 'webgpu',
    kernelFunc: sparseSegmentMean,
};

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
function sparseSegmentSum(args) {
    const { inputs, backend } = args;
    const { data, indices, segmentIds } = inputs;
    return sparseSegmentReduce(data, indices, segmentIds, true, backend);
}
const sparseSegmentSumConfig = {
    kernelName: SparseSegmentSum,
    backendName: 'webgpu',
    kernelFunc: sparseSegmentSum,
};

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
class TileProgram {
    constructor(aShape, reps) {
        this.variableNames = ['A'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const outputShape = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[i] * reps[i];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.rank = this.outputShape.length;
        this.shaderKey = 'tile';
    }
    getUserCode() {
        const sourceCoords = getSourceCoords(this.rank, 'uniforms.');
        const userCode = `
      ${getMainHeaderString('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          setOutputAtIndex(index, getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
function getSourceCoords(rank, uniformPrefix = '') {
    if (rank >= 5) {
        throw Error(`Tile for rank ${rank} is not yet supported`);
    }
    if (rank === 1) {
        return `(resRC % ${uniformPrefix}aShape)`;
    }
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < rank; i++) {
        sourceCoords.push(`(${currentCoords[i]} % ${uniformPrefix}aShape[${i}])`);
    }
    return sourceCoords.join();
}

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
function tile(params) {
    const { inputs, backend, attrs } = params;
    const { x } = inputs;
    const { reps } = attrs;
    // tile gpu program cannot handle rank >= 5 case.
    if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string' ||
        x.shape.length >= 5) {
        // Even thought string tensor is always on CPU, just to be consistent on how
        // to access tensor data.
        const data = backend.readSync(x.dataId);
        const value = x.dtype === 'string' ?
            data.map(d => util.decodeString(d)) :
            data;
        const buf = buffer(x.shape, x.dtype, value);
        const outBuf = tileImplCPU(buf, reps);
        return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
    }
    const program = new TileProgram(x.shape, reps);
    const output = backend.runWebGPUProgram(program, [x], x.dtype);
    return output;
}
const tileConfig = {
    kernelName: Tile,
    backendName: 'webgpu',
    kernelFunc: tile,
};

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
function sparseToDense(args) {
    const { inputs, backend, attrs } = args;
    const { sparseIndices, sparseValues, defaultValue } = inputs;
    const { outputShape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    if (sparseValues.dtype === 'string') {
        const indicesBuf = backend.bufferSync(sparseIndices);
        const updatesBuf = backend.bufferSync(sparseValues);
        const $defaultValue = util.decodeString(backend.readSync(defaultValue.dataId)[0]);
        const outBuf = scatterImplCPU(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
        return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
    }
    const flattenShape = [outputSize / sliceSize, sliceSize];
    const $sparseIndices = reshape({
        inputs: { x: sparseIndices },
        backend,
        attrs: { shape: [numUpdates, sliceRank] }
    });
    const $sparseValues = sparseValues.shape.length ?
        reshape({
            inputs: { x: sparseValues },
            backend,
            attrs: { shape: [numUpdates, sliceSize] }
        }) :
        identity({ inputs: { x: sparseValues }, backend });
    const type = $sparseValues.dtype;
    const zero = backend.makeTensorInfo([], type, util.makeZerosTypedArray(1, type));
    // Fill output tensor with the default value.
    const $defaultValue = reshape({
        inputs: { x: defaultValue },
        backend,
        attrs: { shape: Array(flattenShape.length).fill(1) }
    });
    const $denseValues = tile({ inputs: { x: $defaultValue }, backend, attrs: { reps: flattenShape } });
    const size = util.sizeFromShape([numUpdates, sliceSize]);
    const uniformData = [
        { type: 'int32', data: [sliceRank] },
        { type: 'int32', data: strides },
        { type: 'int32', data: [size] },
    ];
    switch (numUpdates) {
        case 0:
            break;
        case 1:
            {
                const program = new ScatterProgram([numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length, $sparseValues.shape.length, strides, flattenShape, type, sumDupeIndices);
                backend.runWebGPUProgram(program, [$sparseValues, $sparseIndices], type, uniformData, $denseValues);
            }
            break;
        default:
            {
                // First replace the default value with 0 at indices.
                const program = new ScatterProgram([numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length, zero.shape.length, strides, flattenShape, type, sumDupeIndices);
                backend.runWebGPUProgram(program, [zero, $sparseIndices], type, uniformData, $denseValues);
            }
            {
                // Then replace 0 with the (sum of) sparse value(s) at indices.
                const program = new ScatterProgram([numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length, $sparseValues.shape.length, strides, flattenShape, type);
                backend.runWebGPUProgram(program, [$sparseValues, $sparseIndices], type, uniformData, $denseValues);
            }
    }
    const denseValues = reshape({ inputs: { x: $denseValues }, backend, attrs: { shape: outputShape } });
    backend.disposeData($sparseIndices.dataId);
    backend.disposeData($sparseValues.dataId);
    backend.disposeData($defaultValue.dataId);
    backend.disposeData(zero.dataId);
    backend.disposeData($denseValues.dataId);
    return denseValues;
}
const sparseToDenseConfig = {
    kernelName: SparseToDense,
    backendName: 'webgpu',
    kernelFunc: sparseToDense
};

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
function splitV(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { numOrSizeSplits, axis } = attrs;
    const $axis = util.parseAxisParam(axis, x.shape)[0];
    const splitSizes = backend_util.prepareSplitSize(x, numOrSizeSplits, $axis);
    const xRank = x.shape.length;
    const begin = new Array(xRank).fill(0);
    const size = x.shape.slice();
    return splitSizes.map(s => {
        const sliceSize = [...size];
        sliceSize[$axis] = s;
        const sliceT = slice({ inputs: { x }, backend, attrs: { begin, size: sliceSize } });
        begin[$axis] += s;
        return sliceT;
    });
}
const splitVConfig = {
    kernelName: SplitV,
    backendName: 'webgpu',
    kernelFunc: splitV
};

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
const sqrt = unaryKernelFunc({ opType: UnaryOpType.SQRT });
const sqrtConfig = {
    kernelName: Sqrt,
    backendName: 'webgpu',
    kernelFunc: sqrt
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const squareConfig = {
    kernelName: Square,
    backendName: 'webgpu',
    kernelFunc: ({ inputs, backend }) => {
        const { x } = inputs;
        const webGPUBackend = backend;
        const program = new UnaryOpProgram(x.shape, UnaryOpType.SQUARE);
        return webGPUBackend.runWebGPUProgram(program, [x], x.dtype);
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
const squaredDifference = binaryKernelFunc({
    opType: BinaryOpType.SQUARED_DIFFERENCE,
});
const squaredDifferenceConfig = {
    kernelName: SquaredDifference,
    backendName: 'webgpu',
    kernelFunc: squaredDifference
};

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
function step({ inputs, attrs, backend }) {
    const { x } = inputs;
    const program = new UnaryOpProgram(x.shape, UnaryOpType.STEP, 'stepAlpha : f32,');
    const uniformData = [{ type: 'float32', data: [attrs.alpha] }];
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}
const stepConfig = {
    kernelName: Step,
    backendName: 'webgpu',
    kernelFunc: step
};

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
class StridedSliceProgram {
    constructor(destSize) {
        this.variableNames = ['x'];
        // TODO(xing.xu): Increase the workPerThread.
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = destSize;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        const dtype = getCoordsDataType(this.outputShape.length);
        this.uniforms = `begin : ${dtype},  strides : ${dtype}, `;
        this.shaderKey = 'stridedSlice';
    }
    getUserCode() {
        const rank = this.outputShape.length;
        let newCoords = '';
        if (rank === 1) {
            newCoords = 'coords * uniforms.strides + uniforms.begin';
        }
        else {
            let outputAxis = 0;
            newCoords =
                this.outputShape
                    .map((_, i) => {
                    outputAxis++;
                    return this.outputShape.length === 1 ?
                        `coords * uniforms.strides[${i}] + uniforms.begin[${i}]` :
                        `coords[${outputAxis - 1}] * uniforms.strides[${i}] + uniforms.begin[${i}]`;
                })
                    .join(',');
        }
        const userCode = `
       ${getMainHeaderString('index')} {
         if (index < uniforms.size) {
           let coords = getCoordsFromIndex(index);
           setOutputAtIndex(index, getX(${newCoords}));
         }
       }
     `;
        return userCode;
    }
}

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
function stridedSlice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask } = attrs;
    const { finalShapeSparse, finalShape, isIdentity, sliceDim0, isSimpleSlice, begin: $begin, end: $end, strides: $strides } = slice_util.sliceInfo(x.shape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    let result;
    if (isIdentity) {
        // Optimization #1, slice is a no-op plus reshape
        result = reshape({ inputs: { x }, backend, attrs: { shape: finalShape } });
    }
    else if (sliceDim0 || isSimpleSlice) {
        // Optimization #2, slice is memory contiguous (only occurs in dim 0)
        util.assert(x.shape.length >= 1, () => `Input must have rank at least 1, got: ${x.shape.length}`);
        const size = slice_util.computeOutShape($begin, $end, $strides);
        // To tolerate begin[0] > end[0] (a 0-output slice), we min(begin, end).
        const sliced = slice({ inputs: { x }, backend, attrs: { begin: $begin, size } });
        result =
            reshape({ inputs: { x: sliced }, backend, attrs: { shape: finalShape } });
        backend.disposeData(sliced.dataId);
    }
    else {
        const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([x]);
        if (shouldExecuteOnCPU) {
            const values = backend.readSync(x.dataId);
            const xBuf = buffer(x.shape, x.dtype, values);
            const resultValues = stridedSliceImplCPU(finalShapeSparse, xBuf, $strides, $begin);
            result = backend.makeTensorInfo(finalShape, x.dtype, resultValues.values);
        }
        else {
            const program = new StridedSliceProgram(finalShapeSparse);
            const uniformData = [{ type: 'int32', data: $begin }, { type: 'int32', data: $strides }];
            const resultValues = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
            result = reshape({ inputs: { x: resultValues }, backend, attrs: { shape: finalShape } });
            backend.disposeData(resultValues.dataId);
        }
    }
    return result;
}
const stridedSliceConfig = {
    kernelName: StridedSlice,
    backendName: 'webgpu',
    kernelFunc: stridedSlice
};

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
function stringNGrams(args) {
    const { inputs, backend, attrs } = args;
    const { separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences } = attrs;
    const { data, dataSplits } = inputs;
    const $data = backend.readSync(data.dataId);
    const $dataSplits = backend.readSync(dataSplits.dataId);
    const [nGrams, nGramsSplits] = stringNGramsImplCPU($data, $dataSplits, separator, nGramWidths, leftPad, rightPad, padWidth, preserveShortSequences);
    return [
        backend.makeTensorInfo([nGrams.length], 'string', nGrams),
        backend.makeTensorInfo(dataSplits.shape, 'int32', nGramsSplits),
    ];
}
const stringNGramsConfig = {
    kernelName: StringNGrams,
    backendName: 'webgpu',
    kernelFunc: stringNGrams,
};

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
const sub = binaryKernelFunc({ opType: BinaryOpType.SUB, cpuKernelImpl: subImplCPU, supportsComplex: true });
const subConfig = {
    kernelName: Sub,
    backendName: 'webgpu',
    kernelFunc: sub
};

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
const tan = unaryKernelFunc({ opType: UnaryOpType.TAN });
const tanConfig = {
    kernelName: Tan,
    backendName: 'webgpu',
    kernelFunc: tan
};

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
const tanh = unaryKernelFunc({ opType: UnaryOpType.TANH });
const tanhConfig = {
    kernelName: Tanh,
    backendName: 'webgpu',
    kernelFunc: tanh
};

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
function tensorScatterUpdate(args) {
    const { inputs, backend, attrs } = args;
    const { tensor, indices, updates } = inputs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(updates, indices, tensor.shape);
    const flattenShape = [outputSize / sliceSize, sliceSize];
    if (outputSize === 0) {
        return backend.makeTensorInfo(tensor.shape, indices.dtype);
    }
    const toDispose = [];
    const flattenIndices = reshape({ inputs: { x: indices }, backend, attrs: { shape: [numUpdates, sliceRank] } });
    toDispose.push(flattenIndices);
    const flattenX = reshape({ inputs: { x: updates }, backend, attrs: { shape: [numUpdates, sliceSize] } });
    toDispose.push(flattenX);
    const flattenTensor = reshape({ inputs: { x: tensor }, backend, attrs: { shape: flattenShape } });
    toDispose.push(flattenTensor);
    const output = tile({
        inputs: { x: flattenTensor },
        backend,
        attrs: { reps: Array(flattenShape.length).fill(1) }
    });
    const program = new ScatterProgram([numUpdates, sliceSize], sliceRank, flattenIndices.shape.length, flattenX.shape.length, strides, flattenShape, tensor.dtype, false);
    const size = util.sizeFromShape([numUpdates, sliceSize]);
    const uniformData = [
        { type: 'int32', data: [sliceRank] },
        { type: 'int32', data: strides },
        { type: 'int32', data: [size] },
    ];
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndices], flattenTensor.dtype, uniformData, output);
    toDispose.push(res);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: tensor.shape } });
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return reshaped;
}
const tensorScatterUpdateConfig = {
    kernelName: TensorScatterUpdate,
    backendName: 'webgpu',
    kernelFunc: tensorScatterUpdate
};

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
// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
// The original algorithm is based on computing the top K only, however
// since for TFJS we require the indices of the top K values as well then the
// algorithm found here is a bit modified. Rather than producing the values
// at each step, the indices containing the top K are generated instead.
// The output values are not generated to reduce the number of outputs in the
// GPU, the values can easily be retrieved from the indices using a gather
// op.
class SwapProgram {
    constructor(shape) {
        this.variableNames = ['x', 'indices'];
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = `inputSize : i32, firstPass : i32, negativeInf : f32,
        dir : i32, inc : i32,`;
        this.shaderKey = 'swap';
    }
    getUserCode() {
        const userCode = `
        ${getMainHeaderString('index')} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // We compare elements pair-wise within a group of size 2 * inc.
            // The comparing rule for each group alternates between ascending
            // and descending. Within each group, we compare each pair at
            // positions i and i+inc. To decide whether an element at position i
            // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
            // inc, it is in the first half of the group, we denote it as x0,
            // otherwise we denote it as x1.
            // For example, as shown in the Bitonic top K paper referenced
            // above, Figure5(a) shows that element[1] is in the second half of
            // the group when group size is 2, but it is in the first half of
            // the group when group size is 4.
            let isFirstInPair = elemIdx % (2 * uniforms.inc) < uniforms.inc;
            var i = 0;
            if (isFirstInPair) {
              i = elemIdx;
            } else {
              i = elemIdx - uniforms.inc;
            }

            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }

            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.inc;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.inc));
            }

            var x0 = f32(0.0);
            var x1 = f32(0.0);
            if (i0 < uniforms.inputSize) {
              x0 = getX(batch, i0);
            } else {
              x0 = uniforms.negativeInf;
            }
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = uniforms.negativeInf;
            }

            let reverse = elemIdx % (2 * uniforms.dir) >= uniforms.dir;
            let isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
            if (reverse == isGreater) {
              // Elements in opposite order of direction
              let iTemp = i0;
              i0 = i1;
              i1 = iTemp;
            }
            if (isFirstInPair) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `;
        return userCode;
    }
}
class MergeProgram {
    constructor(shape) {
        this.variableNames = ['x', 'indices'];
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        // |n| Size of the original input of TopK
        // |firstPass| indicates if this is the first time swap is being used which
        // means no indices input containing the top K is present yet.
        // |k| Top k elements desired
        this.uniforms = `inputSize : i32, firstPass : i32, k : i32,`;
        this.shaderKey = 'merge';
    }
    getUserCode() {
        const userCode = `
        ${getMainHeaderString('index')} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // The output size is half of the previous size.
            // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _
            // (k=4), we only need to output the indices at positions |, the
            // indices at positions _ can be thrown away, see Figure5(b) After
            // Phase 2 (Merge phase) in the Bitonic Top K paper referenced
            // above.
            // For example, the paper shows we only need to output the orange
            // bars. The output sequence should look like this | | | | | | | |.
            // Because the sequence is halved, to map the output index back to
            // the previous sequence to find the corresponding value, we need
            // to double the index. When we double the index, we basically
            // interpolate a position, so 2i looks like
            // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k
            // position of each 2k positions by - elemIdx % k. E.g. for output
            // at index 4,5,6,7, we want to get the corresponding element at
            // original index 8,9,10,11, for output at index 8,9,10,11,
            // we want to get the corresponding element at original index
            // 16,17,18,19, so on and so forth.

            var i = 0;
            if (elemIdx < uniforms.k) {
              i = elemIdx;
            } else {
              i = elemIdx * 2 - elemIdx % uniforms.k;
            }
            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }
            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.k;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.k));
            }

            let x0 = getX(batch, i0);
            var x1 = f32(0.0);
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = x0;
            }

            if (x0 >= x1) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `;
        return userCode;
    }
}

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
function disposeIntermediateTensorInfoOrNull(backend, tensorInfo) {
    if (tensorInfo !== null) {
        backend.disposeData(tensorInfo.dataId);
    }
}
function roundUpToPow2(num) {
    let pow2 = 1;
    while (pow2 < num) {
        pow2 *= 2;
    }
    return pow2;
}
// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
function topK(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { k, sorted } = attrs;
    const xShape = x.shape;
    const lastDim = xShape[xShape.length - 1];
    if (backend.shouldExecuteOnCPU([x])) {
        const xVals = backend.readSync(x.dataId);
        const [allTopKVals, allTopKIndices] = topKImplCPU(xVals, xShape, x.dtype, k, sorted);
        return [
            backend.makeTensorInfo(allTopKVals.shape, allTopKVals.dtype, allTopKVals.values),
            backend.makeTensorInfo(allTopKIndices.shape, allTopKIndices.dtype, allTopKIndices.values)
        ];
    }
    if (k === 0) {
        xShape[xShape.length - 1] = 0;
        return [
            backend.makeTensorInfo(xShape, x.dtype, []),
            backend.makeTensorInfo(xShape, 'int32', [])
        ];
    }
    if (lastDim === 1 /* firstPass */) {
        return [
            x, fill({ attrs: { shape: xShape, dtype: 'int32', value: 0 }, backend })
        ];
    }
    // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
    const xSize = util.sizeFromShape(xShape);
    const batch = xSize / lastDim;
    const x2D = reshape({ inputs: { x }, attrs: { shape: [batch, lastDim] }, backend });
    const kPow2 = roundUpToPow2(k);
    const lastDimPow2 = roundUpToPow2(lastDim);
    // Only the indices containing the top K are kept at every step to reduce
    // number of outputs in the GPU algorithms, so once the final set of indices
    // is computed then gather is used to grab the corresponding values
    // from the original input.
    let indices = null;
    // GPU algorithm always takes in an indices input but this input is not used
    // on the first run of a GPU algorithm, therefore if indices is null we simply
    // pass in x2D instead of it but the value will not actually be used
    const getInputs = () => indices === null ? [x2D, x2D] : [x2D, indices];
    const runSwap = (dir, inc, shape) => {
        const inputs = getInputs();
        const program = new SwapProgram(shape);
        const firstPass = indices === null ? 1 : 0;
        const uniformDataSwap = [
            { type: 'int32', data: [lastDim] },
            { type: 'int32', data: [firstPass] },
            { type: 'float32', data: [Number.NEGATIVE_INFINITY] },
            { type: 'int32', data: [dir] },
            { type: 'int32', data: [inc] }
        ];
        const prevIndices = indices;
        indices = backend.runWebGPUProgram(program, inputs, 'int32', uniformDataSwap);
        disposeIntermediateTensorInfoOrNull(backend, prevIndices);
    };
    // Step 1: local sort
    for (let len = 1; len < kPow2; len *= 2) {
        const dir = len * 2;
        for (let inc = len; inc >= 1; inc /= 2) {
            runSwap(dir, inc, [batch, lastDimPow2]);
        }
    }
    // Step 2: merge
    for (let indicesSize = lastDimPow2; indicesSize > kPow2; indicesSize /= 2) {
        const inputs = getInputs();
        const mergeProgram = new MergeProgram([batch, indicesSize / 2]);
        const firstPass = indices === null ? 1 : 0;
        const uniformDataMerge = [
            { type: 'int32', data: [lastDim] },
            { type: 'int32', data: [firstPass] },
            { type: 'int32', data: [kPow2] }
        ];
        const prevIndices = indices;
        indices = backend.runWebGPUProgram(mergeProgram, inputs, 'int32', uniformDataMerge);
        disposeIntermediateTensorInfoOrNull(backend, prevIndices);
        // Step 3: rebuild
        const len = kPow2 / 2;
        const dir = len * 2;
        for (let inc = len; inc >= 1; inc /= 2) {
            runSwap(dir, inc, indices.shape);
        }
    }
    // Keep only the requested top K results instead of kPow2
    let prevIndices = indices;
    indices = slice({ inputs: { x: indices }, backend, attrs: { begin: 0, size: [batch, k] } });
    disposeIntermediateTensorInfoOrNull(backend, prevIndices);
    // Gather values on last dimension
    let values = gatherV2({ inputs: { x: x2D, indices }, backend, attrs: { axis: 1, batchDims: 1 } });
    disposeIntermediateTensorInfoOrNull(backend, x2D);
    // Reshape back to the original input shape, except that the last
    // dimension is k.
    const newShape = xShape.slice(0, -1);
    newShape.push(k);
    prevIndices = indices;
    indices = reshape({ inputs: { x: indices }, attrs: { shape: newShape }, backend });
    disposeIntermediateTensorInfoOrNull(backend, prevIndices);
    const prevValues = values;
    values = reshape({ inputs: { x: values }, attrs: { shape: newShape }, backend });
    disposeIntermediateTensorInfoOrNull(backend, prevValues);
    return [values, indices];
}
const topKConfig = {
    kernelName: TopK,
    backendName: 'webgpu',
    kernelFunc: topK
};

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
class TransformProgram {
    constructor(outShape) {
        this.variableNames = ['Image', 'Transforms'];
        this.uniforms = 'interpolationModeId : i32, fillModeId : i32, fillValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'transform';
    }
    getUserCode() {
        const userCode = `
          fn mapCoord(outCoord : f32, len : f32) -> f32{
            var inCoord = outCoord;
            if(uniforms.fillModeId == 2) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  if (inCoord < sz2) {
                    inCoord = sz2 * f32(i32(f32(-inCoord / sz2))) +
                    inCoord;
                  }
                  if (inCoord < -len) {
                    inCoord = inCoord + sz2;
                  } else {
                    inCoord = -inCoord - 1.0;
                  }
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz2 = 2.0 * len;
                  inCoord = inCoord - sz2 * f32(i32(f32(inCoord / sz2)));
                  if (inCoord >= len) {
                    inCoord = sz2 - inCoord - 1.0;
                  }
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 3) {
              if (inCoord < 0.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord + len * (f32(i32(f32(-inCoord / sz))) + 1.0);
                }
              } else if (inCoord > len - 1.0) {
                if (len <= 1.0) {
                  inCoord = 0.0;
                } else {
                  let sz = len - 1.0;
                  inCoord = inCoord - len * f32(i32(f32(inCoord / sz)));
                }
              }
              return clamp(inCoord, 0.0, len - 1.0);
            } else if (uniforms.fillModeId == 4) {
              return clamp(outCoord, 0.0, len - 1.0);
            }
            return outCoord;
          }
          fn readWithFillValue(batch : i32, coordY : i32, coordX : i32,
            channel : i32) -> f32 {
            var outputValue : f32;
            if (0 <= coordY && coordY < uniforms.imageShape[1] && 0 <= coordX && coordX < uniforms.imageShape[2]) {
                outputValue = getImage(batch, coordY, coordX, channel);
            } else {
              outputValue = uniforms.fillValue;
            }
            return outputValue;
          }

          ${getMainHeaderString('index')} {
            if (index < uniforms.size) {
              let coords = getCoordsFromIndex(index);
              var outputValue : f32;
              let batch = coords[0];
              let x = coords[2];
              let y = coords[1];
              let channel = coords[3];
              let xf = f32(x);
              let yf = f32(y);
              let a1 = getTransforms(batch, 0);
              let a2 = getTransforms(batch, 1);
              let a3 = getTransforms(batch, 2);
              let b1 = getTransforms(batch, 3);
              let b2 = getTransforms(batch, 4);
              let b3 = getTransforms(batch, 5);
              let c1 = getTransforms(batch, 6);
              let c2 = getTransforms(batch, 7);
              let projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = uniforms.fillValue;
              } else {
                let inX = (a1 * xf + a2 * yf + a3) / projection;
                let inY = (b1 * xf + b2 * yf + b3) / projection;
                let mapX = mapCoord(inX, f32(uniforms.imageShape[2]));
                let mapY = mapCoord(inY, f32(uniforms.imageShape[1]));

                if (uniforms.interpolationModeId == 1) {
                  let coordY = i32(round(mapY));
                  let coordX = i32(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  let yFloor = floor(mapY);
                  let xFloor = floor(mapX);
                  let yCeil = yFloor + 1.0;
                  let xCeil = xFloor + 1.0;
                  let valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yFloor), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yFloor), i32(xCeil), channel);
                  let valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, i32(yCeil), i32(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, i32(yCeil), i32(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutputAtIndex(index, outputValue);
            }
          }
        `;
        return userCode;
    }
}

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
function transform(args) {
    const { inputs, backend, attrs } = args;
    const { image, transforms } = inputs;
    const { interpolation, fillMode, fillValue, outputShape } = attrs;
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;
    const [outHeight, outWidth] = outputShape != null ? outputShape : [imageHeight, imageWidth];
    const outShape = [batch, outHeight, outWidth,
        numChannels];
    const program = new TransformProgram(outShape);
    const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
    let fillModeId;
    switch (fillMode) {
        case 'constant':
            fillModeId = 1;
            break;
        case 'reflect':
            fillModeId = 2;
            break;
        case 'wrap':
            fillModeId = 3;
            break;
        case 'nearest':
            fillModeId = 4;
            break;
        default:
            fillModeId = 1;
            break;
    }
    const uniformData = [
        { type: 'int32', data: [interpolationModeId] },
        { type: 'int32', data: [fillModeId] }, { type: 'float32', data: [fillValue] }
    ];
    return backend.runWebGPUProgram(program, [image, transforms], 'float32', uniformData);
}
const transformConfig = {
    kernelName: Transform,
    backendName: 'webgpu',
    kernelFunc: transform
};

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
function unpack(args) {
    const { inputs, backend, attrs } = args;
    const { value } = inputs;
    let { axis } = attrs;
    if (axis < 0) {
        axis += value.shape.length;
    }
    const x = value;
    const xRank = x.shape.length;
    const num = value.shape[axis];
    const outShape = new Array(xRank - 1);
    let outIndex = 0;
    for (let i = 0; i < xRank; i++) {
        if (i !== axis) {
            outShape[outIndex++] = x.shape[i];
        }
    }
    const toDispose = [];
    const begin = new Array(xRank).fill(0);
    const size = x.shape.slice();
    size[axis] = 1;
    const res = new Array(num);
    for (let i = 0; i < res.length; i++) {
        begin[axis] = i;
        const sliced = slice({ inputs: { x }, backend, attrs: { begin, size } });
        const reshaped = reshape({ inputs: { x: sliced }, backend, attrs: { shape: outShape } });
        res[i] = reshaped;
        toDispose.push(sliced);
    }
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return res;
}
const unpackConfig = {
    kernelName: Unpack,
    backendName: 'webgpu',
    kernelFunc: unpack
};

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
class UnsortedSegmentSumProgram {
    constructor(inShape, outShape, outputDtype) {
        this.outputShape = [];
        this.variableNames = ['x', 'segmentIds'];
        this.uniforms = 'numSegments : i32, xSize: i32,';
        this.workgroupSize = [64, 1, 1];
        this.atomic = true;
        this.outputShape = outShape;
        this.dispatchLayout = flatDispatchLayout(inShape);
        this.dispatch =
            computeDispatch(this.dispatchLayout, inShape, this.workgroupSize);
        if (outputDtype !== 'float32' && outputDtype !== 'int32') {
            throw new Error(`UnsortedSegmentSum only supports float32 and int32
              types, does not support ${outputDtype} type.`);
        }
        this.type = outputDtype;
        this.shaderKey = 'unsortedSegmentSum';
    }
    getUserCode() {
        const userCode = `
    ${getMainHeaderString('index')} {
      if (index < uniforms.xSize) {
        let coords = getXCoordsFromIndex(index);
        let b = coords[0];
        let inCol = coords[1];

        let segmentId = i32(getSegmentIds(inCol));
        if (segmentId >= 0) {
          let flatIndex = b * uniforms.numSegments + segmentId % uniforms.numSegments;
          let value = getX(b, inCol);

          ${atomicAddSnippet('&result[flatIndex]', 'value', this.type)}
        }
      }
    }
  `;
        return userCode;
    }
}

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
function unsortedSegmentSum(args) {
    const { inputs, backend, attrs } = args;
    const { x, segmentIds } = inputs;
    const { numSegments } = attrs;
    const xRank = x.shape.length;
    const toDispose = [];
    let axis = 0;
    const permutation = backend_util.getAxesPermutation([axis], xRank);
    let permutedX = x;
    if (permutation != null) {
        permutedX = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
        toDispose.push(permutedX);
        axis = backend_util.getInnerMostAxes(1, xRank)[0];
    }
    const outShape = backend_util.segment_util.computeOutShape(permutedX.shape, axis, numSegments);
    const inSize = util.sizeFromShape([permutedX.shape[axis]]);
    const a2D = reshape({ inputs: { x: permutedX }, backend, attrs: { shape: [-1, inSize] } });
    toDispose.push(a2D);
    const dtype = x.dtype;
    const shape = [a2D.shape[0], numSegments];
    const output = fill({ backend, attrs: { shape, value: 0, dtype } });
    const program = new UnsortedSegmentSumProgram(a2D.shape, shape, dtype);
    const uniformData = [
        { type: 'int32', data: [numSegments] },
        { type: 'int32', data: [util.sizeFromShape(a2D.shape)] }
    ];
    const segResult = backend.runWebGPUProgram(program, [a2D, segmentIds], dtype, uniformData, output);
    const reshaped = reshape({ inputs: { x: segResult }, backend, attrs: { shape: outShape } });
    toDispose.push(segResult);
    let result = reshaped;
    if (permutation != null) {
        toDispose.push(reshaped);
        const perm = backend_util.getUndoAxesPermutation(permutation);
        result = transpose({ inputs: { x: result }, backend, attrs: { perm } });
    }
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return result;
}
const unsortedSegmentSumConfig = {
    kernelName: UnsortedSegmentSum,
    backendName: 'webgpu',
    kernelFunc: unsortedSegmentSum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
// List all kernel configs here
const kernelConfigs = [
    _fusedMatMulConfig,
    absConfig,
    acosConfig,
    acoshConfig,
    addConfig,
    addNConfig,
    allConfig,
    anyConfig,
    argMaxConfig,
    argMinConfig,
    asinConfig,
    asinhConfig,
    atanConfig,
    atan2Config,
    atanhConfig,
    avgPoolConfig,
    avgPool3DConfig,
    avgPool3DGradConfig,
    avgPoolGradConfig,
    batchMatMulConfig,
    batchToSpaceNDConfig,
    bincountConfig,
    broadcastArgsConfig,
    castConfig,
    ceilConfig,
    clipByValueConfig,
    complexConfig,
    complexAbsConfig,
    concatConfig,
    conv2DConfig,
    conv2DBackpropFilterConfig,
    conv2DBackpropInputConfig,
    conv3DConfig,
    conv3DBackpropFilterV2Config,
    conv3DBackpropInputV2Config,
    cosConfig,
    coshConfig,
    cropAndResizeConfig,
    cumprodConfig,
    cumsumConfig,
    denseBincountConfig,
    depthToSpaceConfig,
    depthwiseConv2dNativeBackpropFilterConfig,
    depthwiseConv2dNativeBackpropInputConfig,
    depthwiseConv2dNativeConfig,
    diagConfig,
    dilation2DConfig,
    dilation2DBackpropFilterConfig,
    dilation2DBackpropInputConfig,
    drawConfig,
    einsumConfig,
    eluConfig,
    eluGradConfig,
    equalConfig,
    erfConfig,
    expConfig,
    expandDimsConfig,
    expm1Config,
    fftConfig,
    fillConfig,
    flipLeftRightConfig,
    fromPixelsConfig,
    floorConfig,
    floorDivConfig,
    fusedBatchNormConfig,
    fusedConv2DConfig,
    fusedDepthwiseConv2DConfig,
    gatherNdConfig,
    gatherV2Config,
    greaterConfig,
    greaterEqualConfig,
    identityConfig,
    ifftConfig,
    imagConfig,
    isFiniteConfig,
    isInfConfig,
    isNaNConfig,
    leakyReluConfig,
    lessConfig,
    lessEqualConfig,
    linSpaceConfig,
    log1pConfig,
    logConfig,
    logicalAndConfig,
    logicalNotConfig,
    logicalOrConfig,
    lrnConfig,
    lrnGradConfig,
    maxConfig,
    maximumConfig,
    maxPoolConfig,
    maxPoolGradConfig,
    maxPool3DConfig,
    maxPool3DGradConfig,
    maxPoolWithArgmaxConfig,
    meanConfig,
    minConfig,
    minimumConfig,
    mirrorPadConfig,
    modConfig,
    multinomialConfig,
    multiplyConfig,
    negConfig,
    nonMaxSuppressionV3Config,
    nonMaxSuppressionV5Config,
    notEqualConfig,
    oneHotConfig,
    onesLikeConfig,
    packConfig,
    padV2Config,
    powConfig,
    preluConfig,
    prodConfig,
    rangeConfig,
    realConfig,
    realDivConfig,
    reciprocalConfig,
    reluConfig,
    relu6Config,
    reshapeConfig,
    resizeBilinearConfig,
    resizeBilinearGradConfig,
    resizeNearestNeighborConfig,
    resizeNearestNeighborGradConfig,
    reverseConfig,
    rotateWithOffsetConfig,
    roundConfig,
    rsqrtConfig,
    scatterNdConfig,
    searchSortedConfig,
    selectConfig,
    seluConfig,
    sigmoidConfig,
    signConfig,
    sinConfig,
    sinhConfig,
    sliceConfig,
    stepConfig,
    stridedSliceConfig,
    stringNGramsConfig,
    softmaxConfig,
    softplusConfig,
    spaceToBatchNDConfig,
    sparseSegmentMeanConfig,
    sparseSegmentSumConfig,
    sparseToDenseConfig,
    splitVConfig,
    sqrtConfig,
    squareConfig,
    squaredDifferenceConfig,
    subConfig,
    sumConfig,
    tanConfig,
    tanhConfig,
    tensorScatterUpdateConfig,
    tileConfig,
    topKConfig,
    transformConfig,
    transposeConfig,
    unpackConfig,
    unsortedSegmentSumConfig,
    zerosLikeConfig
];
for (const kernelConfig of kernelConfigs) {
    registerKernel(kernelConfig);
}

export { WebGPUBackend, webgpu_util };
//# sourceMappingURL=tf-backend-webgpu.fesm.js.map
