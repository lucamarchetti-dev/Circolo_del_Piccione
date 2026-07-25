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
import './flags_webgpu';
import { backend_util, buffer, DataStorage, engine, env, KernelBackend, util } from '@tensorflow/tfjs-core';
import { AdapterInfo } from './adapter_info';
import { BufferManager } from './buffer_manager';
import { TextureManager } from './texture_manager';
import * as webgpu_program from './webgpu_program';
import * as webgpu_util from './webgpu_util';
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
        if (!webgpu_util.isWebGPUSupported()) {
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
        const size = webgpu_util.GPUBytesPerElement(tensorData.dtype) *
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
        const size = webgpu_util.GPUBytesPerElement(tensorData.dtype) *
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
            webgpu_program.makeShaderKey(program, inputsData, output);
        const parallelCompilation = env().getBool('WEBGPU_ENGINE_COMPILE_ONLY');
        if (!(program.shaderKey in this.pipelineCache)) {
            this.pipelineCache[program.shaderKey] = webgpu_program.compileProgram(this.device, program, inputsData, output, parallelCompilation);
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
            program.pixelsOpType === webgpu_program.PixelsOpType.DRAW) {
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
export { WebGPUBackend };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93ZWJncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9iYWNrZW5kX3dlYmdwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLGdCQUFnQixDQUFDO0FBRXhCLE9BQU8sRUFBQyxZQUFZLEVBQWlCLE1BQU0sRUFBRSxXQUFXLEVBQVksTUFBTSxFQUFFLEdBQUcsRUFBVyxhQUFhLEVBQTRGLElBQUksRUFBYSxNQUFNLHVCQUF1QixDQUFDO0FBRWxQLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUMzQyxPQUFPLEVBQUMsYUFBYSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDL0MsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ2pELE9BQU8sS0FBSyxjQUFjLE1BQU0sa0JBQWtCLENBQUM7QUFDbkQsT0FBTyxLQUFLLFdBQVcsTUFBTSxlQUFlLENBQUM7QUF1QzdDLCtFQUErRTtBQUMvRSw0QkFBNEI7QUFDNUIsTUFBTSwwQkFBMEIsR0FDNUIsR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLG1DQUFtQyxDQUFDLENBQUM7QUFFekQsaURBQWlEO0FBQ2pELE1BQU0sZUFBZSxHQUNqQixDQUFDLE1BQWlCLEVBQ2pCLE9BQXFDLEVBQTRCLEVBQUU7SUFDbEUsTUFBTSx1Q0FBdUMsR0FDekMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxnQ0FBZ0MsQ0FBQztJQUNuRCxNQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsZ0JBQWdCLENBQUMsQ0FBQztJQUN6QyxNQUFNLFFBQVEsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFDLENBQUM7SUFDckMsSUFBSSxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLElBQUksdUNBQXVDLENBQUMsRUFBRTtR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QUMsT0FBTyxFQUFFLENBQUM7UUFDN0IsSUFBSSxDQUFDLGNBQWMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUM5QixJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQztJQUN2QixDQUFDOztBQXQ2QmMsd0JBQVUsR0FBRyxDQUFDLEFBQUosQ0FBSztTQW5CbkIsYUFBYSIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICcuL2ZsYWdzX3dlYmdwdSc7XG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBCYWNrZW5kVmFsdWVzLCBidWZmZXIsIERhdGFTdG9yYWdlLCBEYXRhVHlwZSwgZW5naW5lLCBlbnYsIEdQVURhdGEsIEtlcm5lbEJhY2tlbmQsIFJhbmssIFJlY3Vyc2l2ZUFycmF5LCBTaGFwZU1hcCwgVGVuc29yLCBUZW5zb3JCdWZmZXIsIFRlbnNvckluZm8sIFRpbWluZ0luZm8sIFR5cGVkQXJyYXksIHV0aWwsIFdlYkdQVURhdGF9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QWRhcHRlckluZm99IGZyb20gJy4vYWRhcHRlcl9pbmZvJztcbmltcG9ydCB7QnVmZmVyTWFuYWdlcn0gZnJvbSAnLi9idWZmZXJfbWFuYWdlcic7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB3ZWJncHVfcHJvZ3JhbSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCAqIGFzIHdlYmdwdV91dGlsIGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdQVU1lbW9yeUluZm8gZXh0ZW5kcyBiYWNrZW5kX3V0aWwuTWVtb3J5SW5mbyB7XG4gIG51bUJ5dGVzSW5HUFU6IG51bWJlcjtcbiAgbnVtQnl0ZXNBbGxvY2F0ZWRJbkdQVTogbnVtYmVyO1xuICB1bnJlbGlhYmxlOiBib29sZWFuO1xufVxuXG50eXBlIFRlbnNvckRhdGEgPSB7XG4gIHZhbHVlczogQmFja2VuZFZhbHVlcyxcbiAgZHR5cGU6IERhdGFUeXBlLFxuICBzaGFwZTogbnVtYmVyW10sXG4gIHJlZkNvdW50OiBudW1iZXIsXG4gIHJlc291cmNlPzogR1BVQnVmZmVyfEdQVVRleHR1cmV8R1BVRXh0ZXJuYWxUZXh0dXJlLFxuICAvLyBleHRlcm5hbCBpcyB0cnVlIG1lYW5zIHdlIHVzZSB0aGUgcmVzb3VyY2UgcHJvdmlkZWQgYnkgdXNlcnMgZGlyZWN0bHlcbiAgLy8gKHdpdGhvdXQgYSBjb3B5KSwgc28gdXNlcnMgc2hvdWxkIGJlIHJlc3BvbnNpYmxlIGZvciBpdHMgcmVsZWFzZS5cbiAgZXh0ZXJuYWw/OiBib29sZWFuLFxuICAvLyBGb3IgY29tcGxleCBudW1iZXJzLCB0aGUgcmVhbCBhbmQgaW1hZ2luYXJ5IHBhcnRzIGFyZSBzdG9yZWQgYXMgdGhlaXIgb3duXG4gIC8vIGluZGl2aWR1YWwgdGVuc29ycywgd2l0aCBhIHBhcmVudCBqb2luaW5nIHRoZSB0d28gd2l0aCB0aGVcbiAgLy8gY29tcGxleFRlbnNvckluZm9zIGZpZWxkLlxuICBjb21wbGV4VGVuc29ySW5mb3M/OiB7cmVhbDogVGVuc29ySW5mbywgaW1hZzogVGVuc29ySW5mb31cbn07XG5cbmludGVyZmFjZSBEYXRhSWQge31cblxuZXhwb3J0IHR5cGUgV2ViR1BVS2VybmVsSW5mbyA9IHtcbiAgbmFtZTogc3RyaW5nLFxuICBxdWVyeTogUHJvbWlzZTxudW1iZXI+LFxufTtcblxuZXhwb3J0IHR5cGUgVGltZXJOb2RlID0gUmVjdXJzaXZlQXJyYXk8V2ViR1BVS2VybmVsSW5mbz58V2ViR1BVS2VybmVsSW5mbztcblxuZXhwb3J0IGludGVyZmFjZSBXZWJHUFVUaW1pbmdJbmZvIGV4dGVuZHMgVGltaW5nSW5mbyB7XG4gIHVwbG9hZFdhaXRNczogbnVtYmVyO1xuICBkb3dubG9hZFdhaXRNczogbnVtYmVyO1xufVxuXG50eXBlIFByb2dyYW1Vbmlmb3JtID0gQXJyYXk8e3R5cGU6IHN0cmluZzsgZGF0YTogbnVtYmVyW119PjtcblxuLy8gRW1waXJpY2FsbHkgZGV0ZXJtaW5lZCBjb25zdGFudCB1c2VkIHRvIGRldGVybWluZSBzaXplIHRocmVzaG9sZCBmb3IgaGFuZGluZ1xuLy8gb2ZmIGV4ZWN1dGlvbiB0byB0aGUgQ1BVLlxuY29uc3QgQ1BVX0hBTkRPRkZfU0laRV9USFJFU0hPTEQgPVxuICAgIGVudigpLmdldE51bWJlcignV0VCR1BVX0NQVV9IQU5ET0ZGX1NJWkVfVEhSRVNIT0xEJyk7XG5cbi8vIFJlc2hhcGUgZGlzcGF0Y2gsIG5vdCB0byBleGNlZWQgZGV2aWNlIGxpbWl0cy5cbmNvbnN0IHJlc2hhcGVEaXNwYXRjaCA9XG4gICAgKGRldmljZTogR1BVRGV2aWNlLFxuICAgICBwcm9ncmFtOiB3ZWJncHVfcHJvZ3JhbS5XZWJHUFVQcm9ncmFtKTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0+IHtcbiAgICAgIGNvbnN0IE1BWF9DT01QVVRFX1BFUl9ESU1FTlNJT05fRElTUEFUQ0hfU0laRSA9XG4gICAgICAgICAgZGV2aWNlLmxpbWl0cy5tYXhDb21wdXRlV29ya2dyb3Vwc1BlckRpbWVuc2lvbjtcbiAgICAgIGNvbnN0IGxheW91dCA9IHByb2dyYW1bJ2Rpc3BhdGNoTGF5b3V0J107XG4gICAgICBjb25zdCBkaXNwYXRjaCA9IHByb2dyYW1bJ2Rpc3BhdGNoJ107XG4gICAgICBpZiAoZGlzcGF0Y2guZXZlcnkoKGQpID0+IGQgPD0gTUFYX0NPTVBVVEVfUEVSX0RJTUVOU0lPTl9ESVNQQVRDSF9TSVpFKSkge1xuICAgICAgICByZXR1cm4gZGlzcGF0Y2g7XG4gICAgICB9XG5cbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGRpc3BhdGNoWzBdID4gTUFYX0NPTVBVVEVfUEVSX0RJTUVOU0lPTl9ESVNQQVRDSF9TSVpFICYmXG4gICAgICAgICAgICAgIGxheW91dC55ID09PSB1bmRlZmluZWQgJiYgbGF5b3V0LnogPT09IHVuZGVmaW5lZCxcbiAgICAgICAgICAoKSA9PiAnRGlzcGF0Y2ggc2l6ZSBleGNlZWRzIFdlYkdQVSBsaW1pdHMgaW4gWSBvciBaIGRpbWVuc2lvbi4nKTtcblxuICAgICAgbGV0IGRpc3BhdGNoQXZlcmFnZSA9IE1hdGguY2VpbChNYXRoLnNxcnQoZGlzcGF0Y2hbMF0pKTtcbiAgICAgIGlmIChkaXNwYXRjaEF2ZXJhZ2UgPiBNQVhfQ09NUFVURV9QRVJfRElNRU5TSU9OX0RJU1BBVENIX1NJWkUpIHtcbiAgICAgICAgZGlzcGF0Y2hBdmVyYWdlID0gTWF0aC5jZWlsKE1hdGguY2JydChkaXNwYXRjaFswXSkpO1xuICAgICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAgIGRpc3BhdGNoQXZlcmFnZSA8PSBNQVhfQ09NUFVURV9QRVJfRElNRU5TSU9OX0RJU1BBVENIX1NJWkUsXG4gICAgICAgICAgICAoKSA9PiAnVG90YWwgZGlzcGF0Y2ggc2l6ZSBleGNlZWRzIFdlYkdQVSBtYXhpbXVtLicpO1xuICAgICAgICByZXR1cm4gW2Rpc3BhdGNoQXZlcmFnZSwgZGlzcGF0Y2hBdmVyYWdlLCBkaXNwYXRjaEF2ZXJhZ2VdO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIFtkaXNwYXRjaEF2ZXJhZ2UsIGRpc3BhdGNoQXZlcmFnZSwgMV07XG4gICAgICB9XG4gICAgfTtcblxuZXhwb3J0IGNsYXNzIFdlYkdQVUJhY2tlbmQgZXh0ZW5kcyBLZXJuZWxCYWNrZW5kIHtcbiAgYnVmZmVyTWFuYWdlcjogQnVmZmVyTWFuYWdlcjtcbiAgYWRhcHRlckluZm86IEFkYXB0ZXJJbmZvO1xuICBkZXZpY2U6IEdQVURldmljZTtcbiAgcXVldWU6IEdQVVF1ZXVlO1xuICB0ZW5zb3JNYXA6IERhdGFTdG9yYWdlPFRlbnNvckRhdGE+O1xuICB0ZXh0dXJlTWFuYWdlcjogVGV4dHVyZU1hbmFnZXI7XG4gIHRocmVzaG9sZFRvSW5jcmVhc2VXb3JrZ3JvdXBzOiBudW1iZXI7XG5cbiAgcHJpdmF0ZSBhY3RpdmVUaW1lcnM6IFRpbWVyTm9kZVtdO1xuICBwcml2YXRlIGNvbW1hbmRFbmNvZGVyOiBHUFVDb21tYW5kRW5jb2RlcjtcbiAgcHJpdmF0ZSBjb21wdXRlUGFzc0VuY29kZXI6IEdQVUNvbXB1dGVQYXNzRW5jb2RlcjtcbiAgcHJpdmF0ZSBjb21tYW5kUXVldWVPd25lZElkcyA9IG5ldyBXZWFrU2V0PERhdGFJZD4oKTtcbiAgcHJpdmF0ZSBkaXNwYXRjaENvdW50SW5QYXNzID0gMDtcbiAgcHJpdmF0ZSBkaXNwb3NlZCA9IGZhbHNlO1xuICBwcml2YXRlIGRvd25sb2FkV2FpdE1zID0gMDtcbiAgcHJpdmF0ZSBkdW1teUNhbnZhczogSFRNTENhbnZhc0VsZW1lbnQ7XG4gIHByaXZhdGUgZHVtbXlDb250ZXh0OiBHUFVDYW52YXNDb250ZXh0O1xuICBwcml2YXRlIHRlbnNvckRhdGFQZW5kaW5nRGlzcG9zYWw6IERhdGFJZFtdID0gW107XG4gIHByaXZhdGUgc3RhdGljIG5leHREYXRhSWQgPSAwO1xuICBwcml2YXRlIHBpcGVsaW5lQ2FjaGU6XG4gICAgICB7W2tleTogc3RyaW5nXTogR1BVQ29tcHV0ZVBpcGVsaW5lfFByb21pc2U8R1BVQ29tcHV0ZVBpcGVsaW5lPn07XG4gIHByaXZhdGUgcHJvZ3JhbVRpbWVyc1N0YWNrOiBUaW1lck5vZGVbXTtcbiAgcHJpdmF0ZSBxdWVyeVJlc29sdmVCdWZmZXI6IEdQVUJ1ZmZlciA9IG51bGw7XG4gIHByaXZhdGUgcXVlcnlTZXQ6IEdQVVF1ZXJ5U2V0ID0gbnVsbDtcbiAgcHJpdmF0ZSBxdWVyeVNldENvdW50ID0gMjtcbiAgcHJpdmF0ZSBzdGFnaW5nUGVuZGluZ0Rpc3Bvc2FsOiBHUFVCdWZmZXJbXSA9IFtdO1xuICBwcml2YXRlIHN1cHBvcnRUaW1lc3RhbXBRdWVyeTogYm9vbGVhbjtcbiAgcHJpdmF0ZSB1bmlmb3JtUGVuZGluZ0Rpc3Bvc2FsOiBHUFVCdWZmZXJbXSA9IFtdO1xuICBwcml2YXRlIHVwbG9hZFdhaXRNcyA9IDA7XG4gIHByaXZhdGUgaGFzUmVhZFN5bmNXYXJuZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSBoYXNUaW1lc3RhbXBRdWVyeVdhcm5lZCA9IGZhbHNlO1xuXG4gIHByaXZhdGUgbmV4dERhdGFJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBXZWJHUFVCYWNrZW5kLm5leHREYXRhSWQrKztcbiAgfVxuXG4gIGNvbnN0cnVjdG9yKGRldmljZTogR1BVRGV2aWNlLCBhZGFwdGVySW5mbz86IEdQVUFkYXB0ZXJJbmZvKSB7XG4gICAgc3VwZXIoKTtcbiAgICBpZiAoIXdlYmdwdV91dGlsLmlzV2ViR1BVU3VwcG9ydGVkKCkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignV2ViR1BVIGlzIG5vdCBzdXBwb3J0ZWQgb24gdGhpcyBkZXZpY2UnKTtcbiAgICB9XG4gICAgdGhpcy5waXBlbGluZUNhY2hlID0ge307XG4gICAgdGhpcy5kZXZpY2UgPSBkZXZpY2U7XG4gICAgdGhpcy5xdWV1ZSA9IGRldmljZS5xdWV1ZTtcbiAgICB0aGlzLmNvbW1hbmRFbmNvZGVyID0gbnVsbDtcbiAgICB0aGlzLmNvbXB1dGVQYXNzRW5jb2RlciA9IG51bGw7XG4gICAgdGhpcy5hZGFwdGVySW5mbyA9IG5ldyBBZGFwdGVySW5mbyhhZGFwdGVySW5mbyk7XG4gICAgdGhpcy5zdXBwb3J0VGltZXN0YW1wUXVlcnkgPSB0aGlzLmRldmljZS5mZWF0dXJlcy5oYXMoJ3RpbWVzdGFtcC1xdWVyeScpO1xuICAgIHRoaXMudGhyZXNob2xkVG9JbmNyZWFzZVdvcmtncm91cHMgPVxuICAgICAgICB0aGlzLmFkYXB0ZXJJbmZvLmludGVsR1BVR2VuZXJhdGlvbiA+PSAxMiA/IDE2IDogODtcblxuICAgIHRoaXMuYnVmZmVyTWFuYWdlciA9IG5ldyBCdWZmZXJNYW5hZ2VyKHRoaXMuZGV2aWNlKTtcbiAgICB0aGlzLnRleHR1cmVNYW5hZ2VyID0gbmV3IFRleHR1cmVNYW5hZ2VyKHRoaXMuZGV2aWNlKTtcbiAgICB0aGlzLnRlbnNvck1hcCA9IG5ldyBEYXRhU3RvcmFnZSh0aGlzLCBlbmdpbmUoKSk7XG5cbiAgICAvLyBQcm9maWxpbmcgdG9vbHMgbGlrZSBQSVggbmVlZHMgdGhpcyBkdW1teSBjYW52YXMgdG9cbiAgICAvLyB0cmlnZ2VyIGNhcHR1cmluZyBhIGZyYW1lLlxuICAgIGlmIChlbnYoKS5nZXRCb29sKCdXRUJHUFVfVVNFX1BST0ZJTEVfVE9PTCcpKSB7XG4gICAgICB0aGlzLmR1bW15Q2FudmFzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgICB0aGlzLmR1bW15Q2FudmFzLndpZHRoID0gMTtcbiAgICAgIHRoaXMuZHVtbXlDYW52YXMuaGVpZ2h0ID0gMTtcblxuICAgICAgdGhpcy5kdW1teUNvbnRleHQgPSB0aGlzLmR1bW15Q2FudmFzLmdldENvbnRleHQoJ3dlYmdwdScpO1xuICAgICAgdGhpcy5kdW1teUNvbnRleHQuY29uZmlndXJlKHtcbiAgICAgICAgZGV2aWNlLFxuICAgICAgICBmb3JtYXQ6ICdiZ3JhOHVub3JtJyxcbiAgICAgIH0pO1xuXG4gICAgICBkb2N1bWVudC5ib2R5LmFwcGVuZENoaWxkKHRoaXMuZHVtbXlDYW52YXMpO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGZsb2F0UHJlY2lzaW9uKCk6IDMyIHtcbiAgICByZXR1cm4gMzI7XG4gIH1cblxuICAvKipcbiAgICogRGlzcG9zZSB0aGUgbWVtb3J5IGlmIHRoZSBkYXRhSWQgaGFzIDAgcmVmQ291bnQuIFJldHVybiB0cnVlIGlmIHRoZSBtZW1vcnlcbiAgICogaXMgcmVsZWFzZWQgb3IgZGVsYXllZCBpbiB0aGlzIGJhY2tlbmQsIGZhbHNlIGlmIHRoZXJlIGFyZSBzdGlsbFxuICAgKiByZWZlcmVuY2VzLlxuICAgKiBAcGFyYW0gZGF0YUlkXG4gICAqIEBvYXJhbSBmb3JjZSBPcHRpb25hbCwgcmVtb3ZlIHRoZSBkYXRhIHJlZ2FyZGxlc3Mgb2YgcmVmQ291bnRcbiAgICovXG4gIG92ZXJyaWRlIGRpc3Bvc2VEYXRhKGRhdGFJZDogRGF0YUlkLCBmb3JjZSA9IGZhbHNlKTogYm9vbGVhbiB7XG4gICAgLy8gTm8tb3AgaWYgYWxyZWFkeSBkaXNwb3NlZC5cbiAgICBpZiAoIXRoaXMudGVuc29yTWFwLmhhcyhkYXRhSWQpKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG5cbiAgICBjb25zdCB0ZW5zb3JEYXRhID0gdGhpcy50ZW5zb3JNYXAuZ2V0KGRhdGFJZCk7XG4gICAgaWYgKGZvcmNlKSB7XG4gICAgICB0ZW5zb3JEYXRhLnJlZkNvdW50ID0gMDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGVuc29yRGF0YS5yZWZDb3VudC0tO1xuICAgIH1cblxuICAgIGlmICh0ZW5zb3JEYXRhLnJlZkNvdW50ID4gMCkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIGlmICh0ZW5zb3JEYXRhLmNvbXBsZXhUZW5zb3JJbmZvcyAhPSBudWxsKSB7XG4gICAgICB0aGlzLmRpc3Bvc2VEYXRhKHRlbnNvckRhdGEuY29tcGxleFRlbnNvckluZm9zLnJlYWwuZGF0YUlkKTtcbiAgICAgIHRoaXMuZGlzcG9zZURhdGEodGVuc29yRGF0YS5jb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQpO1xuICAgIH1cblxuICAgIGlmICh0aGlzLmNvbW1hbmRRdWV1ZU93bmVkSWRzLmhhcyhkYXRhSWQpKSB7XG4gICAgICB0aGlzLnRlbnNvckRhdGFQZW5kaW5nRGlzcG9zYWwucHVzaChkYXRhSWQpO1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgdGhpcy5yZWxlYXNlUmVzb3VyY2UoZGF0YUlkKTtcbiAgICB0aGlzLnRlbnNvck1hcC5kZWxldGUoZGF0YUlkKTtcblxuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgb3ZlcnJpZGUgbWVtb3J5KCk6IFdlYkdQVU1lbW9yeUluZm8ge1xuICAgIHJldHVybiB7XG4gICAgICBudW1CeXRlc0luR1BVOiB0aGlzLmJ1ZmZlck1hbmFnZXIubnVtQnl0ZXNVc2VkLFxuICAgICAgbnVtQnl0ZXNBbGxvY2F0ZWRJbkdQVTogdGhpcy5idWZmZXJNYW5hZ2VyLm51bUJ5dGVzQWxsb2NhdGVkLFxuICAgICAgdW5yZWxpYWJsZTogZmFsc2VcbiAgICB9IGFzIFdlYkdQVU1lbW9yeUluZm87XG4gIH1cblxuICBwcml2YXRlIHJlbGVhc2VSZXNvdXJjZShkYXRhSWQ6IERhdGFJZCkge1xuICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRlbnNvck1hcC5nZXQoZGF0YUlkKTtcbiAgICBpZiAoIXRlbnNvckRhdGEgfHwgIXRlbnNvckRhdGEucmVzb3VyY2UpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICAvLyBJZiB0ZW5zb3IncyByZXNvdXJjZSBpcyBmcm9tIGV4dGVybmFsLCBkbyBub3QgcmVsZWFzZS5cbiAgICBpZiAodGVuc29yRGF0YS5leHRlcm5hbCkge1xuICAgICAgdGVuc29yRGF0YS5yZXNvdXJjZSA9IG51bGw7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0ZW5zb3JEYXRhLnJlc291cmNlIGluc3RhbmNlb2YgR1BVQnVmZmVyKSB7XG4gICAgICB0aGlzLmJ1ZmZlck1hbmFnZXIucmVsZWFzZUJ1ZmZlcih0ZW5zb3JEYXRhLnJlc291cmNlKTtcbiAgICB9IGVsc2UgaWYgKHRlbnNvckRhdGEucmVzb3VyY2UgaW5zdGFuY2VvZiBHUFVUZXh0dXJlKSB7XG4gICAgICB0aGlzLnRleHR1cmVNYW5hZ2VyLnJlbGVhc2VUZXh0dXJlKHRlbnNvckRhdGEucmVzb3VyY2UpO1xuICAgIH1cbiAgICB0ZW5zb3JEYXRhLnJlc291cmNlID0gbnVsbDtcbiAgfVxuXG4gIC8qKiBSZXR1cm4gcmVmQ291bnQgb2YgYSBgVGVuc29yRGF0YWAuICovXG4gIG92ZXJyaWRlIHJlZkNvdW50KGRhdGFJZDogRGF0YUlkKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy50ZW5zb3JNYXAuaGFzKGRhdGFJZCkpIHtcbiAgICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRlbnNvck1hcC5nZXQoZGF0YUlkKTtcbiAgICAgIHJldHVybiB0ZW5zb3JEYXRhLnJlZkNvdW50O1xuICAgIH1cbiAgICByZXR1cm4gMDtcbiAgfVxuXG4gIC8qKiBJbmNyZWFzZSByZWZDb3VudCBvZiBhIGBUZW5zb3JEYXRhYC4gKi9cbiAgb3ZlcnJpZGUgaW5jUmVmKGRhdGFJZDogRGF0YUlkKTogdm9pZCB7XG4gICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGVuc29yTWFwLmdldChkYXRhSWQpO1xuICAgIHRlbnNvckRhdGEucmVmQ291bnQrKztcbiAgfVxuXG4gIC8qKiBEZWNyZWFzZSByZWZDb3VudCBvZiBhIGBUZW5zb3JEYXRhYC4gKi9cbiAgZGVjUmVmKGRhdGFJZDogRGF0YUlkKTogdm9pZCB7XG4gICAgaWYgKHRoaXMudGVuc29yTWFwLmhhcyhkYXRhSWQpKSB7XG4gICAgICBjb25zdCB0ZW5zb3JEYXRhID0gdGhpcy50ZW5zb3JNYXAuZ2V0KGRhdGFJZCk7XG4gICAgICB0ZW5zb3JEYXRhLnJlZkNvdW50LS07XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgd3JpdGUodmFsdWVzOiBCYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6XG4gICAgICBEYXRhSWQge1xuICAgIGlmIChkdHlwZSA9PT0gJ2NvbXBsZXg2NCcgJiYgdmFsdWVzICE9IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQ2Fubm90IHdyaXRlIHRvIGEgY29tcGxleDY0IGR0eXBlLiBgICtcbiAgICAgICAgICBgUGxlYXNlIHVzZSB0Zi5jb21wbGV4KHJlYWwsIGltYWcpLmApO1xuICAgIH1cbiAgICBjb25zdCBkYXRhSWQgPSB7aWQ6IHRoaXMubmV4dERhdGFJZCgpfTtcbiAgICB0aGlzLnRlbnNvck1hcC5zZXQoZGF0YUlkLCB7ZHR5cGUsIHNoYXBlLCB2YWx1ZXMsIHJlZkNvdW50OiAxfSk7XG4gICAgcmV0dXJuIGRhdGFJZDtcbiAgfVxuXG4gIG92ZXJyaWRlIG1vdmUoXG4gICAgICBkYXRhSWQ6IERhdGFJZCwgdmFsdWVzOiBCYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIHJlZkNvdW50OiBudW1iZXIpOiB2b2lkIHtcbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCArXG4gICAgICAgICAgYFBsZWFzZSB1c2UgdGYuY29tcGxleChyZWFsLCBpbWFnKS5gKTtcbiAgICB9XG4gICAgdGhpcy50ZW5zb3JNYXAuc2V0KGRhdGFJZCwge2R0eXBlLCBzaGFwZSwgdmFsdWVzLCByZWZDb3VudH0pO1xuICB9XG5cbiAgc3VibWl0UXVldWUoKSB7XG4gICAgdGhpcy5xdWV1ZS5zdWJtaXQoW3RoaXMuY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICB0aGlzLmNvbW1hbmRFbmNvZGVyID0gbnVsbDtcbiAgICB0aGlzLmRpc3BhdGNoQ291bnRJblBhc3MgPSAwO1xuXG4gICAgdGhpcy5jb21tYW5kUXVldWVPd25lZElkcyA9IG5ldyBXZWFrU2V0PERhdGFJZD4oKTtcblxuICAgIHRoaXMudGVuc29yRGF0YVBlbmRpbmdEaXNwb3NhbC5mb3JFYWNoKGQgPT4ge1xuICAgICAgdGhpcy5yZWxlYXNlUmVzb3VyY2UoZCk7XG4gICAgICB0aGlzLnRlbnNvck1hcC5kZWxldGUoZCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVuaWZvcm1QZW5kaW5nRGlzcG9zYWwuZm9yRWFjaChcbiAgICAgICAgYiA9PiB0aGlzLmJ1ZmZlck1hbmFnZXIucmVsZWFzZUJ1ZmZlcihiKSk7XG4gICAgdGhpcy5zdGFnaW5nUGVuZGluZ0Rpc3Bvc2FsLmZvckVhY2goXG4gICAgICAgIGIgPT4gdGhpcy5idWZmZXJNYW5hZ2VyLnJlbGVhc2VCdWZmZXIoYiwgZmFsc2UpKTtcblxuICAgIHRoaXMudGVuc29yRGF0YVBlbmRpbmdEaXNwb3NhbCA9IFtdO1xuICAgIHRoaXMudW5pZm9ybVBlbmRpbmdEaXNwb3NhbCA9IFtdO1xuICAgIHRoaXMuc3RhZ2luZ1BlbmRpbmdEaXNwb3NhbCA9IFtdO1xuICB9XG5cbiAgZW5zdXJlQ29tbWFuZEVuY29kZXJSZWFkeSgpIHtcbiAgICBpZiAoIXRoaXMuY29tbWFuZEVuY29kZXIpIHtcbiAgICAgIHRoaXMuY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgIH1cbiAgfVxuXG4gIGVuZENvbXB1dGVQYXNzRW5jb2RlcigpIHtcbiAgICBpZiAodGhpcy5jb21wdXRlUGFzc0VuY29kZXIpIHtcbiAgICAgIHRoaXMuY29tcHV0ZVBhc3NFbmNvZGVyLmVuZCgpO1xuICAgICAgdGhpcy5jb21wdXRlUGFzc0VuY29kZXIgPSBudWxsO1xuICAgIH1cbiAgfVxuXG4gIC8vIENoZWNrIGlmIHBhcmFsbGVsIGNvbXBpbGF0aW9uIGlzIGRvbmUuXG4gIGFzeW5jIGNoZWNrQ29tcGlsZUNvbXBsZXRpb25Bc3luYygpIHtcbiAgICBsZXQgcGlwZWxpbmVzOiBHUFVDb21wdXRlUGlwZWxpbmVbXTtcbiAgICB0cnkge1xuICAgICAgcGlwZWxpbmVzID0gYXdhaXQgUHJvbWlzZS5hbGwoT2JqZWN0LnZhbHVlcyh0aGlzLnBpcGVsaW5lQ2FjaGUpKTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICAvLyBUT0RPOiBBZGQgdGVzdCBjYXNlIHRvIGNhdGNoIHRoaXMgZXhjZXB0aW9uLlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGUubWVzc2FnZSk7XG4gICAgfVxuICAgIE9iamVjdC5rZXlzKHRoaXMucGlwZWxpbmVDYWNoZSkubWFwKChrZXksIGkpID0+IHtcbiAgICAgIHRoaXMucGlwZWxpbmVDYWNoZVtrZXldID0gcGlwZWxpbmVzW2ldO1xuICAgIH0pO1xuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldEJ1ZmZlckRhdGEoYnVmZmVyOiBHUFVCdWZmZXIpOiBQcm9taXNlPEFycmF5QnVmZmVyPiB7XG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdQVV9FTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnVGhlIGRhdGEgbWF5IGJlIGludmFsaWQgc2luY2UgV0VCR1BVX0VOR0lORV9DT01QSUxFX09OTFkgaXMgdHJ1ZSwgdGhpcyBjYW4gb25seSBiZSBjYWxsZWQgd2hlbiBXRUJHUFVfRU5HSU5FX0NPTVBJTEVfT05MWSBpcyBmYWxzZScpO1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGNvbnN0IHNpemUgPSBidWZmZXIuc2l6ZTtcbiAgICBjb25zdCBzdGFnaW5nQnVmZmVyID0gdGhpcy5idWZmZXJNYW5hZ2VyLmFjcXVpcmVCdWZmZXIoXG4gICAgICAgIHNpemUsIEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNUIHwgR1BVQnVmZmVyVXNhZ2UuTUFQX1JFQUQpO1xuICAgIHRoaXMuZW5zdXJlQ29tbWFuZEVuY29kZXJSZWFkeSgpO1xuICAgIHRoaXMuZW5kQ29tcHV0ZVBhc3NFbmNvZGVyKCk7XG4gICAgdGhpcy5jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoYnVmZmVyLCAwLCBzdGFnaW5nQnVmZmVyLCAwLCBzaXplKTtcbiAgICB0aGlzLnN1Ym1pdFF1ZXVlKCk7XG5cbiAgICBhd2FpdCBzdGFnaW5nQnVmZmVyLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCk7XG4gICAgY29uc3QgdmFsdWVzID0gc3RhZ2luZ0J1ZmZlci5nZXRNYXBwZWRSYW5nZSgpLnNsaWNlKDApO1xuXG4gICAgc3RhZ2luZ0J1ZmZlci51bm1hcCgpO1xuICAgIGlmIChzdGFnaW5nQnVmZmVyICE9IG51bGwpIHtcbiAgICAgIHRoaXMuYnVmZmVyTWFuYWdlci5yZWxlYXNlQnVmZmVyKHN0YWdpbmdCdWZmZXIpO1xuICAgIH1cblxuICAgIC8vIE5lZWQgdG8gZ2V0IHRleHR1cmUgZnJvbSBzd2FwQ2hhaW4gdG8gZW5hYmxlIHByb2ZpbGluZyB0b29sXG4gICAgLy8gdG8gY2FwdHVyZSBhIGZyYW1lXG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdQVV9VU0VfUFJPRklMRV9UT09MJykpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIHRoaXMuZHVtbXlDb250ZXh0ICE9PSB1bmRlZmluZWQsXG4gICAgICAgICAgKCkgPT4gYEZhaWwgdG8gZ2V0IGNvbnRleHQgZm9yIHByb2ZpbGluZyB0b29sYCk7XG4gICAgICB0aGlzLmR1bW15Q29udGV4dC5nZXRDdXJyZW50VGV4dHVyZSgpO1xuICAgIH1cblxuICAgIHJldHVybiB2YWx1ZXM7XG4gIH1cblxuICBwcml2YXRlIGNvbnZlcnRBbmRDYWNoZU9uQ1BVKGRhdGFJZDogRGF0YUlkLCBkYXRhOiBCYWNrZW5kVmFsdWVzKTpcbiAgICAgIEJhY2tlbmRWYWx1ZXMge1xuICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRlbnNvck1hcC5nZXQoZGF0YUlkKTtcbiAgICB0ZW5zb3JEYXRhLnZhbHVlcyA9IGRhdGE7XG4gICAgcmV0dXJuIHRlbnNvckRhdGEudmFsdWVzO1xuICB9XG5cbiAgb3ZlcnJpZGUgcmVhZFN5bmMoZGF0YUlkOiBvYmplY3QpOiBCYWNrZW5kVmFsdWVzIHtcbiAgICBjb25zdCB0ZW5zb3JEYXRhID0gdGhpcy50ZW5zb3JNYXAuZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qge3ZhbHVlcywgY29tcGxleFRlbnNvckluZm9zfSA9IHRlbnNvckRhdGE7XG5cbiAgICBpZiAodmFsdWVzICE9IG51bGwgfHwgdGVuc29yRGF0YS5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHJldHVybiB2YWx1ZXM7XG4gICAgfVxuXG4gICAgaWYgKHRlbnNvckRhdGEuZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICBjb25zdCByZWFsVmFsdWVzID1cbiAgICAgICAgICB0aGlzLnJlYWRTeW5jKGNvbXBsZXhUZW5zb3JJbmZvcy5yZWFsLmRhdGFJZCkgYXMgRmxvYXQzMkFycmF5O1xuICAgICAgY29uc3QgaW1hZ1ZhbHVlcyA9XG4gICAgICAgICAgdGhpcy5yZWFkU3luYyhjb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQpIGFzIEZsb2F0MzJBcnJheTtcbiAgICAgIGNvbnN0IGNvbXBsZXhWYWxzID0gdXRpbC5jb252ZXJ0QmFja2VuZFZhbHVlc0FuZEFycmF5QnVmZmVyKFxuICAgICAgICAgIGJhY2tlbmRfdXRpbC5tZXJnZVJlYWxBbmRJbWFnQXJyYXlzKHJlYWxWYWx1ZXMsIGltYWdWYWx1ZXMpLmJ1ZmZlcixcbiAgICAgICAgICAnZmxvYXQzMicpO1xuICAgICAgdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQsIGNvbXBsZXhWYWxzKTtcbiAgICAgIHJldHVybiBjb21wbGV4VmFscztcbiAgICB9XG5cbiAgICBpZiAoIXRoaXMuaGFzUmVhZFN5bmNXYXJuZWQpIHtcbiAgICAgIHRoaXMuaGFzUmVhZFN5bmNXYXJuZWQgPSB0cnVlO1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBUaGUgcGVyZm9ybWFuY2Ugb2Ygc3luY2hyb25vdXNseSByZWFkaW5nIGRhdGEgZnJvbSBHUFUgdG8gQ1BVIGlzIGAgK1xuICAgICAgICAgIGBwb29yIG9uIHRoZSB3ZWJncHUgYmFja2VuZCwgcGxlYXNlIHVzZSBhc3luY2hyb25vdXMgQVBJcyBpbnN0ZWFkLmApO1xuICAgIH1cblxuICAgIGNvbnN0IGFscGhhTW9kZXM6IEdQVUNhbnZhc0FscGhhTW9kZVtdID0gWydvcGFxdWUnLCAncHJlbXVsdGlwbGllZCddO1xuXG4gICAgY29uc3QgYnVmZmVyID0gdGVuc29yRGF0YS5yZXNvdXJjZSBhcyBHUFVCdWZmZXI7XG4gICAgY29uc3QgYnVmZmVyU2l6ZSA9IGJ1ZmZlci5zaXplO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBidWZmZXJTaXplICUgNCA9PT0gMCxcbiAgICAgICAgKCkgPT4gJ0JlY2F1c2UgdGhlcmUgaXMgNCBieXRlcyBmb3IgJyArXG4gICAgICAgICAgICAnb25lIHBpeGVsLCBidWZmZXIgc2l6ZSBtdXN0IGJlIG11bHRpcGxlIG9mIDQuJyk7XG4gICAgY29uc3QgcGl4ZWxzU2l6ZSA9IGJ1ZmZlclNpemUgLyA0O1xuICAgIGNvbnN0IHZhbHNHUFUgPSBuZXcgQXJyYXlCdWZmZXIoYnVmZmVyU2l6ZSk7XG4gICAgLy8gVE9ETzogYWRqdXN0IHRoZSByZWFkaW5nIHdpbmRvdyBzaXplIGFjY29yZGluZyB0aGUgYGJ1ZmZlclNpemVgLlxuICAgIGNvbnN0IGNhbnZhc1dpZHRoID0gMjU2LCBjYW52YXNIZWlnaHQgPSAyNTY7XG4gICAgY29uc3Qgc3RhZ2luZ0RldmljZVN0b3JhZ2U6IE9mZnNjcmVlbkNhbnZhc1tdID1cbiAgICAgICAgYWxwaGFNb2Rlcy5tYXAoXyA9PiBuZXcgT2Zmc2NyZWVuQ2FudmFzKGNhbnZhc1dpZHRoLCBjYW52YXNIZWlnaHQpKTtcbiAgICBjb25zdCBzdGFnaW5nSG9zdFN0b3JhZ2UgPSBuZXcgT2Zmc2NyZWVuQ2FudmFzKGNhbnZhc1dpZHRoLCBjYW52YXNIZWlnaHQpO1xuXG4gICAgdGhpcy5lbmRDb21wdXRlUGFzc0VuY29kZXIoKTtcbiAgICBzdGFnaW5nRGV2aWNlU3RvcmFnZVxuICAgICAgICAubWFwKChzdG9yYWdlLCBpbmRleCkgPT4ge1xuICAgICAgICAgIGNvbnN0IGNvbnRleHQgPSBzdG9yYWdlLmdldENvbnRleHQoJ3dlYmdwdScpO1xuICAgICAgICAgIC8vIFRPRE86IHVzZSByZ2JhOHVub3JtIGZvcm1hdCB3aGVuIHRoaXMgZm9ybWF0IGlzIHN1cHBvcnRlZCBvbiBNYWMuXG4gICAgICAgICAgLy8gaHR0cHM6Ly9idWdzLmNocm9taXVtLm9yZy9wL2Nocm9taXVtL2lzc3Vlcy9kZXRhaWw/aWQ9MTI5ODYxOFxuICAgICAgICAgIGNvbnRleHQuY29uZmlndXJlKHtcbiAgICAgICAgICAgIGRldmljZTogdGhpcy5kZXZpY2UsXG4gICAgICAgICAgICBmb3JtYXQ6ICdiZ3JhOHVub3JtJyxcbiAgICAgICAgICAgIHVzYWdlOiBHUFVUZXh0dXJlVXNhZ2UuQ09QWV9EU1QsXG4gICAgICAgICAgICBhbHBoYU1vZGU6IGFscGhhTW9kZXNbaW5kZXhdLFxuICAgICAgICAgIH0pO1xuICAgICAgICAgIHJldHVybiBjb250ZXh0LmdldEN1cnJlbnRUZXh0dXJlKCk7XG4gICAgICAgIH0pXG4gICAgICAgIC5tYXAoKHRleHR1cmUsIGluZGV4KSA9PiB7XG4gICAgICAgICAgY29uc3QgYnl0ZXNQZXJSb3cgPSBjYW52YXNXaWR0aCAqIDQ7XG4gICAgICAgICAgY29uc3QgcmVhZERhdGFHUFVUb0NQVSA9XG4gICAgICAgICAgICAgICh3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlciwgb2Zmc2V0OiBudW1iZXIpID0+IHtcbiAgICAgICAgICAgICAgICB0aGlzLmVuc3VyZUNvbW1hbmRFbmNvZGVyUmVhZHkoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb1RleHR1cmUoXG4gICAgICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgICBidWZmZXIsXG4gICAgICAgICAgICAgICAgICAgICAgYnl0ZXNQZXJSb3csXG4gICAgICAgICAgICAgICAgICAgICAgb2Zmc2V0LFxuICAgICAgICAgICAgICAgICAgICB9LFxuICAgICAgICAgICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgICAgICAgdGV4dHVyZSxcbiAgICAgICAgICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICAgICAgICAge1xuICAgICAgICAgICAgICAgICAgICAgIHdpZHRoLFxuICAgICAgICAgICAgICAgICAgICAgIGhlaWdodCxcbiAgICAgICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgdGhpcy5zdWJtaXRRdWV1ZSgpO1xuXG4gICAgICAgICAgICAgICAgY29uc3QgY29udGV4dCA9IHN0YWdpbmdIb3N0U3RvcmFnZS5nZXRDb250ZXh0KCcyZCcsIHtcbiAgICAgICAgICAgICAgICAgIHdpbGxSZWFkRnJlcXVlbnRseTogdHJ1ZSxcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICBjb250ZXh0LmNsZWFyUmVjdCgwLCAwLCB3aWR0aCwgaGVpZ2h0KTtcbiAgICAgICAgICAgICAgICBjb250ZXh0LmRyYXdJbWFnZShzdGFnaW5nRGV2aWNlU3RvcmFnZVtpbmRleF0sIDAsIDApO1xuICAgICAgICAgICAgICAgIGNvbnN0IHN0YWdpbmdWYWx1ZXMgPVxuICAgICAgICAgICAgICAgICAgICBjb250ZXh0LmdldEltYWdlRGF0YSgwLCAwLCB3aWR0aCwgaGVpZ2h0KS5kYXRhO1xuICAgICAgICAgICAgICAgIGNvbnN0IGFscGhhTW9kZSA9IGFscGhhTW9kZXNbaW5kZXhdO1xuICAgICAgICAgICAgICAgIGNvbnN0IHNwYW4gPVxuICAgICAgICAgICAgICAgICAgICBuZXcgVWludDhDbGFtcGVkQXJyYXkodmFsc0dQVSwgb2Zmc2V0LCB3aWR0aCAqIGhlaWdodCAqIDQpO1xuICAgICAgICAgICAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgc3Bhbi5sZW5ndGg7IGsgKz0gNCkge1xuICAgICAgICAgICAgICAgICAgaWYgKGFscGhhTW9kZSA9PT0gJ3ByZW11bHRpcGxpZWQnKSB7XG4gICAgICAgICAgICAgICAgICAgIHNwYW5bayArIDNdID0gc3RhZ2luZ1ZhbHVlc1trICsgM107XG4gICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBjb25zdCB2YWx1ZSA9IHN0YWdpbmdWYWx1ZXNba107XG4gICAgICAgICAgICAgICAgICAgIHNwYW5ba10gPSBzdGFnaW5nVmFsdWVzW2sgKyAyXTtcbiAgICAgICAgICAgICAgICAgICAgc3BhbltrICsgMV0gPSBzdGFnaW5nVmFsdWVzW2sgKyAxXTtcbiAgICAgICAgICAgICAgICAgICAgc3BhbltrICsgMl0gPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH07XG5cbiAgICAgICAgICBjb25zdCBmdWxseVJlYWRDb3VudCA9XG4gICAgICAgICAgICAgIE1hdGguZmxvb3IocGl4ZWxzU2l6ZSAvIChjYW52YXNXaWR0aCAqIGNhbnZhc0hlaWdodCkpO1xuICAgICAgICAgIGxldCB3aWR0aCA9IGNhbnZhc1dpZHRoLCBoZWlnaHQgPSBjYW52YXNIZWlnaHQsIG9mZnNldCA9IDA7XG4gICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBmdWxseVJlYWRDb3VudDsgaSsrKSB7XG4gICAgICAgICAgICAvLyBSZWFkIHRoZSBidWZmZXIgZGF0YSwgd2hpY2ggZnVsbHkgZmlsbCB0aGUgd2hvbGUgY2FudmFzLlxuICAgICAgICAgICAgcmVhZERhdGFHUFVUb0NQVSh3aWR0aCwgaGVpZ2h0LCBvZmZzZXQpO1xuICAgICAgICAgICAgb2Zmc2V0ICs9IGNhbnZhc1dpZHRoICogY2FudmFzSGVpZ2h0ICogNDtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBjb25zdCByZW1haW5TaXplID0gcGl4ZWxzU2l6ZSAlIChjYW52YXNXaWR0aCAqIGNhbnZhc0hlaWdodCk7XG4gICAgICAgICAgaGVpZ2h0ID0gTWF0aC5mbG9vcihyZW1haW5TaXplIC8gY2FudmFzV2lkdGgpO1xuICAgICAgICAgIGlmIChoZWlnaHQgPiAwKSB7XG4gICAgICAgICAgICAvLyBSZWFkIHRoZSBidWZmZXIgZGF0YSwgd2hpY2ggZnVsbHkgZmlsbCBjZXJ0YWluIHJvd3Mgb2YgY2FudmFzLlxuICAgICAgICAgICAgcmVhZERhdGFHUFVUb0NQVSh3aWR0aCwgaGVpZ2h0LCBvZmZzZXQpO1xuICAgICAgICAgICAgb2Zmc2V0ICs9IGhlaWdodCAqIChjYW52YXNXaWR0aCAqIDQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHdpZHRoID0gcmVtYWluU2l6ZSAlIGNhbnZhc1dpZHRoO1xuICAgICAgICAgIGlmICh3aWR0aCA+IDApIHtcbiAgICAgICAgICAgIC8vIFJlYWQgdGhlIGJ1ZmZlciBkYXRhLCB3aGljaCBub3QgZnVsbHkgZmlsbCBvbmUgcm93IG9mIGNhbnZhcy5cbiAgICAgICAgICAgIHJlYWREYXRhR1BVVG9DUFUod2lkdGgsIDEsIG9mZnNldCk7XG4gICAgICAgICAgfVxuICAgICAgICB9KTtcblxuICAgIGNvbnN0IHZhbHMgPVxuICAgICAgICB1dGlsLmNvbnZlcnRCYWNrZW5kVmFsdWVzQW5kQXJyYXlCdWZmZXIodmFsc0dQVSwgdGVuc29yRGF0YS5kdHlwZSk7XG4gICAgdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQsIHZhbHMpO1xuICAgIHJldHVybiB2YWxzO1xuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgcmVhZChkYXRhSWQ6IG9iamVjdCk6IFByb21pc2U8QmFja2VuZFZhbHVlcz4ge1xuICAgIGlmICghdGhpcy50ZW5zb3JNYXAuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVGVuc29yICR7ZGF0YUlkfSB3YXMgbm90IHJlZ2lzdGVyZWQhYCk7XG4gICAgfVxuICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRlbnNvck1hcC5nZXQoZGF0YUlkKTtcblxuICAgIGNvbnN0IHt2YWx1ZXN9ID0gdGVuc29yRGF0YTtcblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHZhbHVlcztcbiAgICB9XG5cbiAgICAvLyBEb3dubG9hZCB0aGUgdmFsdWVzIGZyb20gdGhlIEdQVS5cbiAgICBsZXQgdmFsczogQmFja2VuZFZhbHVlcztcbiAgICBpZiAodGVuc29yRGF0YS5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIGNvbnN0IHBzID0gYXdhaXQgUHJvbWlzZS5hbGwoW1xuICAgICAgICB0aGlzLnJlYWQodGVuc29yRGF0YS5jb21wbGV4VGVuc29ySW5mb3MucmVhbC5kYXRhSWQpLFxuICAgICAgICB0aGlzLnJlYWQodGVuc29yRGF0YS5jb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQpXG4gICAgICBdKTtcblxuICAgICAgY29uc3QgcmVhbFZhbHVlcyA9IHBzWzBdO1xuICAgICAgY29uc3QgaW1hZ1ZhbHVlcyA9IHBzWzFdO1xuICAgICAgdmFscyA9IGJhY2tlbmRfdXRpbC5tZXJnZVJlYWxBbmRJbWFnQXJyYXlzKFxuICAgICAgICAgIHJlYWxWYWx1ZXMgYXMgRmxvYXQzMkFycmF5LCBpbWFnVmFsdWVzIGFzIEZsb2F0MzJBcnJheSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGRhdGEgPSBhd2FpdCB0aGlzLmdldEJ1ZmZlckRhdGEodGVuc29yRGF0YS5yZXNvdXJjZSBhcyBHUFVCdWZmZXIpO1xuICAgICAgdmFscyA9IHV0aWwuY29udmVydEJhY2tlbmRWYWx1ZXNBbmRBcnJheUJ1ZmZlcihkYXRhLCB0ZW5zb3JEYXRhLmR0eXBlKTtcbiAgICB9XG4gICAgdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQsIHZhbHMpO1xuICAgIHJldHVybiB2YWxzO1xuICB9XG5cbiAgLy8gVGhlIHNvdXJjZSBHUFVCdWZmZXIgYW5kIGRlc3RpbmF0aW9uIEdQVUJ1ZmZlciBoYXZlIHRoZSBzYW1lIHNpemUgYW5kXG4gIC8vIHVzYWdlLlxuICBwcml2YXRlIGNvcHlCdWZmZXIoc3JjQnVmZmVyOiBHUFVCdWZmZXIpIHtcbiAgICBjb25zdCBzaXplID0gc3JjQnVmZmVyLnNpemU7XG4gICAgY29uc3QgdXNhZ2UgPSBzcmNCdWZmZXIudXNhZ2U7XG4gICAgY29uc3QgZHN0QnVmZmVyID0gdGhpcy5idWZmZXJNYW5hZ2VyLmFjcXVpcmVCdWZmZXIoc2l6ZSwgdXNhZ2UpO1xuICAgIHRoaXMuZW5zdXJlQ29tbWFuZEVuY29kZXJSZWFkeSgpO1xuICAgIHRoaXMuZW5kQ29tcHV0ZVBhc3NFbmNvZGVyKCk7XG4gICAgdGhpcy5jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoc3JjQnVmZmVyLCAwLCBkc3RCdWZmZXIsIDAsIHNpemUpO1xuICAgIHRoaXMuc3VibWl0UXVldWUoKTtcbiAgICByZXR1cm4gZHN0QnVmZmVyO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIFRGLmpzIHRlbnNvciBvdXQgb2YgYW4gZXhpc3RpbmcgV2ViR1BVIGJ1ZmZlci5cbiAgICovXG4gIG92ZXJyaWRlIGNyZWF0ZVRlbnNvckZyb21HUFVEYXRhKFxuICAgICAgd2ViR1BVRGF0YTogV2ViR1BVRGF0YSwgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIGxldCBidWZmZXIgPSB3ZWJHUFVEYXRhLmJ1ZmZlcjtcbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGFJZCA9IHtpZDogdGhpcy5uZXh0RGF0YUlkKCl9O1xuICAgIHRoaXMudGVuc29yTWFwLnNldChkYXRhSWQsIHtcbiAgICAgIGR0eXBlLFxuICAgICAgc2hhcGUsXG4gICAgICB2YWx1ZXM6IG51bGwsXG4gICAgICByZWZDb3VudDogMSxcbiAgICAgIGV4dGVybmFsOiB3ZWJHUFVEYXRhLnplcm9Db3B5XG4gICAgfSk7XG4gICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGVuc29yTWFwLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHNpemUgPSB3ZWJncHVfdXRpbC5HUFVCeXRlc1BlckVsZW1lbnQodGVuc29yRGF0YS5kdHlwZSkgKlxuICAgICAgICB1dGlsLnNpemVGcm9tU2hhcGUodGVuc29yRGF0YS5zaGFwZSk7XG4gICAgaWYgKHdlYkdQVURhdGEuYnVmZmVyLnNpemUgPCBzaXplKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEdQVUJ1ZmZlciBzaXplKCR7XG4gICAgICAgICAgd2ViR1BVRGF0YS5idWZmZXIuc2l6ZX0pIGlzIHNtYWxsZXIgdGhhbiB0ZW5zb3Igc2l6ZSgke3NpemV9KSFgKTtcbiAgICB9IGVsc2UgaWYgKFxuICAgICAgICAod2ViR1BVRGF0YS5idWZmZXIudXNhZ2UgJlxuICAgICAgICAgKEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQykpICE9PVxuICAgICAgICAoR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICdHUFVCdWZmZXIudXNhZ2Ugc2hvdWxkIGluY2x1ZGUgR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDIScpO1xuICAgIH1cblxuICAgIC8vIERvIGJ1ZmZlciBjb3B5IGJ5IGRlZmF1bHQuXG4gICAgaWYgKHdlYkdQVURhdGEuemVyb0NvcHkgIT09IHRydWUpIHtcbiAgICAgIGJ1ZmZlciA9IHRoaXMuY29weUJ1ZmZlcihidWZmZXIpO1xuICAgIH1cbiAgICB0ZW5zb3JEYXRhLnJlc291cmNlID0gYnVmZmVyO1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbURhdGFJZChkYXRhSWQsIHNoYXBlLCBkdHlwZSwgdGhpcyk7XG4gIH1cblxuICAvKipcbiAgICogUmVhZCB0ZW5zb3IgdG8gYSBuZXcgR1BVQnVmZmVyLlxuICAgKiBAcGFyYW0gZGF0YUlkIFRoZSBzb3VyY2UgdGVuc29yLlxuICAgKi9cbiAgb3ZlcnJpZGUgcmVhZFRvR1BVKGRhdGFJZDogRGF0YUlkKTogR1BVRGF0YSB7XG4gICAgY29uc3Qgc3JjVGVuc29yRGF0YSA9IHRoaXMudGVuc29yTWFwLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHt2YWx1ZXMsIGR0eXBlLCBzaGFwZSwgcmVzb3VyY2V9ID0gc3JjVGVuc29yRGF0YTtcblxuICAgIGlmIChkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignRG9lcyBub3Qgc3VwcG9ydCByZWFkaW5nIGJ1ZmZlciBmb3IgY29tcGxleDY0IGR0eXBlLicpO1xuICAgIH1cblxuICAgIGlmIChyZXNvdXJjZSA9PSBudWxsKSB7XG4gICAgICBpZiAodmFsdWVzICE9IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdEYXRhIGlzIG5vdCBvbiBHUFUgYnV0IG9uIENQVS4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignVGhlcmUgaXMgbm8gZGF0YSBvbiBHUFUgb3IgQ1BVLicpO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IHNyY0J1ZmZlciA9IHJlc291cmNlIGFzIEdQVUJ1ZmZlcjtcbiAgICBjb25zdCBzaXplID0gc3JjQnVmZmVyLnNpemU7XG4gICAgY29uc3QgdXNhZ2UgPSBzcmNCdWZmZXIudXNhZ2U7XG4gICAgY29uc3QgYnVmZmVyID0gdGhpcy5idWZmZXJNYW5hZ2VyLmFjcXVpcmVCdWZmZXIoc2l6ZSwgdXNhZ2UpO1xuICAgIHRoaXMuZW5zdXJlQ29tbWFuZEVuY29kZXJSZWFkeSgpO1xuICAgIHRoaXMuZW5kQ29tcHV0ZVBhc3NFbmNvZGVyKCk7XG4gICAgdGhpcy5jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICAgIHJlc291cmNlIGFzIEdQVUJ1ZmZlciwgMCwgYnVmZmVyLCAwLCBzaXplKTtcbiAgICB0aGlzLnN1Ym1pdFF1ZXVlKCk7XG5cbiAgICBjb25zdCB0ZW5zb3JJbmZvID0gdGhpcy5tYWtlVGVuc29ySW5mbyhzaGFwZSwgZHR5cGUpO1xuICAgIC8vIE1ha2UgZW5naW5lIHRyYWNrIHRoaXMgdGVuc29yLCBzbyB0aGF0IHdlIGNhbiBkaXNwb3NlIGl0IGxhdGVyLlxuICAgIGNvbnN0IHRlbnNvclJlZiA9IGVuZ2luZSgpLm1ha2VUZW5zb3JGcm9tVGVuc29ySW5mbyh0ZW5zb3JJbmZvKTtcblxuICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRlbnNvck1hcC5nZXQodGVuc29ySW5mby5kYXRhSWQpO1xuICAgIHRlbnNvckRhdGEucmVzb3VyY2UgPSBidWZmZXI7XG5cbiAgICByZXR1cm4ge3RlbnNvclJlZiwgYnVmZmVyfTtcbiAgfVxuXG4gIGJ1ZmZlclN5bmM8UiBleHRlbmRzIFJhbmssIEQgZXh0ZW5kcyBEYXRhVHlwZT4odDogVGVuc29ySW5mbyk6XG4gICAgICBUZW5zb3JCdWZmZXI8UiwgRD4ge1xuICAgIGNvbnN0IGRhdGEgPSB0aGlzLnJlYWRTeW5jKHQuZGF0YUlkKTtcbiAgICBpZiAodC5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIC8vIERlY29kZSB0aGUgYnl0ZXMgaW50byBzdHJpbmcuXG4gICAgICAgIGNvbnN0IHN0cmluZ3MgPSAoZGF0YSBhcyBVaW50OEFycmF5W10pLm1hcChkID0+IHV0aWwuZGVjb2RlU3RyaW5nKGQpKTtcbiAgICAgICAgcmV0dXJuIGJ1ZmZlcih0LnNoYXBlIGFzIFNoYXBlTWFwW1JdLCB0LmR0eXBlLCBzdHJpbmdzKSBhc1xuICAgICAgICAgICAgVGVuc29yQnVmZmVyPFIsIEQ+O1xuICAgICAgfSBjYXRjaCB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignRmFpbGVkIHRvIGRlY29kZSBlbmNvZGVkIHN0cmluZyBieXRlcyBpbnRvIHV0Zi04Jyk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBidWZmZXIodC5zaGFwZSBhcyBTaGFwZU1hcFtSXSwgdC5kdHlwZSwgZGF0YSBhcyBUeXBlZEFycmF5KSBhc1xuICAgICAgICBUZW5zb3JCdWZmZXI8UiwgRD47XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyB0aW1lKGY6ICgpID0+IHZvaWQpOiBQcm9taXNlPFdlYkdQVVRpbWluZ0luZm8+IHtcbiAgICBpZiAoIXRoaXMuc3VwcG9ydFRpbWVzdGFtcFF1ZXJ5ICYmICF0aGlzLmhhc1RpbWVzdGFtcFF1ZXJ5V2FybmVkKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFRoaXMgZGV2aWNlIGRvZXNuJ3Qgc3VwcG9ydCB0aW1lc3RhbXAtcXVlcnkgZXh0ZW5zaW9uLiBgICtcbiAgICAgICAgICBgU3RhcnQgQ2hyb21lIGJyb3dzZXIgd2l0aCBmbGFnIGAgK1xuICAgICAgICAgIGAtLWVuYWJsZS1kYXduLWZlYXR1cmVzPWFsbG93X3Vuc2FmZV9hcGlzIHRvIHRyeSBpdCBhZ2Fpbi4gYCArXG4gICAgICAgICAgYE90aGVyd2lzZSwgemVybyB3aWxsIGJlIHNob3duIGZvciB0aGUga2VybmVsIHRpbWUgd2hlbiBwcm9maWxpbmcgYCArXG4gICAgICAgICAgYG1vZGUgaXMgZW5hYmxlZC5gKTtcbiAgICAgIHRoaXMuaGFzVGltZXN0YW1wUXVlcnlXYXJuZWQgPSB0cnVlO1xuICAgIH1cblxuICAgIGNvbnN0IG9sZEFjdGl2ZVRpbWVycyA9IHRoaXMuYWN0aXZlVGltZXJzO1xuICAgIGNvbnN0IG5ld0FjdGl2ZVRpbWVyczogVGltZXJOb2RlW10gPSBbXTtcblxuICAgIGxldCBvdXRlck1vc3RUaW1lID0gZmFsc2U7XG4gICAgaWYgKHRoaXMucHJvZ3JhbVRpbWVyc1N0YWNrID09IG51bGwpIHtcbiAgICAgIHRoaXMucHJvZ3JhbVRpbWVyc1N0YWNrID0gbmV3QWN0aXZlVGltZXJzO1xuICAgICAgb3V0ZXJNb3N0VGltZSA9IHRydWU7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuYWN0aXZlVGltZXJzLnB1c2gobmV3QWN0aXZlVGltZXJzKTtcbiAgICB9XG4gICAgdGhpcy5hY3RpdmVUaW1lcnMgPSBuZXdBY3RpdmVUaW1lcnM7XG5cbiAgICBmKCk7XG5cbiAgICBjb25zdCBmbGF0dGVuZWRBY3RpdmVUaW1lclF1ZXJpZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBXZWJHUFVLZXJuZWxJbmZvKSA9PiBkLnF1ZXJ5KSlcbiAgICAgICAgICAgIC5maWx0ZXIoZCA9PiBkICE9IG51bGwpO1xuICAgIGNvbnN0IGZsYXR0ZW5lZEFjdGl2ZVRpbWVyTmFtZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBXZWJHUFVLZXJuZWxJbmZvKSA9PiBkLm5hbWUpKVxuICAgICAgICAgICAgLmZpbHRlcihkID0+IGQgIT0gbnVsbCk7XG5cbiAgICB0aGlzLmFjdGl2ZVRpbWVycyA9IG9sZEFjdGl2ZVRpbWVycztcblxuICAgIGlmIChvdXRlck1vc3RUaW1lKSB7XG4gICAgICB0aGlzLnByb2dyYW1UaW1lcnNTdGFjayA9IG51bGw7XG4gICAgfVxuICAgIGNvbnN0IHJlczogV2ViR1BVVGltaW5nSW5mbyA9IHtcbiAgICAgIHVwbG9hZFdhaXRNczogdGhpcy51cGxvYWRXYWl0TXMsXG4gICAgICBkb3dubG9hZFdhaXRNczogdGhpcy5kb3dubG9hZFdhaXRNcyxcbiAgICAgIGtlcm5lbE1zOiBudWxsLFxuICAgICAgd2FsbE1zOiBudWxsXG4gICAgfTtcblxuICAgIGNvbnN0IGtlcm5lbE1zID0gYXdhaXQgUHJvbWlzZS5hbGwoZmxhdHRlbmVkQWN0aXZlVGltZXJRdWVyaWVzKTtcbiAgICByZXNbJ2tlcm5lbE1zJ10gPSB1dGlsLnN1bShrZXJuZWxNcyk7XG4gICAgcmVzWydnZXRFeHRyYVByb2ZpbGVJbmZvJ10gPSAoKSA9PlxuICAgICAgICBrZXJuZWxNcy5tYXAoKGQsIGkpID0+ICh7bmFtZTogZmxhdHRlbmVkQWN0aXZlVGltZXJOYW1lc1tpXSwgbXM6IGR9KSlcbiAgICAgICAgICAgIC5tYXAoZCA9PiBgJHtkLm5hbWV9OiAke2QubXN9YClcbiAgICAgICAgICAgIC5qb2luKCcsICcpO1xuICAgIHRoaXMudXBsb2FkV2FpdE1zID0gMDtcbiAgICB0aGlzLmRvd25sb2FkV2FpdE1zID0gMDtcbiAgICByZXR1cm4gcmVzO1xuICB9XG5cbiAgbWFrZVRlbnNvckluZm8oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIHZhbHVlcz86IEJhY2tlbmRWYWx1ZXN8c3RyaW5nW10pOiBUZW5zb3JJbmZvIHtcbiAgICBpZiAoZHR5cGUgPT09ICdzdHJpbmcnICYmIHZhbHVlcyAhPSBudWxsICYmIHZhbHVlcy5sZW5ndGggPiAwICYmXG4gICAgICAgIHV0aWwuaXNTdHJpbmcodmFsdWVzWzBdKSkge1xuICAgICAgdmFsdWVzID0gKHZhbHVlcyBhcyB1bmtub3duIGFzIHN0cmluZ1tdKS5tYXAoZCA9PiB1dGlsLmVuY29kZVN0cmluZyhkKSk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGFJZCA9IHRoaXMud3JpdGUodmFsdWVzIGFzIEJhY2tlbmRWYWx1ZXMsIHNoYXBlLCBkdHlwZSk7XG4gICAgcmV0dXJuIHtkYXRhSWQsIHNoYXBlLCBkdHlwZX07XG4gIH1cblxuICBwcml2YXRlIHRlbnNvclRvQmluZGluZyh0ZW5zb3I/OiBUZW5zb3JJbmZvKTogR1BVQmluZGluZ1Jlc291cmNlIHtcbiAgICBpZiAoIXRlbnNvcikge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuXG4gICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGVuc29yTWFwLmdldCh0ZW5zb3IuZGF0YUlkKTtcbiAgICBjb25zdCByZXNvdXJjZSA9IHRlbnNvckRhdGEucmVzb3VyY2U7XG5cbiAgICBpZiAocmVzb3VyY2UgaW5zdGFuY2VvZiBHUFVCdWZmZXIpIHtcbiAgICAgIHJldHVybiB7YnVmZmVyOiByZXNvdXJjZX07XG4gICAgfVxuICAgIGlmIChyZXNvdXJjZSBpbnN0YW5jZW9mIEdQVVRleHR1cmUpIHtcbiAgICAgIHJldHVybiByZXNvdXJjZS5jcmVhdGVWaWV3KCk7XG4gICAgfVxuICAgIC8vIEdQVUV4dGVybmFsVGV4dHVyZVxuICAgIHJldHVybiByZXNvdXJjZTtcbiAgfVxuXG4gIHVwbG9hZFRvR1BVKGRhdGFJZDogRGF0YUlkKTogdm9pZCB7XG4gICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGVuc29yTWFwLmdldChkYXRhSWQpO1xuICAgIC8vIEFscmVhZHkgb24gdGhlIEdQVS5cbiAgICBpZiAodGVuc29yRGF0YS5yZXNvdXJjZSAhPSBudWxsKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgc2l6ZSA9IHdlYmdwdV91dGlsLkdQVUJ5dGVzUGVyRWxlbWVudCh0ZW5zb3JEYXRhLmR0eXBlKSAqXG4gICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZSh0ZW5zb3JEYXRhLnNoYXBlKTtcbiAgICBsZXQgYnVmZmVyO1xuICAgIGNvbnN0IHVzYWdlID0gR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDIHxcbiAgICAgICAgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1Q7XG4gICAgaWYgKHRlbnNvckRhdGEudmFsdWVzKSB7XG4gICAgICBidWZmZXIgPSB0aGlzLmJ1ZmZlck1hbmFnZXIuYWNxdWlyZUJ1ZmZlcihzaXplLCB1c2FnZSwgdHJ1ZSk7XG4gICAgICBpZiAoYnVmZmVyLm1hcFN0YXRlID09PSAndW5tYXBwZWQnKSB7XG4gICAgICAgIGNvbnN0IHN0YWdpbmdCdWZmZXIgPSB0aGlzLmJ1ZmZlck1hbmFnZXIuYWNxdWlyZUJ1ZmZlcihcbiAgICAgICAgICAgIHNpemUsIEdQVUJ1ZmZlclVzYWdlLk1BUF9XUklURSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDLCB0cnVlLFxuICAgICAgICAgICAgZmFsc2UpO1xuICAgICAgICBjb25zdCBhcnJheUJ1ZmZlciA9IHN0YWdpbmdCdWZmZXIuZ2V0TWFwcGVkUmFuZ2UoKTtcbiAgICAgICAgaWYgKHRlbnNvckRhdGEuZHR5cGUgPT09ICdpbnQzMicgfHwgdGVuc29yRGF0YS5kdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgICAgICAgbmV3IEludDMyQXJyYXkoYXJyYXlCdWZmZXIpLnNldCh0ZW5zb3JEYXRhLnZhbHVlcyBhcyBUeXBlZEFycmF5KTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBuZXcgRmxvYXQzMkFycmF5KGFycmF5QnVmZmVyKS5zZXQodGVuc29yRGF0YS52YWx1ZXMgYXMgRmxvYXQzMkFycmF5KTtcbiAgICAgICAgfVxuICAgICAgICBzdGFnaW5nQnVmZmVyLnVubWFwKCk7XG4gICAgICAgIHRoaXMuZW5zdXJlQ29tbWFuZEVuY29kZXJSZWFkeSgpO1xuICAgICAgICB0aGlzLmVuZENvbXB1dGVQYXNzRW5jb2RlcigpO1xuICAgICAgICB0aGlzLmNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgICAgICAgIHN0YWdpbmdCdWZmZXIsIDAsIGJ1ZmZlciwgMCwgc2l6ZSk7XG5cbiAgICAgICAgdGhpcy5zdGFnaW5nUGVuZGluZ0Rpc3Bvc2FsLnB1c2goc3RhZ2luZ0J1ZmZlcik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBhcnJheUJ1ZmZlciA9IGJ1ZmZlci5nZXRNYXBwZWRSYW5nZSgpO1xuICAgICAgICBpZiAodGVuc29yRGF0YS5kdHlwZSA9PT0gJ2ludDMyJyB8fCB0ZW5zb3JEYXRhLmR0eXBlID09PSAnYm9vbCcpIHtcbiAgICAgICAgICBuZXcgSW50MzJBcnJheShhcnJheUJ1ZmZlcikuc2V0KHRlbnNvckRhdGEudmFsdWVzIGFzIFR5cGVkQXJyYXkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIG5ldyBGbG9hdDMyQXJyYXkoYXJyYXlCdWZmZXIpLnNldCh0ZW5zb3JEYXRhLnZhbHVlcyBhcyBGbG9hdDMyQXJyYXkpO1xuICAgICAgICB9XG4gICAgICAgIGJ1ZmZlci51bm1hcCgpO1xuICAgICAgfVxuXG4gICAgICAvLyBPbmNlIHVwbG9hZGVkLCBkb24ndCBzdG9yZSB0aGUgdmFsdWVzIG9uIGNwdS5cbiAgICAgIHRlbnNvckRhdGEudmFsdWVzID0gbnVsbDtcbiAgICB9IGVsc2Uge1xuICAgICAgYnVmZmVyID0gdGhpcy5idWZmZXJNYW5hZ2VyLmFjcXVpcmVCdWZmZXIoc2l6ZSwgdXNhZ2UpO1xuICAgIH1cbiAgICB0ZW5zb3JEYXRhLnJlc291cmNlID0gYnVmZmVyO1xuICB9XG5cbiAgcHJpdmF0ZSBtYWtlVW5pZm9ybXMocHJvZ3JhbVVuaWZvcm06IFByb2dyYW1Vbmlmb3JtKTogR1BVQmluZGluZ1Jlc291cmNlIHtcbiAgICBsZXQgY3VycmVudE9mZnNldCA9IDA7XG4gICAgbGV0IHByZUxlbmd0aCA9IDA7XG4gICAgY29uc3Qgb2Zmc2V0czogbnVtYmVyW10gPSBbXTtcbiAgICBsZXQgbWF4QWxpZ25tZW50T2ZGaWVsZCA9IDE7XG4gICAgcHJvZ3JhbVVuaWZvcm0uZm9yRWFjaCgoZCkgPT4ge1xuICAgICAgaWYgKGQuZGF0YS5sZW5ndGggPT09IDApIHtcbiAgICAgICAgZC5kYXRhID0gWzFdO1xuICAgICAgfVxuICAgICAgLy8gaHR0cHM6Ly93d3cudzMub3JnL1RSL1dHU0wvI2FsaWdub2ZcbiAgICAgIGxldCBiYXNlQWxpZ25tZW50OiBudW1iZXI7XG4gICAgICBzd2l0Y2ggKGQuZGF0YS5sZW5ndGgpIHtcbiAgICAgICAgY2FzZSAxOlxuICAgICAgICAgIGJhc2VBbGlnbm1lbnQgPSA0O1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICBjYXNlIDI6XG4gICAgICAgICAgYmFzZUFsaWdubWVudCA9IDg7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgMzpcbiAgICAgICAgICBiYXNlQWxpZ25tZW50ID0gMTY7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgNDpcbiAgICAgICAgICBiYXNlQWxpZ25tZW50ID0gMTY7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgNTpcbiAgICAgICAgICBiYXNlQWxpZ25tZW50ID0gMTY7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgNjpcbiAgICAgICAgICBiYXNlQWxpZ25tZW50ID0gMTY7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGRlZmF1bHQ6XG4gICAgICAgICAgdXRpbC5hc3NlcnQoZmFsc2UsICgpID0+IGBVbnN1cHBvcnRlZCAke2QuZGF0YS5sZW5ndGh9RCBzaGFwZWApO1xuICAgICAgfVxuXG4gICAgICBpZiAocHJlTGVuZ3RoID09PSA1IHx8IHByZUxlbmd0aCA9PT0gNikge1xuICAgICAgICBiYXNlQWxpZ25tZW50ID0gMTY7XG4gICAgICB9XG4gICAgICBpZiAoYmFzZUFsaWdubWVudCA+IG1heEFsaWdubWVudE9mRmllbGQpIHtcbiAgICAgICAgbWF4QWxpZ25tZW50T2ZGaWVsZCA9IGJhc2VBbGlnbm1lbnQ7XG4gICAgICB9XG4gICAgICBjdXJyZW50T2Zmc2V0ID0gTWF0aC5jZWlsKGN1cnJlbnRPZmZzZXQgLyBiYXNlQWxpZ25tZW50KSAqIGJhc2VBbGlnbm1lbnQ7XG4gICAgICBwcmVMZW5ndGggPSBkLmRhdGEubGVuZ3RoO1xuICAgICAgb2Zmc2V0cy5wdXNoKGN1cnJlbnRPZmZzZXQpO1xuICAgICAgY3VycmVudE9mZnNldCArPSBkLmRhdGEubGVuZ3RoICogNDtcbiAgICB9KTtcblxuICAgIGN1cnJlbnRPZmZzZXQgPVxuICAgICAgICBNYXRoLmNlaWwoY3VycmVudE9mZnNldCAvIG1heEFsaWdubWVudE9mRmllbGQpICogbWF4QWxpZ25tZW50T2ZGaWVsZDtcbiAgICBjb25zdCBhcnJheUJ1ZmZlciA9IG5ldyBBcnJheUJ1ZmZlcihjdXJyZW50T2Zmc2V0KTtcbiAgICBwcm9ncmFtVW5pZm9ybS5mb3JFYWNoKChkLCBpKSA9PiB7XG4gICAgICBjb25zdCBvZmZzZXQgPSBvZmZzZXRzW2ldO1xuICAgICAgaWYgKGQudHlwZSA9PT0gJ2ludDMyJykge1xuICAgICAgICBuZXcgSW50MzJBcnJheShhcnJheUJ1ZmZlciwgb2Zmc2V0LCBkLmRhdGEubGVuZ3RoKS5zZXQoZC5kYXRhKTtcbiAgICAgIH0gZWxzZSBpZiAoZC50eXBlID09PSAndWludDMyJykge1xuICAgICAgICBuZXcgVWludDMyQXJyYXkoYXJyYXlCdWZmZXIsIG9mZnNldCwgZC5kYXRhLmxlbmd0aCkuc2V0KGQuZGF0YSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBuZXcgRmxvYXQzMkFycmF5KGFycmF5QnVmZmVyLCBvZmZzZXQsIGQuZGF0YS5sZW5ndGgpLnNldChkLmRhdGEpO1xuICAgICAgfVxuICAgIH0pO1xuXG4gICAgY29uc3QgdW5pZm9ybUJ1ZmZlciA9IHRoaXMuYnVmZmVyTWFuYWdlci5hY3F1aXJlQnVmZmVyKFxuICAgICAgICBjdXJyZW50T2Zmc2V0LCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCB8IEdQVUJ1ZmZlclVzYWdlLlVOSUZPUk0pO1xuICAgIHRoaXMucXVldWUud3JpdGVCdWZmZXIodW5pZm9ybUJ1ZmZlciwgMCwgYXJyYXlCdWZmZXIsIDAsIGN1cnJlbnRPZmZzZXQpO1xuICAgIHRoaXMudW5pZm9ybVBlbmRpbmdEaXNwb3NhbC5wdXNoKHVuaWZvcm1CdWZmZXIpO1xuXG4gICAgcmV0dXJuIHtvZmZzZXQ6IDAsIHNpemU6IGN1cnJlbnRPZmZzZXQsIGJ1ZmZlcjogdW5pZm9ybUJ1ZmZlcn07XG4gIH1cblxuICBwdWJsaWMgcnVuV2ViR1BVUHJvZ3JhbShcbiAgICAgIHByb2dyYW06IHdlYmdwdV9wcm9ncmFtLldlYkdQVVByb2dyYW0sIGlucHV0czogVGVuc29ySW5mb1tdLFxuICAgICAgb3V0cHV0RHR5cGU6IERhdGFUeXBlLCBwcm9ncmFtRGVmaW5lZFVuaWZvcm0/OiBQcm9ncmFtVW5pZm9ybSxcbiAgICAgIG91dHB1dD86IFRlbnNvckluZm8pOiBUZW5zb3JJbmZvIHtcbiAgICBpZiAoIW91dHB1dCkge1xuICAgICAgb3V0cHV0ID0gdGhpcy5tYWtlVGVuc29ySW5mbyhwcm9ncmFtLm91dHB1dFNoYXBlLCBvdXRwdXREdHlwZSk7XG4gICAgfVxuICAgIGlmICh1dGlsLnNpemVGcm9tU2hhcGUob3V0cHV0LnNoYXBlKSA9PT0gMCkge1xuICAgICAgLy8gU2hvcnQtY2lyY3VpdCB0aGUgY29tcHV0YXRpb24gc2luY2UgdGhlIHJlc3VsdCBpcyBlbXB0eSAoaGFzIDAgaW4gaXRzXG4gICAgICAvLyBzaGFwZSkuXG4gICAgICB0aGlzLnRlbnNvck1hcC5nZXQob3V0cHV0LmRhdGFJZCkudmFsdWVzID1cbiAgICAgICAgICB1dGlsLmdldFR5cGVkQXJyYXlGcm9tRFR5cGUob3V0cHV0LmR0eXBlIGFzICdmbG9hdDMyJywgMCk7XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH1cbiAgICB0aGlzLnVwbG9hZFRvR1BVKG91dHB1dC5kYXRhSWQpO1xuICAgIHByb2dyYW0uZGlzcGF0Y2ggPSByZXNoYXBlRGlzcGF0Y2godGhpcy5kZXZpY2UsIHByb2dyYW0pO1xuXG4gICAgY29uc3QgaW5wdXRzRGF0YSA9IGlucHV0cy5tYXAoKGlucHV0OiBUZW5zb3JJbmZvLCBpOiBudW1iZXIpID0+IHtcbiAgICAgIGlmIChpbnB1dC5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgYEdQR1BVUHJvZ3JhbSBkb2VzIG5vdCBzdXBwb3J0IGNvbXBsZXg2NCBpbnB1dC4gRm9yIGNvbXBsZXg2NCBgICtcbiAgICAgICAgICAgIGBkdHlwZXMsIHBsZWFzZSBzZXBhcmF0ZSB0aGUgcHJvZ3JhbSBpbnRvIHJlYWwgYW5kIGltYWdpbmFyeSBgICtcbiAgICAgICAgICAgIGBwYXJ0cy5gKTtcbiAgICAgIH1cbiAgICAgIHRoaXMudXBsb2FkVG9HUFUoaW5wdXQuZGF0YUlkKTtcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgLy8gUmV0dXJuaW5nIGR0eXBlIGZyb20gdGVuc29yTWFwIGJlY2F1c2UgaXQgcmVmbGVjdHMgZHR5cGVcbiAgICAgICAgLy8gb2YgdW5kZXJseWluZyBidWZmZXIsIHJhdGhlciB0aGFuIGFic3RyYWN0IGR0eXBlLlxuICAgICAgICBkdHlwZTogdGhpcy50ZW5zb3JNYXAuZ2V0KGlucHV0LmRhdGFJZCkuZHR5cGUsXG4gICAgICAgIHNoYXBlOiBpbnB1dC5zaGFwZSxcbiAgICAgICAgbmFtZTogcHJvZ3JhbS52YXJpYWJsZU5hbWVzW2ldXG4gICAgICB9O1xuICAgIH0pO1xuXG4gICAgcHJvZ3JhbS5zaGFkZXJLZXkgPVxuICAgICAgICB3ZWJncHVfcHJvZ3JhbS5tYWtlU2hhZGVyS2V5KHByb2dyYW0sIGlucHV0c0RhdGEsIG91dHB1dCk7XG5cbiAgICBjb25zdCBwYXJhbGxlbENvbXBpbGF0aW9uID0gZW52KCkuZ2V0Qm9vbCgnV0VCR1BVX0VOR0lORV9DT01QSUxFX09OTFknKTtcbiAgICBpZiAoIShwcm9ncmFtLnNoYWRlcktleSBpbiB0aGlzLnBpcGVsaW5lQ2FjaGUpKSB7XG4gICAgICB0aGlzLnBpcGVsaW5lQ2FjaGVbcHJvZ3JhbS5zaGFkZXJLZXldID0gd2ViZ3B1X3Byb2dyYW0uY29tcGlsZVByb2dyYW0oXG4gICAgICAgICAgdGhpcy5kZXZpY2UsIHByb2dyYW0sIGlucHV0c0RhdGEsIG91dHB1dCwgcGFyYWxsZWxDb21waWxhdGlvbik7XG4gICAgfVxuICAgIHByb2dyYW0ucGlwZWxpbmUgPSB0aGlzLnBpcGVsaW5lQ2FjaGVbcHJvZ3JhbS5zaGFkZXJLZXldO1xuXG4gICAgaWYgKCFwYXJhbGxlbENvbXBpbGF0aW9uKSB7XG4gICAgICB0aGlzLnJlY29yZEFuZFN1Ym1pdChwcm9ncmFtLCBvdXRwdXQsIGlucHV0cywgcHJvZ3JhbURlZmluZWRVbmlmb3JtKTtcbiAgICB9XG4gICAgcmV0dXJuIG91dHB1dDtcbiAgfVxuXG4gIHByaXZhdGUgcmVjb3JkQW5kU3VibWl0KFxuICAgICAgcHJvZ3JhbTogd2ViZ3B1X3Byb2dyYW0uV2ViR1BVUHJvZ3JhbSwgb3V0cHV0OiBUZW5zb3JJbmZvLFxuICAgICAgaW5wdXRzOiBUZW5zb3JJbmZvW10sIHByb2dyYW1EZWZpbmVkVW5pZm9ybT86IFByb2dyYW1Vbmlmb3JtKSB7XG4gICAgaWYgKHByb2dyYW0ucGlwZWxpbmUgaW5zdGFuY2VvZiBQcm9taXNlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ1BsZWFzZSBjYWxsIGNoZWNrQ29tcGlsZUNvbXBsZXRpb25Bc3luYyB0byBlbnN1cmUgcGFyYWxsZWwgY29tcGlsYXRpb24gaXMgZG9uZSEnKTtcbiAgICB9XG4gICAgLy8gVGhlcmUgYXJlIHNpeCBraW5kcyBvZiB1bmlmb3JtczogTkFOLCBJTkZJTklUWSwgc2hhcGVzLCBzaGFwZSBzdHJpZGVzLFxuICAgIC8vIHByb2dyYW0gc2l6ZSwgcHJvZ3JhbSBkZWZpbmVkIHVuaWZvcm1zLlxuICAgIGxldCBwcm9ncmFtVW5pZm9ybTogUHJvZ3JhbVVuaWZvcm0gPSBbXTtcbiAgICBsZXQgYnVmZmVyU2hhcGVzOiBudW1iZXJbXVtdID0gW107XG4gICAgY29uc3QgdW5pZm9ybXNUeXBlID0gJ2ludDMyJztcbiAgICBpZiAocHJvZ3JhbS5waXhlbHNPcFR5cGUgPT0gbnVsbCkge1xuICAgICAgcHJvZ3JhbVVuaWZvcm0ucHVzaChcbiAgICAgICAgICB7dHlwZTogJ2Zsb2F0MzInLCBkYXRhOiBbTmFOXX0sIHt0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFtJbmZpbml0eV19KTtcbiAgICAgIGJ1ZmZlclNoYXBlcyA9IGlucHV0cy5jb25jYXQob3V0cHV0KS5tYXAoZCA9PiBkLnNoYXBlKTtcbiAgICAgIGNvbnN0IHVuaWZvcm1zVHlwZSA9ICdpbnQzMic7XG4gICAgICBidWZmZXJTaGFwZXMubWFwKGQgPT4ge1xuICAgICAgICBwcm9ncmFtVW5pZm9ybS5wdXNoKHt0eXBlOiB1bmlmb3Jtc1R5cGUsIGRhdGE6IGR9KTtcbiAgICAgICAgY29uc3Qgc3RyaWRlcyA9IHV0aWwuY29tcHV0ZVN0cmlkZXMoZCk7XG4gICAgICAgIHByb2dyYW1Vbmlmb3JtLnB1c2goe3R5cGU6IHVuaWZvcm1zVHlwZSwgZGF0YTogc3RyaWRlc30pO1xuICAgICAgfSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IHN0cmlkZXMgPSB1dGlsLmNvbXB1dGVTdHJpZGVzKG91dHB1dC5zaGFwZSk7XG4gICAgICBwcm9ncmFtVW5pZm9ybS5wdXNoKHt0eXBlOiB1bmlmb3Jtc1R5cGUsIGRhdGE6IHN0cmlkZXN9KTtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0uc2l6ZSkge1xuICAgICAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShwcm9ncmFtLm91dHB1dFNoYXBlKTtcbiAgICAgIHByb2dyYW1Vbmlmb3JtLnB1c2goe1xuICAgICAgICB0eXBlOiB1bmlmb3Jtc1R5cGUsXG4gICAgICAgIGRhdGE6IFtwcm9ncmFtLm91dHB1dENvbXBvbmVudCA/IHNpemUgLyBwcm9ncmFtLm91dHB1dENvbXBvbmVudCA6IHNpemVdXG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAocHJvZ3JhbURlZmluZWRVbmlmb3JtKSB7XG4gICAgICBwcm9ncmFtVW5pZm9ybSA9IFsuLi5wcm9ncmFtVW5pZm9ybSwgLi4ucHJvZ3JhbURlZmluZWRVbmlmb3JtXTtcbiAgICB9XG4gICAgY29uc3QgYmluZGluZ3MgPSBbXG4gICAgICB0aGlzLnRlbnNvclRvQmluZGluZyhvdXRwdXQpLCAuLi5pbnB1dHMubWFwKHQgPT4gdGhpcy50ZW5zb3JUb0JpbmRpbmcodCkpLFxuICAgICAgdGhpcy5tYWtlVW5pZm9ybXMocHJvZ3JhbVVuaWZvcm0pXG4gICAgXTtcblxuICAgIGlucHV0cy5mb3JFYWNoKGlucHV0ID0+IHtcbiAgICAgIHRoaXMuY29tbWFuZFF1ZXVlT3duZWRJZHMuYWRkKGlucHV0LmRhdGFJZCk7XG4gICAgfSk7XG4gICAgdGhpcy5jb21tYW5kUXVldWVPd25lZElkcy5hZGQob3V0cHV0LmRhdGFJZCk7XG5cbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiBwcm9ncmFtLnBpcGVsaW5lLmdldEJpbmRHcm91cExheW91dCgwKSxcbiAgICAgIGVudHJpZXM6IGJpbmRpbmdzLm1hcCgoYiwgaSkgPT4gKHtiaW5kaW5nOiBpLCByZXNvdXJjZTogYn0pKSxcbiAgICB9KTtcblxuICAgIGNvbnN0IHNob3VsZFRpbWVQcm9ncmFtID0gdGhpcy5hY3RpdmVUaW1lcnMgIT0gbnVsbDtcbiAgICB0aGlzLmVuc3VyZUNvbW1hbmRFbmNvZGVyUmVhZHkoKTtcblxuICAgIGNvbnN0IGNvbXB1dGVQYXNzRGVzY3JpcHRvcjogR1BVQ29tcHV0ZVBhc3NEZXNjcmlwdG9yID0ge307XG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtICYmIHRoaXMuc3VwcG9ydFRpbWVzdGFtcFF1ZXJ5KSB7XG4gICAgICB0aGlzLmVuZENvbXB1dGVQYXNzRW5jb2RlcigpO1xuICAgICAgaWYgKHRoaXMucXVlcnlTZXQgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnF1ZXJ5U2V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlUXVlcnlTZXQoe1xuICAgICAgICAgIHR5cGU6ICd0aW1lc3RhbXAnLFxuICAgICAgICAgIGNvdW50OiB0aGlzLnF1ZXJ5U2V0Q291bnQsXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgICAgY29tcHV0ZVBhc3NEZXNjcmlwdG9yLnRpbWVzdGFtcFdyaXRlcyA9IHtcbiAgICAgICAgcXVlcnlTZXQ6IHRoaXMucXVlcnlTZXQsXG4gICAgICAgIGJlZ2lubmluZ09mUGFzc1dyaXRlSW5kZXg6IDAsXG4gICAgICAgIGVuZE9mUGFzc1dyaXRlSW5kZXg6IDEsXG4gICAgICB9O1xuICAgICAgdGhpcy5jb21wdXRlUGFzc0VuY29kZXIgPVxuICAgICAgICAgIHRoaXMuY29tbWFuZEVuY29kZXIuYmVnaW5Db21wdXRlUGFzcyhjb21wdXRlUGFzc0Rlc2NyaXB0b3IpO1xuICAgIH0gZWxzZSBpZiAoIXRoaXMuY29tcHV0ZVBhc3NFbmNvZGVyKSB7XG4gICAgICB0aGlzLmNvbXB1dGVQYXNzRW5jb2RlciA9XG4gICAgICAgICAgdGhpcy5jb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKGNvbXB1dGVQYXNzRGVzY3JpcHRvcik7XG4gICAgfVxuXG4gICAgdGhpcy5jb21wdXRlUGFzc0VuY29kZXIuc2V0UGlwZWxpbmUocHJvZ3JhbS5waXBlbGluZSk7XG4gICAgdGhpcy5jb21wdXRlUGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgdGhpcy5jb21wdXRlUGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKFxuICAgICAgICBwcm9ncmFtLmRpc3BhdGNoWzBdLCBwcm9ncmFtLmRpc3BhdGNoWzFdLCBwcm9ncmFtLmRpc3BhdGNoWzJdKTtcbiAgICB0aGlzLmRpc3BhdGNoQ291bnRJblBhc3MrKztcblxuICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSB8fFxuICAgICAgICBlbnYoKS5nZXQoJ1dFQkdQVV9ERUZFUlJFRF9TVUJNSVRfQkFUQ0hfU0laRScpIGFzXG4gICAgICAgICAgICBudW1iZXIgPD0gdGhpcy5kaXNwYXRjaENvdW50SW5QYXNzIHx8XG4gICAgICAgIHByb2dyYW0ucGl4ZWxzT3BUeXBlID09PSB3ZWJncHVfcHJvZ3JhbS5QaXhlbHNPcFR5cGUuRFJBVykge1xuICAgICAgdGhpcy5lbmRDb21wdXRlUGFzc0VuY29kZXIoKTtcbiAgICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgICB0aGlzLmFjdGl2ZVRpbWVycy5wdXNoKFxuICAgICAgICAgICAge25hbWU6IHByb2dyYW0uY29uc3RydWN0b3IubmFtZSwgcXVlcnk6IHRoaXMuZ2V0UXVlcnlUaW1lKCl9KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMuc3VibWl0UXVldWUoKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBhc3luYyBnZXRRdWVyeVRpbWUoKTogUHJvbWlzZTxudW1iZXI+IHtcbiAgICBpZiAoIXRoaXMuc3VwcG9ydFRpbWVzdGFtcFF1ZXJ5KSB7XG4gICAgICByZXR1cm4gMDtcbiAgICB9XG5cbiAgICBpZiAodGhpcy5xdWVyeVJlc29sdmVCdWZmZXIgPT0gbnVsbCkge1xuICAgICAgdGhpcy5xdWVyeVJlc29sdmVCdWZmZXIgPSB0aGlzLmJ1ZmZlck1hbmFnZXIuYWNxdWlyZUJ1ZmZlcihcbiAgICAgICAgICB0aGlzLnF1ZXJ5U2V0Q291bnQgKiA4LFxuICAgICAgICAgIEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QgfFxuICAgICAgICAgICAgICBHUFVCdWZmZXJVc2FnZS5RVUVSWV9SRVNPTFZFKTtcbiAgICB9XG4gICAgdGhpcy5jb21tYW5kRW5jb2Rlci5yZXNvbHZlUXVlcnlTZXQoXG4gICAgICAgIHRoaXMucXVlcnlTZXQsIDAsIHRoaXMucXVlcnlTZXRDb3VudCwgdGhpcy5xdWVyeVJlc29sdmVCdWZmZXIsIDApO1xuXG4gICAgY29uc3QgcXVlcnlTdGFnaW5nQnVmZmVyID0gdGhpcy5idWZmZXJNYW5hZ2VyLmFjcXVpcmVCdWZmZXIoXG4gICAgICAgIHRoaXMucXVlcnlTZXRDb3VudCAqIDgsXG4gICAgICAgIEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QpO1xuXG4gICAgdGhpcy5jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICAgIHRoaXMucXVlcnlSZXNvbHZlQnVmZmVyLCAwLCBxdWVyeVN0YWdpbmdCdWZmZXIsIDAsXG4gICAgICAgIHRoaXMucXVlcnlTZXRDb3VudCAqIDgpO1xuXG4gICAgdGhpcy5zdWJtaXRRdWV1ZSgpO1xuXG4gICAgYXdhaXQgcXVlcnlTdGFnaW5nQnVmZmVyLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCk7XG4gICAgY29uc3QgYXJyYXlCdWZmZXIgPSBuZXcgQmlnVWludDY0QXJyYXkocXVlcnlTdGFnaW5nQnVmZmVyLmdldE1hcHBlZFJhbmdlKCkpO1xuICAgIGNvbnN0IHRpbWUgPSBOdW1iZXIoYXJyYXlCdWZmZXJbMV0gLSBhcnJheUJ1ZmZlclswXSkgLyAxMDAwMDAwO1xuICAgIHF1ZXJ5U3RhZ2luZ0J1ZmZlci51bm1hcCgpO1xuICAgIHRoaXMuYnVmZmVyTWFuYWdlci5yZWxlYXNlQnVmZmVyKHF1ZXJ5U3RhZ2luZ0J1ZmZlcik7XG4gICAgcmV0dXJuIHRpbWU7XG4gIH1cblxuICBzaG91bGRFeGVjdXRlT25DUFUoXG4gICAgICBpbnB1dHM6IFRlbnNvckluZm9bXSxcbiAgICAgIHNpemVUaHJlc2hvbGQgPSBDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiBlbnYoKS5nZXRCb29sKCdXRUJHUFVfQ1BVX0ZPUldBUkQnKSAmJlxuICAgICAgICBpbnB1dHMuZXZlcnkoXG4gICAgICAgICAgICBpbnB1dCA9PiB0aGlzLnRlbnNvck1hcC5nZXQoaW5wdXQuZGF0YUlkKS5yZXNvdXJjZSA9PSBudWxsICYmXG4gICAgICAgICAgICAgICAgdXRpbC5zaXplRnJvbVNoYXBlKGlucHV0LnNoYXBlKSA8IHNpemVUaHJlc2hvbGQpO1xuICB9XG5cbiAgb3ZlcnJpZGUgbnVtRGF0YUlkcygpIHtcbiAgICByZXR1cm4gdGhpcy50ZW5zb3JNYXAubnVtRGF0YUlkcygpIC0gdGhpcy50ZW5zb3JEYXRhUGVuZGluZ0Rpc3Bvc2FsLmxlbmd0aDtcbiAgfVxuXG4gIG92ZXJyaWRlIGRpc3Bvc2UoKSB7XG4gICAgaWYgKHRoaXMuZGlzcG9zZWQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHRoaXMucXVlcnlTZXQgIT0gbnVsbCkge1xuICAgICAgdGhpcy5xdWVyeVNldC5kZXN0cm95KCk7XG4gICAgfVxuICAgIHRoaXMuYnVmZmVyTWFuYWdlci5kaXNwb3NlKCk7XG4gICAgdGhpcy50ZXh0dXJlTWFuYWdlci5kaXNwb3NlKCk7XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cbn1cbiJdfQ==