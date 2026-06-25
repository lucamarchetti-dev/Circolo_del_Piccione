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
import { atomicAddSnippet } from './shader_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class Dilation2DBackpropInputProgram {
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
       ${main('index')} {
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
export class Dilation2DBackpropFilterProgram {
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
       ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGlsYXRpb25fYmFja3Byb3Bfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvZGlsYXRpb25fYmFja3Byb3Bfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUMvQyxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLDhCQUE4QjtJQVl6QyxZQUFZLFFBQWlDLEVBQUUsV0FBcUI7UUFQcEUsa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDakMsYUFBUSxHQUNKLGdHQUFnRyxDQUFDO1FBQ3JHLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxXQUFNLEdBQUcsSUFBSSxDQUFDO1FBSVosSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDO1FBQ3BDLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDO1FBQzVELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLFFBQVEsQ0FBQyxRQUFRLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRWhFLElBQUksV0FBVyxLQUFLLFNBQVMsSUFBSSxXQUFXLEtBQUssT0FBTyxFQUFFO1lBQ3hELE1BQU0sSUFBSSxLQUFLLENBQUM7b0NBQ2MsV0FBVyxRQUFRLENBQUMsQ0FBQztTQUNwRDtRQUNELElBQUksQ0FBQyxJQUFJLEdBQUcsV0FBVyxDQUFDO1FBQ3hCLElBQUksQ0FBQyxTQUFTLEdBQUcseUJBQXlCLENBQUM7SUFDN0MsQ0FBQztJQUVELFdBQVc7UUFDVCw4REFBOEQ7UUFDOUQsc0dBQXNHO1FBQ3RHLE1BQU0sUUFBUSxHQUFHO1NBQ1osSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O2FBd0NkLGdCQUFnQixDQUNaLHNCQUFzQixFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsSUFBMkIsQ0FBQzs7O01BR3hFLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0Y7QUFFRCxNQUFNLE9BQU8sK0JBQStCO0lBWTFDLFlBQ0ksUUFBaUMsRUFBRSxLQUFlLEVBQ2xELFdBQXFCO1FBVHpCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ2pDLGFBQVEsR0FDSixnR0FBZ0csQ0FBQztRQUNyRyxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsV0FBTSxHQUFHLElBQUksQ0FBQztRQU1aLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztRQUN4QyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUM1RCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxRQUFRLENBQUMsUUFBUSxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUVoRSxJQUFJLFdBQVcsS0FBSyxTQUFTLElBQUksV0FBVyxLQUFLLE9BQU8sRUFBRTtZQUN4RCxNQUFNLElBQUksS0FBSyxDQUFDO29DQUNjLFdBQVcsUUFBUSxDQUFDLENBQUM7U0FDcEQ7UUFDRCxJQUFJLENBQUMsSUFBSSxHQUFHLFdBQVcsQ0FBQztRQUN4QixJQUFJLENBQUMsU0FBUyxHQUFHLDBCQUEwQixDQUFDO0lBQzlDLENBQUM7SUFFRCxXQUFXO1FBQ1QsOERBQThEO1FBQzlELHNHQUFzRztRQUN0RyxNQUFNLFFBQVEsR0FBRztTQUNaLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O2FBdUNkLGdCQUFnQixDQUNaLHNCQUFzQixFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsSUFBMkIsQ0FBQzs7O01BR3hFLENBQUM7UUFDSCxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBEYXRhVHlwZX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHthdG9taWNBZGRTbmlwcGV0fSBmcm9tICcuL3NoYWRlcl91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgRGlsYXRpb24yREJhY2twcm9wSW5wdXRQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ3cnLCAnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgJ2ZpbHRlckRpbXM6IHZlYzI8aTMyPiwgcGFkczogdmVjMjxpMzI+LCBzdHJpZGVzOiB2ZWMyPGkzMj4sIGRpbGF0aW9uczogdmVjMjxpMzI+LCBkeVNpemU6IGkzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBhdG9taWMgPSB0cnVlO1xuICB0eXBlOiBEYXRhVHlwZTtcblxuICBjb25zdHJ1Y3Rvcihjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8sIG91dHB1dER0eXBlOiBEYXRhVHlwZSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5pblNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQoY29udkluZm8ub3V0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIGNvbnZJbmZvLm91dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgaWYgKG91dHB1dER0eXBlICE9PSAnZmxvYXQzMicgJiYgb3V0cHV0RHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgRGlsYXRpb24yREJhY2twcm9wSW5wdXQgb25seSBzdXBwb3J0cyBmbG9hdDMyIGFuZCBpbnQzMlxuICAgICAgICAgIHR5cGVzLCBkb2VzIG5vdCBzdXBwb3J0ICR7b3V0cHV0RHR5cGV9IHR5cGUuYCk7XG4gICAgfVxuICAgIHRoaXMudHlwZSA9IG91dHB1dER0eXBlO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ2RpbGF0aW9uMkRCYWNrcHJvcElucHV0JztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgLy8gVGhpcyBpbXBsZW1lbnRhdGlvbiBmb2xsb3dzIHRoZSBURiBjKysgY3VkYSBpbXBsZW1lbnRhdGlvbjpcbiAgICAvLyBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZW5zb3JmbG93L2Jsb2IvbWFzdGVyL3RlbnNvcmZsb3cvY29yZS9rZXJuZWxzL2RpbGF0aW9uX29wc19ncHUuY3UuY2NcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLmR5U2l6ZSkge1xuICAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0RHlDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgICBsZXQgYiA9IGNvb3Jkc1swXTtcbiAgICAgICAgICAgbGV0IHIgPSBjb29yZHNbMV07XG4gICAgICAgICAgIGxldCBjID0gY29vcmRzWzJdO1xuICAgICAgICAgICBsZXQgZCA9IGNvb3Jkc1szXTtcblxuICAgICAgICAgICBsZXQgZHlDb3JuZXIgPSB2ZWMyPGkzMj4ociwgYykgKiB1bmlmb3Jtcy5zdHJpZGVzIC0gdW5pZm9ybXMucGFkcztcbiAgICAgICAgICAgdmFyIGN1clZhbCA9IC0zLjRlMzg7ICAvLyBuZWdfaW5maW5pdHlcbiAgICAgICAgICAgdmFyIHhSTWF4ID0gMDtcbiAgICAgICAgICAgdmFyIHhDTWF4ID0gMDtcblxuICAgICAgICAgICAvLyBJbiB0aGUgY2FzZSBvZiBtdWx0aXBsZSBhcmdtYXggYnJhbmNoZXMsIHdlIG9ubHkgYmFjay1wcm9wYWdhdGVcbiAgICAgICAgICAgLy8gYWxvbmcgdGhlIGxhc3QgYnJhbmNoLCBpLmUuLCB0aGUgb25lIHdpdGggbGFyZ2VzdCB2YWx1ZSBvZlxuICAgICAgICAgICAvLyAnd1IgKiB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdICsgd0MnLCBzaW1pbGFybHkgdG8gdGhlIG1heC1wb29saW5nXG4gICAgICAgICAgIC8vIGJhY2t3YXJkIHJvdXRpbmVzLlxuICAgICAgICAgICBmb3IgKHZhciB3UiA9IDA7IHdSIDwgdW5pZm9ybXMuZmlsdGVyRGltc1swXTsgd1IrKykge1xuICAgICAgICAgICAgIGxldCB4UiA9IGR5Q29ybmVyLnggKyB3UiAqIHVuaWZvcm1zLmRpbGF0aW9uc1swXTtcblxuICAgICAgICAgICAgIGlmICh4UiA+PSAwICYmIHhSIDwgdW5pZm9ybXMueFNoYXBlWzFdKSB7XG4gICAgICAgICAgICAgICBmb3IgKHZhciB3QyA9IDA7IHdDIDwgdW5pZm9ybXMuZmlsdGVyRGltc1sxXTsgd0MrKykge1xuICAgICAgICAgICAgICAgICBsZXQgeEMgPSBkeUNvcm5lci55ICsgd0MgKiB1bmlmb3Jtcy5kaWxhdGlvbnNbMV07XG5cbiAgICAgICAgICAgICAgICAgaWYgKHhDID49IDAgJiYgeEMgPCB1bmlmb3Jtcy54U2hhcGVbMl0pIHtcbiAgICAgICAgICAgICAgICAgICBsZXQgdmFsID0gZ2V0WChiLCB4UiwgeEMsIGQpICsgZ2V0Vyh3Uiwgd0MsIGQpO1xuICAgICAgICAgICAgICAgICAgIGlmICh2YWwgPiBjdXJWYWwpIHtcbiAgICAgICAgICAgICAgICAgICAgIGN1clZhbCA9IHZhbDtcbiAgICAgICAgICAgICAgICAgICAgIHhSTWF4ID0geFI7XG4gICAgICAgICAgICAgICAgICAgICB4Q01heCA9IHhDO1xuICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgIH1cbiAgICAgICAgICAgfVxuXG4gICAgICAgICAgIGxldCBmbGF0SW5kZXhJbiA9IGQgKyB1bmlmb3Jtcy54U2hhcGVbM10gKlxuICAgICAgICAgICAgICAgKHhDTWF4ICsgdW5pZm9ybXMueFNoYXBlWzJdICogKHhSTWF4ICsgdW5pZm9ybXMueFNoYXBlWzFdICogYikpO1xuICAgICAgICAgICBsZXQgdmFsdWUgPSBnZXREeShiLCByLCBjLCBkKTtcbiAgICAgICAgICAgJHtcbiAgICAgICAgYXRvbWljQWRkU25pcHBldChcbiAgICAgICAgICAgICcmcmVzdWx0W2ZsYXRJbmRleEluXScsICd2YWx1ZScsIHRoaXMudHlwZSBhcyAnZmxvYXQzMicgfCAnaW50MzInKX1cbiAgICAgICAgIH1cbiAgICAgICB9XG4gICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBEaWxhdGlvbjJEQmFja3Byb3BGaWx0ZXJQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ3cnLCAnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgJ2ZpbHRlckRpbXM6IHZlYzI8aTMyPiwgcGFkczogdmVjMjxpMzI+LCBzdHJpZGVzOiB2ZWMyPGkzMj4sIGRpbGF0aW9uczogdmVjMjxpMzI+LCBkeVNpemU6IGkzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBhdG9taWMgPSB0cnVlO1xuICB0eXBlOiBEYXRhVHlwZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbywgc2hhcGU6IG51bWJlcltdLFxuICAgICAgb3V0cHV0RHR5cGU6IERhdGFUeXBlKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZJbmZvLmZpbHRlclNoYXBlO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQoY29udkluZm8ub3V0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIGNvbnZJbmZvLm91dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgaWYgKG91dHB1dER0eXBlICE9PSAnZmxvYXQzMicgJiYgb3V0cHV0RHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgRGlsYXRpb24yREJhY2twcm9wRmlsdGVyIG9ubHkgc3VwcG9ydHMgZmxvYXQzMiBhbmQgaW50MzJcbiAgICAgICAgICB0eXBlcywgZG9lcyBub3Qgc3VwcG9ydCAke291dHB1dER0eXBlfSB0eXBlLmApO1xuICAgIH1cbiAgICB0aGlzLnR5cGUgPSBvdXRwdXREdHlwZTtcbiAgICB0aGlzLnNoYWRlcktleSA9ICdkaWxhdGlvbjJEQmFja3Byb3BGaWx0ZXInO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICAvLyBUaGlzIGltcGxlbWVudGF0aW9uIGZvbGxvd3MgdGhlIFRGIGMrKyBjdWRhIGltcGxlbWVudGF0aW9uOlxuICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RlbnNvcmZsb3cvYmxvYi9tYXN0ZXIvdGVuc29yZmxvdy9jb3JlL2tlcm5lbHMvZGlsYXRpb25fb3BzX2dwdS5jdS5jY1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuZHlTaXplKSB7XG4gICAgICAgICAgIGxldCBjb29yZHMgPSBnZXREeUNvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgIGxldCBiID0gY29vcmRzWzBdO1xuICAgICAgICAgICBsZXQgciA9IGNvb3Jkc1sxXTtcbiAgICAgICAgICAgbGV0IGMgPSBjb29yZHNbMl07XG4gICAgICAgICAgIGxldCBkID0gY29vcmRzWzNdO1xuXG4gICAgICAgICAgIGxldCBkeUNvcm5lciA9IHZlYzI8aTMyPihyLCBjKSAqIHVuaWZvcm1zLnN0cmlkZXMgLSB1bmlmb3Jtcy5wYWRzO1xuICAgICAgICAgICB2YXIgY3VyVmFsID0gLTMuNGUzODsgIC8vIG5lZ19pbmZpbml0eVxuICAgICAgICAgICB2YXIgd1JNYXggPSAwO1xuICAgICAgICAgICB2YXIgd0NNYXggPSAwO1xuXG4gICAgICAgICAgIC8vIEluIHRoZSBjYXNlIG9mIG11bHRpcGxlIGFyZ21heCBicmFuY2hlcywgd2Ugb25seSBiYWNrLXByb3BhZ2F0ZVxuICAgICAgICAgICAvLyBhbG9uZyB0aGUgbGFzdCBicmFuY2gsIGkuZS4sIHRoZSBvbmUgd2l0aCBsYXJnZXN0IHZhbHVlIG9mXG4gICAgICAgICAgIC8vICd3UiAqIHVuaWZvcm1zLmZpbHRlckRpbXNbMV0gKyB3QycsIHNpbWlsYXJseSB0byB0aGUgbWF4LXBvb2xpbmdcbiAgICAgICAgICAgLy8gYmFja3dhcmQgcm91dGluZXMuXG4gICAgICAgICAgIGZvciAodmFyIHdSID0gMDsgd1IgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyB3UisrKSB7XG4gICAgICAgICAgICAgbGV0IHhSID0gZHlDb3JuZXIueCArIHdSICogdW5pZm9ybXMuZGlsYXRpb25zWzBdO1xuXG4gICAgICAgICAgICAgaWYgKHhSID49IDAgJiYgeFIgPCB1bmlmb3Jtcy54U2hhcGVbMV0pIHtcbiAgICAgICAgICAgICAgIGZvciAodmFyIHdDID0gMDsgd0MgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzFdOyB3QysrKSB7XG4gICAgICAgICAgICAgICAgIGxldCB4QyA9IGR5Q29ybmVyLnkgKyB3QyAqIHVuaWZvcm1zLmRpbGF0aW9uc1sxXTtcblxuICAgICAgICAgICAgICAgICBpZiAoeEMgPj0gMCAmJiB4QyA8IHVuaWZvcm1zLnhTaGFwZVsyXSkge1xuICAgICAgICAgICAgICAgICAgIGxldCB2YWwgPSBnZXRYKGIsIHhSLCB4QywgZCkgKyBnZXRXKHdSLCB3QywgZCk7XG4gICAgICAgICAgICAgICAgICAgaWYgKHZhbCA+IGN1clZhbCkge1xuICAgICAgICAgICAgICAgICAgICAgY3VyVmFsID0gdmFsO1xuICAgICAgICAgICAgICAgICAgICAgd1JNYXggPSB3UjtcbiAgICAgICAgICAgICAgICAgICAgIHdDTWF4ID0gd0M7XG4gICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgfVxuICAgICAgICAgICB9XG5cbiAgICAgICAgICAgbGV0IGZsYXRJbmRleEluID0gZCArIHVuaWZvcm1zLndTaGFwZVsyXSAqICh3Q01heCArIHdSTWF4ICogdW5pZm9ybXMud1NoYXBlWzFdKTtcbiAgICAgICAgICAgbGV0IHZhbHVlID0gZ2V0RHkoYiwgciwgYywgZCk7XG4gICAgICAgICAgICR7XG4gICAgICAgIGF0b21pY0FkZFNuaXBwZXQoXG4gICAgICAgICAgICAnJnJlc3VsdFtmbGF0SW5kZXhJbl0nLCAndmFsdWUnLCB0aGlzLnR5cGUgYXMgJ2Zsb2F0MzInIHwgJ2ludDMyJyl9XG4gICAgICAgICB9XG4gICAgICAgfVxuICAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19