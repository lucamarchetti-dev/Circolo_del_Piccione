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
export class UnsortedSegmentSumProgram {
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
    ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidW5zb3J0ZWRfc2VnbWVudF9zdW1fd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvdW5zb3J0ZWRfc2VnbWVudF9zdW1fd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUMvQyxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLHlCQUF5QjtJQVdwQyxZQUFZLE9BQWlCLEVBQUUsUUFBa0IsRUFBRSxXQUFxQjtRQVZ4RSxnQkFBVyxHQUFhLEVBQUUsQ0FBQztRQUkzQixrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLFlBQVksQ0FBQyxDQUFDO1FBQ3BDLGFBQVEsR0FBRyxnQ0FBZ0MsQ0FBQztRQUM1QyxrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsV0FBTSxHQUFHLElBQUksQ0FBQztRQUlaLElBQUksQ0FBQyxXQUFXLEdBQUcsUUFBUSxDQUFDO1FBQzVCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDbEQsSUFBSSxDQUFDLFFBQVE7WUFDVCxlQUFlLENBQUMsSUFBSSxDQUFDLGNBQWMsRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQ3RFLElBQUksV0FBVyxLQUFLLFNBQVMsSUFBSSxXQUFXLEtBQUssT0FBTyxFQUFFO1lBQ3hELE1BQU0sSUFBSSxLQUFLLENBQUM7d0NBQ2tCLFdBQVcsUUFBUSxDQUFDLENBQUM7U0FDeEQ7UUFDRCxJQUFJLENBQUMsSUFBSSxHQUFHLFdBQVcsQ0FBQztRQUN4QixJQUFJLENBQUMsU0FBUyxHQUFHLG9CQUFvQixDQUFDO0lBQ3hDLENBQUM7SUFFRCxXQUFXO1FBQ1QsTUFBTSxRQUFRLEdBQUc7TUFDZixJQUFJLENBQUMsT0FBTyxDQUFDOzs7Ozs7Ozs7OztZQVlYLGdCQUFnQixDQUNaLG9CQUFvQixFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsSUFBMkIsQ0FBQzs7OztHQUl6RSxDQUFDO1FBQ0EsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0RhdGFUeXBlfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2F0b21pY0FkZFNuaXBwZXR9IGZyb20gJy4vc2hhZGVyX3V0aWwnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBVbnNvcnRlZFNlZ21lbnRTdW1Qcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnc2VnbWVudElkcyddO1xuICB1bmlmb3JtcyA9ICdudW1TZWdtZW50cyA6IGkzMiwgeFNpemU6IGkzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBhdG9taWMgPSB0cnVlO1xuICB0eXBlOiBEYXRhVHlwZTtcblxuICBjb25zdHJ1Y3RvcihpblNoYXBlOiBudW1iZXJbXSwgb3V0U2hhcGU6IG51bWJlcltdLCBvdXRwdXREdHlwZTogRGF0YVR5cGUpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dChpblNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID1cbiAgICAgICAgY29tcHV0ZURpc3BhdGNoKHRoaXMuZGlzcGF0Y2hMYXlvdXQsIGluU2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgaWYgKG91dHB1dER0eXBlICE9PSAnZmxvYXQzMicgJiYgb3V0cHV0RHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zb3J0ZWRTZWdtZW50U3VtIG9ubHkgc3VwcG9ydHMgZmxvYXQzMiBhbmQgaW50MzJcbiAgICAgICAgICAgICAgdHlwZXMsIGRvZXMgbm90IHN1cHBvcnQgJHtvdXRwdXREdHlwZX0gdHlwZS5gKTtcbiAgICB9XG4gICAgdGhpcy50eXBlID0gb3V0cHV0RHR5cGU7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAndW5zb3J0ZWRTZWdtZW50U3VtJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy54U2l6ZSkge1xuICAgICAgICBsZXQgY29vcmRzID0gZ2V0WENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgIGxldCBiID0gY29vcmRzWzBdO1xuICAgICAgICBsZXQgaW5Db2wgPSBjb29yZHNbMV07XG5cbiAgICAgICAgbGV0IHNlZ21lbnRJZCA9IGkzMihnZXRTZWdtZW50SWRzKGluQ29sKSk7XG4gICAgICAgIGlmIChzZWdtZW50SWQgPj0gMCkge1xuICAgICAgICAgIGxldCBmbGF0SW5kZXggPSBiICogdW5pZm9ybXMubnVtU2VnbWVudHMgKyBzZWdtZW50SWQgJSB1bmlmb3Jtcy5udW1TZWdtZW50cztcbiAgICAgICAgICBsZXQgdmFsdWUgPSBnZXRYKGIsIGluQ29sKTtcblxuICAgICAgICAgICR7XG4gICAgICAgIGF0b21pY0FkZFNuaXBwZXQoXG4gICAgICAgICAgICAnJnJlc3VsdFtmbGF0SW5kZXhdJywgJ3ZhbHVlJywgdGhpcy50eXBlIGFzICdmbG9hdDMyJyB8ICdpbnQzMicpfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19