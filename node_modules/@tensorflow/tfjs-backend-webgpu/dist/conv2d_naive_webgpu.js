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
import { activationFnSnippet, biasActivationSnippet } from './activation_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch } from './webgpu_util';
export class Conv2DNaiveProgram {
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
       ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkX25haXZlX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2NvbnYyZF9uYWl2ZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsT0FBTyxFQUFDLG1CQUFtQixFQUFFLHFCQUFxQixFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDN0UsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRTlDLE1BQU0sT0FBTyxrQkFBa0I7SUFjN0IsWUFDSSxRQUFpQyxFQUFFLE9BQU8sR0FBRyxLQUFLLEVBQ2xELGFBQXNDLElBQUksRUFDMUMseUJBQXlCLEdBQUcsS0FBSztRQVpyQyxrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBQzNCLGFBQVEsR0FDSixtRkFBbUYsQ0FBQztRQUN4RixrQkFBYSxHQUE2QixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFVbEQsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDO1FBQ3JDLElBQUksQ0FBQyxjQUFjLEdBQUcsUUFBUSxDQUFDLFVBQVUsS0FBSyxjQUFjLENBQUM7UUFDN0QsSUFBSSxDQUFDLGNBQWMsR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBQyxDQUFDLENBQUM7WUFDN0IsRUFBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUMsQ0FBQztRQUN4RSxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsT0FBTyxHQUFHLE9BQU8sQ0FBQztRQUN2QixJQUFJLENBQUMsVUFBVSxHQUFHLFVBQVUsQ0FBQztRQUM3QixJQUFJLENBQUMseUJBQXlCLEdBQUcseUJBQXlCLENBQUM7UUFFM0QsSUFBSSxPQUFPLEVBQUU7WUFDWCxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztTQUNqQztRQUVELElBQUkseUJBQXlCLEVBQUU7WUFDN0IsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQztTQUNuRDtRQUVELElBQUksQ0FBQyxTQUFTLEdBQUcsZUFBZSxJQUFJLENBQUMsVUFBVSxJQUFJLElBQUksQ0FBQyxjQUFjLEVBQUUsQ0FBQztJQUMzRSxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1NBRWIsbUJBQW1CLENBQ2YsSUFBSSxDQUFDLFVBQVUsRUFBRSxJQUFJLENBQUMseUJBQXlCLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7O3dCQW1COUQsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsbUNBQW1DLENBQUMsQ0FBQztZQUNyQyxtQ0FBbUM7OzthQUdwRCxxQkFBcUIsQ0FBQyxJQUFJLENBQUMsT0FBTyxFQUFFLElBQUksQ0FBQyxVQUFVLENBQUM7Ozs7U0FJeEQsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7OzRCQUdNLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsWUFBWTt3QkFDckQsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxZQUFZO3dCQUNqRCxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLFlBQVk7Ozs7OztpREFPakUsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMscUJBQXFCLENBQUMsQ0FBQztZQUN2QixxQkFBcUI7aUJBRTNDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLCtDQUErQyxDQUFDLENBQUM7WUFDakQsK0NBQStDOzs7Ozs7OztNQVF2RSxDQUFDO1FBQ0gsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHthY3RpdmF0aW9uRm5TbmlwcGV0LCBiaWFzQWN0aXZhdGlvblNuaXBwZXR9IGZyb20gJy4vYWN0aXZhdGlvbl91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNofSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIENvbnYyRE5haXZlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdLCB5OiBudW1iZXJbXSwgejogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ1cnXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgJ2ZpbHRlckRpbXM6IHZlYzI8aTMyPiwgcGFkczogdmVjMjxpMzI+LCBzdHJpZGVzOiB2ZWMyPGkzMj4sIGRpbGF0aW9uczogdmVjMjxpMzI+LCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs0LCA0LCA4XTtcbiAgYWRkQmlhczogYm9vbGVhbjtcbiAgYWN0aXZhdGlvbjogYmFja2VuZF91dGlsLkFjdGl2YXRpb247XG4gIGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHM6IGJvb2xlYW47XG4gIGlzQ2hhbm5lbHNMYXN0OiBib29sZWFuO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvLCBhZGRCaWFzID0gZmFsc2UsXG4gICAgICBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbiA9IG51bGwsXG4gICAgICBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gZmFsc2UpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gY29udkluZm8ub3V0U2hhcGU7XG4gICAgdGhpcy5pc0NoYW5uZWxzTGFzdCA9IGNvbnZJbmZvLmRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0xhc3QnO1xuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSB0aGlzLmlzQ2hhbm5lbHNMYXN0ID8ge3g6IFsyXSwgeTogWzFdLCB6OiBbMCwgM119IDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHt4OiBbM10sIHk6IFsyXSwgejogWzAsIDFdfTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuYWRkQmlhcyA9IGFkZEJpYXM7XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gYWN0aXZhdGlvbjtcbiAgICB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzO1xuXG4gICAgaWYgKGFkZEJpYXMpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdiaWFzJyk7XG4gICAgfVxuXG4gICAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdwcmVsdUFjdGl2YXRpb25XZWlnaHRzJyk7XG4gICAgfVxuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgY29udjJkbmFpdmVfJHt0aGlzLmFjdGl2YXRpb259XyR7dGhpcy5pc0NoYW5uZWxzTGFzdH1gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAke1xuICAgICAgICBhY3RpdmF0aW9uRm5TbmlwcGV0KFxuICAgICAgICAgICAgdGhpcy5hY3RpdmF0aW9uLCB0aGlzLmhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMsIGZhbHNlLCA0KX1cbiAgICAgICBmbiByZWFkSW5wKGJhdGNoIDogaTMyLCByb3cgOiBpMzIsIGNvbCA6IGkzMiwgY2hhbiA6IGkzMikgLT4gZjMye1xuICAgICAgICAgbGV0IGNvb3JkcyA9IHZlYzQ8aTMyPihiYXRjaCwgcm93LCBjb2wsIGNoYW4pO1xuICAgICAgICAgaWYgKGNvb3Jkc0luQm91bmRzNEQoY29vcmRzLCB1bmlmb3Jtcy54U2hhcGUpKSB7XG4gICAgICAgICAgIHJldHVybiAgZ2V0WChiYXRjaCwgcm93LCBjb2wsIGNoYW4pO1xuICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICByZXR1cm4gMC4wO1xuICAgICAgICAgfVxuICAgICAgIH1cbiAgICAgICBmbiByZWFkRmlsdChyb3cgOiBpMzIsIGNvbCA6IGkzMiwgeENoYW5uZWwgOiBpMzIsIG91dENoYW5uZWwgOiBpMzIpIC0+IGYzMntcbiAgICAgICAgIGxldCBjb29yZHMgPSB2ZWM0PGkzMj4ocm93LCBjb2wsIHhDaGFubmVsLCBvdXRDaGFubmVsKTtcbiAgICAgICAgIGlmKGNvb3Jkc0luQm91bmRzNEQoY29vcmRzLCB1bmlmb3Jtcy53U2hhcGUpKSB7XG4gICAgICAgICAgIHJldHVybiBnZXRXKHJvdywgY29sLCB4Q2hhbm5lbCwgb3V0Q2hhbm5lbCk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiAwLjA7XG4gICAgICAgICAgfVxuICAgICAgIH1cbiAgICAgICBmbiB3cml0ZVJlc3VsdChiYXRjaCA6IGkzMiwgcm93IDogaTMyLCBjb2wgOiBpMzIsIGNoYW4gOiBpMzIsIHZhbHVlSW4gOiBmMzIpIHtcbiAgICAgICAgIGxldCBjb29yZHMgPSAke1xuICAgICAgICB0aGlzLmlzQ2hhbm5lbHNMYXN0ID8gYHZlYzQ8aTMyPihiYXRjaCwgcm93LCBjb2wsIGNoYW4pO2AgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYHZlYzQ8aTMyPihiYXRjaCwgY2hhbiwgcm93LCBjb2wpO2B9XG4gICAgICAgICBpZiAoY29vcmRzSW5Cb3VuZHM0RChjb29yZHMsIHVuaWZvcm1zLm91dFNoYXBlKSkge1xuICAgICAgICAgICB2YXIgdmFsdWUgPSB2YWx1ZUluO1xuICAgICAgICAgICAke2JpYXNBY3RpdmF0aW9uU25pcHBldCh0aGlzLmFkZEJpYXMsIHRoaXMuYWN0aXZhdGlvbil9XG4gICAgICAgICAgIHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkcy54LCBjb29yZHMueSwgY29vcmRzLnosIGNvb3Jkcy53LCB2YWx1ZSk7XG4gICAgICAgICB9XG4gICAgICAgfVxuICAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICAgbGV0IGJhdGNoID0gY29vcmRzWzBdO1xuICAgICAgICAgbGV0IG91dENoYW5uZWwgPSAke3RoaXMuaXNDaGFubmVsc0xhc3QgPyBgY29vcmRzWzNdO2AgOiBgY29vcmRzWzFdO2B9XG4gICAgICAgICBsZXQgb3V0Um93ID0gJHt0aGlzLmlzQ2hhbm5lbHNMYXN0ID8gYGNvb3Jkc1sxXTtgIDogYGNvb3Jkc1syXTtgfVxuICAgICAgICAgbGV0IG91dENvbCA9ICR7dGhpcy5pc0NoYW5uZWxzTGFzdCA/IGBjb29yZHNbMl07YCA6IGBjb29yZHNbM107YH1cbiAgICAgICAgIHZhciBhY2MgOiBmMzIgPSAwLjA7XG4gICAgICAgICBmb3IgKHZhciByb3cgPSAwOyByb3cgPCB1bmlmb3Jtcy5maWx0ZXJEaW1zWzBdOyByb3cgPSByb3cgKyAxKSB7XG4gICAgICAgICAgIGZvciAodmFyIGNvbCA9IDA7IGNvbCA8IHVuaWZvcm1zLmZpbHRlckRpbXNbMV07IGNvbCA9IGNvbCArIDEpIHtcbiAgICAgICAgICAgICBsZXQgeFJvdyA9IG91dFJvdyAqIHVuaWZvcm1zLnN0cmlkZXNbMF0gKyB1bmlmb3Jtcy5kaWxhdGlvbnNbMF0gKiByb3cgLSB1bmlmb3Jtcy5wYWRzWzBdO1xuICAgICAgICAgICAgIGxldCB4Q29sID0gb3V0Q29sICogdW5pZm9ybXMuc3RyaWRlc1sxXSArIHVuaWZvcm1zLmRpbGF0aW9uc1sxXSAqIGNvbCAtIHVuaWZvcm1zLnBhZHNbMV07XG4gICAgICAgICAgICAgZm9yICh2YXIgeENoYW5uZWwgPSAwOyB4Q2hhbm5lbCA8ICR7XG4gICAgICAgIHRoaXMuaXNDaGFubmVsc0xhc3QgPyBgdW5pZm9ybXMueFNoYXBlWzNdO2AgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYHVuaWZvcm1zLnhTaGFwZVsxXTtgfSB4Q2hhbm5lbCA9IHhDaGFubmVsICsgMSkge1xuICAgICAgICAgICAgICAgJHtcbiAgICAgICAgdGhpcy5pc0NoYW5uZWxzTGFzdCA/IGBsZXQgdiA9IHJlYWRJbnAoYmF0Y2gsIHhSb3csIHhDb2wsIHhDaGFubmVsKTtgIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGBsZXQgdiA9IHJlYWRJbnAoYmF0Y2gsIHhDaGFubmVsLCB4Um93LCB4Q29sKTtgfVxuICAgICAgICAgICAgICAgbGV0IGYgPSByZWFkRmlsdChyb3csIGNvbCwgeENoYW5uZWwsIG91dENoYW5uZWwpO1xuICAgICAgICAgICAgICAgYWNjID0gYWNjICsgdiAqIGY7XG4gICAgICAgICAgICAgfVxuICAgICAgICAgICB9XG4gICAgICAgICB9XG4gICAgICAgICB3cml0ZVJlc3VsdChiYXRjaCwgb3V0Um93LCBvdXRDb2wsIG91dENoYW5uZWwsIGFjYyk7XG4gICAgICAgfVxuICAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19