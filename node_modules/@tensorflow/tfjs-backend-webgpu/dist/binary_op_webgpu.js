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
import { backend_util, util } from '@tensorflow/tfjs-core';
import { getBinaryOpString } from './binary_op_util';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class BinaryOpProgram {
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
        ${main('index')} {
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
       ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmluYXJ5X29wX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2JpbmFyeV9vcF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV6RCxPQUFPLEVBQWUsaUJBQWlCLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUNqRSxPQUFPLEVBQUMsbUJBQW1CLElBQUksSUFBSSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzVFLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLGVBQWU7SUFpQjFCLFlBQVksRUFBZ0IsRUFBRSxNQUFnQixFQUFFLE1BQWdCO1FBVmhFLFNBQUksR0FBRyxJQUFJLENBQUM7UUFDWixrQkFBYSxHQUFHLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBVXpCLElBQUksQ0FBQyxXQUFXLEdBQUcsWUFBWSxDQUFDLDBCQUEwQixDQUFDLE1BQU0sRUFBRSxNQUFNLENBQUMsQ0FBQztRQUMzRSxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsRUFBRSxHQUFHLEVBQUUsQ0FBQztRQUViLElBQUksQ0FBQyxvQkFBb0I7WUFDckIsTUFBTSxDQUFDLE1BQU0sSUFBSSxDQUFDLElBQUksTUFBTSxDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLEdBQUcsQ0FBQztRQUMvRCxJQUFJLENBQUMsb0JBQW9CO1lBQ3JCLE1BQU0sQ0FBQyxNQUFNLElBQUksQ0FBQyxJQUFJLE1BQU0sQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJ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cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCaW5hcnlPcFR5cGUsIGdldEJpbmFyeU9wU3RyaW5nfSBmcm9tICcuL2JpbmFyeV9vcF91dGlsJztcbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgQmluYXJ5T3BQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBvdXRwdXRDb21wb25lbnQ6IG51bWJlcjtcbiAgb3A6IEJpbmFyeU9wVHlwZTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgc2l6ZSA9IHRydWU7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ0EnLCAnQiddO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlQ29tcG9uZW50czogbnVtYmVyW107XG5cbiAgcHJpdmF0ZSBsYXN0RGltZW5zaW9uU2l6ZTogbnVtYmVyO1xuICBwcml2YXRlIHVzZVNoYXJlZE1lbW9yeVdpdGhBOiBib29sZWFuO1xuICBwcml2YXRlIHVzZVNoYXJlZE1lbW9yeVdpdGhCOiBib29sZWFuO1xuICBwcml2YXRlIHR5cGU6IHN0cmluZztcblxuICBjb25zdHJ1Y3RvcihvcDogQmluYXJ5T3BUeXBlLCBhU2hhcGU6IG51bWJlcltdLCBiU2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGJhY2tlbmRfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZShhU2hhcGUsIGJTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLm9wID0gb3A7XG5cbiAgICB0aGlzLnVzZVNoYXJlZE1lbW9yeVdpdGhBID1cbiAgICAgICAgYVNoYXBlLmxlbmd0aCA8PSAxICYmIGJTaGFwZS5sZW5ndGggPiAxICYmIGFTaGFwZVswXSA8IDEyODtcbiAgICB0aGlzLnVzZVNoYXJlZE1lbW9yeVdpdGhCID1cbiAgICAgICAgYlNoYXBlLmxlbmd0aCA8PSAxICYmIGFTaGFwZS5sZW5ndGggPiAxICYmIGJTaGFwZVswXSA8IDEyODtcblxuICAgIGlmICh0aGlzLnVzZVNoYXJlZE1lbW9yeVdpdGhBIHx8IHRoaXMudXNlU2hhcmVkTWVtb3J5V2l0aEIpIHtcbiAgICAgIHRoaXMub3V0cHV0Q29tcG9uZW50ID0gMTtcbiAgICAgIHRoaXMudmFyaWFibGVDb21wb25lbnRzID0gWzEsIDFdO1xuICAgICAgLy8gbGFzdERpbWVuc2lvblNpemUgaXMgdXNlZCBhcyBzaGFyZWRCdWYgYXJyYXkgc2l6ZSwgc28gY2FuIG5vdCBiZVxuICAgICAgLy8gdXNlZCBhcyB1bmlmb3JtLlxuICAgICAgdGhpcy5sYXN0RGltZW5zaW9uU2l6ZSA9XG4gICAgICAgICAgdGhpcy51c2VTaGFyZWRNZW1vcnlXaXRoQiA/IGJTaGFwZVswXSA6IGFTaGFwZVswXTtcbiAgICAgIHRoaXMuc2hhZGVyS2V5ID0gYGJpbmFyeV8ke29wfV8ke3RoaXMubGFzdERpbWVuc2lvblNpemV9YDtcbiAgICAgIHRoaXMudHlwZSA9ICdzaGFyZWQnO1xuICAgICAgLy8gVGhpcyBpcyBhbiBleHBlcmltZW50YWwgdmFsdWUgd2hlbiB1c2luZyBzaGFyZWQgbWVtb3J5LlxuICAgICAgLy8gTm90ZSB0aGF0IHRoZSBtYXhpbXVtIG9mIHdvcmtncm91cCBYIGRpbWVuc2lvbiBpcyAyNTYuXG4gICAgICB0aGlzLndvcmtncm91cFNpemUgPSBbMjU2LCAxLCAxXTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgYURpdmlzaWJsZUJ5NCA9XG4gICAgICAgICAgYVNoYXBlLmxlbmd0aCA+IDAgJiYgYVNoYXBlW2FTaGFwZS5sZW5ndGggLSAxXSAlIDQgPT09IDA7XG4gICAgICBjb25zdCBiRGl2aXNpYmxlQnk0ID1cbiAgICAgICAgICBiU2hhcGUubGVuZ3RoID4gMCAmJiBiU2hhcGVbYlNoYXBlLmxlbmd0aCAtIDFdICUgNCA9PT0gMDtcbiAgICAgIGlmIChhRGl2aXNpYmxlQnk0ICYmIGJEaXZpc2libGVCeTQpIHtcbiAgICAgICAgdGhpcy5vdXRwdXRDb21wb25lbnQgPSA0O1xuICAgICAgICB0aGlzLnZhcmlhYmxlQ29tcG9uZW50cyA9IFs0LCA0XTtcbiAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgICAgKGFEaXZpc2libGVCeTQgJiZcbiAgICAgICAgICAgKHV0aWwuaXNTY2FsYXJTaGFwZShiU2hhcGUpIHx8IGJTaGFwZVtiU2hhcGUubGVuZ3RoIC0gMV0gPT09IDEpKSB8fFxuICAgICAgICAgIChiRGl2aXNpYmxlQnk0ICYmXG4gICAgICAgICAgICh1dGlsLmlzU2NhbGFyU2hhcGUoYVNoYXBlKSB8fCBhU2hhcGVbYVNoYXBlLmxlbmd0aCAtIDFdID09PSAxKSkpIHtcbiAgICAgICAgdGhpcy5vdXRwdXRDb21wb25lbnQgPSA0O1xuICAgICAgICB0aGlzLnZhcmlhYmxlQ29tcG9uZW50cyA9IGFEaXZpc2libGVCeTQgPyBbNCwgMV0gOiBbMSwgNF07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLm91dHB1dENvbXBvbmVudCA9IDE7XG4gICAgICAgIHRoaXMudmFyaWFibGVDb21wb25lbnRzID0gWzEsIDFdO1xuICAgICAgfVxuICAgICAgdGhpcy50eXBlID0gJ25vbnNoYXJlZCc7XG4gICAgICB0aGlzLnNoYWRlcktleSA9IGBiaW5hcnlfJHtvcH1fJHt0aGlzLnZhcmlhYmxlQ29tcG9uZW50c31gO1xuICAgICAgLy8gVE9ETyhqaWFqaWEucWluQGludGVsLmNvbSk6IEhldXJpc3RpY2FsbHkgc2VsZWN0IGEgZ29vZCB3b3JrIGdyb3VwXG4gICAgICAvLyBzaXplLlxuICAgICAgdGhpcy53b3JrZ3JvdXBTaXplID0gWzEyOCwgMSwgMV07XG4gICAgfVxuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSxcbiAgICAgICAgW3RoaXMub3V0cHV0Q29tcG9uZW50LCAxLCAxXSk7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGxldCB1c2VyQ29kZTtcbiAgICBjb25zdCBkVHlwZSA9IHRoaXMub3V0cHV0Q29tcG9uZW50ID09PSA0ID8gJ3ZlYzQ8ZjMyPicgOiAnZjMyJztcbiAgICBjb25zdCBvcEZuU3RyID0gYFxuICAgIGZuIGJpbmFyeU9wZXJhdGlvbihhIDogJHtkVHlwZX0sIGIgOiAke2RUeXBlfSkgLT4gJHtkVHlwZX0ge1xuICAgICAgJHtnZXRCaW5hcnlPcFN0cmluZyh0aGlzLm9wLCB0aGlzLm91dHB1dENvbXBvbmVudCA9PT0gNCl9XG4gICAgfTtcbiAgICBgO1xuXG4gICAgaWYgKHRoaXMudHlwZSA9PT0gJ3NoYXJlZCcpIHtcbiAgICAgIGNvbnN0IHNoYXJlZEluZGV4U25pcHBldCA9IHRoaXMubGFzdERpbWVuc2lvblNpemUgPiAxID9cbiAgICAgICAgICBgY29vcmRzWyR7dGhpcy5vdXRwdXRTaGFwZS5sZW5ndGggLSAxfV1gIDpcbiAgICAgICAgICAnMCc7XG4gICAgICBjb25zdCBhY2Nlc3NEYXRhU25pcHBldCA9IHRoaXMudXNlU2hhcmVkTWVtb3J5V2l0aEIgP1xuICAgICAgICAgIGBsZXQgYSA9IGdldEFCeU91dHB1dEluZGV4KGluZGV4KTtcbiAgICAgICAgICBsZXQgYiA9IHNoYXJlZEJ1Zlske3NoYXJlZEluZGV4U25pcHBldH1dO2AgOlxuICAgICAgICAgIGBsZXQgYSA9IHNoYXJlZEJ1Zlske3NoYXJlZEluZGV4U25pcHBldH1dO1xuICAgICAgICAgIGxldCBiID0gZ2V0QkJ5T3V0cHV0SW5kZXgoaW5kZXgpO2A7XG4gICAgICB1c2VyQ29kZSA9IGBcbiAgICAgICAgJHtvcEZuU3RyfVxuICAgICAgICB2YXI8d29ya2dyb3VwPiBzaGFyZWRCdWYgOiBhcnJheTxmMzIsICR7dGhpcy5sYXN0RGltZW5zaW9uU2l6ZX0+O1xuICAgICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgICAvLyBGaWxsIGluIHRoZSBzaGFyZWQgbWVtb3J5IGJ1ZmZlci5cbiAgICAgICAgICBsZXQgbG9jYWxJbmRleCA9IGkzMihsb2NhbElkLngpO1xuICAgICAgICAgIGlmKGxvY2FsSW5kZXggPCAke3RoaXMubGFzdERpbWVuc2lvblNpemV9KSB7XG4gICAgICAgICAgICBzaGFyZWRCdWZbbG9jYWxJbmRleF0gPSBmMzIoJHtcbiAgICAgICAgICB0aGlzLnVzZVNoYXJlZE1lbW9yeVdpdGhCID8gJ0InIDogJ0EnfVtsb2NhbEluZGV4XSk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHdvcmtncm91cEJhcnJpZXIoKTtcblxuICAgICAgICAgIGlmKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgICAke2FjY2Vzc0RhdGFTbmlwcGV0fVxuICAgICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgYmluYXJ5T3BlcmF0aW9uKGEsIGIpKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgYDtcbiAgICB9IGVsc2Uge1xuICAgICAgdXNlckNvZGUgPSBgXG4gICAgICAgJHtvcEZuU3RyfVxuICAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4ICogJHt0aGlzLm91dHB1dENvbXBvbmVudH0pO1xuICAgICAgICAgICBsZXQgYSA9ICR7ZFR5cGV9KGdldEFCeU91dHB1dENvb3Jkcyhjb29yZHMpKTtcbiAgICAgICAgICAgbGV0IGIgPSAke2RUeXBlfShnZXRCQnlPdXRwdXRDb29yZHMoY29vcmRzKSk7XG4gICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGJpbmFyeU9wZXJhdGlvbihhLCBiKSk7XG4gICAgICAgICB9XG4gICAgICAgfVxuICAgICAgIGA7XG4gICAgfVxuXG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=