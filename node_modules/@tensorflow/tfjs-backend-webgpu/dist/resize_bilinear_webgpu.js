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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ResizeBilinearProgram {
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
      ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX2JpbGluZWFyX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3Jlc2l6ZV9iaWxpbmVhcl93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxxQkFBcUI7SUFVaEMsWUFDSSxVQUE0QyxFQUFFLFNBQWlCLEVBQy9ELFFBQWdCO1FBUHBCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN0QixhQUFRLEdBQUcsd0RBQXdELENBQUM7UUFDcEUsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFLVixJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkUsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxnQkFBZ0IsQ0FBQztJQUNwQyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0tBMENoQixDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFJlc2l6ZUJpbGluZWFyUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCddO1xuICB1bmlmb3JtcyA9ICdhZGp1c3RIZWlnaHRXaWR0aCA6IHZlYzI8ZjMyPiwgaGFsZlBpeGVsQ2VudGVycyA6IGYzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBuZXdIZWlnaHQ6IG51bWJlcixcbiAgICAgIG5ld1dpZHRoOiBudW1iZXIpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW2lucHV0U2hhcGVbMF0sIG5ld0hlaWdodCwgbmV3V2lkdGgsIGlucHV0U2hhcGVbM11dO1xuXG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcblxuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLnNoYWRlcktleSA9IGByZXNpemVCaWxpbmVhcmA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgbGV0IGIgPSBjb29yZHNbMF07XG4gICAgICAgICAgbGV0IGQgPSBjb29yZHNbM107XG4gICAgICAgICAgbGV0IHJjID0gY29vcmRzLnl6O1xuXG4gICAgICAgICAgbGV0IGVmZmVjdGl2ZUluU2l6ZSA9IHZlYzI8ZjMyPihcbiAgICAgICAgICAgIGYzMih1bmlmb3Jtcy54U2hhcGUueSkgLSB1bmlmb3Jtcy5hZGp1c3RIZWlnaHRXaWR0aFswXSxcbiAgICAgICAgICAgIGYzMih1bmlmb3Jtcy54U2hhcGUueikgLSB1bmlmb3Jtcy5hZGp1c3RIZWlnaHRXaWR0aFsxXSk7XG5cbiAgICAgICAgICBsZXQgZWZmZWN0aXZlT3V0U2l6ZSA9IHZlYzI8ZjMyPihcbiAgICAgICAgICAgIGYzMih1bmlmb3Jtcy5vdXRTaGFwZS55KSAtIHVuaWZvcm1zLmFkanVzdEhlaWdodFdpZHRoWzBdLFxuICAgICAgICAgICAgZjMyKHVuaWZvcm1zLm91dFNoYXBlLnopIC0gdW5pZm9ybXMuYWRqdXN0SGVpZ2h0V2lkdGhbMV0pO1xuXG4gICAgICAgICAgbGV0IGVmZmVjdGl2ZUlucHV0T3Zlck91dHB1dFJhdGlvUkMgPVxuICAgICAgICAgICAgICBlZmZlY3RpdmVJblNpemUgLyBlZmZlY3RpdmVPdXRTaXplO1xuXG4gICAgICAgICAgLy8gRnJhY3Rpb25hbCBzb3VyY2UgaW5kZXhcbiAgICAgICAgICBsZXQgc291cmNlRnJhY0luZGV4UkMgPVxuICAgICAgICAgICAgKHZlYzI8ZjMyPihyYykgKyB2ZWMyPGYzMj4odW5pZm9ybXMuaGFsZlBpeGVsQ2VudGVycykpICpcbiAgICAgICAgICAgIGVmZmVjdGl2ZUlucHV0T3Zlck91dHB1dFJhdGlvUkMgLSB2ZWMyPGYzMj4odW5pZm9ybXMuaGFsZlBpeGVsQ2VudGVycyk7XG5cbiAgICAgICAgICAvLyBDb21wdXRlIHRoZSBmb3VyIGludGVnZXIgaW5kaWNlcy5cbiAgICAgICAgICBsZXQgc291cmNlRmxvb3JSQyA9IHZlYzI8aTMyPihzb3VyY2VGcmFjSW5kZXhSQyk7XG4gICAgICAgICAgbGV0IHNvdXJjZUNlaWxSQyA9IHZlYzI8aTMyPihcbiAgICAgICAgICAgIG1pbih2ZWMyPGYzMj4odW5pZm9ybXMueFNoYXBlLnl6KSAtIHZlYzI8ZjMyPigxLjApLCBjZWlsKHNvdXJjZUZyYWNJbmRleFJDKSkpO1xuXG4gICAgICAgICAgbGV0IHRvcExlZnQgPSBnZXRYKGIsIHNvdXJjZUZsb29yUkMueCwgc291cmNlRmxvb3JSQy55LCBkKTtcbiAgICAgICAgICBsZXQgYm90dG9tTGVmdCA9IGdldFgoYiwgc291cmNlQ2VpbFJDLngsIHNvdXJjZUZsb29yUkMueSwgZCk7XG4gICAgICAgICAgbGV0IHRvcFJpZ2h0ID0gZ2V0WChiLCBzb3VyY2VGbG9vclJDLngsIHNvdXJjZUNlaWxSQy55LCBkKTtcbiAgICAgICAgICBsZXQgYm90dG9tUmlnaHQgPSBnZXRYKGIsIHNvdXJjZUNlaWxSQy54LCBzb3VyY2VDZWlsUkMueSwgZCk7XG5cbiAgICAgICAgICBsZXQgZnJhY1JDID0gc291cmNlRnJhY0luZGV4UkMgLSB2ZWMyPGYzMj4oc291cmNlRmxvb3JSQyk7XG5cbiAgICAgICAgICBsZXQgdG9wID0gdG9wTGVmdCArICh0b3BSaWdodCAtIHRvcExlZnQpICogZnJhY1JDLnk7XG4gICAgICAgICAgbGV0IGJvdHRvbSA9IGJvdHRvbUxlZnQgKyAoYm90dG9tUmlnaHQgLSBib3R0b21MZWZ0KSAqIGZyYWNSQy55O1xuICAgICAgICAgIGxldCBuZXdWYWx1ZSA9IHRvcCArIChib3R0b20gLSB0b3ApICogZnJhY1JDLng7XG5cbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBuZXdWYWx1ZSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19