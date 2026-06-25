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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ResizeNearestNeighborProgram {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX25lYXJlc3RfbmVpZ2hib3Jfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvcmVzaXplX25lYXJlc3RfbmVpZ2hib3Jfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sNEJBQTRCO0lBV3ZDLFlBQ0ksVUFBNEMsRUFBRSxTQUFpQixFQUMvRCxRQUFnQixFQUFFLGdCQUF5QjtRQVIvQyxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUFHLGlEQUFpRCxDQUFDO1FBQzdELGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBS1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRXZFLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRTNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRS9ELElBQUksQ0FBQyxnQkFBZ0IsR0FBRyxnQkFBZ0IsQ0FBQztRQUN6QyxJQUFJLENBQUMsU0FBUyxHQUFHLGlCQUFpQixnQkFBZ0IsRUFBRSxDQUFDO0lBQ3ZELENBQUM7SUFFRCxXQUFXO1FBQ1QsSUFBSSxpQkFBeUIsQ0FBQztRQUM5QixJQUFJLElBQUksQ0FBQyxnQkFBZ0IsRUFBRTtZQUN6QixpQkFBaUI7Z0JBQ2Isd0VBQXdFO29CQUN4RSxtQkFBbUIsQ0FBQztTQUN6QjthQUFNO1lBQ0wsaUJBQWlCLEdBQUcsaURBQWlELENBQUM7U0FDdkU7UUFFRCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7b0NBbUJlLGlCQUFpQjs7Ozs7Ozs7Ozs7S0FXaEQsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBSZXNpemVOZWFyZXN0TmVpZ2hib3JQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zID0gJ2FkanVzdEhlaWdodFdpZHRoIDogdmVjMjxmMzI+LCByb3VuZEJhc2UgOiBmMzIsJztcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgaGFsZlBpeGVsQ2VudGVyczogYm9vbGVhbjtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgbmV3SGVpZ2h0OiBudW1iZXIsXG4gICAgICBuZXdXaWR0aDogbnVtYmVyLCBoYWxmUGl4ZWxDZW50ZXJzOiBib29sZWFuKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IFtpbnB1dFNoYXBlWzBdLCBuZXdIZWlnaHQsIG5ld1dpZHRoLCBpbnB1dFNoYXBlWzNdXTtcblxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG5cbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5oYWxmUGl4ZWxDZW50ZXJzID0gaGFsZlBpeGVsQ2VudGVycztcbiAgICB0aGlzLnNoYWRlcktleSA9IGByZXNpemVOZWFyZXN0XyR7aGFsZlBpeGVsQ2VudGVyc31gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBsZXQgc291cmNlRnJhY0luZGV4UkM6IHN0cmluZztcbiAgICBpZiAodGhpcy5oYWxmUGl4ZWxDZW50ZXJzKSB7XG4gICAgICBzb3VyY2VGcmFjSW5kZXhSQyA9XG4gICAgICAgICAgYG1heCgodmVjMjxmMzI+KHJjKSArIHZlYzI8ZjMyPigwLjUpKSAqIGVmZmVjdGl2ZUlucHV0T3Zlck91dHB1dFJhdGlvUkNgICtcbiAgICAgICAgICBgLCB2ZWMyPGYzMj4oMC4wKSlgO1xuICAgIH0gZWxzZSB7XG4gICAgICBzb3VyY2VGcmFjSW5kZXhSQyA9IGB2ZWMyPGYzMj4ocmMpICogZWZmZWN0aXZlSW5wdXRPdmVyT3V0cHV0UmF0aW9SQ2A7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCBiID0gY29vcmRzWzBdO1xuICAgICAgICAgIGxldCBkID0gY29vcmRzWzNdO1xuICAgICAgICAgIGxldCByYyA9IGNvb3Jkcy55ejtcblxuICAgICAgICAgIGxldCBlZmZlY3RpdmVJblNpemUgPSB2ZWMyPGYzMj4oXG4gICAgICAgICAgICBmMzIodW5pZm9ybXMueFNoYXBlLnkpIC0gdW5pZm9ybXMuYWRqdXN0SGVpZ2h0V2lkdGhbMF0sXG4gICAgICAgICAgICBmMzIodW5pZm9ybXMueFNoYXBlLnopIC0gdW5pZm9ybXMuYWRqdXN0SGVpZ2h0V2lkdGhbMV0pO1xuXG4gICAgICAgICAgbGV0IGVmZmVjdGl2ZU91dFNpemUgPSB2ZWMyPGYzMj4oXG4gICAgICAgICAgICBmMzIodW5pZm9ybXMub3V0U2hhcGUueSkgLSB1bmlmb3Jtcy5hZGp1c3RIZWlnaHRXaWR0aFswXSxcbiAgICAgICAgICAgIGYzMih1bmlmb3Jtcy5vdXRTaGFwZS56KSAtIHVuaWZvcm1zLmFkanVzdEhlaWdodFdpZHRoWzFdKTtcblxuICAgICAgICAgIGxldCBlZmZlY3RpdmVJbnB1dE92ZXJPdXRwdXRSYXRpb1JDID1cbiAgICAgICAgICAgICAgZWZmZWN0aXZlSW5TaXplIC8gZWZmZWN0aXZlT3V0U2l6ZTtcblxuICAgICAgICAgIC8vIEZyYWN0aW9uYWwgc291cmNlIGluZGV4XG4gICAgICAgICAgbGV0IHNvdXJjZUZyYWNJbmRleFJDID0gJHtzb3VyY2VGcmFjSW5kZXhSQ307XG5cbiAgICAgICAgICAvLyBDb21wdXRlIHRoZSBjb29yZGluYXRvcnMgb2YgbmVhcmVzdCBuZWlnaGJvciBwb2ludC5cbiAgICAgICAgICBsZXQgaW5wdXRTaGFwZVJDID0gdmVjMjxmMzI+KGYzMih1bmlmb3Jtcy54U2hhcGUueSksIGYzMih1bmlmb3Jtcy54U2hhcGUueikpO1xuICAgICAgICAgIGxldCBzb3VyY2VOZWFyZXN0UkMgPSB2ZWMyPGkzMj4oXG4gICAgICAgICAgICBtaW4oaW5wdXRTaGFwZVJDIC0gMS4wLCBmbG9vcihzb3VyY2VGcmFjSW5kZXhSQyArIHVuaWZvcm1zLnJvdW5kQmFzZSkpKTtcbiAgICAgICAgICBsZXQgbmV3VmFsdWUgPSBnZXRYKGIsIHNvdXJjZU5lYXJlc3RSQy54LCBzb3VyY2VOZWFyZXN0UkMueSwgZCk7XG5cbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBuZXdWYWx1ZSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19