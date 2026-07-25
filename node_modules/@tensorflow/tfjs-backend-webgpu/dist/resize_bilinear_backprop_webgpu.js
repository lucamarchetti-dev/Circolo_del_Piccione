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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ResizeBilinearBackpropProgram {
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
      ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX2JpbGluZWFyX2JhY2twcm9wX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3Jlc2l6ZV9iaWxpbmVhcl9iYWNrcHJvcF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyw2QkFBNkI7SUFheEMsWUFDSSxVQUE0QyxFQUFFLFlBQXFCO1FBVHZFLGtCQUFhLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN2QixhQUFRLEdBQ0o7bUZBQzZFLENBQUM7UUFDbEYsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBRXJELFNBQUksR0FBRyxJQUFJLENBQUM7UUFJVixJQUFJLENBQUMsV0FBVyxHQUFHLFVBQVUsQ0FBQztRQUU5QixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsWUFBWSxHQUFHLFlBQVksQ0FBQztRQUNqQyxJQUFJLENBQUMsU0FBUyxHQUFHLDBCQUEwQixZQUFZLEVBQUUsQ0FBQztJQUM1RCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQXlFaEIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBSZXNpemVCaWxpbmVhckJhY2twcm9wUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgYGVmZmVjdGl2ZVhTaXplIDogdmVjMjxpMzI+LCBlZmZlY3RpdmVZU2l6ZSA6IHZlYzI8aTMyPiwgaGVpZ2h0U2NhbGUgOiBmMzIsIHdpZHRoU2NhbGUgOiBmMzIsXG4gICAgICAgaW52SGVpZ2h0U2NhbGUgOiBmMzIsIGludldpZHRoU2NhbGUgOiBmMzIsIHdpbkhlaWdodCA6IGkzMiwgd2luV2lkdGggOiBpMzIsYDtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgYWxpZ25Db3JuZXJzOiBib29sZWFuO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBhbGlnbkNvcm5lcnM6IGJvb2xlYW4pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gaW5wdXRTaGFwZTtcblxuICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQgPSBmbGF0RGlzcGF0Y2hMYXlvdXQodGhpcy5vdXRwdXRTaGFwZSk7XG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuYWxpZ25Db3JuZXJzID0gYWxpZ25Db3JuZXJzO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYHJlc2l6ZUJpbGluZWFyQmFja3Byb3BfJHthbGlnbkNvcm5lcnN9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgICBsZXQgYiA9IGNvb3Jkc1swXTtcbiAgICAgICAgICBsZXQgZCA9IGNvb3Jkc1szXTtcbiAgICAgICAgICBsZXQgciA9IGNvb3Jkc1sxXTtcbiAgICAgICAgICBsZXQgYyA9IGNvb3Jkc1syXTtcblxuICAgICAgICAgIHZhciBhY2N1bXVsYXRvciA9IDAuMDtcblxuICAgICAgICAgIC8vIENvbXB1dGUgYm91bmRzIGZvciB3aGVyZSBpbiBkeSB3ZSB3aWxsIGxvb2tcbiAgICAgICAgICBsZXQgc3RhcnRSTGVycCA9IGZsb29yKGYzMihyKSAqIHVuaWZvcm1zLmludkhlaWdodFNjYWxlKTtcbiAgICAgICAgICBsZXQgc3RhcnREeVIgPSBpMzIoc3RhcnRSTGVycCAtIGYzMih1bmlmb3Jtcy53aW5IZWlnaHQgLyAyKSk7XG5cbiAgICAgICAgICBsZXQgc3RhcnRDTGVycCA9IGZsb29yKGYzMihjKSAqIHVuaWZvcm1zLmludldpZHRoU2NhbGUpO1xuICAgICAgICAgIGxldCBzdGFydER5QyA9IGkzMihzdGFydENMZXJwIC0gZjMyKHVuaWZvcm1zLndpbldpZHRoIC8gMikpO1xuXG4gICAgICAgICAgLy8gTG9vcCBvdmVyIGR5XG4gICAgICAgICAgZm9yICh2YXIgZHlST2Zmc2V0ID0gMDsgZHlST2Zmc2V0IDwgdW5pZm9ybXMud2luSGVpZ2h0OyBkeVJPZmZzZXQrKykge1xuICAgICAgICAgICAgbGV0IGR5UiA9IHN0YXJ0RHlSICsgZHlST2Zmc2V0O1xuXG4gICAgICAgICAgICAvLyBHdWFyZCBhZ2FpbnN0IHRoZSB3aW5kb3cgZXhjZWVkaW5nIHRoZSBib3VuZHMgb2YgZHlcbiAgICAgICAgICAgIGlmIChkeVIgPCAwIHx8IGR5UiA+PSB1bmlmb3Jtcy5keVNoYXBlWzFdKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmb3IgKHZhciBkeUNPZmZzZXQgPSAwOyBkeUNPZmZzZXQgPCB1bmlmb3Jtcy53aW5XaWR0aDsgZHlDT2Zmc2V0KyspIHtcbiAgICAgICAgICAgICAgbGV0IGR5QyA9IHN0YXJ0RHlDICsgZHlDT2Zmc2V0O1xuXG4gICAgICAgICAgICAgIC8vIEd1YXJkIGFnYWluc3QgdGhlIHdpbmRvdyBleGNlZWRpbmcgdGhlIGJvdW5kcyBvZiBkeVxuICAgICAgICAgICAgICBpZiAoZHlDIDwgMCB8fCBkeUMgPj0gdW5pZm9ybXMuZHlTaGFwZVsyXSkge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgbGV0IGR4UiA9IGYzMihkeVIpICogdW5pZm9ybXMuaGVpZ2h0U2NhbGU7XG4gICAgICAgICAgICAgIGxldCB0b3BEeFJJbmRleCA9IGkzMihmbG9vcihkeFIpKTtcbiAgICAgICAgICAgICAgbGV0IGJvdHRvbUR4UkluZGV4ID0gaTMyKG1pbihjZWlsKGR4UiksIGYzMih1bmlmb3Jtcy5vdXRTaGFwZVsxXSAtIDEpKSk7XG4gICAgICAgICAgICAgIGxldCBkeFJMZXJwID0gZHhSIC0gZjMyKHRvcER4UkluZGV4KTtcbiAgICAgICAgICAgICAgbGV0IGludmVyc2VEeFJMZXJwID0gMS4wIC0gZHhSTGVycDtcblxuICAgICAgICAgICAgICBsZXQgZHhDID0gZjMyKGR5QykgKiB1bmlmb3Jtcy53aWR0aFNjYWxlO1xuICAgICAgICAgICAgICBsZXQgbGVmdER4Q0luZGV4ID0gaTMyKGZsb29yKGR4QykpO1xuICAgICAgICAgICAgICBsZXQgcmlnaHREeENJbmRleCA9IGkzMihtaW4oY2VpbChkeEMpLCBmMzIodW5pZm9ybXMub3V0U2hhcGVbMl0gLSAxKSkpO1xuICAgICAgICAgICAgICBsZXQgZHhDTGVycCA9IGR4QyAtIGYzMihsZWZ0RHhDSW5kZXgpO1xuICAgICAgICAgICAgICBsZXQgaW52ZXJzZUR4Q0xlcnAgPSAxLjAgLSBkeENMZXJwO1xuXG4gICAgICAgICAgICAgIGlmIChyID09IHRvcER4UkluZGV4ICYmIGMgPT0gbGVmdER4Q0luZGV4KSB7XG4gICAgICAgICAgICAgICAgLy8gdG9wTGVmdFxuICAgICAgICAgICAgICAgIGFjY3VtdWxhdG9yICs9XG4gICAgICAgICAgICAgICAgICBnZXREeShiLCBkeVIsIGR5QywgZCkgKiBpbnZlcnNlRHhSTGVycCAqIGludmVyc2VEeENMZXJwO1xuICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgaWYgKHIgPT0gdG9wRHhSSW5kZXggJiYgYyA9PSByaWdodER4Q0luZGV4KSB7XG4gICAgICAgICAgICAgICAgLy8gdG9wUmlnaHRcbiAgICAgICAgICAgICAgICBhY2N1bXVsYXRvciArPSBnZXREeShiLCBkeVIsIGR5QywgZCkgKiBpbnZlcnNlRHhSTGVycCAqIGR4Q0xlcnA7XG4gICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICBpZiAociA9PSBib3R0b21EeFJJbmRleCAmJiBjID09IGxlZnREeENJbmRleCkge1xuICAgICAgICAgICAgICAgIC8vIGJvdHRvbUxlZnRcbiAgICAgICAgICAgICAgICBhY2N1bXVsYXRvciArPSBnZXREeShiLCBkeVIsIGR5QywgZCkgKiBkeFJMZXJwICogaW52ZXJzZUR4Q0xlcnA7XG4gICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICBpZiAociA9PSBib3R0b21EeFJJbmRleCAmJiBjID09IHJpZ2h0RHhDSW5kZXgpIHtcbiAgICAgICAgICAgICAgICAvLyBib3R0b21SaWdodFxuICAgICAgICAgICAgICAgIGFjY3VtdWxhdG9yICs9IGdldER5KGIsIGR5UiwgZHlDLCBkKSAqIGR4UkxlcnAgKiBkeENMZXJwO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIC8vIEVuZCBsb29wIG92ZXIgZHlcblxuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGFjY3VtdWxhdG9yKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=