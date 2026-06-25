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
export class ResizeNearestNeigborBackpropProgram {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX25lYXJlc3RfbmVpZ2hib3JfYmFja3Byb3Bfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvcmVzaXplX25lYXJlc3RfbmVpZ2hib3JfYmFja3Byb3Bfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sbUNBQW1DO0lBYTlDLFlBQ0ksVUFBNEMsRUFBRSxZQUFxQjtRQVR2RSxrQkFBYSxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDdkIsYUFBUSxHQUNKO3dDQUNrQyxDQUFDO1FBQ3ZDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBSVYsSUFBSSxDQUFDLFdBQVcsR0FBRyxVQUFVLENBQUM7UUFFOUIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFlBQVksR0FBRyxZQUFZLENBQUM7UUFDakMsSUFBSSxDQUFDLFNBQVMsR0FBRyxnQ0FBZ0MsWUFBWSxFQUFFLENBQUM7SUFDbEUsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztvQkEyQ2IsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsNEJBQTRCLENBQUMsQ0FBQztZQUM5QixzQkFBc0I7Ozs7b0JBSzFDLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLDRCQUE0QixDQUFDLENBQUM7WUFDOUIsc0JBQXNCOzs7Ozs7Ozs7Ozs7S0FZN0MsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBSZXNpemVOZWFyZXN0TmVpZ2JvckJhY2twcm9wUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsnZHknXTtcbiAgdW5pZm9ybXMgPVxuICAgICAgYGVmZmVjdGl2ZVhTaXplIDogdmVjMjxpMzI+LCBlZmZlY3RpdmVZU2l6ZSA6IHZlYzI8aTMyPiwgaW52SGVpZ2h0U2NhbGUgOiBmMzIsIGludldpZHRoU2NhbGUgOiBmMzIsXG4gICAgICAgd2luSGVpZ2h0IDogaTMyLCB3aW5XaWR0aCA6IGkzMixgO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBhbGlnbkNvcm5lcnM6IGJvb2xlYW47XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGFsaWduQ29ybmVyczogYm9vbGVhbikge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBpbnB1dFNoYXBlO1xuXG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5hbGlnbkNvcm5lcnMgPSBhbGlnbkNvcm5lcnM7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgcmVzaXplTmVhcmVzdE5laWdib3JCYWNrcHJvcF8ke2FsaWduQ29ybmVyc31gO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICAgIGxldCBiID0gY29vcmRzWzBdO1xuICAgICAgICAgIGxldCBkID0gY29vcmRzWzNdO1xuICAgICAgICAgIGxldCByID0gY29vcmRzWzFdO1xuICAgICAgICAgIGxldCBjID0gY29vcmRzWzJdO1xuXG4gICAgICAgICAgdmFyIGFjY3VtdWxhdG9yID0gMC4wO1xuXG4gICAgICAgICAgLy8gQ29tcHV0ZSBib3VuZHMgZm9yIHdoZXJlIGluIGR5IHdlIHdpbGwgbG9va1xuICAgICAgICAgIGxldCBzdGFydFJMZXJwID0gZmxvb3IoZjMyKHIpICogdW5pZm9ybXMuaW52SGVpZ2h0U2NhbGUpO1xuICAgICAgICAgIGxldCBzdGFydER5UiA9IGkzMihmbG9vcihzdGFydFJMZXJwIC0gZjMyKHVuaWZvcm1zLndpbkhlaWdodCAvIDIpKSk7XG5cbiAgICAgICAgICBsZXQgc3RhcnRDTGVycCA9IGZsb29yKGYzMihjKSAqIHVuaWZvcm1zLmludldpZHRoU2NhbGUpO1xuICAgICAgICAgIGxldCBzdGFydER5QyA9IGkzMihmbG9vcihzdGFydENMZXJwIC0gZjMyKHVuaWZvcm1zLndpbldpZHRoIC8gMikpKTtcblxuICAgICAgICAgIC8vIExvb3Agb3ZlciBkeVxuICAgICAgICAgIGZvciAodmFyIGR5Uk9mZnNldCA9IDA7IGR5Uk9mZnNldCA8IHVuaWZvcm1zLndpbkhlaWdodDsgZHlST2Zmc2V0KyspIHtcbiAgICAgICAgICAgIGxldCBkeVIgPSBzdGFydER5UiArIGR5Uk9mZnNldDtcblxuICAgICAgICAgICAgLy8gR3VhcmQgYWdhaW5zdCB0aGUgd2luZG93IGV4Y2VlZGluZyB0aGUgYm91bmRzIG9mIGR5XG4gICAgICAgICAgICBpZiAoZHlSIDwgMCB8fCBkeVIgPj0gdW5pZm9ybXMuZHlTaGFwZVsxXSkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZm9yICh2YXIgZHlDT2Zmc2V0ID0gMDsgZHlDT2Zmc2V0IDwgdW5pZm9ybXMud2luV2lkdGg7IGR5Q09mZnNldCsrKSB7XG4gICAgICAgICAgICAgIGxldCBkeUMgPSBzdGFydER5QyArIGR5Q09mZnNldDtcblxuICAgICAgICAgICAgICAvLyBHdWFyZCBhZ2FpbnN0IHRoZSB3aW5kb3cgZXhjZWVkaW5nIHRoZSBib3VuZHMgb2YgZHlcbiAgICAgICAgICAgICAgaWYgKGR5QyA8IDAgfHwgZHlDID49IHVuaWZvcm1zLmR5U2hhcGVbMl0pIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGxldCBzb3VyY2VGcmFjUm93ID0gZjMyKHVuaWZvcm1zLmVmZmVjdGl2ZVhTaXplWzBdKSAqXG4gICAgICAgICAgICAgICAgICAoZjMyKGR5UikgLyBmMzIodW5pZm9ybXMuZWZmZWN0aXZlWVNpemVbMF0pKTtcblxuICAgICAgICAgICAgICBsZXQgc291cmNlRnJhY0NvbCA9IGYzMih1bmlmb3Jtcy5lZmZlY3RpdmVYU2l6ZVsxXSkgKlxuICAgICAgICAgICAgICAgICAgKGYzMihkeUMpIC8gZjMyKHVuaWZvcm1zLmVmZmVjdGl2ZVlTaXplWzFdKSk7XG5cbiAgICAgICAgICAgICAgbGV0IHNvdXJjZU5lYXJlc3RSb3cgPVxuICAgICAgICAgICAgICAgICAgaTMyKG1pbihmMzIodW5pZm9ybXMub3V0U2hhcGVbMV0gLSAxKSxcbiAgICAgICAgICAgICAgICAgICR7XG4gICAgICAgIHRoaXMuYWxpZ25Db3JuZXJzID8gJ2Zsb29yKHNvdXJjZUZyYWNSb3cgKyAwLjUpJyA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2Zsb29yKHNvdXJjZUZyYWNSb3cpJ30pKTtcblxuICAgICAgICAgICAgICBsZXQgc291cmNlTmVhcmVzdENvbCA9XG4gICAgICAgICAgICAgICAgICBpMzIobWluKGYzMih1bmlmb3Jtcy5vdXRTaGFwZVsyXSAtIDEpLFxuICAgICAgICAgICAgICAgICAgJHtcbiAgICAgICAgdGhpcy5hbGlnbkNvcm5lcnMgPyAnZmxvb3Ioc291cmNlRnJhY0NvbCArIDAuNSknIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAnZmxvb3Ioc291cmNlRnJhY0NvbCknfSkpO1xuXG4gICAgICAgICAgICAgIGlmIChyID09IHNvdXJjZU5lYXJlc3RSb3cgJiYgYyA9PSBzb3VyY2VOZWFyZXN0Q29sKSB7XG4gICAgICAgICAgICAgICAgYWNjdW11bGF0b3IgKz0gZ2V0RHkoYiwgZHlSLCBkeUMsIGQpO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIC8vIEVuZCBsb29wIG92ZXIgZHlcblxuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGFjY3VtdWxhdG9yKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=