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
import { getMainHeaderString as main, PixelsOpType } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class FromPixelsProgram {
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
      ${main('index')} {
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZnJvbV9waXhlbHNfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvZnJvbV9waXhlbHNfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQUUsWUFBWSxFQUFnQixNQUFNLGtCQUFrQixDQUFDO0FBQzFGLE9BQU8sRUFBQyxlQUFlLEVBQUUsa0JBQWtCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFbEUsTUFBTSxPQUFPLGlCQUFpQjtJQVc1QixZQUFZLFdBQXFCLEVBQUUsV0FBbUIsRUFBRSxXQUFXLEdBQUcsS0FBSztRQVIzRSxpQkFBWSxHQUFHLFlBQVksQ0FBQyxXQUFXLENBQUM7UUFDeEMsZ0JBQVcsR0FBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRzVCLGtCQUFhLEdBQWEsRUFBRSxDQUFDO1FBQzdCLGtCQUFhLEdBQ1QsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUUsdUJBQXVCO1FBR3ZDLElBQUksQ0FBQyxXQUFXLEdBQUcsV0FBVyxDQUFDO1FBQy9CLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsRUFDekQsQ0FBQyxXQUFXLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFekIsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLFNBQVMsR0FBRyxjQUFjLElBQUksQ0FBQyxXQUFXLEVBQUUsQ0FBQztJQUNwRCxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sV0FBVyxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztZQUNsQyx5Q0FBeUMsQ0FBQyxDQUFDO1lBQzNDLDJDQUEyQyxDQUFDO1FBQ2hELE1BQU0sV0FBVyxHQUNiLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLGtCQUFrQixDQUFDLENBQUMsQ0FBQyxpQkFBaUIsQ0FBQztRQUM5RCxPQUFPO3VDQUM0QixXQUFXO1FBQzFDLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7eUJBSUksV0FBVzs7Ozs7O0dBTWpDLENBQUM7SUFDRixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBQaXhlbHNPcFR5cGUsIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBGcm9tUGl4ZWxzUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgcGl4ZWxzT3BUeXBlID0gUGl4ZWxzT3BUeXBlLkZST01fUElYRUxTO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbMF07XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBpbXBvcnRWaWRlbzogYm9vbGVhbjtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW10gPSBbXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgIFsyNTYsIDEsIDFdOyAgLy8gVGhlIGVtcGlyaWNhbCB2YWx1ZS5cblxuICBjb25zdHJ1Y3RvcihvdXRwdXRTaGFwZTogbnVtYmVyW10sIG51bUNoYW5uZWxzOiBudW1iZXIsIGltcG9ydFZpZGVvID0gZmFsc2UpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUsXG4gICAgICAgIFtudW1DaGFubmVscywgMSwgMV0pO1xuXG4gICAgdGhpcy5pbXBvcnRWaWRlbyA9IGltcG9ydFZpZGVvO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGZyb21QaXhlbHNfJHt0aGlzLmltcG9ydFZpZGVvfWA7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHRleHR1cmVMb2FkID0gdGhpcy5pbXBvcnRWaWRlbyA/XG4gICAgICAgICd0ZXh0dXJlTG9hZChzcmMsIHZlYzI8aTMyPihjb29yZHMueXgpKTsnIDpcbiAgICAgICAgJ3RleHR1cmVMb2FkKHNyYywgdmVjMjxpMzI+KGNvb3Jkcy55eCksIDApJztcbiAgICBjb25zdCB0ZXh0dXJlVHlwZSA9XG4gICAgICAgIHRoaXMuaW1wb3J0VmlkZW8gPyAndGV4dHVyZV9leHRlcm5hbCcgOiAndGV4dHVyZV8yZDxmMzI+JztcbiAgICByZXR1cm4gYFxuICAgICAgQGJpbmRpbmcoMSkgQGdyb3VwKDApIHZhciBzcmM6ICR7dGV4dHVyZVR5cGV9O1xuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGxldCBmbGF0SW5kZXggPSBpbmRleCAqIHVuaWZvcm1zLm51bUNoYW5uZWxzO1xuICAgICAgICBpZiAoZmxhdEluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoZmxhdEluZGV4KTtcbiAgICAgICAgICBsZXQgdmFsdWVzID0gJHt0ZXh0dXJlTG9hZH07XG4gICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB1bmlmb3Jtcy5udW1DaGFubmVsczsgaSA9IGkgKyAxKSB7XG4gICAgICAgICAgICByZXN1bHRbZmxhdEluZGV4ICsgaV0gPSBpMzIoZmxvb3IoMjU1LjAgKiB2YWx1ZXNbaV0pKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgYDtcbiAgfVxufVxuIl19