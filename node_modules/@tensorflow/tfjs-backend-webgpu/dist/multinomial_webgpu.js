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
export class MultinomialProgram {
    constructor(batchSize, numSamples) {
        this.variableNames = ['probs'];
        this.outputShape = [];
        this.uniforms = 'seed : f32, numOutcomes: i32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = [batchSize, numSamples];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'multinomial';
    }
    getUserCode() {
        const userCode = `
    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    fn random (seed : f32, resultUV : vec2<f32>) -> f32 {
      let HASHSCALE1 = 443.8975;
      let p = resultUV * seed;
      var p3  = fract(vec3<f32>(p.xyx) * HASHSCALE1);
      p3 = p3 + dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getOutputCoords();
        let batch = coords[0];

        let resUV = vec2<f32>(f32(coords[1]) / f32(uniforms.outShape[1]),
            f32(coords[0]) / f32(uniforms.outShape[0]));
        let r = random(uniforms.seed, resUV);
        var cdf = 0.0;
        for (var i = 0; i < uniforms.numOutcomes - 1; i = i + 1) {
          cdf = cdf + getProbs(batch, i);

          if (r < cdf) {
            setOutputAtIndexI32(index, i);
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutputAtIndexI32(index, uniforms.numOutcomes - 1);
      }
    }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibXVsdGlub21pYWxfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvbXVsdGlub21pYWxfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sa0JBQWtCO0lBVTdCLFlBQVksU0FBaUIsRUFBRSxVQUFrQjtRQVRqRCxrQkFBYSxHQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDcEMsZ0JBQVcsR0FBYSxFQUFFLENBQUM7UUFJM0IsYUFBUSxHQUFHLCtCQUErQixDQUFDO1FBQzNDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxDQUFDLFNBQVMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUMzQyxJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLENBQUMsU0FBUyxHQUFHLGFBQWEsQ0FBQztJQUNqQyxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHOzs7Ozs7Ozs7OztNQVdmLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FzQmhCLENBQUM7UUFDQSxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgTXVsdGlub21pYWxQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXM6IHN0cmluZ1tdID0gWydwcm9icyddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB1bmlmb3JtcyA9ICdzZWVkIDogZjMyLCBudW1PdXRjb21lczogaTMyLCc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGJhdGNoU2l6ZTogbnVtYmVyLCBudW1TYW1wbGVzOiBudW1iZXIpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW2JhdGNoU2l6ZSwgbnVtU2FtcGxlc107XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnbXVsdGlub21pYWwnO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAvL0Jhc2VkIG9uIHRoZSB3b3JrIG9mIERhdmUgSG9za2luc1xuICAgIC8vaHR0cHM6Ly93d3cuc2hhZGVydG95LmNvbS92aWV3LzRkalNSV1xuICAgIGZuIHJhbmRvbSAoc2VlZCA6IGYzMiwgcmVzdWx0VVYgOiB2ZWMyPGYzMj4pIC0+IGYzMiB7XG4gICAgICBsZXQgSEFTSFNDQUxFMSA9IDQ0My44OTc1O1xuICAgICAgbGV0IHAgPSByZXN1bHRVViAqIHNlZWQ7XG4gICAgICB2YXIgcDMgID0gZnJhY3QodmVjMzxmMzI+KHAueHl4KSAqIEhBU0hTQ0FMRTEpO1xuICAgICAgcDMgPSBwMyArIGRvdChwMywgcDMueXp4ICsgMTkuMTkpO1xuICAgICAgcmV0dXJuIGZyYWN0KChwMy54ICsgcDMueSkgKiBwMy56KTtcbiAgICB9XG5cbiAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgbGV0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG5cbiAgICAgICAgbGV0IHJlc1VWID0gdmVjMjxmMzI+KGYzMihjb29yZHNbMV0pIC8gZjMyKHVuaWZvcm1zLm91dFNoYXBlWzFdKSxcbiAgICAgICAgICAgIGYzMihjb29yZHNbMF0pIC8gZjMyKHVuaWZvcm1zLm91dFNoYXBlWzBdKSk7XG4gICAgICAgIGxldCByID0gcmFuZG9tKHVuaWZvcm1zLnNlZWQsIHJlc1VWKTtcbiAgICAgICAgdmFyIGNkZiA9IDAuMDtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB1bmlmb3Jtcy5udW1PdXRjb21lcyAtIDE7IGkgPSBpICsgMSkge1xuICAgICAgICAgIGNkZiA9IGNkZiArIGdldFByb2JzKGJhdGNoLCBpKTtcblxuICAgICAgICAgIGlmIChyIDwgY2RmKSB7XG4gICAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4STMyKGluZGV4LCBpKTtcbiAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICAvLyBJZiBubyBvdGhlciBldmVudCBoYXBwZW5lZCwgbGFzdCBldmVudCBoYXBwZW5lZC5cbiAgICAgICAgc2V0T3V0cHV0QXRJbmRleEkzMihpbmRleCwgdW5pZm9ybXMubnVtT3V0Y29tZXMgLSAxKTtcbiAgICAgIH1cbiAgICB9XG4gIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=