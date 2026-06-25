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
// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
// The original algorithm is based on computing the top K only, however
// since for TFJS we require the indices of the top K values as well then the
// algorithm found here is a bit modified. Rather than producing the values
// at each step, the indices containing the top K are generated instead.
// The output values are not generated to reduce the number of outputs in the
// GPU, the values can easily be retrieved from the indices using a gather
// op.
export class SwapProgram {
    constructor(shape) {
        this.variableNames = ['x', 'indices'];
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.uniforms = `inputSize : i32, firstPass : i32, negativeInf : f32,
        dir : i32, inc : i32,`;
        this.shaderKey = 'swap';
    }
    getUserCode() {
        const userCode = `
        ${main('index')} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // We compare elements pair-wise within a group of size 2 * inc.
            // The comparing rule for each group alternates between ascending
            // and descending. Within each group, we compare each pair at
            // positions i and i+inc. To decide whether an element at position i
            // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
            // inc, it is in the first half of the group, we denote it as x0,
            // otherwise we denote it as x1.
            // For example, as shown in the Bitonic top K paper referenced
            // above, Figure5(a) shows that element[1] is in the second half of
            // the group when group size is 2, but it is in the first half of
            // the group when group size is 4.
            let isFirstInPair = elemIdx % (2 * uniforms.inc) < uniforms.inc;
            var i = 0;
            if (isFirstInPair) {
              i = elemIdx;
            } else {
              i = elemIdx - uniforms.inc;
            }

            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }

            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.inc;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.inc));
            }

            var x0 = f32(0.0);
            var x1 = f32(0.0);
            if (i0 < uniforms.inputSize) {
              x0 = getX(batch, i0);
            } else {
              x0 = uniforms.negativeInf;
            }
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = uniforms.negativeInf;
            }

            let reverse = elemIdx % (2 * uniforms.dir) >= uniforms.dir;
            let isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
            if (reverse == isGreater) {
              // Elements in opposite order of direction
              let iTemp = i0;
              i0 = i1;
              i1 = iTemp;
            }
            if (isFirstInPair) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `;
        return userCode;
    }
}
export class MergeProgram {
    constructor(shape) {
        this.variableNames = ['x', 'indices'];
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        // |n| Size of the original input of TopK
        // |firstPass| indicates if this is the first time swap is being used which
        // means no indices input containing the top K is present yet.
        // |k| Top k elements desired
        this.uniforms = `inputSize : i32, firstPass : i32, k : i32,`;
        this.shaderKey = 'merge';
    }
    getUserCode() {
        const userCode = `
        ${main('index')} {
          if (index < uniforms.size) {
            let outC = getCoordsFromIndex(index);
            let batch = outC[0];
            let elemIdx = outC[1];
            // The output size is half of the previous size.
            // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _
            // (k=4), we only need to output the indices at positions |, the
            // indices at positions _ can be thrown away, see Figure5(b) After
            // Phase 2 (Merge phase) in the Bitonic Top K paper referenced
            // above.
            // For example, the paper shows we only need to output the orange
            // bars. The output sequence should look like this | | | | | | | |.
            // Because the sequence is halved, to map the output index back to
            // the previous sequence to find the corresponding value, we need
            // to double the index. When we double the index, we basically
            // interpolate a position, so 2i looks like
            // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k
            // position of each 2k positions by - elemIdx % k. E.g. for output
            // at index 4,5,6,7, we want to get the corresponding element at
            // original index 8,9,10,11, for output at index 8,9,10,11,
            // we want to get the corresponding element at original index
            // 16,17,18,19, so on and so forth.

            var i = 0;
            if (elemIdx < uniforms.k) {
              i = elemIdx;
            } else {
              i = elemIdx * 2 - elemIdx % uniforms.k;
            }
            var i0 = 0;
            if (uniforms.firstPass == 1) {
              i0 = i;
            } else {
              i0 = i32(getIndices(batch, i));
            }
            var i1 = 0;
            if (uniforms.firstPass == 1) {
              i1 = i + uniforms.k;
            } else {
              i1 = i32(getIndices(batch, i + uniforms.k));
            }

            let x0 = getX(batch, i0);
            var x1 = f32(0.0);
            if (i1 < uniforms.inputSize) {
              x1 = getX(batch, i1);
            } else {
              x1 = x0;
            }

            if (x0 >= x1) {
              setOutputAtIndex(index, f32(i0));
            } else {
              setOutputAtIndex(index, f32(i1));
            }
          }
        }
      `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidG9wX2tfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvdG9wX2tfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSw4Q0FBOEM7QUFDOUMsNkRBQTZEO0FBQzdELHVFQUF1RTtBQUN2RSw2RUFBNkU7QUFDN0UsMkVBQTJFO0FBQzNFLHdFQUF3RTtBQUN4RSw2RUFBNkU7QUFDN0UsMEVBQTBFO0FBQzFFLE1BQU07QUFFTixNQUFNLE9BQU8sV0FBVztJQVV0QixZQUFZLEtBQWU7UUFMM0Isa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxTQUFTLENBQUMsQ0FBQztRQUVqQyxrQkFBYSxHQUE2QixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDdEQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUdWLElBQUksQ0FBQyxXQUFXLEdBQUcsS0FBSyxDQUFDO1FBQ3pCLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxRQUFRLEdBQUc7OEJBQ1UsQ0FBQztRQUMzQixJQUFJLENBQUMsU0FBUyxHQUFHLE1BQU0sQ0FBQztJQUMxQixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1VBQ1gsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O09Ba0VoQixDQUFDO1FBQ0osT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGO0FBRUQsTUFBTSxPQUFPLFlBQVk7SUFVdkIsWUFBWSxLQUFlO1FBTDNCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFFakMsa0JBQWEsR0FBNkIsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3RELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLEtBQUssQ0FBQztRQUN6QixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUMvRCx5Q0FBeUM7UUFDekMsMkVBQTJFO1FBQzNFLDhEQUE4RDtRQUM5RCw2QkFBNkI7UUFDN0IsSUFBSSxDQUFDLFFBQVEsR0FBRyw0Q0FBNEMsQ0FBQztRQUM3RCxJQUFJLENBQUMsU0FBUyxHQUFHLE9BQU8sQ0FBQztJQUMzQixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFHO1VBQ1gsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQTBEaEIsQ0FBQztRQUNKLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbi8vIEJhc2VkIG9uIEFsZ29yaXRobSAyIG9mIEJpdG9uaWMgVG9wIEssIHJlZjpcbi8vIGh0dHBzOi8vYW5pbHNoYW5iaGFnLmluL3N0YXRpYy9wYXBlcnMvZ3B1dG9wa19zaWdtb2QxOC5wZGZcbi8vIFRoZSBvcmlnaW5hbCBhbGdvcml0aG0gaXMgYmFzZWQgb24gY29tcHV0aW5nIHRoZSB0b3AgSyBvbmx5LCBob3dldmVyXG4vLyBzaW5jZSBmb3IgVEZKUyB3ZSByZXF1aXJlIHRoZSBpbmRpY2VzIG9mIHRoZSB0b3AgSyB2YWx1ZXMgYXMgd2VsbCB0aGVuIHRoZVxuLy8gYWxnb3JpdGhtIGZvdW5kIGhlcmUgaXMgYSBiaXQgbW9kaWZpZWQuIFJhdGhlciB0aGFuIHByb2R1Y2luZyB0aGUgdmFsdWVzXG4vLyBhdCBlYWNoIHN0ZXAsIHRoZSBpbmRpY2VzIGNvbnRhaW5pbmcgdGhlIHRvcCBLIGFyZSBnZW5lcmF0ZWQgaW5zdGVhZC5cbi8vIFRoZSBvdXRwdXQgdmFsdWVzIGFyZSBub3QgZ2VuZXJhdGVkIHRvIHJlZHVjZSB0aGUgbnVtYmVyIG9mIG91dHB1dHMgaW4gdGhlXG4vLyBHUFUsIHRoZSB2YWx1ZXMgY2FuIGVhc2lseSBiZSByZXRyaWV2ZWQgZnJvbSB0aGUgaW5kaWNlcyB1c2luZyBhIGdhdGhlclxuLy8gb3AuXG5cbmV4cG9ydCBjbGFzcyBTd2FwUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdpbmRpY2VzJ107XG4gIHVuaWZvcm1zOiBzdHJpbmc7XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsyNTYsIDEsIDFdO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihzaGFwZTogbnVtYmVyW10pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gc2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMudW5pZm9ybXMgPSBgaW5wdXRTaXplIDogaTMyLCBmaXJzdFBhc3MgOiBpMzIsIG5lZ2F0aXZlSW5mIDogZjMyLFxuICAgICAgICBkaXIgOiBpMzIsIGluYyA6IGkzMixgO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3N3YXAnO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgICAgbGV0IG91dEMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICAgICAgbGV0IGJhdGNoID0gb3V0Q1swXTtcbiAgICAgICAgICAgIGxldCBlbGVtSWR4ID0gb3V0Q1sxXTtcbiAgICAgICAgICAgIC8vIFdlIGNvbXBhcmUgZWxlbWVudHMgcGFpci13aXNlIHdpdGhpbiBhIGdyb3VwIG9mIHNpemUgMiAqIGluYy5cbiAgICAgICAgICAgIC8vIFRoZSBjb21wYXJpbmcgcnVsZSBmb3IgZWFjaCBncm91cCBhbHRlcm5hdGVzIGJldHdlZW4gYXNjZW5kaW5nXG4gICAgICAgICAgICAvLyBhbmQgZGVzY2VuZGluZy4gV2l0aGluIGVhY2ggZ3JvdXAsIHdlIGNvbXBhcmUgZWFjaCBwYWlyIGF0XG4gICAgICAgICAgICAvLyBwb3NpdGlvbnMgaSBhbmQgaStpbmMuIFRvIGRlY2lkZSB3aGV0aGVyIGFuIGVsZW1lbnQgYXQgcG9zaXRpb24gaVxuICAgICAgICAgICAgLy8gaXMgeDAgb3IgeDEsIHdlIG1vZCBpdCBieSAyICogaW5jLCBpZiB0aGUgcmVzdWx0IGlzIHNtYWxsZXIgdGhhblxuICAgICAgICAgICAgLy8gaW5jLCBpdCBpcyBpbiB0aGUgZmlyc3QgaGFsZiBvZiB0aGUgZ3JvdXAsIHdlIGRlbm90ZSBpdCBhcyB4MCxcbiAgICAgICAgICAgIC8vIG90aGVyd2lzZSB3ZSBkZW5vdGUgaXQgYXMgeDEuXG4gICAgICAgICAgICAvLyBGb3IgZXhhbXBsZSwgYXMgc2hvd24gaW4gdGhlIEJpdG9uaWMgdG9wIEsgcGFwZXIgcmVmZXJlbmNlZFxuICAgICAgICAgICAgLy8gYWJvdmUsIEZpZ3VyZTUoYSkgc2hvd3MgdGhhdCBlbGVtZW50WzFdIGlzIGluIHRoZSBzZWNvbmQgaGFsZiBvZlxuICAgICAgICAgICAgLy8gdGhlIGdyb3VwIHdoZW4gZ3JvdXAgc2l6ZSBpcyAyLCBidXQgaXQgaXMgaW4gdGhlIGZpcnN0IGhhbGYgb2ZcbiAgICAgICAgICAgIC8vIHRoZSBncm91cCB3aGVuIGdyb3VwIHNpemUgaXMgNC5cbiAgICAgICAgICAgIGxldCBpc0ZpcnN0SW5QYWlyID0gZWxlbUlkeCAlICgyICogdW5pZm9ybXMuaW5jKSA8IHVuaWZvcm1zLmluYztcbiAgICAgICAgICAgIHZhciBpID0gMDtcbiAgICAgICAgICAgIGlmIChpc0ZpcnN0SW5QYWlyKSB7XG4gICAgICAgICAgICAgIGkgPSBlbGVtSWR4O1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgaSA9IGVsZW1JZHggLSB1bmlmb3Jtcy5pbmM7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIHZhciBpMCA9IDA7XG4gICAgICAgICAgICBpZiAodW5pZm9ybXMuZmlyc3RQYXNzID09IDEpIHtcbiAgICAgICAgICAgICAgaTAgPSBpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgaTAgPSBpMzIoZ2V0SW5kaWNlcyhiYXRjaCwgaSkpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB2YXIgaTEgPSAwO1xuICAgICAgICAgICAgaWYgKHVuaWZvcm1zLmZpcnN0UGFzcyA9PSAxKSB7XG4gICAgICAgICAgICAgIGkxID0gaSArIHVuaWZvcm1zLmluYztcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGkxID0gaTMyKGdldEluZGljZXMoYmF0Y2gsIGkgKyB1bmlmb3Jtcy5pbmMpKTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgdmFyIHgwID0gZjMyKDAuMCk7XG4gICAgICAgICAgICB2YXIgeDEgPSBmMzIoMC4wKTtcbiAgICAgICAgICAgIGlmIChpMCA8IHVuaWZvcm1zLmlucHV0U2l6ZSkge1xuICAgICAgICAgICAgICB4MCA9IGdldFgoYmF0Y2gsIGkwKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIHgwID0gdW5pZm9ybXMubmVnYXRpdmVJbmY7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAoaTEgPCB1bmlmb3Jtcy5pbnB1dFNpemUpIHtcbiAgICAgICAgICAgICAgeDEgPSBnZXRYKGJhdGNoLCBpMSk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICB4MSA9IHVuaWZvcm1zLm5lZ2F0aXZlSW5mO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgcmV2ZXJzZSA9IGVsZW1JZHggJSAoMiAqIHVuaWZvcm1zLmRpcikgPj0gdW5pZm9ybXMuZGlyO1xuICAgICAgICAgICAgbGV0IGlzR3JlYXRlciA9IHgwID4geDEgfHwgKHgwID09IHgxICYmIGkxID4gaTApO1xuICAgICAgICAgICAgaWYgKHJldmVyc2UgPT0gaXNHcmVhdGVyKSB7XG4gICAgICAgICAgICAgIC8vIEVsZW1lbnRzIGluIG9wcG9zaXRlIG9yZGVyIG9mIGRpcmVjdGlvblxuICAgICAgICAgICAgICBsZXQgaVRlbXAgPSBpMDtcbiAgICAgICAgICAgICAgaTAgPSBpMTtcbiAgICAgICAgICAgICAgaTEgPSBpVGVtcDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmIChpc0ZpcnN0SW5QYWlyKSB7XG4gICAgICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGYzMihpMCkpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgZjMyKGkxKSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgTWVyZ2VQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ2luZGljZXMnXTtcbiAgdW5pZm9ybXM6IHN0cmluZztcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzI1NiwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBudW1iZXJbXSkge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBzaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgLy8gfG58IFNpemUgb2YgdGhlIG9yaWdpbmFsIGlucHV0IG9mIFRvcEtcbiAgICAvLyB8Zmlyc3RQYXNzfCBpbmRpY2F0ZXMgaWYgdGhpcyBpcyB0aGUgZmlyc3QgdGltZSBzd2FwIGlzIGJlaW5nIHVzZWQgd2hpY2hcbiAgICAvLyBtZWFucyBubyBpbmRpY2VzIGlucHV0IGNvbnRhaW5pbmcgdGhlIHRvcCBLIGlzIHByZXNlbnQgeWV0LlxuICAgIC8vIHxrfCBUb3AgayBlbGVtZW50cyBkZXNpcmVkXG4gICAgdGhpcy51bmlmb3JtcyA9IGBpbnB1dFNpemUgOiBpMzIsIGZpcnN0UGFzcyA6IGkzMiwgayA6IGkzMixgO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ21lcmdlJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICAgIGxldCBvdXRDID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgICAgIGxldCBiYXRjaCA9IG91dENbMF07XG4gICAgICAgICAgICBsZXQgZWxlbUlkeCA9IG91dENbMV07XG4gICAgICAgICAgICAvLyBUaGUgb3V0cHV0IHNpemUgaXMgaGFsZiBvZiB0aGUgcHJldmlvdXMgc2l6ZS5cbiAgICAgICAgICAgIC8vIElmIHRoZSBwcmV2aW91cyBzZXF1ZW5jZSBpcyB8IHwgfCB8IF8gXyBfIF8gIHwgfCB8IHwgIF8gXyBfIF9cbiAgICAgICAgICAgIC8vIChrPTQpLCB3ZSBvbmx5IG5lZWQgdG8gb3V0cHV0IHRoZSBpbmRpY2VzIGF0IHBvc2l0aW9ucyB8LCB0aGVcbiAgICAgICAgICAgIC8vIGluZGljZXMgYXQgcG9zaXRpb25zIF8gY2FuIGJlIHRocm93biBhd2F5LCBzZWUgRmlndXJlNShiKSBBZnRlclxuICAgICAgICAgICAgLy8gUGhhc2UgMiAoTWVyZ2UgcGhhc2UpIGluIHRoZSBCaXRvbmljIFRvcCBLIHBhcGVyIHJlZmVyZW5jZWRcbiAgICAgICAgICAgIC8vIGFib3ZlLlxuICAgICAgICAgICAgLy8gRm9yIGV4YW1wbGUsIHRoZSBwYXBlciBzaG93cyB3ZSBvbmx5IG5lZWQgdG8gb3V0cHV0IHRoZSBvcmFuZ2VcbiAgICAgICAgICAgIC8vIGJhcnMuIFRoZSBvdXRwdXQgc2VxdWVuY2Ugc2hvdWxkIGxvb2sgbGlrZSB0aGlzIHwgfCB8IHwgfCB8IHwgfC5cbiAgICAgICAgICAgIC8vIEJlY2F1c2UgdGhlIHNlcXVlbmNlIGlzIGhhbHZlZCwgdG8gbWFwIHRoZSBvdXRwdXQgaW5kZXggYmFjayB0b1xuICAgICAgICAgICAgLy8gdGhlIHByZXZpb3VzIHNlcXVlbmNlIHRvIGZpbmQgdGhlIGNvcnJlc3BvbmRpbmcgdmFsdWUsIHdlIG5lZWRcbiAgICAgICAgICAgIC8vIHRvIGRvdWJsZSB0aGUgaW5kZXguIFdoZW4gd2UgZG91YmxlIHRoZSBpbmRleCwgd2UgYmFzaWNhbGx5XG4gICAgICAgICAgICAvLyBpbnRlcnBvbGF0ZSBhIHBvc2l0aW9uLCBzbyAyaSBsb29rcyBsaWtlXG4gICAgICAgICAgICAvLyB8IF8gfCBfIHwgXyB8IF8gfCBfIHwgXyB8IF8uIFdlIG1vdmUgdGhlIHwgdG8gdGhlIGZpcnN0IGtcbiAgICAgICAgICAgIC8vIHBvc2l0aW9uIG9mIGVhY2ggMmsgcG9zaXRpb25zIGJ5IC0gZWxlbUlkeCAlIGsuIEUuZy4gZm9yIG91dHB1dFxuICAgICAgICAgICAgLy8gYXQgaW5kZXggNCw1LDYsNywgd2Ugd2FudCB0byBnZXQgdGhlIGNvcnJlc3BvbmRpbmcgZWxlbWVudCBhdFxuICAgICAgICAgICAgLy8gb3JpZ2luYWwgaW5kZXggOCw5LDEwLDExLCBmb3Igb3V0cHV0IGF0IGluZGV4IDgsOSwxMCwxMSxcbiAgICAgICAgICAgIC8vIHdlIHdhbnQgdG8gZ2V0IHRoZSBjb3JyZXNwb25kaW5nIGVsZW1lbnQgYXQgb3JpZ2luYWwgaW5kZXhcbiAgICAgICAgICAgIC8vIDE2LDE3LDE4LDE5LCBzbyBvbiBhbmQgc28gZm9ydGguXG5cbiAgICAgICAgICAgIHZhciBpID0gMDtcbiAgICAgICAgICAgIGlmIChlbGVtSWR4IDwgdW5pZm9ybXMuaykge1xuICAgICAgICAgICAgICBpID0gZWxlbUlkeDtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIGkgPSBlbGVtSWR4ICogMiAtIGVsZW1JZHggJSB1bmlmb3Jtcy5rO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIGkwID0gMDtcbiAgICAgICAgICAgIGlmICh1bmlmb3Jtcy5maXJzdFBhc3MgPT0gMSkge1xuICAgICAgICAgICAgICBpMCA9IGk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICBpMCA9IGkzMihnZXRJbmRpY2VzKGJhdGNoLCBpKSk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB2YXIgaTEgPSAwO1xuICAgICAgICAgICAgaWYgKHVuaWZvcm1zLmZpcnN0UGFzcyA9PSAxKSB7XG4gICAgICAgICAgICAgIGkxID0gaSArIHVuaWZvcm1zLms7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICBpMSA9IGkzMihnZXRJbmRpY2VzKGJhdGNoLCBpICsgdW5pZm9ybXMuaykpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgeDAgPSBnZXRYKGJhdGNoLCBpMCk7XG4gICAgICAgICAgICB2YXIgeDEgPSBmMzIoMC4wKTtcbiAgICAgICAgICAgIGlmIChpMSA8IHVuaWZvcm1zLmlucHV0U2l6ZSkge1xuICAgICAgICAgICAgICB4MSA9IGdldFgoYmF0Y2gsIGkxKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIHgxID0geDA7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGlmICh4MCA+PSB4MSkge1xuICAgICAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBmMzIoaTApKTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIGYzMihpMSkpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==