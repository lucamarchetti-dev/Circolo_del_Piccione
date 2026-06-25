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
import { backend_util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class ConcatProgram {
    constructor(shapes) {
        this.uniforms = '';
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape =
            backend_util.computeOutShape(shapes, 1 /* axis */);
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.offsetLength = shapes.length - 1;
        for (let i = 0; i < this.offsetLength; i++) {
            this.uniforms += `offset${i} : i32,`;
        }
        this.shaderKey = 'concat';
    }
    getUserCode() {
        const snippets = [];
        if (this.offsetLength > 0) {
            snippets.push(`if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }`);
            for (let i = 1; i < this.offsetLength; i++) {
                snippets.push(`else if (yC < uniforms.offset${[i]}){ ` +
                    `setOutputAtCoords(coords.x, coords.y, getT${i}(yR, yC - uniforms.offset${i - 1})); }`);
            }
            const lastIndex = this.offsetLength;
            const lastShiftIndex = this.offsetLength - 1;
            snippets.push(`else { setOutputAtCoords(coords.x, coords.y, getT${lastIndex}(yR, yC - uniforms.offset${lastShiftIndex})); }`);
        }
        else {
            snippets.push(`setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));`);
        }
        const userCode = `
      ${main('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uY2F0X3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2NvbmNhdF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ25ELE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sYUFBYTtJQVl4QixZQUFZLE1BQStCO1FBTjNDLGFBQVEsR0FBRyxFQUFFLENBQUM7UUFDZCxrQkFBYSxHQUFHLENBQUMsQ0FBQztRQUNsQixrQkFBYSxHQUE2QixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDckQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUlWLElBQUksQ0FBQyxXQUFXO1lBQ1osWUFBWSxDQUFDLGVBQWUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLFVBQVUsQ0FBcUIsQ0FBQztRQUMzRSxJQUFJLENBQUMsYUFBYSxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDbkQsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxFQUN6RCxDQUFDLElBQUksQ0FBQyxhQUFhLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFaEMsSUFBSSxDQUFDLFlBQVksR0FBRyxNQUFNLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztRQUN0QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLFlBQVksRUFBRSxDQUFDLEVBQUUsRUFBRTtZQUMxQyxJQUFJLENBQUMsUUFBUSxJQUFJLFNBQVMsQ0FBQyxTQUFTLENBQUM7U0FDdEM7UUFDRCxJQUFJLENBQUMsU0FBUyxHQUFHLFFBQVEsQ0FBQztJQUM1QixDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sUUFBUSxHQUFhLEVBQUUsQ0FBQztRQUM5QixJQUFJLElBQUksQ0FBQyxZQUFZLEdBQUcsQ0FBQyxFQUFFO1lBQ3pCLFFBQVEsQ0FBQyxJQUFJLENBQ1QscUZBQXFGLENBQUMsQ0FBQztZQUMzRixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLFlBQVksRUFBRSxDQUFDLEVBQUUsRUFBRTtnQkFDMUMsUUFBUSxDQUFDLElBQUksQ0FDVCxnQ0FBZ0MsQ0FBQyxDQUFDLENBQUMsS0FBSztvQkFDeEMsNkNBQ0ksQ0FBQyw0QkFBNEIsQ0FBQyxHQUFHLENBQUMsT0FBTyxDQUFDLENBQUM7YUFDcEQ7WUFDRCxNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDO1lBQ3BDLE1BQU0sY0FBYyxHQUFHLElBQUksQ0FBQyxZQUFZLEdBQUcsQ0FBQyxDQUFDO1lBQzdDLFFBQVEsQ0FBQyxJQUFJLENBQUMsb0RBQ1YsU0FBUyw0QkFBNEIsY0FBYyxPQUFPLENBQUMsQ0FBQztTQUNqRTthQUFNO1lBQ0wsUUFBUSxDQUFDLElBQUksQ0FBQyx1REFBdUQsQ0FBQyxDQUFDO1NBQ3hFO1FBRUQsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzZCQUNRLElBQUksQ0FBQyxhQUFhO29DQUNYLElBQUksQ0FBQyxhQUFhOzs7Ozs7Y0FNeEMsUUFBUSxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUM7Ozs7S0FJcEMsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIENvbmNhdFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXM6IHN0cmluZ1tdO1xuICB1bmlmb3JtcyA9ICcnO1xuICB3b3JrUGVyVGhyZWFkID0gMTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG4gIG9mZnNldExlbmd0aDogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlczogQXJyYXk8W251bWJlciwgbnVtYmVyXT4pIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID1cbiAgICAgICAgYmFja2VuZF91dGlsLmNvbXB1dGVPdXRTaGFwZShzaGFwZXMsIDEgLyogYXhpcyAqLykgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgICB0aGlzLnZhcmlhYmxlTmFtZXMgPSBzaGFwZXMubWFwKChfLCBpKSA9PiBgVCR7aX1gKTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSxcbiAgICAgICAgW3RoaXMud29ya1BlclRocmVhZCwgMSwgMV0pO1xuXG4gICAgdGhpcy5vZmZzZXRMZW5ndGggPSBzaGFwZXMubGVuZ3RoIC0gMTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMub2Zmc2V0TGVuZ3RoOyBpKyspIHtcbiAgICAgIHRoaXMudW5pZm9ybXMgKz0gYG9mZnNldCR7aX0gOiBpMzIsYDtcbiAgICB9XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnY29uY2F0JztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3Qgc25pcHBldHM6IHN0cmluZ1tdID0gW107XG4gICAgaWYgKHRoaXMub2Zmc2V0TGVuZ3RoID4gMCkge1xuICAgICAgc25pcHBldHMucHVzaChcbiAgICAgICAgICBgaWYgKHlDIDwgdW5pZm9ybXMub2Zmc2V0MCl7IHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkcy54LCBjb29yZHMueSwgZ2V0VDAoeVIsIHlDKSk7IH1gKTtcbiAgICAgIGZvciAobGV0IGkgPSAxOyBpIDwgdGhpcy5vZmZzZXRMZW5ndGg7IGkrKykge1xuICAgICAgICBzbmlwcGV0cy5wdXNoKFxuICAgICAgICAgICAgYGVsc2UgaWYgKHlDIDwgdW5pZm9ybXMub2Zmc2V0JHtbaV19KXsgYCArXG4gICAgICAgICAgICBgc2V0T3V0cHV0QXRDb29yZHMoY29vcmRzLngsIGNvb3Jkcy55LCBnZXRUJHtcbiAgICAgICAgICAgICAgICBpfSh5UiwgeUMgLSB1bmlmb3Jtcy5vZmZzZXQke2kgLSAxfSkpOyB9YCk7XG4gICAgICB9XG4gICAgICBjb25zdCBsYXN0SW5kZXggPSB0aGlzLm9mZnNldExlbmd0aDtcbiAgICAgIGNvbnN0IGxhc3RTaGlmdEluZGV4ID0gdGhpcy5vZmZzZXRMZW5ndGggLSAxO1xuICAgICAgc25pcHBldHMucHVzaChgZWxzZSB7IHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkcy54LCBjb29yZHMueSwgZ2V0VCR7XG4gICAgICAgICAgbGFzdEluZGV4fSh5UiwgeUMgLSB1bmlmb3Jtcy5vZmZzZXQke2xhc3RTaGlmdEluZGV4fSkpOyB9YCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHNuaXBwZXRzLnB1c2goYHNldE91dHB1dEF0Q29vcmRzKGNvb3Jkcy54LCBjb29yZHMueSwgZ2V0VDAoeVIsIHlDKSk7YCk7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgZm9yKHZhciBpID0gMDsgaSA8ICR7dGhpcy53b3JrUGVyVGhyZWFkfTsgaSA9IGkgKyAxKSB7XG4gICAgICAgICAgbGV0IGZsYXRJbmRleCA9IGluZGV4ICogJHt0aGlzLndvcmtQZXJUaHJlYWR9ICsgaTtcbiAgICAgICAgICBpZihmbGF0SW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGZsYXRJbmRleCk7XG4gICAgICAgICAgICBsZXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgICAgIGxldCB5QyA9IGNvb3Jkcy55O1xuXG4gICAgICAgICAgICAke3NuaXBwZXRzLmpvaW4oJ1xcbiAgICAgICAgJyl9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==