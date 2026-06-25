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
export class AddNPackedProgram {
    constructor(shapes) {
        this.workPerThread = 1;
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shapes[0];
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [this.workPerThread, 1, 1]);
        this.shaderKey = 'addN';
    }
    getUserCode() {
        const snippets = [];
        // Get target elements from every input tensor.
        this.variableNames.forEach(variable => {
            snippets.push(`let v${variable} = get${variable}ByOutputCoords(coords);`);
        });
        // Calculate the sum of all elements.
        const operation = this.variableNames
            .map(variable => {
            return `v${variable}`;
        })
            .join(' + ');
        const userCode = `
      ${main('index')} {
        for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if (flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            ${snippets.join('\n        ')}
            setOutputAtIndex(flatIndex, ${operation});
          }
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRkbl9wYWNrZWRfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvYWRkbl9wYWNrZWRfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8saUJBQWlCO0lBVTVCLFlBQVksTUFBa0I7UUFKOUIsa0JBQWEsR0FBRyxDQUFDLENBQUM7UUFDbEIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFHVixJQUFJLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM3QixJQUFJLENBQUMsYUFBYSxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDbkQsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxFQUN6RCxDQUFDLElBQUksQ0FBQyxhQUFhLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDaEMsSUFBSSxDQUFDLFNBQVMsR0FBRyxNQUFNLENBQUM7SUFDMUIsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBYSxFQUFFLENBQUM7UUFDOUIsK0NBQStDO1FBQy9DLElBQUksQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLFFBQVEsQ0FBQyxFQUFFO1lBQ3BDLFFBQVEsQ0FBQyxJQUFJLENBQUMsUUFBUSxRQUFRLFNBQVMsUUFBUSx5QkFBeUIsQ0FBQyxDQUFDO1FBQzVFLENBQUMsQ0FBQyxDQUFDO1FBQ0gscUNBQXFDO1FBQ3JDLE1BQU0sU0FBUyxHQUFHLElBQUksQ0FBQyxhQUFhO2FBQ2IsR0FBRyxDQUFDLFFBQVEsQ0FBQyxFQUFFO1lBQ2QsT0FBTyxJQUFJLFFBQVEsRUFBRSxDQUFDO1FBQ3hCLENBQUMsQ0FBQzthQUNELElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUVuQyxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7OEJBQ1MsSUFBSSxDQUFDLGFBQWE7b0NBQ1osSUFBSSxDQUFDLGFBQWE7OztjQUd4QyxRQUFRLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQzswQ0FDQyxTQUFTOzs7O0tBSTlDLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgQWRkTlBhY2tlZFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXM6IHN0cmluZ1tdO1xuICB3b3JrUGVyVGhyZWFkID0gMTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGVzOiBudW1iZXJbXVtdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IHNoYXBlc1swXTtcbiAgICB0aGlzLnZhcmlhYmxlTmFtZXMgPSBzaGFwZXMubWFwKChfLCBpKSA9PiBgVCR7aX1gKTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSxcbiAgICAgICAgW3RoaXMud29ya1BlclRocmVhZCwgMSwgMV0pO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ2FkZE4nO1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCBzbmlwcGV0czogc3RyaW5nW10gPSBbXTtcbiAgICAvLyBHZXQgdGFyZ2V0IGVsZW1lbnRzIGZyb20gZXZlcnkgaW5wdXQgdGVuc29yLlxuICAgIHRoaXMudmFyaWFibGVOYW1lcy5mb3JFYWNoKHZhcmlhYmxlID0+IHtcbiAgICAgIHNuaXBwZXRzLnB1c2goYGxldCB2JHt2YXJpYWJsZX0gPSBnZXQke3ZhcmlhYmxlfUJ5T3V0cHV0Q29vcmRzKGNvb3Jkcyk7YCk7XG4gICAgfSk7XG4gICAgLy8gQ2FsY3VsYXRlIHRoZSBzdW0gb2YgYWxsIGVsZW1lbnRzLlxuICAgIGNvbnN0IG9wZXJhdGlvbiA9IHRoaXMudmFyaWFibGVOYW1lc1xuICAgICAgICAgICAgICAgICAgICAgICAgICAubWFwKHZhcmlhYmxlID0+IHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gYHYke3ZhcmlhYmxlfWA7XG4gICAgICAgICAgICAgICAgICAgICAgICAgIH0pXG4gICAgICAgICAgICAgICAgICAgICAgICAgIC5qb2luKCcgKyAnKTtcblxuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgJHt0aGlzLndvcmtQZXJUaHJlYWR9OyBpID0gaSArIDEpIHtcbiAgICAgICAgICBsZXQgZmxhdEluZGV4ID0gaW5kZXggKiAke3RoaXMud29ya1BlclRocmVhZH0gKyBpO1xuICAgICAgICAgIGlmIChmbGF0SW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGZsYXRJbmRleCk7XG4gICAgICAgICAgICAke3NuaXBwZXRzLmpvaW4oJ1xcbiAgICAgICAgJyl9XG4gICAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGZsYXRJbmRleCwgJHtvcGVyYXRpb259KTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19