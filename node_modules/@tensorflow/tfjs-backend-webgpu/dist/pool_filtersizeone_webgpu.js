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
export class PoolWithFilterSizeEqualsOneProgram {
    constructor(convInfo) {
        this.variableNames = ['x'];
        this.uniforms = `strides : vec2<i32>,`;
        this.workgroupSize = [256, 1, 1];
        this.size = true;
        this.outputShape = convInfo.outShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = 'poolWithFilterSizeEqualsOne';
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];

          let xRCCorner = coords.yz * uniforms.strides;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutputAtIndex(index, value);
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbF9maWx0ZXJzaXplb25lX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3Bvb2xfZmlsdGVyc2l6ZW9uZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxrQ0FBa0M7SUFVN0MsWUFBWSxRQUFpQztRQUw3QyxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEIsYUFBUSxHQUFHLHNCQUFzQixDQUFDO1FBQ2xDLGtCQUFhLEdBQTZCLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUN0RCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDO1FBQ3JDLElBQUksQ0FBQyxjQUFjLEdBQUcsa0JBQWtCLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRTNELElBQUksQ0FBQyxRQUFRLEdBQUcsZUFBZSxDQUMzQixJQUFJLENBQUMsY0FBYyxFQUFFLElBQUksQ0FBQyxXQUFXLEVBQUUsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1FBRS9ELElBQUksQ0FBQyxTQUFTLEdBQUcsNkJBQTZCLENBQUM7SUFDakQsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs7Ozs7Ozs7O0tBY2hCLENBQUM7UUFDRixPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBQb29sV2l0aEZpbHRlclNpemVFcXVhbHNPbmVQcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW119O1xuICBkaXNwYXRjaDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHVuaWZvcm1zID0gYHN0cmlkZXMgOiB2ZWMyPGkzMj4sYDtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzI1NiwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuXG4gIGNvbnN0cnVjdG9yKGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbykge1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5vdXRTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuXG4gICAgdGhpcy5kaXNwYXRjaCA9IGNvbXB1dGVEaXNwYXRjaChcbiAgICAgICAgdGhpcy5kaXNwYXRjaExheW91dCwgdGhpcy5vdXRwdXRTaGFwZSwgdGhpcy53b3JrZ3JvdXBTaXplKTtcblxuICAgIHRoaXMuc2hhZGVyS2V5ID0gJ3Bvb2xXaXRoRmlsdGVyU2l6ZUVxdWFsc09uZSc7XG4gIH1cblxuICBnZXRVc2VyQ29kZSgpOiBzdHJpbmcge1xuICAgIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICAgIGlmIChpbmRleCA8IHVuaWZvcm1zLnNpemUpIHtcbiAgICAgICAgICBsZXQgY29vcmRzID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgICBsZXQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgICAgbGV0IGQgPSBjb29yZHNbM107XG5cbiAgICAgICAgICBsZXQgeFJDQ29ybmVyID0gY29vcmRzLnl6ICogdW5pZm9ybXMuc3RyaWRlcztcbiAgICAgICAgICBsZXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgICBsZXQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAgIGxldCB2YWx1ZSA9IGdldFgoYmF0Y2gsIHhSQ29ybmVyLCB4Q0Nvcm5lciwgZCk7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgdmFsdWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==