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
export class BatchNormProgram {
    constructor(xShape, meanShape, varianceShape, offsetShape, scaleShape) {
        this.uniforms = 'varianceEpsilon : f32,';
        // This is an experimental value.
        this.workgroupSize = [128, 1, 1];
        this.size = true;
        this.variableNames = ['x', 'mean', 'variance'];
        backend_util.assertAndGetBroadcastShape(xShape, meanShape);
        backend_util.assertAndGetBroadcastShape(xShape, varianceShape);
        this.outputShape = xShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        if (offsetShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, offsetShape);
            this.variableNames.push('offset');
        }
        if (scaleShape != null) {
            backend_util.assertAndGetBroadcastShape(xShape, scaleShape);
            this.variableNames.push('scale');
        }
        this.offsetShape = offsetShape;
        this.scaleShape = scaleShape;
        this.shaderKey = 'batchNorm';
    }
    getUserCode() {
        let offsetSnippet = '0.0';
        if (this.offsetShape != null) {
            offsetSnippet = 'getOffsetByOutputIndex(index)';
        }
        let scaleSnippet = '1.0';
        if (this.scaleShape != null) {
            scaleSnippet = 'getScaleByOutputIndex(index)';
        }
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size)
        {
          let xValue = getXByOutputIndex(index);
          let meanValue = getMeanByOutputIndex(index);
          let varianValue = getVarianceByOutputIndex(index);
          let offsetValue = ${offsetSnippet};
          let scaleValue = ${scaleSnippet};
          let inv = scaleValue * inverseSqrt(varianValue + f32(uniforms.varianceEpsilon));
          setOutputAtIndex(index,dot(vec3<f32>(xValue, -meanValue, offsetValue), vec3<f32>(inv, inv, 1.0)));
        }
      }
  `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmF0Y2hub3JtX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2JhdGNobm9ybV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ25ELE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sZ0JBQWdCO0lBYzNCLFlBQ0ksTUFBZ0IsRUFBRSxTQUFtQixFQUFFLGFBQXVCLEVBQzlELFdBQTBCLEVBQUUsVUFBeUI7UUFWekQsYUFBUSxHQUFHLHdCQUF3QixDQUFDO1FBQ3BDLGlDQUFpQztRQUNqQyxrQkFBYSxHQUE2QixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFJdEQsU0FBSSxHQUFHLElBQUksQ0FBQztRQUtWLElBQUksQ0FBQyxhQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsTUFBTSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1FBQy9DLFlBQVksQ0FBQywwQkFBMEIsQ0FBQyxNQUFNLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDM0QsWUFBWSxDQUFDLDBCQUEwQixDQUFDLE1BQU0sRUFBRSxhQUFhLENBQUMsQ0FBQztRQUMvRCxJQUFJLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQztRQUMxQixJQUFJLENBQUMsY0FBYyxHQUFHLGtCQUFrQixDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUMzRCxJQUFJLENBQUMsUUFBUSxHQUFHLGVBQWUsQ0FDM0IsSUFBSSxDQUFDLGNBQWMsRUFBRSxJQUFJLENBQUMsV0FBVyxFQUFFLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUUvRCxJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7WUFDdkIsWUFBWSxDQUFDLDBCQUEwQixDQUFDLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQztZQUM3RCxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQztTQUNuQztRQUNELElBQUksVUFBVSxJQUFJLElBQUksRUFBRTtZQUN0QixZQUFZLENBQUMsMEJBQTBCLENBQUMsTUFBTSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1lBQzVELElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1NBQ2xDO1FBQ0QsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7UUFDN0IsSUFBSSxDQUFDLFNBQVMsR0FBRyxXQUFXLENBQUM7SUFDL0IsQ0FBQztJQUVELFdBQVc7UUFDVCxJQUFJLGFBQWEsR0FBRyxLQUFLLENBQUM7UUFDMUIsSUFBSSxJQUFJLENBQUMsV0FBVyxJQUFJLElBQUksRUFBRTtZQUM1QixhQUFhLEdBQUcsK0JBQStCLENBQUM7U0FDakQ7UUFFRCxJQUFJLFlBQVksR0FBRyxLQUFLLENBQUM7UUFDekIsSUFBSSxJQUFJLENBQUMsVUFBVSxJQUFJLElBQUksRUFBRTtZQUMzQixZQUFZLEdBQUcsOEJBQThCLENBQUM7U0FDL0M7UUFFRCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7Ozs4QkFNUyxhQUFhOzZCQUNkLFlBQVk7Ozs7O0dBS3RDLENBQUM7UUFDQSxPQUFPLFFBQVEsQ0FBQztJQUNsQixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBCYXRjaE5vcm1Qcm9ncmFtIGltcGxlbWVudHMgV2ViR1BVUHJvZ3JhbSB7XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgc2hhZGVyS2V5OiBzdHJpbmc7XG4gIGRpc3BhdGNoTGF5b3V0OiB7eDogbnVtYmVyW10sIHk/OiBudW1iZXJbXSwgej86IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW107XG4gIHVuaWZvcm1zID0gJ3ZhcmlhbmNlRXBzaWxvbiA6IGYzMiwnO1xuICAvLyBUaGlzIGlzIGFuIGV4cGVyaW1lbnRhbCB2YWx1ZS5cbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzEyOCwgMSwgMV07XG4gIG9mZnNldFNoYXBlOiBudW1iZXJbXXxudWxsO1xuICBzY2FsZVNoYXBlOiBudW1iZXJbXXxudWxsO1xuICB2YXJpYW5jZUVwc2lsb246IG51bWJlcjtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICB4U2hhcGU6IG51bWJlcltdLCBtZWFuU2hhcGU6IG51bWJlcltdLCB2YXJpYW5jZVNoYXBlOiBudW1iZXJbXSxcbiAgICAgIG9mZnNldFNoYXBlOiBudW1iZXJbXXxudWxsLCBzY2FsZVNoYXBlOiBudW1iZXJbXXxudWxsKSB7XG4gICAgdGhpcy52YXJpYWJsZU5hbWVzID0gWyd4JywgJ21lYW4nLCAndmFyaWFuY2UnXTtcbiAgICBiYWNrZW5kX3V0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGUoeFNoYXBlLCBtZWFuU2hhcGUpO1xuICAgIGJhY2tlbmRfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZSh4U2hhcGUsIHZhcmlhbmNlU2hhcGUpO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSB4U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgaWYgKG9mZnNldFNoYXBlICE9IG51bGwpIHtcbiAgICAgIGJhY2tlbmRfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZSh4U2hhcGUsIG9mZnNldFNoYXBlKTtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdvZmZzZXQnKTtcbiAgICB9XG4gICAgaWYgKHNjYWxlU2hhcGUgIT0gbnVsbCkge1xuICAgICAgYmFja2VuZF91dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdFNoYXBlKHhTaGFwZSwgc2NhbGVTaGFwZSk7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgnc2NhbGUnKTtcbiAgICB9XG4gICAgdGhpcy5vZmZzZXRTaGFwZSA9IG9mZnNldFNoYXBlO1xuICAgIHRoaXMuc2NhbGVTaGFwZSA9IHNjYWxlU2hhcGU7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnYmF0Y2hOb3JtJztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgbGV0IG9mZnNldFNuaXBwZXQgPSAnMC4wJztcbiAgICBpZiAodGhpcy5vZmZzZXRTaGFwZSAhPSBudWxsKSB7XG4gICAgICBvZmZzZXRTbmlwcGV0ID0gJ2dldE9mZnNldEJ5T3V0cHV0SW5kZXgoaW5kZXgpJztcbiAgICB9XG5cbiAgICBsZXQgc2NhbGVTbmlwcGV0ID0gJzEuMCc7XG4gICAgaWYgKHRoaXMuc2NhbGVTaGFwZSAhPSBudWxsKSB7XG4gICAgICBzY2FsZVNuaXBwZXQgPSAnZ2V0U2NhbGVCeU91dHB1dEluZGV4KGluZGV4KSc7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSlcbiAgICAgICAge1xuICAgICAgICAgIGxldCB4VmFsdWUgPSBnZXRYQnlPdXRwdXRJbmRleChpbmRleCk7XG4gICAgICAgICAgbGV0IG1lYW5WYWx1ZSA9IGdldE1lYW5CeU91dHB1dEluZGV4KGluZGV4KTtcbiAgICAgICAgICBsZXQgdmFyaWFuVmFsdWUgPSBnZXRWYXJpYW5jZUJ5T3V0cHV0SW5kZXgoaW5kZXgpO1xuICAgICAgICAgIGxldCBvZmZzZXRWYWx1ZSA9ICR7b2Zmc2V0U25pcHBldH07XG4gICAgICAgICAgbGV0IHNjYWxlVmFsdWUgPSAke3NjYWxlU25pcHBldH07XG4gICAgICAgICAgbGV0IGludiA9IHNjYWxlVmFsdWUgKiBpbnZlcnNlU3FydCh2YXJpYW5WYWx1ZSArIGYzMih1bmlmb3Jtcy52YXJpYW5jZUVwc2lsb24pKTtcbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LGRvdCh2ZWMzPGYzMj4oeFZhbHVlLCAtbWVhblZhbHVlLCBvZmZzZXRWYWx1ZSksIHZlYzM8ZjMyPihpbnYsIGludiwgMS4wKSkpO1xuICAgICAgICB9XG4gICAgICB9XG4gIGA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG59XG4iXX0=