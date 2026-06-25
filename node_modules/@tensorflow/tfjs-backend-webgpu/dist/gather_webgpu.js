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
export class GatherProgram {
    constructor(aShape, outputShape) {
        this.variableNames = ['A', 'indices'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = aShape.slice();
        this.aShape = aShape;
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `gather`;
    }
    getUserCode() {
        const sourceCoords = getSourceCoords(this.aShape);
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let indexZ = i32(getIndices(resRC.x, resRC.z));
          let inBounds = select(0.0, 1.0, indexZ >= 0 && indexZ < uniforms.aShape[2]);
          setOutputAtIndex(index, inBounds * getA(${sourceCoords}));
        }
      }
    `;
        return userCode;
    }
}
// The input and output are always flattened into rank 4 tensors.
function getSourceCoords(aShape) {
    const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
    const sourceCoords = [];
    for (let i = 0; i < aShape.length; i++) {
        if (i === 2) {
            sourceCoords.push('indexZ');
        }
        else {
            sourceCoords.push(`${currentCoords[i]}`);
        }
    }
    return sourceCoords.join();
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2F0aGVyX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2dhdGhlcl93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxhQUFhO0lBVXhCLFlBQVksTUFBZ0IsRUFBRSxXQUFxQjtRQUxuRCxrQkFBYSxHQUFhLENBQUMsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQzNDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUVyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxNQUFNLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDbEMsSUFBSSxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7UUFDckIsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxRQUFRLENBQUM7SUFDNUIsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFlBQVksR0FBRyxlQUFlLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2xELE1BQU0sUUFBUSxHQUFHO1FBQ2IsSUFBSSxDQUFDLE9BQU8sQ0FBQzs7Ozs7b0RBSytCLFlBQVk7OztLQUczRCxDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGO0FBRUQsaUVBQWlFO0FBQ2pFLFNBQVMsZUFBZSxDQUFDLE1BQWdCO0lBQ3ZDLE1BQU0sYUFBYSxHQUFHLENBQUMsU0FBUyxFQUFFLFNBQVMsRUFBRSxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDbkUsTUFBTSxZQUFZLEdBQUcsRUFBRSxDQUFDO0lBQ3hCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ3RDLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRTtZQUNYLFlBQVksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUM7U0FDN0I7YUFBTTtZQUNMLFlBQVksQ0FBQyxJQUFJLENBQUMsR0FBRyxhQUFhLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1NBQzFDO0tBQ0Y7SUFDRCxPQUFPLFlBQVksQ0FBQyxJQUFJLEVBQUUsQ0FBQztBQUM3QixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIEdhdGhlclByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHZhcmlhYmxlTmFtZXM6IHN0cmluZ1tdID0gWydBJywgJ2luZGljZXMnXTtcbiAgd29ya2dyb3VwU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gWzY0LCAxLCAxXTtcbiAgYVNoYXBlOiBudW1iZXJbXTtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoYVNoYXBlOiBudW1iZXJbXSwgb3V0cHV0U2hhcGU6IG51bWJlcltdKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGFTaGFwZS5zbGljZSgpO1xuICAgIHRoaXMuYVNoYXBlID0gYVNoYXBlO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBvdXRwdXRTaGFwZTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgZ2F0aGVyYDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3Qgc291cmNlQ29vcmRzID0gZ2V0U291cmNlQ29vcmRzKHRoaXMuYVNoYXBlKTtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IHJlc1JDID0gZ2V0Q29vcmRzRnJvbUluZGV4KGluZGV4KTtcbiAgICAgICAgICBsZXQgaW5kZXhaID0gaTMyKGdldEluZGljZXMocmVzUkMueCwgcmVzUkMueikpO1xuICAgICAgICAgIGxldCBpbkJvdW5kcyA9IHNlbGVjdCgwLjAsIDEuMCwgaW5kZXhaID49IDAgJiYgaW5kZXhaIDwgdW5pZm9ybXMuYVNoYXBlWzJdKTtcbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBpbkJvdW5kcyAqIGdldEEoJHtzb3VyY2VDb29yZHN9KSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuXG4vLyBUaGUgaW5wdXQgYW5kIG91dHB1dCBhcmUgYWx3YXlzIGZsYXR0ZW5lZCBpbnRvIHJhbmsgNCB0ZW5zb3JzLlxuZnVuY3Rpb24gZ2V0U291cmNlQ29vcmRzKGFTaGFwZTogbnVtYmVyW10pOiBzdHJpbmcge1xuICBjb25zdCBjdXJyZW50Q29vcmRzID0gWydyZXNSQy54JywgJ3Jlc1JDLnknLCAncmVzUkMueicsICdyZXNSQy53J107XG4gIGNvbnN0IHNvdXJjZUNvb3JkcyA9IFtdO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGFTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIGlmIChpID09PSAyKSB7XG4gICAgICBzb3VyY2VDb29yZHMucHVzaCgnaW5kZXhaJyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHNvdXJjZUNvb3Jkcy5wdXNoKGAke2N1cnJlbnRDb29yZHNbaV19YCk7XG4gICAgfVxuICB9XG4gIHJldHVybiBzb3VyY2VDb29yZHMuam9pbigpO1xufVxuIl19