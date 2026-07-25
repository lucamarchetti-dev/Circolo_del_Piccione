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
import { getMainHeaderString as main } from './webgpu_program';
import { computeDispatch, flatDispatchLayout } from './webgpu_util';
export class SelectProgram {
    constructor(cRank, shape, rank) {
        this.variableNames = ['c', 'a', 'b'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.cRank = cRank;
        this.rank = rank;
        this.shaderKey = 'select';
    }
    getUserCode() {
        // TODO(WGSL): below code can be merged with getUserCode.
        let cCoords;
        let abCoords;
        if (this.rank > 4) {
            throw Error(`Where for rank ${this.rank} is not yet supported`);
        }
        if (this.rank === 1) {
            abCoords = `resRC`;
            cCoords = `resRC`;
        }
        else {
            const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
            const cCoordVars = [];
            const abCoordVars = [];
            for (let i = 0; i < this.outputShape.length; i++) {
                abCoordVars.push(`${currentCoords[i]}`);
                if (i < this.cRank) {
                    cCoordVars.push(`${currentCoords[i]}`);
                }
            }
            cCoords = cCoordVars.join();
            abCoords = abCoordVars.join();
        }
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let resRC = getCoordsFromIndex(index);
          let cVal = getC(${cCoords});
          if (cVal >= 1.0) {
            setOutputAtIndex(index, getA(${abCoords}));
          } else {
            setOutputAtIndex(index, getB(${abCoords}));
          }
        }
      }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2VsZWN0X3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL3NlbGVjdF93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxhQUFhO0lBV3hCLFlBQVksS0FBYSxFQUFFLEtBQWUsRUFBRSxJQUFZO1FBVnhELGtCQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1FBS2hDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUdyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBR1YsSUFBSSxDQUFDLFdBQVcsR0FBRyxLQUFLLENBQUM7UUFDekIsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLEtBQUssR0FBRyxLQUFLLENBQUM7UUFDbkIsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7UUFDakIsSUFBSSxDQUFDLFNBQVMsR0FBRyxRQUFRLENBQUM7SUFDNUIsQ0FBQztJQUVELFdBQVc7UUFDVCx5REFBeUQ7UUFDekQsSUFBSSxPQUFPLENBQUM7UUFDWixJQUFJLFFBQVEsQ0FBQztRQUNiLElBQUksSUFBSSxDQUFDLElBQUksR0FBRyxDQUFDLEVBQUU7WUFDakIsTUFBTSxLQUFLLENBQUMsa0JBQWtCLElBQUksQ0FBQyxJQUFJLHVCQUF1QixDQUFDLENBQUM7U0FDakU7UUFFRCxJQUFJLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1lBQ25CLFFBQVEsR0FBRyxPQUFPLENBQUM7WUFDbkIsT0FBTyxHQUFHLE9BQU8sQ0FBQztTQUNuQjthQUFNO1lBQ0wsTUFBTSxhQUFhLEdBQUcsQ0FBQyxTQUFTLEVBQUUsU0FBUyxFQUFFLFNBQVMsRUFBRSxTQUFTLENBQUMsQ0FBQztZQUNuRSxNQUFNLFVBQVUsR0FBRyxFQUFFLENBQUM7WUFDdEIsTUFBTSxXQUFXLEdBQUcsRUFBRSxDQUFDO1lBQ3ZCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtnQkFDaEQsV0FBVyxDQUFDLElBQUksQ0FBQyxHQUFHLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3hDLElBQUksQ0FBQyxHQUFHLElBQUksQ0FBQyxLQUFLLEVBQUU7b0JBQ2xCLFVBQVUsQ0FBQyxJQUFJLENBQUMsR0FBRyxhQUFhLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2lCQUN4QzthQUNGO1lBQ0QsT0FBTyxHQUFHLFVBQVUsQ0FBQyxJQUFJLEVBQUUsQ0FBQztZQUM1QixRQUFRLEdBQUcsV0FBVyxDQUFDLElBQUksRUFBRSxDQUFDO1NBQy9CO1FBRUQsTUFBTSxRQUFRLEdBQUc7UUFDYixJQUFJLENBQUMsT0FBTyxDQUFDOzs7NEJBR08sT0FBTzs7MkNBRVEsUUFBUTs7MkNBRVIsUUFBUTs7OztLQUk5QyxDQUFDO1FBQ0YsT0FBTyxRQUFRLENBQUM7SUFDbEIsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2dldE1haW5IZWFkZXJTdHJpbmcgYXMgbWFpbiwgV2ViR1BVUHJvZ3JhbX0gZnJvbSAnLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge2NvbXB1dGVEaXNwYXRjaCwgZmxhdERpc3BhdGNoTGF5b3V0fSBmcm9tICcuL3dlYmdwdV91dGlsJztcblxuZXhwb3J0IGNsYXNzIFNlbGVjdFByb2dyYW0gaW1wbGVtZW50cyBXZWJHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsnYycsICdhJywgJ2InXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIGNSYW5rOiBudW1iZXI7XG4gIHJhbms6IG51bWJlcjtcbiAgc2l6ZSA9IHRydWU7XG5cbiAgY29uc3RydWN0b3IoY1Jhbms6IG51bWJlciwgc2hhcGU6IG51bWJlcltdLCByYW5rOiBudW1iZXIpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gc2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuXG4gICAgdGhpcy5jUmFuayA9IGNSYW5rO1xuICAgIHRoaXMucmFuayA9IHJhbms7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSAnc2VsZWN0JztcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgLy8gVE9ETyhXR1NMKTogYmVsb3cgY29kZSBjYW4gYmUgbWVyZ2VkIHdpdGggZ2V0VXNlckNvZGUuXG4gICAgbGV0IGNDb29yZHM7XG4gICAgbGV0IGFiQ29vcmRzO1xuICAgIGlmICh0aGlzLnJhbmsgPiA0KSB7XG4gICAgICB0aHJvdyBFcnJvcihgV2hlcmUgZm9yIHJhbmsgJHt0aGlzLnJhbmt9IGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMucmFuayA9PT0gMSkge1xuICAgICAgYWJDb29yZHMgPSBgcmVzUkNgO1xuICAgICAgY0Nvb3JkcyA9IGByZXNSQ2A7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGN1cnJlbnRDb29yZHMgPSBbJ3Jlc1JDLngnLCAncmVzUkMueScsICdyZXNSQy56JywgJ3Jlc1JDLncnXTtcbiAgICAgIGNvbnN0IGNDb29yZFZhcnMgPSBbXTtcbiAgICAgIGNvbnN0IGFiQ29vcmRWYXJzID0gW107XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgYWJDb29yZFZhcnMucHVzaChgJHtjdXJyZW50Q29vcmRzW2ldfWApO1xuICAgICAgICBpZiAoaSA8IHRoaXMuY1JhbmspIHtcbiAgICAgICAgICBjQ29vcmRWYXJzLnB1c2goYCR7Y3VycmVudENvb3Jkc1tpXX1gKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgY0Nvb3JkcyA9IGNDb29yZFZhcnMuam9pbigpO1xuICAgICAgYWJDb29yZHMgPSBhYkNvb3JkVmFycy5qb2luKCk7XG4gICAgfVxuXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgICAke21haW4oJ2luZGV4Jyl9IHtcbiAgICAgICAgaWYgKGluZGV4IDwgdW5pZm9ybXMuc2l6ZSkge1xuICAgICAgICAgIGxldCByZXNSQyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgbGV0IGNWYWwgPSBnZXRDKCR7Y0Nvb3Jkc30pO1xuICAgICAgICAgIGlmIChjVmFsID49IDEuMCkge1xuICAgICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgZ2V0QSgke2FiQ29vcmRzfSkpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBnZXRCKCR7YWJDb29yZHN9KSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgYDtcbiAgICByZXR1cm4gdXNlckNvZGU7XG4gIH1cbn1cbiJdfQ==