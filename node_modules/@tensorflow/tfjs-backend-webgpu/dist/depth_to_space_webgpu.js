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
export class DepthToSpaceProgram {
    constructor(outputShape, dataFormat) {
        this.variableNames = ['x'];
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        this.uniforms = 'blockSize : i32,';
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `depthToSpace_${dataFormat}`;
        this.dataFormat = dataFormat;
    }
    getUserCode() {
        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let h = ${this.getHeightCoordString()};
          let w = ${this.getWidthCoordString()};
          let d = ${this.getDepthCoordString()};

          let in_h = h / uniforms.blockSize;
          let offset_h = h % uniforms.blockSize;
          let in_w = w / uniforms.blockSize;
          let offset_w = w % uniforms.blockSize;
          let offset_d = (offset_h * uniforms.blockSize + offset_w) *
            ${this.getOutputDepthSize()};
          let in_d = d + offset_d;

          let rlt = ${this.getInputSamplingString()};
          setOutputAtIndex(index, rlt);
        }
      }`;
        return userCode;
    }
    getHeightCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[1]`;
        }
        else {
            return `coords[2]`;
        }
    }
    getWidthCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[2]`;
        }
        else {
            return `coords[3]`;
        }
    }
    getDepthCoordString() {
        if (this.dataFormat === 'NHWC') {
            return `coords[3]`;
        }
        else {
            return `coords[1]`;
        }
    }
    getOutputDepthSize() {
        if (this.dataFormat === 'NHWC') {
            return `uniforms.outShape[3]`;
        }
        else {
            return `uniforms.outShape[1]`;
        }
    }
    getInputSamplingString() {
        if (this.dataFormat === 'NHWC') {
            return `getX(b, in_h, in_w, in_d)`;
        }
        else {
            return `getX(b, in_d, in_h, in_w)`;
        }
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGhfdG9fc3BhY2Vfd2ViZ3B1LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvZGVwdGhfdG9fc3BhY2Vfd2ViZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxtQkFBbUIsSUFBSSxJQUFJLEVBQWdCLE1BQU0sa0JBQWtCLENBQUM7QUFDNUUsT0FBTyxFQUFDLGVBQWUsRUFBRSxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUVsRSxNQUFNLE9BQU8sbUJBQW1CO0lBVzlCLFlBQVksV0FBcUIsRUFBRSxVQUF5QjtRQVY1RCxrQkFBYSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7UUFNdEIsa0JBQWEsR0FBNkIsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3JELFNBQUksR0FBRyxJQUFJLENBQUM7UUFDWixhQUFRLEdBQUcsa0JBQWtCLENBQUM7UUFHNUIsSUFBSSxDQUFDLFdBQVcsR0FBRyxXQUFXLENBQUM7UUFDL0IsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFDL0QsSUFBSSxDQUFDLFNBQVMsR0FBRyxnQkFBZ0IsVUFBVSxFQUFFLENBQUM7UUFDOUMsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7SUFDL0IsQ0FBQztJQUVELFdBQVc7UUFDVCxNQUFNLFFBQVEsR0FBRztRQUNiLElBQUksQ0FBQyxPQUFPLENBQUM7Ozs7b0JBSUQsSUFBSSxDQUFDLG9CQUFvQixFQUFFO29CQUMzQixJQUFJLENBQUMsbUJBQW1CLEVBQUU7b0JBQzFCLElBQUksQ0FBQyxtQkFBbUIsRUFBRTs7Ozs7OztjQU9oQyxJQUFJLENBQUMsa0JBQWtCLEVBQUU7OztzQkFHakIsSUFBSSxDQUFDLHNCQUFzQixFQUFFOzs7UUFHM0MsQ0FBQztRQUNMLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7SUFFTyxvQkFBb0I7UUFDMUIsSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLE1BQU0sRUFBRTtZQUM5QixPQUFPLFdBQVcsQ0FBQztTQUNwQjthQUFNO1lBQ0wsT0FBTyxXQUFXLENBQUM7U0FDcEI7SUFDSCxDQUFDO0lBRU8sbUJBQW1CO1FBQ3pCLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxNQUFNLEVBQUU7WUFDOUIsT0FBTyxXQUFXLENBQUM7U0FDcEI7YUFBTTtZQUNMLE9BQU8sV0FBVyxDQUFDO1NBQ3BCO0lBQ0gsQ0FBQztJQUVPLG1CQUFtQjtRQUN6QixJQUFJLElBQUksQ0FBQyxVQUFVLEtBQUssTUFBTSxFQUFFO1lBQzlCLE9BQU8sV0FBVyxDQUFDO1NBQ3BCO2FBQU07WUFDTCxPQUFPLFdBQVcsQ0FBQztTQUNwQjtJQUNILENBQUM7SUFFTyxrQkFBa0I7UUFDeEIsSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLE1BQU0sRUFBRTtZQUM5QixPQUFPLHNCQUFzQixDQUFDO1NBQy9CO2FBQU07WUFDTCxPQUFPLHNCQUFzQixDQUFDO1NBQy9CO0lBQ0gsQ0FBQztJQUVPLHNCQUFzQjtRQUM1QixJQUFJLElBQUksQ0FBQyxVQUFVLEtBQUssTUFBTSxFQUFFO1lBQzlCLE9BQU8sMkJBQTJCLENBQUM7U0FDcEM7YUFBTTtZQUNMLE9BQU8sMkJBQTJCLENBQUM7U0FDcEM7SUFDSCxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Z2V0TWFpbkhlYWRlclN0cmluZyBhcyBtYWluLCBXZWJHUFVQcm9ncmFtfSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcbmltcG9ydCB7Y29tcHV0ZURpc3BhdGNoLCBmbGF0RGlzcGF0Y2hMYXlvdXR9IGZyb20gJy4vd2ViZ3B1X3V0aWwnO1xuXG5leHBvcnQgY2xhc3MgRGVwdGhUb1NwYWNlUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgZGF0YUZvcm1hdDogc3RyaW5nO1xuICBzaGFkZXJLZXk6IHN0cmluZztcbiAgZGlzcGF0Y2hMYXlvdXQ6IHt4OiBudW1iZXJbXX07XG4gIGRpc3BhdGNoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHdvcmtncm91cFNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFs2NCwgMSwgMV07XG4gIHNpemUgPSB0cnVlO1xuICB1bmlmb3JtcyA9ICdibG9ja1NpemUgOiBpMzIsJztcblxuICBjb25zdHJ1Y3RvcihvdXRwdXRTaGFwZTogbnVtYmVyW10sIGRhdGFGb3JtYXQ6ICdOSFdDJ3wnTkNIVycpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gb3V0cHV0U2hhcGU7XG4gICAgdGhpcy5kaXNwYXRjaExheW91dCA9IGZsYXREaXNwYXRjaExheW91dCh0aGlzLm91dHB1dFNoYXBlKTtcbiAgICB0aGlzLmRpc3BhdGNoID0gY29tcHV0ZURpc3BhdGNoKFxuICAgICAgICB0aGlzLmRpc3BhdGNoTGF5b3V0LCB0aGlzLm91dHB1dFNoYXBlLCB0aGlzLndvcmtncm91cFNpemUpO1xuICAgIHRoaXMuc2hhZGVyS2V5ID0gYGRlcHRoVG9TcGFjZV8ke2RhdGFGb3JtYXR9YDtcbiAgICB0aGlzLmRhdGFGb3JtYXQgPSBkYXRhRm9ybWF0O1xuICB9XG5cbiAgZ2V0VXNlckNvZGUoKTogc3RyaW5nIHtcbiAgICBjb25zdCB1c2VyQ29kZSA9IGBcbiAgICAgICR7bWFpbignaW5kZXgnKX0ge1xuICAgICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgICAgbGV0IGNvb3JkcyA9IGdldENvb3Jkc0Zyb21JbmRleChpbmRleCk7XG4gICAgICAgICAgbGV0IGIgPSBjb29yZHNbMF07XG4gICAgICAgICAgbGV0IGggPSAke3RoaXMuZ2V0SGVpZ2h0Q29vcmRTdHJpbmcoKX07XG4gICAgICAgICAgbGV0IHcgPSAke3RoaXMuZ2V0V2lkdGhDb29yZFN0cmluZygpfTtcbiAgICAgICAgICBsZXQgZCA9ICR7dGhpcy5nZXREZXB0aENvb3JkU3RyaW5nKCl9O1xuXG4gICAgICAgICAgbGV0IGluX2ggPSBoIC8gdW5pZm9ybXMuYmxvY2tTaXplO1xuICAgICAgICAgIGxldCBvZmZzZXRfaCA9IGggJSB1bmlmb3Jtcy5ibG9ja1NpemU7XG4gICAgICAgICAgbGV0IGluX3cgPSB3IC8gdW5pZm9ybXMuYmxvY2tTaXplO1xuICAgICAgICAgIGxldCBvZmZzZXRfdyA9IHcgJSB1bmlmb3Jtcy5ibG9ja1NpemU7XG4gICAgICAgICAgbGV0IG9mZnNldF9kID0gKG9mZnNldF9oICogdW5pZm9ybXMuYmxvY2tTaXplICsgb2Zmc2V0X3cpICpcbiAgICAgICAgICAgICR7dGhpcy5nZXRPdXRwdXREZXB0aFNpemUoKX07XG4gICAgICAgICAgbGV0IGluX2QgPSBkICsgb2Zmc2V0X2Q7XG5cbiAgICAgICAgICBsZXQgcmx0ID0gJHt0aGlzLmdldElucHV0U2FtcGxpbmdTdHJpbmcoKX07XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgcmx0KTtcbiAgICAgICAgfVxuICAgICAgfWA7XG4gICAgcmV0dXJuIHVzZXJDb2RlO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRIZWlnaHRDb29yZFN0cmluZygpOiBzdHJpbmcge1xuICAgIGlmICh0aGlzLmRhdGFGb3JtYXQgPT09ICdOSFdDJykge1xuICAgICAgcmV0dXJuIGBjb29yZHNbMV1gO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gYGNvb3Jkc1syXWA7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRXaWR0aENvb3JkU3RyaW5nKCk6IHN0cmluZyB7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ05IV0MnKSB7XG4gICAgICByZXR1cm4gYGNvb3Jkc1syXWA7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiBgY29vcmRzWzNdYDtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGdldERlcHRoQ29vcmRTdHJpbmcoKTogc3RyaW5nIHtcbiAgICBpZiAodGhpcy5kYXRhRm9ybWF0ID09PSAnTkhXQycpIHtcbiAgICAgIHJldHVybiBgY29vcmRzWzNdYDtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIGBjb29yZHNbMV1gO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZ2V0T3V0cHV0RGVwdGhTaXplKCk6IHN0cmluZyB7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ05IV0MnKSB7XG4gICAgICByZXR1cm4gYHVuaWZvcm1zLm91dFNoYXBlWzNdYDtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIGB1bmlmb3Jtcy5vdXRTaGFwZVsxXWA7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRJbnB1dFNhbXBsaW5nU3RyaW5nKCk6IHN0cmluZyB7XG4gICAgaWYgKHRoaXMuZGF0YUZvcm1hdCA9PT0gJ05IV0MnKSB7XG4gICAgICByZXR1cm4gYGdldFgoYiwgaW5faCwgaW5fdywgaW5fZClgO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gYGdldFgoYiwgaW5fZCwgaW5faCwgaW5fdylgO1xuICAgIH1cbiAgfVxufVxuIl19