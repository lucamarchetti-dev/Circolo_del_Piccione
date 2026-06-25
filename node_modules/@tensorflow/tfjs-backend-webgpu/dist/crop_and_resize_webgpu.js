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
export class CropAndResizeProgram {
    constructor(channnel, boxShape, cropSize, method) {
        this.variableNames = ['Image', 'Boxes', 'BoxInd'];
        this.uniforms = 'extrapolationValue : f32,';
        this.workgroupSize = [64, 1, 1];
        this.size = true;
        const [numBoxes,] = boxShape;
        this.outputShape = [numBoxes, cropSize[0], cropSize[1], channnel];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.methodId = method === 'bilinear' ? 1 : 0;
        this.cropHeightBiggerThan1 = this.outputShape[1] > 1;
        this.cropWidthBiggerThan1 = this.outputShape[2] > 1;
        this.shaderKey = `cropAndResize_${this.methodId}_${this.cropHeightBiggerThan1}_${this.cropWidthBiggerThan1}`;
    }
    getUserCode() {
        const [inputHeightFloat, inputWidthFloat] = [`f32(uniforms.imageShape[1] - 1)`, `f32(uniforms.imageShape[2] - 1)`];
        const [heightRatio, heightScale, inY] = this.cropHeightBiggerThan1 ?
            [
                `(${inputHeightFloat} / f32(uniforms.outShape[1] - 1))`,
                '(y2-y1) * height_ratio',
                `y1*${inputHeightFloat} + f32(y)*(height_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (y1+y2) * ${inputHeightFloat}`,
            ];
        const [widthRatio, widthScale, inX] = this.cropWidthBiggerThan1 ?
            [
                `(${inputWidthFloat} / f32(uniforms.outShape[2] - 1))`,
                '(x2-x1) * width_ratio',
                `x1*${inputWidthFloat} + f32(x)*(width_scale)`,
            ] :
            [
                '0.0',
                '0.0',
                `0.5 * (x1+x2) * ${inputWidthFloat}`,
            ];
        // Reference implementation
        // tslint:disable-next-line:max-line-length
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op_gpu.cu.cc
        const userCode = `
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        let height_ratio = f32(${heightRatio});
        let width_ratio = f32(${widthRatio});
        let b = coords[0];
        let y = coords[1];
        let x = coords[2];
        let d = coords[3];
        // get box vals
        let y1 = getBoxes(b, 0);
        let x1 = getBoxes(b, 1);
        let y2 = getBoxes(b, 2);
        let x2 = getBoxes(b, 3);
        // get image in batch index
        let bInd = i32(round(getBoxInd(b)));
        if(bInd < 0 || bInd >= uniforms.outShape[0]) {
          return;
        }
        let height_scale = ${heightScale};
        let width_scale = ${widthScale};
        let in_y = ${inY};
        if( in_y < 0.0 || in_y > ${inputHeightFloat} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let in_x = ${inX};
        if( in_x < 0.0 || in_x > ${inputWidthFloat} ) {
          setOutputAtIndex(index, uniforms.extrapolationValue);
          return;
        }
        let sourceFracIndexCR = vec2<f32>(in_x,in_y);
        if(${this.methodId} == 1) {
          // Compute the four integer indices.
          let sourceFloorCR = vec2<i32>(sourceFracIndexCR);
          let sourceCeilCR = vec2<i32>(ceil(sourceFracIndexCR));
          let topLeft = getImage(bInd, sourceFloorCR.y, sourceFloorCR.x, d);
          let bottomLeft = getImage(bInd, sourceCeilCR.y, sourceFloorCR.x, d);
          let topRight = getImage(bInd, sourceFloorCR.y, sourceCeilCR.x, d);
          let bottomRight = getImage(bInd, sourceCeilCR.y, sourceCeilCR.x, d);
          let fracCR = sourceFracIndexCR - vec2<f32>(sourceFloorCR);
          let top = topLeft + (topRight - topLeft) * fracCR.x;
          let bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          let newValue = top + (bottom - top) * fracCR.y;
          setOutputAtIndex(index, newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          let sourceNearestCR = vec2<i32>(floor(
            sourceFracIndexCR + vec2<f32>(0.5,0.5)));
          let newValue = getImage(
            bInd, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutputAtIndex(index, newValue);
        }
      }
    }
    `;
        return userCode;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY3JvcF9hbmRfcmVzaXplX3dlYmdwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2Nyb3BfYW5kX3Jlc2l6ZV93ZWJncHUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLG1CQUFtQixJQUFJLElBQUksRUFBZ0IsTUFBTSxrQkFBa0IsQ0FBQztBQUM1RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGtCQUFrQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRWxFLE1BQU0sT0FBTyxvQkFBb0I7SUFhL0IsWUFDSSxRQUFnQixFQUFFLFFBQTBCLEVBQUUsUUFBMEIsRUFDeEUsTUFBNEI7UUFWaEMsa0JBQWEsR0FBRyxDQUFDLE9BQU8sRUFBRSxPQUFPLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDN0MsYUFBUSxHQUFHLDJCQUEyQixDQUFDO1FBQ3ZDLGtCQUFhLEdBQTZCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUlyRCxTQUFJLEdBQUcsSUFBSSxDQUFDO1FBS1YsTUFBTSxDQUFDLFFBQVEsRUFBRyxHQUFHLFFBQVEsQ0FBQztRQUM5QixJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsUUFBUSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDbEUsSUFBSSxDQUFDLGNBQWMsR0FBRyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDM0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxlQUFlLENBQzNCLElBQUksQ0FBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7UUFFL0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxNQUFNLEtBQUssVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM5QyxJQUFJLENBQUMscUJBQXFCLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDckQsSUFBSSxDQUFDLG9CQUFvQixHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3BELElBQUksQ0FBQyxTQUFTLEdBQUcsaUJBQWlCLElBQUksQ0FBQyxRQUFRLElBQzNDLElBQUksQ0FBQyxxQkFBcUIsSUFBSSxJQUFJLENBQUMsb0JBQW9CLEVBQUUsQ0FBQztJQUNoRSxDQUFDO0lBRUQsV0FBVztRQUNULE1BQU0sQ0FBQyxnQkFBZ0IsRUFBRSxlQUFlLENBQUMsR0FDckMsQ0FBQyxpQ0FBaUMsRUFBRSxpQ0FBaUMsQ0FBQyxDQUFDO1FBRTNFLE1BQU0sQ0FBQyxXQUFXLEVBQUUsV0FBVyxFQUFFLEdBQUcsQ0FBQyxHQUFHLElBQUksQ0FBQyxxQkFBcUIsQ0FBQyxDQUFDO1lBQ2hFO2dCQUNFLElBQUksZ0JBQWdCLG1DQUFtQztnQkFDdkQsd0JBQXdCO2dCQUN4QixNQUFNLGdCQUFnQiwwQkFBMEI7YUFDakQsQ0FBQyxDQUFDO1lBQ0g7Z0JBQ0UsS0FBSztnQkFDTCxLQUFLO2dCQUNMLG1CQUFtQixnQkFBZ0IsRUFBRTthQUN0QyxDQUFDO1FBQ04sTUFBTSxDQUFDLFVBQVUsRUFBRSxVQUFVLEVBQUUsR0FBRyxDQUFDLEdBQUcsSUFBSSxDQUFDLG9CQUFvQixDQUFDLENBQUM7WUFDN0Q7Z0JBQ0UsSUFBSSxlQUFlLG1DQUFtQztnQkFDdEQsdUJBQXVCO2dCQUN2QixNQUFNLGVBQWUseUJBQXlCO2FBQy9DLENBQUMsQ0FBQztZQUNIO2dCQUNFLEtBQUs7Z0JBQ0wsS0FBSztnQkFDTCxtQkFBbUIsZUFBZSxFQUFFO2FBQ3JDLENBQUM7UUFFTiwyQkFBMkI7UUFDM0IsMkNBQTJDO1FBQzNDLDRHQUE0RztRQUM1RyxNQUFNLFFBQVEsR0FBRztNQUNmLElBQUksQ0FBQyxPQUFPLENBQUM7OztpQ0FHYyxXQUFXO2dDQUNaLFVBQVU7Ozs7Ozs7Ozs7Ozs7Ozs2QkFlYixXQUFXOzRCQUNaLFVBQVU7cUJBQ2pCLEdBQUc7bUNBQ1csZ0JBQWdCOzs7O3FCQUk5QixHQUFHO21DQUNXLGVBQWU7Ozs7O2FBS3JDLElBQUksQ0FBQyxRQUFROzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztLQXVCckIsQ0FBQztRQUNGLE9BQU8sUUFBUSxDQUFDO0lBQ2xCLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtnZXRNYWluSGVhZGVyU3RyaW5nIGFzIG1haW4sIFdlYkdQVVByb2dyYW19IGZyb20gJy4vd2ViZ3B1X3Byb2dyYW0nO1xuaW1wb3J0IHtjb21wdXRlRGlzcGF0Y2gsIGZsYXREaXNwYXRjaExheW91dH0gZnJvbSAnLi93ZWJncHVfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBDcm9wQW5kUmVzaXplUHJvZ3JhbSBpbXBsZW1lbnRzIFdlYkdQVVByb2dyYW0ge1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHNoYWRlcktleTogc3RyaW5nO1xuICBkaXNwYXRjaExheW91dDoge3g6IG51bWJlcltdfTtcbiAgZGlzcGF0Y2g6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgdmFyaWFibGVOYW1lcyA9IFsnSW1hZ2UnLCAnQm94ZXMnLCAnQm94SW5kJ107XG4gIHVuaWZvcm1zID0gJ2V4dHJhcG9sYXRpb25WYWx1ZSA6IGYzMiwnO1xuICB3b3JrZ3JvdXBTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbNjQsIDEsIDFdO1xuICBtZXRob2RJZDogbnVtYmVyO1xuICBjcm9wSGVpZ2h0QmlnZ2VyVGhhbjE6IGJvb2xlYW47XG4gIGNyb3BXaWR0aEJpZ2dlclRoYW4xOiBib29sZWFuO1xuICBzaXplID0gdHJ1ZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIGNoYW5ubmVsOiBudW1iZXIsIGJveFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCBjcm9wU2l6ZTogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIG1ldGhvZDogJ2JpbGluZWFyJ3wnbmVhcmVzdCcpIHtcbiAgICBjb25zdCBbbnVtQm94ZXMsIF0gPSBib3hTaGFwZTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW251bUJveGVzLCBjcm9wU2l6ZVswXSwgY3JvcFNpemVbMV0sIGNoYW5ubmVsXTtcbiAgICB0aGlzLmRpc3BhdGNoTGF5b3V0ID0gZmxhdERpc3BhdGNoTGF5b3V0KHRoaXMub3V0cHV0U2hhcGUpO1xuICAgIHRoaXMuZGlzcGF0Y2ggPSBjb21wdXRlRGlzcGF0Y2goXG4gICAgICAgIHRoaXMuZGlzcGF0Y2hMYXlvdXQsIHRoaXMub3V0cHV0U2hhcGUsIHRoaXMud29ya2dyb3VwU2l6ZSk7XG5cbiAgICB0aGlzLm1ldGhvZElkID0gbWV0aG9kID09PSAnYmlsaW5lYXInID8gMSA6IDA7XG4gICAgdGhpcy5jcm9wSGVpZ2h0QmlnZ2VyVGhhbjEgPSB0aGlzLm91dHB1dFNoYXBlWzFdID4gMTtcbiAgICB0aGlzLmNyb3BXaWR0aEJpZ2dlclRoYW4xID0gdGhpcy5vdXRwdXRTaGFwZVsyXSA+IDE7XG4gICAgdGhpcy5zaGFkZXJLZXkgPSBgY3JvcEFuZFJlc2l6ZV8ke3RoaXMubWV0aG9kSWR9XyR7XG4gICAgICAgIHRoaXMuY3JvcEhlaWdodEJpZ2dlclRoYW4xfV8ke3RoaXMuY3JvcFdpZHRoQmlnZ2VyVGhhbjF9YDtcbiAgfVxuXG4gIGdldFVzZXJDb2RlKCk6IHN0cmluZyB7XG4gICAgY29uc3QgW2lucHV0SGVpZ2h0RmxvYXQsIGlucHV0V2lkdGhGbG9hdF0gPVxuICAgICAgICBbYGYzMih1bmlmb3Jtcy5pbWFnZVNoYXBlWzFdIC0gMSlgLCBgZjMyKHVuaWZvcm1zLmltYWdlU2hhcGVbMl0gLSAxKWBdO1xuXG4gICAgY29uc3QgW2hlaWdodFJhdGlvLCBoZWlnaHRTY2FsZSwgaW5ZXSA9IHRoaXMuY3JvcEhlaWdodEJpZ2dlclRoYW4xID9cbiAgICAgICAgW1xuICAgICAgICAgIGAoJHtpbnB1dEhlaWdodEZsb2F0fSAvIGYzMih1bmlmb3Jtcy5vdXRTaGFwZVsxXSAtIDEpKWAsXG4gICAgICAgICAgJyh5Mi15MSkgKiBoZWlnaHRfcmF0aW8nLFxuICAgICAgICAgIGB5MSoke2lucHV0SGVpZ2h0RmxvYXR9ICsgZjMyKHkpKihoZWlnaHRfc2NhbGUpYCxcbiAgICAgICAgXSA6XG4gICAgICAgIFtcbiAgICAgICAgICAnMC4wJyxcbiAgICAgICAgICAnMC4wJyxcbiAgICAgICAgICBgMC41ICogKHkxK3kyKSAqICR7aW5wdXRIZWlnaHRGbG9hdH1gLFxuICAgICAgICBdO1xuICAgIGNvbnN0IFt3aWR0aFJhdGlvLCB3aWR0aFNjYWxlLCBpblhdID0gdGhpcy5jcm9wV2lkdGhCaWdnZXJUaGFuMSA/XG4gICAgICAgIFtcbiAgICAgICAgICBgKCR7aW5wdXRXaWR0aEZsb2F0fSAvIGYzMih1bmlmb3Jtcy5vdXRTaGFwZVsyXSAtIDEpKWAsXG4gICAgICAgICAgJyh4Mi14MSkgKiB3aWR0aF9yYXRpbycsXG4gICAgICAgICAgYHgxKiR7aW5wdXRXaWR0aEZsb2F0fSArIGYzMih4KSood2lkdGhfc2NhbGUpYCxcbiAgICAgICAgXSA6XG4gICAgICAgIFtcbiAgICAgICAgICAnMC4wJyxcbiAgICAgICAgICAnMC4wJyxcbiAgICAgICAgICBgMC41ICogKHgxK3gyKSAqICR7aW5wdXRXaWR0aEZsb2F0fWAsXG4gICAgICAgIF07XG5cbiAgICAvLyBSZWZlcmVuY2UgaW1wbGVtZW50YXRpb25cbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bWF4LWxpbmUtbGVuZ3RoXG4gICAgLy8gaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGVuc29yZmxvdy9ibG9iL21hc3Rlci90ZW5zb3JmbG93L2NvcmUva2VybmVscy9jcm9wX2FuZF9yZXNpemVfb3BfZ3B1LmN1LmNjXG4gICAgY29uc3QgdXNlckNvZGUgPSBgXG4gICAgJHttYWluKCdpbmRleCcpfSB7XG4gICAgICBpZiAoaW5kZXggPCB1bmlmb3Jtcy5zaXplKSB7XG4gICAgICAgIGxldCBjb29yZHMgPSBnZXRDb29yZHNGcm9tSW5kZXgoaW5kZXgpO1xuICAgICAgICBsZXQgaGVpZ2h0X3JhdGlvID0gZjMyKCR7aGVpZ2h0UmF0aW99KTtcbiAgICAgICAgbGV0IHdpZHRoX3JhdGlvID0gZjMyKCR7d2lkdGhSYXRpb30pO1xuICAgICAgICBsZXQgYiA9IGNvb3Jkc1swXTtcbiAgICAgICAgbGV0IHkgPSBjb29yZHNbMV07XG4gICAgICAgIGxldCB4ID0gY29vcmRzWzJdO1xuICAgICAgICBsZXQgZCA9IGNvb3Jkc1szXTtcbiAgICAgICAgLy8gZ2V0IGJveCB2YWxzXG4gICAgICAgIGxldCB5MSA9IGdldEJveGVzKGIsIDApO1xuICAgICAgICBsZXQgeDEgPSBnZXRCb3hlcyhiLCAxKTtcbiAgICAgICAgbGV0IHkyID0gZ2V0Qm94ZXMoYiwgMik7XG4gICAgICAgIGxldCB4MiA9IGdldEJveGVzKGIsIDMpO1xuICAgICAgICAvLyBnZXQgaW1hZ2UgaW4gYmF0Y2ggaW5kZXhcbiAgICAgICAgbGV0IGJJbmQgPSBpMzIocm91bmQoZ2V0Qm94SW5kKGIpKSk7XG4gICAgICAgIGlmKGJJbmQgPCAwIHx8IGJJbmQgPj0gdW5pZm9ybXMub3V0U2hhcGVbMF0pIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgbGV0IGhlaWdodF9zY2FsZSA9ICR7aGVpZ2h0U2NhbGV9O1xuICAgICAgICBsZXQgd2lkdGhfc2NhbGUgPSAke3dpZHRoU2NhbGV9O1xuICAgICAgICBsZXQgaW5feSA9ICR7aW5ZfTtcbiAgICAgICAgaWYoIGluX3kgPCAwLjAgfHwgaW5feSA+ICR7aW5wdXRIZWlnaHRGbG9hdH0gKSB7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgdW5pZm9ybXMuZXh0cmFwb2xhdGlvblZhbHVlKTtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgbGV0IGluX3ggPSAke2luWH07XG4gICAgICAgIGlmKCBpbl94IDwgMC4wIHx8IGluX3ggPiAke2lucHV0V2lkdGhGbG9hdH0gKSB7XG4gICAgICAgICAgc2V0T3V0cHV0QXRJbmRleChpbmRleCwgdW5pZm9ybXMuZXh0cmFwb2xhdGlvblZhbHVlKTtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgbGV0IHNvdXJjZUZyYWNJbmRleENSID0gdmVjMjxmMzI+KGluX3gsaW5feSk7XG4gICAgICAgIGlmKCR7dGhpcy5tZXRob2RJZH0gPT0gMSkge1xuICAgICAgICAgIC8vIENvbXB1dGUgdGhlIGZvdXIgaW50ZWdlciBpbmRpY2VzLlxuICAgICAgICAgIGxldCBzb3VyY2VGbG9vckNSID0gdmVjMjxpMzI+KHNvdXJjZUZyYWNJbmRleENSKTtcbiAgICAgICAgICBsZXQgc291cmNlQ2VpbENSID0gdmVjMjxpMzI+KGNlaWwoc291cmNlRnJhY0luZGV4Q1IpKTtcbiAgICAgICAgICBsZXQgdG9wTGVmdCA9IGdldEltYWdlKGJJbmQsIHNvdXJjZUZsb29yQ1IueSwgc291cmNlRmxvb3JDUi54LCBkKTtcbiAgICAgICAgICBsZXQgYm90dG9tTGVmdCA9IGdldEltYWdlKGJJbmQsIHNvdXJjZUNlaWxDUi55LCBzb3VyY2VGbG9vckNSLngsIGQpO1xuICAgICAgICAgIGxldCB0b3BSaWdodCA9IGdldEltYWdlKGJJbmQsIHNvdXJjZUZsb29yQ1IueSwgc291cmNlQ2VpbENSLngsIGQpO1xuICAgICAgICAgIGxldCBib3R0b21SaWdodCA9IGdldEltYWdlKGJJbmQsIHNvdXJjZUNlaWxDUi55LCBzb3VyY2VDZWlsQ1IueCwgZCk7XG4gICAgICAgICAgbGV0IGZyYWNDUiA9IHNvdXJjZUZyYWNJbmRleENSIC0gdmVjMjxmMzI+KHNvdXJjZUZsb29yQ1IpO1xuICAgICAgICAgIGxldCB0b3AgPSB0b3BMZWZ0ICsgKHRvcFJpZ2h0IC0gdG9wTGVmdCkgKiBmcmFjQ1IueDtcbiAgICAgICAgICBsZXQgYm90dG9tID0gYm90dG9tTGVmdCArIChib3R0b21SaWdodCAtIGJvdHRvbUxlZnQpICogZnJhY0NSLng7XG4gICAgICAgICAgbGV0IG5ld1ZhbHVlID0gdG9wICsgKGJvdHRvbSAtIHRvcCkgKiBmcmFjQ1IueTtcbiAgICAgICAgICBzZXRPdXRwdXRBdEluZGV4KGluZGV4LCBuZXdWYWx1ZSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgLy8gQ29tcHV0ZSB0aGUgY29vcmRpbmF0b3JzIG9mIG5lYXJlc3QgbmVpZ2hib3IgcG9pbnQuXG4gICAgICAgICAgbGV0IHNvdXJjZU5lYXJlc3RDUiA9IHZlYzI8aTMyPihmbG9vcihcbiAgICAgICAgICAgIHNvdXJjZUZyYWNJbmRleENSICsgdmVjMjxmMzI+KDAuNSwwLjUpKSk7XG4gICAgICAgICAgbGV0IG5ld1ZhbHVlID0gZ2V0SW1hZ2UoXG4gICAgICAgICAgICBiSW5kLCBzb3VyY2VOZWFyZXN0Q1IueSwgc291cmNlTmVhcmVzdENSLngsIGQpO1xuICAgICAgICAgIHNldE91dHB1dEF0SW5kZXgoaW5kZXgsIG5ld1ZhbHVlKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICBgO1xuICAgIHJldHVybiB1c2VyQ29kZTtcbiAgfVxufVxuIl19