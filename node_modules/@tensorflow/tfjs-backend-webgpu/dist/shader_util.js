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
// Generates WGSL that computes strides.
export function symbolicallyComputeStrides(indicesArr, variableName) {
    if (Math.max(...indicesArr) > 5) {
        throw new Error('Cannot symbolically compute strides for rank > 6 tensor.');
    }
    const numCoords = indicesArr.length;
    const indicesStr = 'xyzwuv';
    const shape = indicesArr.map(d => `${variableName}.${indicesStr[d]}`);
    const strides = new Array(numCoords - 1);
    strides[numCoords - 2] = shape[numCoords - 1];
    for (let i = numCoords - 3; i >= 0; --i) {
        strides[i] = `(${strides[i + 1]} * ${shape[i + 1]})`;
    }
    return strides;
}
export const atomicAddSnippet = (ptr, v, type) => {
    if (type === 'int32') {
        return `atomicAdd(${ptr}, bitcast<i32>(${v}));`;
    }
    else {
        // atomicAdd only supports uint/int type. For float, we use
        // atomicCompareExchangeWeak to simulate.
        return `
          {
            var oldValue = 0;
            loop {
              let newValueF32 = bitcast<f32>(oldValue) + (${v});
              let newValue = bitcast<i32>(newValueF32);
              let res = atomicCompareExchangeWeak(${ptr}, oldValue, newValue);
              if res.exchanged {
                break;
              }
              oldValue = res.old_value;
            }
          }`;
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2hhZGVyX3V0aWwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9zaGFkZXJfdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCx3Q0FBd0M7QUFDeEMsTUFBTSxVQUFVLDBCQUEwQixDQUN0QyxVQUFvQixFQUFFLFlBQW9CO0lBQzVDLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxHQUFHLENBQUMsRUFBRTtRQUMvQixNQUFNLElBQUksS0FBSyxDQUFDLDBEQUEwRCxDQUFDLENBQUM7S0FDN0U7SUFFRCxNQUFNLFNBQVMsR0FBRyxVQUFVLENBQUMsTUFBTSxDQUFDO0lBQ3BDLE1BQU0sVUFBVSxHQUFHLFFBQVEsQ0FBQztJQUM1QixNQUFNLEtBQUssR0FBRyxVQUFVLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxZQUFZLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUN0RSxNQUFNLE9BQU8sR0FBRyxJQUFJLEtBQUssQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDekMsT0FBTyxDQUFDLFNBQVMsR0FBRyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUMsU0FBUyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQzlDLEtBQUssSUFBSSxDQUFDLEdBQUcsU0FBUyxHQUFHLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFO1FBQ3ZDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLE9BQU8sQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLE1BQU0sS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDO0tBQ3REO0lBRUQsT0FBTyxPQUFPLENBQUM7QUFDakIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGdCQUFnQixHQUN6QixDQUFDLEdBQVcsRUFBRSxDQUFTLEVBQUUsSUFBdUIsRUFBRSxFQUFFO0lBQ2xELElBQUksSUFBSSxLQUFLLE9BQU8sRUFBRTtRQUNwQixPQUFPLGFBQWEsR0FBRyxrQkFBa0IsQ0FBQyxLQUFLLENBQUM7S0FDakQ7U0FBTTtRQUNMLDJEQUEyRDtRQUMzRCx5Q0FBeUM7UUFDekMsT0FBTzs7Ozs0REFJNkMsQ0FBQzs7b0RBRVQsR0FBRzs7Ozs7O1lBTTNDLENBQUM7S0FDTjtBQUNILENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLy8gR2VuZXJhdGVzIFdHU0wgdGhhdCBjb21wdXRlcyBzdHJpZGVzLlxuZXhwb3J0IGZ1bmN0aW9uIHN5bWJvbGljYWxseUNvbXB1dGVTdHJpZGVzKFxuICAgIGluZGljZXNBcnI6IG51bWJlcltdLCB2YXJpYWJsZU5hbWU6IHN0cmluZyk6IHN0cmluZ1tdIHtcbiAgaWYgKE1hdGgubWF4KC4uLmluZGljZXNBcnIpID4gNSkge1xuICAgIHRocm93IG5ldyBFcnJvcignQ2Fubm90IHN5bWJvbGljYWxseSBjb21wdXRlIHN0cmlkZXMgZm9yIHJhbmsgPiA2IHRlbnNvci4nKTtcbiAgfVxuXG4gIGNvbnN0IG51bUNvb3JkcyA9IGluZGljZXNBcnIubGVuZ3RoO1xuICBjb25zdCBpbmRpY2VzU3RyID0gJ3h5end1dic7XG4gIGNvbnN0IHNoYXBlID0gaW5kaWNlc0Fyci5tYXAoZCA9PiBgJHt2YXJpYWJsZU5hbWV9LiR7aW5kaWNlc1N0cltkXX1gKTtcbiAgY29uc3Qgc3RyaWRlcyA9IG5ldyBBcnJheShudW1Db29yZHMgLSAxKTtcbiAgc3RyaWRlc1tudW1Db29yZHMgLSAyXSA9IHNoYXBlW251bUNvb3JkcyAtIDFdO1xuICBmb3IgKGxldCBpID0gbnVtQ29vcmRzIC0gMzsgaSA+PSAwOyAtLWkpIHtcbiAgICBzdHJpZGVzW2ldID0gYCgke3N0cmlkZXNbaSArIDFdfSAqICR7c2hhcGVbaSArIDFdfSlgO1xuICB9XG5cbiAgcmV0dXJuIHN0cmlkZXM7XG59XG5cbmV4cG9ydCBjb25zdCBhdG9taWNBZGRTbmlwcGV0ID1cbiAgICAocHRyOiBzdHJpbmcsIHY6IHN0cmluZywgdHlwZTogJ2ludDMyJ3wnZmxvYXQzMicpID0+IHtcbiAgICAgIGlmICh0eXBlID09PSAnaW50MzInKSB7XG4gICAgICAgIHJldHVybiBgYXRvbWljQWRkKCR7cHRyfSwgYml0Y2FzdDxpMzI+KCR7dn0pKTtgO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLy8gYXRvbWljQWRkIG9ubHkgc3VwcG9ydHMgdWludC9pbnQgdHlwZS4gRm9yIGZsb2F0LCB3ZSB1c2VcbiAgICAgICAgLy8gYXRvbWljQ29tcGFyZUV4Y2hhbmdlV2VhayB0byBzaW11bGF0ZS5cbiAgICAgICAgcmV0dXJuIGBcbiAgICAgICAgICB7XG4gICAgICAgICAgICB2YXIgb2xkVmFsdWUgPSAwO1xuICAgICAgICAgICAgbG9vcCB7XG4gICAgICAgICAgICAgIGxldCBuZXdWYWx1ZUYzMiA9IGJpdGNhc3Q8ZjMyPihvbGRWYWx1ZSkgKyAoJHt2fSk7XG4gICAgICAgICAgICAgIGxldCBuZXdWYWx1ZSA9IGJpdGNhc3Q8aTMyPihuZXdWYWx1ZUYzMik7XG4gICAgICAgICAgICAgIGxldCByZXMgPSBhdG9taWNDb21wYXJlRXhjaGFuZ2VXZWFrKCR7cHRyfSwgb2xkVmFsdWUsIG5ld1ZhbHVlKTtcbiAgICAgICAgICAgICAgaWYgcmVzLmV4Y2hhbmdlZCB7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgb2xkVmFsdWUgPSByZXMub2xkX3ZhbHVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1gO1xuICAgICAgfVxuICAgIH07XG4iXX0=