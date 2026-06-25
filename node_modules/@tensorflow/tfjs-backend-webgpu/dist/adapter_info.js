/**
 * @license
 * Copyright 2022 Google LLC.
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
export class AdapterInfo {
    constructor(adapterInfo) {
        if (adapterInfo) {
            this.vendor = adapterInfo.vendor;
            this.architecture = adapterInfo.architecture;
            this.intelGPUGeneration = this.getIntelGPUGeneration();
        }
    }
    getIntelGPUGeneration() {
        if (this.isIntel()) {
            if (this.architecture.startsWith('gen')) {
                return Number(this.architecture.match(/\d+/));
            }
            else if (this.architecture.startsWith('xe')) {
                return 12;
            }
        }
        return 0;
    }
    isIntel() {
        return this.vendor === 'intel';
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRhcHRlcl9pbmZvLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvYWRhcHRlcl9pbmZvLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE1BQU0sT0FBTyxXQUFXO0lBS3RCLFlBQVksV0FBMkI7UUFDckMsSUFBSSxXQUFXLEVBQUU7WUFDZixJQUFJLENBQUMsTUFBTSxHQUFHLFdBQVcsQ0FBQyxNQUFNLENBQUM7WUFDakMsSUFBSSxDQUFDLFlBQVksR0FBRyxXQUFXLENBQUMsWUFBWSxDQUFDO1lBQzdDLElBQUksQ0FBQyxrQkFBa0IsR0FBRyxJQUFJLENBQUMscUJBQXFCLEVBQUUsQ0FBQztTQUN4RDtJQUNILENBQUM7SUFFTyxxQkFBcUI7UUFDM0IsSUFBSSxJQUFJLENBQUMsT0FBTyxFQUFFLEVBQUU7WUFDbEIsSUFBSSxJQUFJLENBQUMsWUFBWSxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsRUFBRTtnQkFDdkMsT0FBTyxNQUFNLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQzthQUMvQztpQkFBTSxJQUFJLElBQUksQ0FBQyxZQUFZLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBQyxFQUFFO2dCQUM3QyxPQUFPLEVBQUUsQ0FBQzthQUNYO1NBQ0Y7UUFDRCxPQUFPLENBQUMsQ0FBQztJQUNYLENBQUM7SUFFRCxPQUFPO1FBQ0wsT0FBTyxJQUFJLENBQUMsTUFBTSxLQUFLLE9BQU8sQ0FBQztJQUNqQyxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmV4cG9ydCBjbGFzcyBBZGFwdGVySW5mbyB7XG4gIHByaXZhdGUgdmVuZG9yOiBzdHJpbmc7XG4gIHByaXZhdGUgYXJjaGl0ZWN0dXJlOiBzdHJpbmc7XG4gIHB1YmxpYyBpbnRlbEdQVUdlbmVyYXRpb246IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihhZGFwdGVySW5mbzogR1BVQWRhcHRlckluZm8pIHtcbiAgICBpZiAoYWRhcHRlckluZm8pIHtcbiAgICAgIHRoaXMudmVuZG9yID0gYWRhcHRlckluZm8udmVuZG9yO1xuICAgICAgdGhpcy5hcmNoaXRlY3R1cmUgPSBhZGFwdGVySW5mby5hcmNoaXRlY3R1cmU7XG4gICAgICB0aGlzLmludGVsR1BVR2VuZXJhdGlvbiA9IHRoaXMuZ2V0SW50ZWxHUFVHZW5lcmF0aW9uKCk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRJbnRlbEdQVUdlbmVyYXRpb24oKSB7XG4gICAgaWYgKHRoaXMuaXNJbnRlbCgpKSB7XG4gICAgICBpZiAodGhpcy5hcmNoaXRlY3R1cmUuc3RhcnRzV2l0aCgnZ2VuJykpIHtcbiAgICAgICAgcmV0dXJuIE51bWJlcih0aGlzLmFyY2hpdGVjdHVyZS5tYXRjaCgvXFxkKy8pKTtcbiAgICAgIH0gZWxzZSBpZiAodGhpcy5hcmNoaXRlY3R1cmUuc3RhcnRzV2l0aCgneGUnKSkge1xuICAgICAgICByZXR1cm4gMTI7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiAwO1xuICB9XG5cbiAgaXNJbnRlbCgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy52ZW5kb3IgPT09ICdpbnRlbCc7XG4gIH1cbn1cbiJdfQ==