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
import { BinaryOpType, getBinaryOpString } from './binary_op_util';
import { getUnaryOpString, UnaryOpType } from './unary_op_util';
import { typeSnippet } from './webgpu_program';
export function activationFnSnippet(activation, hasPreluActivationWeights = false, packed = false, coordsLength = 3) {
    if (activation === null) {
        return '';
    }
    let activationOpSnippet = '';
    if (activation === 'linear') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LINEAR);
    }
    else if (activation === 'relu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU, packed);
    }
    else if (activation === 'elu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.ELU, packed);
    }
    else if (activation === 'relu6') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU6, packed);
    }
    else if (activation === 'prelu') {
        activationOpSnippet = getBinaryOpString(BinaryOpType.PRELU, packed);
    }
    else if (activation === 'sigmoid') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.SIGMOID, packed);
    }
    else if (activation === 'leakyrelu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LEAKYRELU, packed);
    }
    else {
        throw new Error(`Activation ${activation} has not been implemented for the WebGPU backend.`);
    }
    const elementSize = packed ? 4 : 1;
    const dataType = typeSnippet(elementSize);
    let activationFnSnippet = '';
    if (hasPreluActivationWeights) {
        activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${activationOpSnippet}
      }`;
    }
    else {
        activationFnSnippet = `
      fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
        ${activationOpSnippet}
      }`;
    }
    return activationFnSnippet;
}
export function biasActivationSnippet(hasBias, activation) {
    return `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWN0aXZhdGlvbl91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMvYWN0aXZhdGlvbl91dGlsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUlILE9BQU8sRUFBQyxZQUFZLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUNqRSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUUsV0FBVyxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDOUQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBRTdDLE1BQU0sVUFBVSxtQkFBbUIsQ0FDL0IsVUFBbUMsRUFBRSx5QkFBeUIsR0FBRyxLQUFLLEVBQ3RFLE1BQU0sR0FBRyxLQUFLLEVBQUUsWUFBWSxHQUFHLENBQUM7SUFDbEMsSUFBSSxVQUFVLEtBQUssSUFBSSxFQUFFO1FBQ3ZCLE9BQU8sRUFBRSxDQUFDO0tBQ1g7SUFFRCxJQUFJLG1CQUFtQixHQUFHLEVBQUUsQ0FBQztJQUM3QixJQUFJLFVBQVUsS0FBSyxRQUFRLEVBQUU7UUFDM0IsbUJBQW1CLEdBQUcsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0tBQzVEO1NBQU0sSUFBSSxVQUFVLEtBQUssTUFBTSxFQUFFO1FBQ2hDLG1CQUFtQixHQUFHLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxJQUFJLEVBQUUsTUFBTSxDQUFDLENBQUM7S0FDbEU7U0FBTSxJQUFJLFVBQVUsS0FBSyxLQUFLLEVBQUU7UUFDL0IsbUJBQW1CLEdBQUcsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLEdBQUcsRUFBRSxNQUFNLENBQUMsQ0FBQztLQUNqRTtTQUFNLElBQUksVUFBVSxLQUFLLE9BQU8sRUFBRTtRQUNqQyxtQkFBbUIsR0FBRyxnQkFBZ0IsQ0FBQyxXQUFXLENBQUMsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0tBQ25FO1NBQU0sSUFBSSxVQUFVLEtBQUssT0FBTyxFQUFFO1FBQ2pDLG1CQUFtQixHQUFHLGlCQUFpQixDQUFDLFlBQVksQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7S0FDckU7U0FBTSxJQUFJLFVBQVUsS0FBSyxTQUFTLEVBQUU7UUFDbkMsbUJBQW1CLEdBQUcsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQztLQUNyRTtTQUFNLElBQUksVUFBVSxLQUFLLFdBQVcsRUFBRTtRQUNyQyxtQkFBbUIsR0FBRyxnQkFBZ0IsQ0FBQyxXQUFXLENBQUMsU0FBUyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0tBQ3ZFO1NBQU07UUFDTCxNQUFNLElBQUksS0FBSyxDQUFDLGNBQ1osVUFBVSxtREFBbUQsQ0FBQyxDQUFDO0tBQ3BFO0lBQ0QsTUFBTSxXQUFXLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNuQyxNQUFNLFFBQVEsR0FBRyxXQUFXLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDMUMsSUFBSSxtQkFBbUIsR0FBRyxFQUFFLENBQUM7SUFDN0IsSUFBSSx5QkFBeUIsRUFBRTtRQUM3QixtQkFBbUIsR0FBRzswQkFDQSxRQUFRLGlCQUFpQixZQUFZLGFBQ3ZELFFBQVE7O1VBRU4sbUJBQW1CO1FBQ3JCLENBQUM7S0FDTjtTQUFNO1FBQ0wsbUJBQW1CLEdBQUc7MEJBQ0EsUUFBUSxpQkFBaUIsWUFBWSxhQUN2RCxRQUFRO1VBQ04sbUJBQW1CO1FBQ3JCLENBQUM7S0FDTjtJQUNELE9BQU8sbUJBQW1CLENBQUM7QUFDN0IsQ0FBQztBQUVELE1BQU0sVUFBVSxxQkFBcUIsQ0FDakMsT0FBZ0IsRUFBRSxVQUFtQztJQUN2RCxPQUFPO1FBQ0QsT0FBTyxDQUFDLENBQUMsQ0FBQyxnREFBZ0QsQ0FBQyxDQUFDLENBQUMsRUFBRTtRQUMvRCxVQUFVLENBQUMsQ0FBQyxDQUFDLG9DQUFvQyxDQUFDLENBQUMsQ0FBQyxFQUFFO09BQ3ZELENBQUM7QUFDUixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCaW5hcnlPcFR5cGUsIGdldEJpbmFyeU9wU3RyaW5nfSBmcm9tICcuL2JpbmFyeV9vcF91dGlsJztcbmltcG9ydCB7Z2V0VW5hcnlPcFN0cmluZywgVW5hcnlPcFR5cGV9IGZyb20gJy4vdW5hcnlfb3BfdXRpbCc7XG5pbXBvcnQge3R5cGVTbmlwcGV0fSBmcm9tICcuL3dlYmdwdV9wcm9ncmFtJztcblxuZXhwb3J0IGZ1bmN0aW9uIGFjdGl2YXRpb25GblNuaXBwZXQoXG4gICAgYWN0aXZhdGlvbjogYmFja2VuZF91dGlsLkFjdGl2YXRpb24sIGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBmYWxzZSxcbiAgICBwYWNrZWQgPSBmYWxzZSwgY29vcmRzTGVuZ3RoID0gMyk6IHN0cmluZyB7XG4gIGlmIChhY3RpdmF0aW9uID09PSBudWxsKSB7XG4gICAgcmV0dXJuICcnO1xuICB9XG5cbiAgbGV0IGFjdGl2YXRpb25PcFNuaXBwZXQgPSAnJztcbiAgaWYgKGFjdGl2YXRpb24gPT09ICdsaW5lYXInKSB7XG4gICAgYWN0aXZhdGlvbk9wU25pcHBldCA9IGdldFVuYXJ5T3BTdHJpbmcoVW5hcnlPcFR5cGUuTElORUFSKTtcbiAgfSBlbHNlIGlmIChhY3RpdmF0aW9uID09PSAncmVsdScpIHtcbiAgICBhY3RpdmF0aW9uT3BTbmlwcGV0ID0gZ2V0VW5hcnlPcFN0cmluZyhVbmFyeU9wVHlwZS5SRUxVLCBwYWNrZWQpO1xuICB9IGVsc2UgaWYgKGFjdGl2YXRpb24gPT09ICdlbHUnKSB7XG4gICAgYWN0aXZhdGlvbk9wU25pcHBldCA9IGdldFVuYXJ5T3BTdHJpbmcoVW5hcnlPcFR5cGUuRUxVLCBwYWNrZWQpO1xuICB9IGVsc2UgaWYgKGFjdGl2YXRpb24gPT09ICdyZWx1NicpIHtcbiAgICBhY3RpdmF0aW9uT3BTbmlwcGV0ID0gZ2V0VW5hcnlPcFN0cmluZyhVbmFyeU9wVHlwZS5SRUxVNiwgcGFja2VkKTtcbiAgfSBlbHNlIGlmIChhY3RpdmF0aW9uID09PSAncHJlbHUnKSB7XG4gICAgYWN0aXZhdGlvbk9wU25pcHBldCA9IGdldEJpbmFyeU9wU3RyaW5nKEJpbmFyeU9wVHlwZS5QUkVMVSwgcGFja2VkKTtcbiAgfSBlbHNlIGlmIChhY3RpdmF0aW9uID09PSAnc2lnbW9pZCcpIHtcbiAgICBhY3RpdmF0aW9uT3BTbmlwcGV0ID0gZ2V0VW5hcnlPcFN0cmluZyhVbmFyeU9wVHlwZS5TSUdNT0lELCBwYWNrZWQpO1xuICB9IGVsc2UgaWYgKGFjdGl2YXRpb24gPT09ICdsZWFreXJlbHUnKSB7XG4gICAgYWN0aXZhdGlvbk9wU25pcHBldCA9IGdldFVuYXJ5T3BTdHJpbmcoVW5hcnlPcFR5cGUuTEVBS1lSRUxVLCBwYWNrZWQpO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgQWN0aXZhdGlvbiAke1xuICAgICAgICBhY3RpdmF0aW9ufSBoYXMgbm90IGJlZW4gaW1wbGVtZW50ZWQgZm9yIHRoZSBXZWJHUFUgYmFja2VuZC5gKTtcbiAgfVxuICBjb25zdCBlbGVtZW50U2l6ZSA9IHBhY2tlZCA/IDQgOiAxO1xuICBjb25zdCBkYXRhVHlwZSA9IHR5cGVTbmlwcGV0KGVsZW1lbnRTaXplKTtcbiAgbGV0IGFjdGl2YXRpb25GblNuaXBwZXQgPSAnJztcbiAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMpIHtcbiAgICBhY3RpdmF0aW9uRm5TbmlwcGV0ID0gYFxuICAgICAgZm4gYWN0aXZhdGlvbihhIDogJHtkYXRhVHlwZX0sIGNvb3JkcyA6IHZlYyR7Y29vcmRzTGVuZ3RofTxpMzI+KSAtPiAke1xuICAgICAgICBkYXRhVHlwZX0ge1xuICAgICAgICBsZXQgYiA9IGdldFByZWx1QWN0aXZhdGlvbldlaWdodHNCeU91dHB1dENvb3Jkcyhjb29yZHMpO1xuICAgICAgICAke2FjdGl2YXRpb25PcFNuaXBwZXR9XG4gICAgICB9YDtcbiAgfSBlbHNlIHtcbiAgICBhY3RpdmF0aW9uRm5TbmlwcGV0ID0gYFxuICAgICAgZm4gYWN0aXZhdGlvbihhIDogJHtkYXRhVHlwZX0sIGNvb3JkcyA6IHZlYyR7Y29vcmRzTGVuZ3RofTxpMzI+KSAtPiAke1xuICAgICAgICBkYXRhVHlwZX0ge1xuICAgICAgICAke2FjdGl2YXRpb25PcFNuaXBwZXR9XG4gICAgICB9YDtcbiAgfVxuICByZXR1cm4gYWN0aXZhdGlvbkZuU25pcHBldDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpYXNBY3RpdmF0aW9uU25pcHBldChcbiAgICBoYXNCaWFzOiBib29sZWFuLCBhY3RpdmF0aW9uOiBiYWNrZW5kX3V0aWwuQWN0aXZhdGlvbik6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgICAke2hhc0JpYXMgPyAndmFsdWUgPSB2YWx1ZSArIGdldEJpYXNCeU91dHB1dENvb3Jkcyhjb29yZHMpOycgOiAnJ31cbiAgICAgICR7YWN0aXZhdGlvbiA/ICd2YWx1ZSA9IGFjdGl2YXRpb24odmFsdWUsIGNvb3Jkcyk7JyA6ICcnfVxuICAgICAgYDtcbn1cbiJdfQ==