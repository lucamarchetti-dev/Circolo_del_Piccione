import type { Pose } from '@tensorflow-models/pose-detection/dist/types';
import { getKeypoint } from '../pose-utils';

export type Lane = 'left' | 'center' | 'right';

const SMOOTHING_FRAMES = 5;
const hipXHistory: number[] = [];

export function detectLane(pose: Pose, frameWidth: number): Lane {
  const leftHip = getKeypoint(pose, 'left_hip');
  const rightHip = getKeypoint(pose, 'right_hip');

  if (!leftHip && !rightHip) return 'center';

  const hipX = leftHip && rightHip
    ? (leftHip.x + rightHip.x) / 2
    : (leftHip?.x ?? rightHip!.x);

  // Accumula la storia e tieni solo gli ultimi N frame
  hipXHistory.push(hipX);
  if (hipXHistory.length > SMOOTHING_FRAMES) hipXHistory.shift();

  // Media mobile per evitare cambi di corsia nervosi
  const smoothedX = hipXHistory.reduce((a, b) => a + b, 0) / hipXHistory.length;

  if (smoothedX < frameWidth / 3) return 'left';
  if (smoothedX > (frameWidth / 3) * 2) return 'right';
  return 'center';
}

export function resetLaneHistory() {
  hipXHistory.length = 0;
}