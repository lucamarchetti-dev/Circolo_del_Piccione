// Mock stub for @mediapipe/pose
// The project uses MoveNet, not BlazePose/MediaPipe,
// but pose-detection's bundle always imports this package.
export class Pose {
  constructor() {}
  setOptions() {}
  onResults() {}
  send() { return Promise.resolve(); }
  initialize() { return Promise.resolve(); }
  close() {}
  reset() {}
}
