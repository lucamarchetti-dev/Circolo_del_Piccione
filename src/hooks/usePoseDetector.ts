import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
import type { PoseDetector } from '@tensorflow-models/pose-detection/dist/pose_detector';
import type { Pose } from '@tensorflow-models/pose-detection/dist/types';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

export function usePoseDetector() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectorRef = useRef<PoseDetector | null>(null);
  const lastPoseRef = useRef<Pose | null>(null);

  useEffect(() => {
    async function initialize() {
      await tf.setBackend('webgl');
      await tf.ready();

      detectorRef.current = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    }

    initialize();

    return () => {
      const video = videoRef.current;
      if (video?.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      }
      detectorRef.current?.dispose();
    };
  }, []);

  async function detectPose(): Promise<Pose | null> {
    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video || !detector || video.readyState < 2) return null;

    const poses = await detector.estimatePoses(video);
    if (poses.length === 0) return null;

    lastPoseRef.current = poses[0];
    return poses[0];
  }

  return { videoRef, detectPose, lastPoseRef, VIDEO_WIDTH, VIDEO_HEIGHT };
}