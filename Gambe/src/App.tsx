import React, { useEffect } from 'react';
import './App.css'
import type { PoseDetector } from '@tensorflow-models/pose-detection/dist/pose_detector';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
import { drawKeypoints, drawArm, getKeypoint } from './pose-utils';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

function App() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const animationFrameIdRef = React.useRef<number | null>(null);
  const detectorRef = React.useRef<PoseDetector | null>(null);
  const wristAboveNouseCounterRef = React.useRef<boolean>(false);
  const lastWristRaiseTimeCount = React.useRef<number>(0);

  const [cameras, setCameras] = React.useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = React.useState<string>("");
  const [wristRaised, setWristRaised] = React.useState<number>(0);

  useEffect(() => {
    async function initialize() {
      try {
        console.log("Initializing TensorFlow...");

        await tf.setBackend('webgl');
        await tf.ready();

        console.log("TensorFlow initialized with WebGL backend.", tf.getBackend());

        await setupPoseDetector();

        console.log("Pose detector initialized.");

        await loadCamera();
        await startCamera();
        
        console.log("Camera initialized.");
      }
      catch (error) {
        console.error('Error initializing camera:', error);
      }
    }

    initialize();

    return () => {
      stopCamera();
      stopLoopDrawing();
      detectorRef.current?.dispose();
    }
  }, []);

  useEffect(() => {
    if (!selectedCameraId)
      return;

    startCamera();
  }, [selectedCameraId])

  async function loadCamera() {
    const devices = await navigator.mediaDevices.enumerateDevices();

    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    setCameras(videoDevices);
  }

  async function startCamera() {
    try {
      const video = videoRef.current;

      if (!video)
        return;

      stopCamera();

      // Source of video data stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedCameraId ?
          {
            width: VIDEO_WIDTH,
            height: VIDEO_HEIGHT,
            deviceId: {
              exact: selectedCameraId
            }
          } : {
            width: VIDEO_WIDTH,
            height: VIDEO_HEIGHT,
            // facingMode: 'user'
          },
        audio: false
      });

      video.srcObject = stream;

      video.onloadedmetadata = () => {
        video.play();
        startLoopDrawing();       //  Sync of the canvas with the video stream
      }
    }
    catch (error) {
      console.error('Error accessing webcam:', error);
    }
  }

  function stopCamera() {
    const video = videoRef.current;

    if (!video)
      return;

    const stream = video.srcObject as MediaStream;

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
  }

  async function drawToCanvas() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;

    if (!video || !canvas)
      return;

    const ctx = canvas.getContext('2d');

    if (!ctx)
      return;

    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    if(!detector)
      return;

    const poses = await detector.estimatePoses(video);

    if(poses.length > 0) {
      console.log(poses[0]);
      drawKeypoints(ctx, poses[0]);

      drawArm(ctx, poses[0], 'left');
      drawArm(ctx, poses[0], 'right');

      updateWristRaised(poses[0], 'left');
      updateWristRaised(poses[0], 'right');
    }
  }

  function startLoopDrawing() {
    async function loop() {
      await drawToCanvas();
      animationFrameIdRef.current = requestAnimationFrame(loop);
    }

    loop();
  }

  function stopLoopDrawing() {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
  }

  async function setupPoseDetector() {
    console.log("Loading pose detector...");

    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
      });

    detectorRef.current = detector;
  }

  function updateWristRaised(pose: poseDetection.Pose, side: 'left' | 'right') {
    const nose = getKeypoint(pose, 'nose');
    const wrist = getKeypoint(pose, `${side}_wrist`);

    if(!nose || !wrist)
      return;

    const cooldownMs = 800;

    const now  = performance.now();
    const canCount = now - lastWristRaiseTimeCount.current > cooldownMs;
    const isWristAboveNose = wrist.y < nose.y;

    if(isWristAboveNose && !wristAboveNouseCounterRef.current && canCount) {
      setWristRaised(prev => prev + 1);
      lastWristRaiseTimeCount.current = now;
      wristAboveNouseCounterRef.current = true;
    }
    else if(!isWristAboveNose && wristAboveNouseCounterRef.current) {
      wristAboveNouseCounterRef.current = false;
    }
  }

  function handleCameraChange(event: React.ChangeEvent<HTMLSelectElement>) {
    setSelectedCameraId(event.target.value);
  }

  return (
    <>
      <h1>Pose Browser Demo</h1>

      <label htmlFor="cameraSelect">Select Camera: </label>
      <select
        id="cameraSelect"
        value={selectedCameraId}
        onChange={handleCameraChange}
      >
        {cameras.map((camera, index) => (
          <option key={index} value={camera.deviceId}>
            {camera.label || `Camera ${index + 1}`}
          </option>
        ))}
      </select>

      <hr />

      <p>Wrist raised count: {wristRaised}</p>

      <div className='stage'>
        <video
          ref={videoRef}
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          playsInline
          muted
          className='hidden-video'
        ></video>

        <canvas
          ref={canvasRef}
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          className='canvas'>
        </canvas>
      </div>
    </>
  )
}

export default App
