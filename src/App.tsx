import React, { useEffect } from 'react';
import './App.css'
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
// Importiamo tutto correttamente da pose-utils, incluso getKeypoint
import { drawKeypoints, drawArm, getKeypoint } from './pose-utils';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const BALL_SIZE = 64;

type Ball = {
  x: number;
  y: number;
  speed: number;
  reset: () => void;
  update: (dt: number) => void;
};

function App() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const detectorRef = React.useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = React.useRef<number | null>(null);
  const lastTimeRef = React.useRef(performance.now());
  const lastPoseRef = React.useRef(0);
  
  const currentPoseRef = React.useRef<poseDetection.Pose | null>(null);
  const ballImageRef = React.useRef<HTMLImageElement | null>(null);

  const wristRef = React.useRef({ x: 0, y: 0 });
  const [score, setScore] = React.useState(0);

  const ballRef = React.useRef<Ball>({
    x: 500,
    y: 100,
    speed: 140,

    reset() {
      this.x = Math.random() * (VIDEO_WIDTH - BALL_SIZE);
      this.y = 0;
    },

    update(dt: number) {
      this.y += this.speed * dt;
      if (this.y > VIDEO_HEIGHT) this.reset();
    },
  });

  async function setupPoseDetector() {
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      }
    );
    detectorRef.current = detector;
  }

  async function startCamera() {
    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT },
      audio: false,
    });

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      loop();
    };
  }

  function stopCamera() {
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream;
    stream?.getTracks().forEach(t => t.stop());
  }

  async function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;

    if (!video || !canvas || !detector) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

    // sfondo con la telecamera
    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    // aggiorna gioco
    const ball = ballRef.current;
    ball.update(dt);

    // rilevamento posa ogni 33ms
    if (now - lastPoseRef.current > 33) {
      lastPoseRef.current = now;
      const poses = await detector.estimatePoses(video);
      if (poses && poses.length > 0) {
        currentPoseRef.current = poses[0];
      }
    }

    // disegno dello scheletro e tracciamento
    const activePose = currentPoseRef.current;
    if (activePose) {
      // usiamo il getKeypoint di pose-utils (restituisce il punto solo se supera la confidence)
      const wrist = getKeypoint(activePose, "left_wrist", 0.3);
      if (wrist) {
        wristRef.current = { x: wrist.x, y: wrist.y };
      }

      // corretto l'ordine dei parametri: (ctx, pose, confidence)
      drawKeypoints(ctx, activePose, 0.3);
      
      // disegniamo sia il braccio destro che sinistro per un effetto rehab completo!
      drawArm(ctx, activePose, 'left', 0.3);
      drawArm(ctx, activePose, 'right', 0.3);
    }

    // coordinate del polso
    const wx = wristRef.current.x;
    const wy = wristRef.current.y;

    // feedback visivo sul polso (cerchio rosso)
    ctx.beginPath();
    ctx.arc(wx, wy, 20, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    // render dell'immagine reale della palla
    if (ballImageRef.current && ballImageRef.current.complete) {
      ctx.drawImage(ballImageRef.current, ball.x, ball.y, BALL_SIZE, BALL_SIZE);
    } else {
      ctx.fillStyle = "green";
      ctx.fillRect(ball.x, ball.y, BALL_SIZE, BALL_SIZE);
    }

    // collisione
    const hit =
      wx >= ball.x &&
      wx <= ball.x + BALL_SIZE &&
      wy >= ball.y &&
      wy <= ball.y + BALL_SIZE;

    if (hit) {
      setScore(s => s + 1);
      ball.reset();
    }

    animationRef.current = requestAnimationFrame(loop);
  }

  useEffect(() => {
    // inizializza l'immagine (ricordati di mettere ball.png dentro la cartella /public)
    const img = new Image();
    img.src = '/ball.png'; 
    ballImageRef.current = img;

    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      await setupPoseDetector();
      await startCamera();
    };

    init();

    return () => {
      stopCamera();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      detectorRef.current?.dispose();
    };
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h2>Ball Blaster - Rehab Edition</h2>
      <p style={{ fontSize: '24px', fontWeight: 'bold' }}>Score: {score}</p>

      <canvas
        ref={canvasRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        style={{ border: "2px solid black", maxWidth: "100%", height: "auto" }}
      />

      <video ref={videoRef} style={{ display: "none" }} />
    </div>
  );
}

export default App;