import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const PNG_W = 64;
const PNG_H = 64;

const GameConfig = {
  VIDEO_WIDTH: 800,
  VIDEO_HEIGHT: 300,
};

interface ObjectEntity {
  x: number;
  y: number;
  width: number;
  height: number;
  speed: number;
  draw: (ctx: CanvasRenderingContext2D) => void;
  update: (dt: number) => void;
  reset: () => void;
}

export default function BallBlaster() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const poseRef = useRef<PoseLandmarker | null>(null);
  const ballImgRef = useRef<HTMLImageElement | null>(null);

  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef(performance.now());
  const lastPoseRef = useRef(0);

  const wristRef = useRef({ x: 0, y: 0 });

  const [score, setScore] = useState(0);

  const ballRef = useRef<ObjectEntity>({
    x: 400,
    y: 50,
    width: PNG_W,
    height: PNG_H,
    speed: 140,

    reset() {
      this.x = 400;
      this.y = 50;
    },

    update(dt: number) {
      this.y += this.speed * dt;
      if (this.y > GameConfig.VIDEO_HEIGHT) this.reset();
    },

    draw(ctx) {
      if (!ballImgRef.current) return;
        
      ctx.fillStyle = "green";
      ctx.fillRect(this.x, this.y, this.width, this.height);
        
      ctx.drawImage(ballImgRef.current, this.x, this.y, this.width, this.height);
    }
  });

  useEffect(() => {
    const img = new Image();
    img.src = "/src/assets/ball.png";
    img.onload = () => {
      ballImgRef.current = img;
    };
  }, []);

  const collision = (wx: number, wy: number, b: ObjectEntity) => {
    return (
      wx >= b.x &&
      wx <= b.x + b.width &&
      wy >= b.y &&
      wy <= b.y + b.height
    );
  };

  const setupPose = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    poseRef.current = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      },
      runningMode: "VIDEO",
    });
  };

  const detectPose = async () => {
    const video = videoRef.current;
    const pose = poseRef.current;
    if (!video || !pose) return;
    if (video.videoWidth === 0) return;

    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;

    const result = await pose.detectForVideo(video, performance.now());
    const landmarks = result.landmarks?.[0];
    if (!landmarks) return;

    const wrist = landmarks[15];

    const x = wrist.x * GameConfig.VIDEO_WIDTH;
    const y = wrist.y * GameConfig.VIDEO_HEIGHT;

    wristRef.current = {
      x: Math.max(0, Math.min(GameConfig.VIDEO_WIDTH, x)),
      y: Math.max(0, Math.min(GameConfig.VIDEO_HEIGHT, y)),
    };
  };

  const loop = (t: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dt = (t - lastTimeRef.current) / 1000;
    lastTimeRef.current = t;

    ctx.clearRect(0, 0, GameConfig.VIDEO_WIDTH, GameConfig.VIDEO_HEIGHT);

    const ball = ballRef.current;

    ball.update(dt);
    ball.draw(ctx);

    if (!ballImgRef.current) return;

    if (videoRef.current) {
      ctx.drawImage(
        videoRef.current,
        0,
        0,
        GameConfig.VIDEO_WIDTH,
        GameConfig.VIDEO_HEIGHT
      );
    }

    // wrist debug
    ctx.beginPath();
    ctx.arc(wristRef.current.x, wristRef.current.y, 20, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.fillText("WRIST", wristRef.current.x + 10, wristRef.current.y);

    // collision
    if (collision(wristRef.current.x, wristRef.current.y, ball)) {
      setScore((s) => s + 1);
      ball.reset();
    }

    // pose throttling (~30fps)
    if (t - lastPoseRef.current > 33) {
      lastPoseRef.current = t;
      detectPose();
    }

    animationRef.current = requestAnimationFrame(loop);
  };

  const startCamera = async () => {
    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: GameConfig.VIDEO_WIDTH,
        height: GameConfig.VIDEO_HEIGHT,
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    animationRef.current = requestAnimationFrame(loop);
  };

  const stopCamera = () => {
    const video = videoRef.current;
    if (!video) return;

    const stream = video.srcObject as MediaStream;
    stream?.getTracks().forEach((t) => t.stop());
  };

  // ---------------- INIT ----------------
  useEffect(() => {
    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      await setupPose();
      await startCamera();
    };

    init();

    return () => {
      stopCamera();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      poseRef.current?.close?.();
    };
  }, []);

  // ---------------- RENDER ----------------
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
      <h3>Punteggio: {score}</h3>

      <canvas
        ref={canvasRef}
        width={GameConfig.VIDEO_WIDTH}
        height={GameConfig.VIDEO_HEIGHT}
        style={{ border: "2px solid black" }}
      />

      <video
        ref={videoRef}
        style={{ display: "none" }}
      />
    </div>
  );
}