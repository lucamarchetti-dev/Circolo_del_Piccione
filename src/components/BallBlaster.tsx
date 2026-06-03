// #region Declaring
import React, { useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const poseRef = useRef<PoseLandmarker | null>(null);
const balls = useRef<HTMLImageElement | null>(null);

useEffect(() => {
  const img = new Image();
  img.src = "/assets/ball.png";
  img.onload = () => {
    balls.current = img;
  };
}, []);

const PNG_W = 64;
const PNG_H = 64;
// #endregion

interface ObjectEntity {
  x: number;
  y: number;
  width: number;
  height: number;
  speed: number;
  active: boolean;

  draw: (ctx: CanvasRenderingContext2D) => void;
  reset: () => void;
  update: (dt: number) => void;
}

const GameConfig = {
  VIDEO_WIDTH: 800,
  VIDEO_HEIGHT: 300,
  Object_Start: 400,
  COLORS: {
      PRIMARY: '#535353',
      ACCENT: '#ff5252',
      FOCUS: '#F59E0B',
      WHITE: '#ffffff',
  } // non so nemmeno se e in cosa è utile ma per il momento lo teniamo

}

const spawnBall = (): ObjectEntity => ({
  x: 400,
  y: -PNG_H,
  width: PNG_W,
  height: PNG_H,
  speed: 140,
  active: true,

  reset() {
    this.x = 400;
    this.y = -PNG_H;
  },

  draw(ctx: CanvasRenderingContext2D) {
    if (!balls.current) return;

    ctx.drawImage(
      balls.current,
      this.x,
      this.y,
      this.width,
      this.height
    );
  },

  update(dt: number) {
    this.y += this.speed * dt;

    if (this.y > GameConfig.VIDEO_HEIGHT) {
      this.reset();
    }
  },
});

const BallBlaster: React.FC = () => {
  const canvasRef =
    useRef<HTMLCanvasElement | null>(null);

  const videoRef =
    useRef<HTMLVideoElement | null>(null);

  const animationRef =
    useRef<number>();

  const ballRef =
    useRef<ObjectEntity>(spawnBall());

  const wristRef = useRef({
    x: 0,
    y: 0,
  });

  const hitLockRef = useRef(false);
  const lastTimeRef = useRef(performance.now());

  const [score, setScore] = useState(0);

  const collision = (
    wx: number,
    wy: number,
    b: ObjectEntity
  ) => {
    return (
      wx >= b.x &&
      wx <= b.x + b.width &&
      wy >= b.y &&
      wy <= b.y + b.height
    );
  };

  const loop = (t: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dt =
      (t - lastTimeRef.current) / 1000;

    lastTimeRef.current = t;

    ctx.clearRect(
      0,
      0,
      GameConfig.VIDEO_WIDTH,
      GameConfig.VIDEO_HEIGHT
    );

    const ball = ballRef.current;

    ball.update(dt);
    ball.draw(ctx);

    /* DEBUG wrist */
    ctx.beginPath();
    ctx.arc(
      wristRef.current.x,
      wristRef.current.y,
      12,
      0,
      Math.PI * 2
    );
    ctx.fillStyle = "red";
    ctx.fill();

    if (
      collision(
        wristRef.current.x,
        wristRef.current.y,
        ball
      )
    ) {
      if (!hitLockRef.current) {
        hitLockRef.current = true;

        setScore((s) => s + 1);

        ball.reset();

        setTimeout(() => {
          hitLockRef.current = false;
        }, 250);
      }
    }

    animationRef.current =
      requestAnimationFrame(loop);
  };

  const initCamera = async () => {
    const stream =
      await navigator.mediaDevices.getUserMedia(
        {
          video: true,
        }
      );

    if (videoRef.current) {
      videoRef.current.srcObject =
        stream;

      await videoRef.current.play();
    }
  };

  const initPoseDetector = async () => {
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

  const result = await pose.detectForVideo(video, performance.now());

  const landmark = result.landmarks?.[0];

  if (landmark) {
    const wrist = landmark[15]; // left wrist
    // or 16 = right wrist

    wristRef.current = {
      x: wrist.x * GameConfig.VIDEO_WIDTH,
      y: wrist.y * GameConfig.VIDEO_HEIGHT,
    };
  }
};

  useEffect(() => {
    initCamera();
    initPoseDetector();

    animationRef.current =
      requestAnimationFrame(loop);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(
          animationRef.current
        );
      }
    };
  }, []);


  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "10px",
      }}
    >
      <h2>Ball Blaster 👋</h2>

      <h3>Punteggio: {score}</h3>

      <canvas
        ref={canvasRef}
        width={GameConfig.VIDEO_WIDTH}
        height={GameConfig.VIDEO_HEIGHT}
        style={{
          border: "2px solid black",
        }}
      />

      <video
        ref={videoRef}
        style={{
          display: "none",
        }}
      />
    </div>
  );
};

// interface ExtendedWindow extends Window {
//     webkitAudioContext?: typeof AudioContext;
// }

// interface WristState {
//     wristBaselineY: number;
//     wristSmoothedY: number;
//     wristNormalizedDelta: number;
// }

// interface PNG {
//     active: boolean;
//     x: number;
//     y: number;
//     width: number;
//     height: number;
//     color?: string;
// }

// interface Base { 
//     canRestart: boolean;
//     score: number;
//     // gameSpeed: number; (prossime versioni)
//     lastTime: number;
//     pngPool: PNG[];
//     objectSpawnTimer: number;
//     animationId: number;
//     visionAnimationId: number;
//     object: ObjectEntity;
//     hasStarted: boolean;
//     cameraReady: boolean;
//     // highScore: number; (prossime versioni)
// }

// sezione high score (prossime versioni)

// #region Object
// type ObjectState = 'ACTIVE' | 'INACTIVE';

export default BallBlaster;
