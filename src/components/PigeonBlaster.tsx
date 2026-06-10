import React, { useEffect } from 'react';
import '../App.css'; 
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
import { drawKeypoints, drawArm, getKeypoint } from '../pose-utils'; 

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const Pigeon_SIZE = 80;

type Pigeon = {
  x: number;
  y: number;
  speed: number;
  reset: () => void;
  update: (dt: number) => void;
};

function PigeonBlaster() {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const detectorRef = React.useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = React.useRef<number | null>(null);
  const lastTimeRef = React.useRef(performance.now());
  const lastPoseRef = React.useRef(0);
  
  const currentPoseRef = React.useRef<poseDetection.Pose | null>(null);
  const PigeonImageRef = React.useRef<HTMLImageElement | null>(null);
  
  // Riferimenti per le immagini dei cuori PNG
  const heartFullImgRef = React.useRef<HTMLImageElement | null>(null);
  const heartDeadImgRef = React.useRef<HTMLImageElement | null>(null);

  // Riferimenti per i calcoli interni al ciclo di animazione
  const wristLeftRef = React.useRef({ x: 0, y: 0 });
  const wristRightRef = React.useRef({ x: 0, y: 0 });
  const scoreRef = React.useRef(0);
  const livesRef = React.useRef(3);
  const highScoreRef = React.useRef(0);
  const isGameOverRef = React.useRef(false);

  // Stati di React per l'interfaccia
  const [score, setScore] = React.useState(0);
  const [lives, setLives] = React.useState(3);
  const [highScore, setHighScore] = React.useState(0);
  const [isGameOver, setIsGameOver] = React.useState(false);

  // Gestione fotocamere multiple
  const [devices, setDevices] = React.useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = React.useState<string>('');

  const PigeonRef = React.useRef<Pigeon>({
    x: 300,
    y: 50,
    speed: 260,

    reset() {
      this.x = Math.random() * (VIDEO_WIDTH - Pigeon_SIZE);
      this.y = 0;
    },

    update(dt: number) {
      this.y += this.speed * dt;
    },
  });

  async function setupPoseDetector() {
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        // modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        // meglio per uso piu' leggero di GPU
        // modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
        // per identificare meglio polsi, more accurate ma piu' pesante
        // modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING
        // per piu' persone, dovrebbe funzionare in multiplayer
      }
    );
    detectorRef.current = detector;
  }

  async function getCameras() {
    try {
      await navigator.mediaDevices.getUserMedia({ video: true });
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error("Errore nel recupero delle fotocamere:", err);
    }
  }

  async function startCamera(deviceId?: string) {
    stopCamera();
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        width: VIDEO_WIDTH, 
        height: VIDEO_HEIGHT,
        deviceId: deviceId ? { exact: deviceId } : undefined
      },
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

  function restartGame() {
    scoreRef.current = 0;
    livesRef.current = 3;
    isGameOverRef.current = false;
    
    setScore(0);
    setLives(3);
    setIsGameOver(false);
    
    PigeonRef.current.reset();
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

    if (isGameOverRef.current) {
      ctx.filter = 'grayscale(100%)';
    } else {
      ctx.filter = 'none';
    }

    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    const Pigeon = PigeonRef.current;
    if (!isGameOverRef.current) {
      Pigeon.update(dt);

      if (Pigeon.y > VIDEO_HEIGHT) {
        livesRef.current -= 1;
        setLives(livesRef.current);
        Pigeon.reset();

        if (livesRef.current <= 0) {
          isGameOverRef.current = true;
          setIsGameOver(true);
          if (scoreRef.current > highScoreRef.current) {
            highScoreRef.current = scoreRef.current;
            setHighScore(highScoreRef.current);
          }
        }
      }
    }

    if (now - lastPoseRef.current > 33) {
      lastPoseRef.current = now;
      const poses = await detector.estimatePoses(video);
      if (poses && poses.length > 0) {
        currentPoseRef.current = poses[0];
      }
    }

    const activePose = currentPoseRef.current;
    if (activePose) {
      const leftWrist = getKeypoint(activePose, "left_wrist", 0.3);
      const rightWrist = getKeypoint(activePose, "right_wrist", 0.3);

      if (leftWrist) wristLeftRef.current = { x: leftWrist.x, y: leftWrist.y };
      if (rightWrist) wristRightRef.current = { x: rightWrist.x, y: rightWrist.y };

      drawKeypoints(ctx, activePose, 0.3);
      drawArm(ctx, activePose, 'left', 0.3);
      drawArm(ctx, activePose, 'right', 0.3);
    }

    const lwx = wristLeftRef.current.x;
    const lwy = wristLeftRef.current.y;
    const rwx = wristRightRef.current.x;
    const rwy = wristRightRef.current.y;

    ctx.beginPath();
    ctx.arc(lwx, lwy, 15, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(rwx, rwy, 15, 0, Math.PI * 2);
    ctx.fillStyle = "blue";
    ctx.fill();

    // DISEGNO DEL PICCIONE CON PROTEZIONE CRASH (Controlla se l'immagine è integra)
    if (PigeonImageRef.current && PigeonImageRef.current.complete && PigeonImageRef.current.naturalWidth > 0) {
      try {
        ctx.drawImage(PigeonImageRef.current, Pigeon.x, Pigeon.y, Pigeon_SIZE, Pigeon_SIZE);
      } catch (e) {
        ctx.fillStyle = "green";
        ctx.fillRect(Pigeon.x, Pigeon.y, Pigeon_SIZE, Pigeon_SIZE);
      }
    } else {
      ctx.fillStyle = "green";
      ctx.fillRect(Pigeon.x, Pigeon.y, Pigeon_SIZE, Pigeon_SIZE);
    }

    if (!isGameOverRef.current) {
      const hitLeft = lwx >= Pigeon.x && lwx <= Pigeon.x + Pigeon_SIZE && lwy >= Pigeon.y && lwy <= Pigeon.y + Pigeon_SIZE;
      const hitRight = rwx >= Pigeon.x && rwx <= Pigeon.x + Pigeon_SIZE && rwy >= Pigeon.y && rwy <= Pigeon.y + Pigeon_SIZE;

      if (hitLeft || hitRight) {
        scoreRef.current += 1;
        setScore(scoreRef.current);
        Pigeon.reset();
      }
    }

    ctx.filter = 'none';

    // Funzione helper per i testi
    const drawTextUnflipped = (text: string, x: number, y: number, font: string, color: string, align: CanvasTextAlign = 'left') => {
      ctx.save();
      const canvasX = VIDEO_WIDTH - x;
      ctx.translate(canvasX, y);
      ctx.scale(-1, 1);
      ctx.font = font;
      ctx.fillStyle = color;
      ctx.textAlign = align;
      ctx.fillText(text, 0, 0);
      ctx.restore();
    };

    // Funzione helper per le immagini specchiate
    const drawImageUnflipped = (img: HTMLImageElement, x: number, y: number, width: number, height: number) => {
      ctx.save();
      const canvasX = VIDEO_WIDTH - x - width;
      ctx.translate(canvasX, y);
      ctx.scale(-1, 1);
      ctx.drawImage(img, 0, 0, width, height);
      ctx.restore();
    };

    // DISEGNO DEI CUORI CON VERIFICA "naturalWidth" CONTRO I FILE CORROTTI/404
    const startX = 30;
    const heartWidth = 45;
    const gap = 10;
    
    for (let i = 0; i < 3; i++) {
      const isAlive = i < livesRef.current;
      const currentX = startX + i * (heartWidth + gap);
      
      if (isAlive && heartFullImgRef.current && heartFullImgRef.current.complete && heartFullImgRef.current.naturalWidth > 0) {
        try {
          drawImageUnflipped(heartFullImgRef.current, currentX, 30, heartWidth, heartWidth);
        } catch (e) {
          drawTextUnflipped("❤️", currentX, 70, "40px Arial", "white");
        }
      } else if (!isAlive && heartDeadImgRef.current && heartDeadImgRef.current.complete && heartDeadImgRef.current.naturalWidth > 0) {
        try {
          drawImageUnflipped(heartDeadImgRef.current, currentX, 30, heartWidth, heartWidth);
        } catch (e) {
          drawTextUnflipped("🖤", currentX, 70, "40px Arial", "white");
        }
      } else {
        // Fallback immediato in emoji se l'immagine manca o restituisce 404
        drawTextUnflipped(isAlive ? "❤️" : "🖤", currentX, 70, "40px Arial", "white");
      }
    }

    drawTextUnflipped(`Punteggio: ${scoreRef.current}`, 30, 115, "bold 24px Arial", "white");
    drawTextUnflipped(`Record: ${highScoreRef.current}`, 30, 150, "bold 20px Arial", "#FFD700");

    if (isGameOverRef.current) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

      drawTextUnflipped("HAI PERSO", VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 - 30, "bold 55px Arial", "#FF3333", "center");
      drawTextUnflipped(`Picconi Presi: ${scoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 20, "bold 24px Arial", "white", "center");
      drawTextUnflipped(`Record Massimo: ${highScoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 55, "bold 22px Arial", "#FFD700", "center");
    }

    animationRef.current = requestAnimationFrame(loop);
  }

  useEffect(() => {
    const img = new Image();
    img.src = 'src/assets/Pigeon.png'; 
    PigeonImageRef.current = img;

    const imgFull = new Image();
    imgFull.src = 'src/assets//heart_full.png';
    heartFullImgRef.current = imgFull;

    const imgDead = new Image();
    imgDead.src = 'src/assets//heart_dead.png';
    heartDeadImgRef.current = imgDead;

    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
      await setupPoseDetector();
      await getCameras();
    };

    init();

    return () => {
      stopCamera();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      detectorRef.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (selectedDeviceId) {
      startCamera(selectedDeviceId);
    }
  }, [selectedDeviceId]);

  return (
    <div style={{ textAlign: "center", fontFamily: "sans-serif", padding: "10px" }}>
      <h2>Pigeon Blaster 🐦</h2>
      
      <div style={{ margin: "10px auto", display: "flex", justifyContent: "center", gap: "25px", fontSize: "18px" }}>
        <span>Punteggio Corrente: <strong>{score}</strong></span>
        <span>Vite: <strong>{lives}/3</strong></span>
        <span>Record Attuale: <strong>{highScore}</strong></span>
      </div>

      <div style={{ marginBottom: "15px", display: "flex", justifyContent: "center", gap: "15px", alignItems: "center" }}>
        <div>
          <label htmlFor="camera-select" style={{ marginRight: "10px", fontWeight: "bold" }}>
            Fotocamera: 
          </label>
          <select
            id="camera-select"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            style={{ padding: "6px 12px", fontSize: "16px", borderRadius: "4px" }}
          >
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${device.deviceId.slice(0, 5)}`}
              </option>
            ))}
          </select>
        </div>

        {isGameOver && (
          <button 
            onClick={restartGame}
            style={{
              padding: "8px 16px",
              fontSize: "16px",
              fontWeight: "bold",
              backgroundColor: "#FF3333",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Riprova 🔄
          </button>
        )}
      </div>

      <canvas
        ref={canvasRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        style={{ 
          border: "3px solid #333", 
          borderRadius: "8px",
          maxWidth: "100%", 
          height: "auto",
          transform: "scaleX(-1)" 
        }}
      />

      <video ref={videoRef} style={{ display: "none" }} />
    </div>
  );
}

export default PigeonBlaster;