import React, { useEffect } from 'react';
import './PigeonBlaster.css'; 
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
import { drawKeypoints, drawArm, getKeypoint } from '../pose-utils'; 
import piccioneTerra from '../assets/png/piccione-terra.png';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const Pigeon_SIZE = 80;

type DifficultyConfig = {
  name: string;
  gameDuration: number;
  pigeonLifetime: number;
  maxLives: number;
  maxPigeons: number;
};

const DIFFICULTIES: Record<string, DifficultyConfig> = {
  easy: {
    name: "Facile",
    gameDuration: 60,
    pigeonLifetime: 2,
    maxLives: 5,
    maxPigeons: 2,
  },

  medium: {
    name: "Media",
    gameDuration: 120,
    pigeonLifetime: 1.5,
    maxLives: 5,
    maxPigeons: 3,
  },

  hard: {
    name: "Difficile",
    gameDuration: 180,
    pigeonLifetime: 1.5,
    maxLives: 3,
    maxPigeons: 3,
  },
};

type Pigeon = {
  id: number;
  x: number;
  y: number;
  age: number;
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
  const livesRef = React.useRef(DIFFICULTIES.easy.maxLives);
  const highScoreRef = React.useRef(0);
  const isGameOverRef = React.useRef(false);
  const gameTimerRef = React.useRef(0);
  const isVictoryRef = React.useRef(false);

  // Stati di React per l'interfaccia
  const [score, setScore] = React.useState(0);
  const [lives, setLives] = React.useState(DIFFICULTIES.easy.maxLives);
  const [highScore, setHighScore] = React.useState(0);
  const [isGameOver, setIsGameOver] = React.useState(false);
  const [isVictory, setIsVictory] = React.useState(false);
  const [difficulty, setDifficulty] = React.useState<keyof typeof DIFFICULTIES>("easy");

  // Gestione fotocamere multiple
  const [devices, setDevices] = React.useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = React.useState<string>('');

  // Riferimenti per la zona di calibrazione iniziale (partenza gioco)
  const isCalibratedRef = React.useRef(false);
  const [isCalibrated, setIsCalibrated] = React.useState(false);
  const calibrationTimerRef = React.useRef(0);
  const previousWristsRef = React.useRef({
    left: { x: 0, y: 0 },
    right: { x: 0, y: 0 },
  });

  const stableTimerRef = React.useRef(0);
    
  const pigeonsRef = React.useRef<Pigeon[]>([]);
  
  const pigeonIdRef = React.useRef(0);
  
  const spawnTimerRef = React.useRef(0);

  function spawnPigeon() {

    pigeonsRef.current.push({
      id: pigeonIdRef.current++,

      x:
        Math.random() *
        (VIDEO_WIDTH - PIGEON_SIZE),

      y:
        Math.random() *
        (VIDEO_HEIGHT - PIGEON_SIZE),

      age: 0,
    });
  }

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

  //#region Camera
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
  //#endregion

  function restartGame() {
    function restartAndResume() {
      restartGame();

      requestAnimationFrame(loop);
    }

    const config = difficultyRef.current;
    scoreRef.current = 0;
    livesRef.current = config.maxLives;
    isGameOverRef.current = false;
    
    setScore(0);
    gameTimerRef.current = 0;
    isVictoryRef.current = false;
    setLives(config.maxLives);
    setIsGameOver(false);
    setIsVictory(false);
    
    pigeonsRef.current = [];
    spawnTimerRef.current = 0;
    // reset piccioni
    pigeonsRef.current = [];

    // reset timer spawn
    spawnTimerRef.current = 0;

    // reset vittoria/game over
    isGameOverRef.current = false;
    isVictoryRef.current = false;

    // reset timer partita
    gameTimerRef.current = 0;
  }

  async function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;
    const difficultyRef = React.useRef(DIFFICULTIES.easy);

    useEffect(() => {
      difficultyRef.current =
        DIFFICULTIES[difficulty];
    }, [difficulty]);

    if (!video || !canvas || !detector) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

    if ( isGameOverRef.current || isVictoryRef.current ) {
      ctx.filter = 'grayscale(100%)';
    } else {
      ctx.filter = 'none';
    }

    if (isVictoryRef.current) {

      ctx.fillStyle = "rgba(0,0,0,0.7)";

      ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    
      drawTextUnflipped("HAI VINTO!", VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 - 40, "bold 60px Arial", "#00FF66", "center");
    
      drawTextUnflipped(`Piccioni presi: ${scoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 20, "bold 28px Arial", "white", "center");
    }

    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    // Definizione del quadrante/box di partenza (Centro dello schermo)
    const boxW = 300;
    const boxH = 300;  
    const boxX = (VIDEO_WIDTH - boxW) / 2;
    const boxY = (VIDEO_HEIGHT - boxH) / 2;

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

    // DINAMICHE DI GIOCO

    //#region CONTROLLO POSIZIONE
    if (!isCalibratedRef.current) {
      let isPlayerReady = false;
      if (activePose) {
      
        const leftWrist = getKeypoint(activePose, "left_wrist", 0.5);
        const rightWrist = getKeypoint(activePose, "right_wrist", 0.5);
      
        if (leftWrist && rightWrist) {
        
          const wristsInsideBox =
            leftWrist.x >= boxX &&
            leftWrist.x <= boxX + boxW &&
            leftWrist.y >= boxY &&
            leftWrist.y <= boxY + boxH &&
            rightWrist.x >= boxX &&
            rightWrist.x <= boxX + boxW &&
            rightWrist.y >= boxY &&
            rightWrist.y <= boxY + boxH;
        
          const movement =
            Math.abs(leftWrist.x - previousWristsRef.current.left.x) +
            Math.abs(leftWrist.y - previousWristsRef.current.left.y) +
            Math.abs(rightWrist.x - previousWristsRef.current.right.x) +
            Math.abs(rightWrist.y - previousWristsRef.current.right.y);
        
          const isStill = movement < 25;
        
          previousWristsRef.current.left = {
            x: leftWrist.x,
            y: leftWrist.y,
          };
        
          previousWristsRef.current.right = {
            x: rightWrist.x,
            y: rightWrist.y,
          };
        
          if (wristsInsideBox && isStill) {
            calibrationTimerRef.current += dt;
            isPlayerReady = true;
          } else {
            calibrationTimerRef.current = 0;
          }
        }
      }

    if (isPlayerReady && calibrationTimerRef.current >= 3) {

      isCalibratedRef.current = true;
      setIsCalibrated(true);

      calibrationTimerRef.current = 3;

      spawnPigeon();
    }
  }
      //#endregion
      //#region GIOCO ATTIVO
      else {
        gameTimerRef.current += dt;
        const gameDuration = difficultyRef.current.gameDuration;

        if (
          gameTimerRef.current >= gameDuration
        ) {
          isVictoryRef.current = true;
          setIsVictory(true);
          return;
        }

        spawnTimerRef.current += dt;

        const maxPigeons = difficultyRef.current.maxPigeons;

        if (
          pigeonsRef.current.length < maxPigeons &&
          spawnTimerRef.current > 0.6 &&
          !isGameOverRef.current &&
          !isVictoryRef.current
        ) {
          spawnPigeon();
          spawnTimerRef.current = 0;
        }

        for (const pigeon of pigeonsRef.current) {
          pigeon.age += dt;
        }

        // controllo scadenza piccioni
        const lifetime = difficultyRef.current.pigeonLifetime;

        const beforeCount = pigeonsRef.current.length;

        pigeonsRef.current = pigeonsRef.current.filter((pigeon) => {
          if (pigeon.age >= lifetime) {
            return false; // rimuove il piccione
          }
          return true;
        });

        const removed = beforeCount - pigeonsRef.current.length;

        if (removed > 0) {
          livesRef.current -= removed;
          setLives(livesRef.current);
          ctx.fillStyle = "rgba(255,0,0,0.2)";
          ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
        }

        if (livesRef.current <= 0) {

          isGameOverRef.current = true;
          setIsGameOver(true);

          if (scoreRef.current > highScoreRef.current) {
            highScoreRef.current = scoreRef.current;
            setHighScore(highScoreRef.current);
          }
        }

        // Se passano più di 2 secondi senza prendere il piccione
        if (Pigeon.timer > 2.0) {
          livesRef.current -= 1;
          setLives(livesRef.current);
          Pigeon.reset(); // Cambia posizione al piccione

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
    

    // Disegno dei mirini dei polsi
      const lwx = wristLeftRef.current.x;
      const lwy = wristLeftRef.current.y;
      const rwx = wristRightRef.current.x;
      const rwy = wristRightRef.current.y;

    // Funzioni helper per i testi specchiati
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

    const drawImageUnflipped = (img: HTMLImageElement, x: number, y: number, width: number, height: number) => {
      ctx.save();
      const canvasX = VIDEO_WIDTH - x - width;
      ctx.translate(canvasX, y);
      ctx.scale(-1, 1);
      ctx.drawImage(img, 0, 0, width, height);
      ctx.restore();
    };

    // RENDERING GRAFICO ELEMENTI DI GIOCO
    if (!isGameOverRef.current && !isCalibratedRef.current) {
      // Disegna il box di calibrazione su schermo
      ctx.strokeStyle = calibrationTimerRef.current > 0 ? "#00FF00" : "#FFCC00";
      ctx.lineWidth = 5;
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = "rgba(255, 204, 0, 0.15)";
      ctx.fillRect(boxX, boxY, boxW, boxH);

      const countdown = Math.max(0, 3 - calibrationTimerRef.current);
      drawTextUnflipped( "POSIZIONA ENTRAMBI I POLSI NEL RIQUADRO", VIDEO_WIDTH / 2, boxY - 20, "bold 22px Arial", "#FFD700", "center");
    }

    // Mostra i piccioni solo se il gioco è calibrato e attivo
    if (isCalibratedRef.current && !isGameOverRef.current) {
    
      if (
        PigeonImageRef.current &&
        PigeonImageRef.current.complete &&
        PigeonImageRef.current.naturalWidth > 0
      ) {
      
        try {
          for (const pigeon of pigeonsRef.current) {
            ctx.drawImage(
              PigeonImageRef.current,
              pigeon.x,
              pigeon.y,
              PIGEON_SIZE,
              PIGEON_SIZE
            );
          }
        
        } catch (e) {
          // fallback: disegna rettangoli se immagine fallisce
          for (const pigeon of pigeonsRef.current) {
            ctx.fillStyle = "green";
            ctx.fillRect(
              pigeon.x,
              pigeon.y,
              PIGEON_SIZE,
              PIGEON_SIZE
            );
          }
        }
      
      } else {
      
        for (const pigeon of pigeonsRef.current) {
          ctx.fillStyle = "green";
          ctx.fillRect(
            pigeon.x,
            pigeon.y,
            PIGEON_SIZE,
            PIGEON_SIZE
          );
        }
      }
    }

    // Controllo Collisioni (Hit)
    pigeonsRef.current = pigeonsRef.current.filter(
      (pigeon) => {
      
        const hitLeft =
          lwx >= pigeon.x &&
          lwx <= pigeon.x + PIGEON_SIZE &&
          lwy >= pigeon.y &&
          lwy <= pigeon.y + PIGEON_SIZE;
      
        const hitRight =
          rwx >= pigeon.x &&
          rwx <= pigeon.x + PIGEON_SIZE &&
          rwy >= pigeon.y &&
          rwy <= pigeon.y + PIGEON_SIZE;
      
        const hit =
          hitLeft || hitRight;
      
        if (hit) {
        
          scoreRef.current++;
        
          setScore(
            scoreRef.current
          );
        
          return false;
        }
      
        return true;
      }
    );

    // DISEGNO DELLE VITE (max 5)
    const startX = 30;
    const heartWidth = 35;
    const gap = 8;

    for (let i = 0; i < 5; i++) {
      const isAlive = i < livesRef.current;
      const currentX = startX + i * (heartWidth + gap);
    
      if (
        isAlive &&
        heartFullImgRef.current &&
        heartFullImgRef.current.complete &&
        heartFullImgRef.current.naturalWidth > 0
      ) {
        drawImageUnflipped(
          heartFullImgRef.current,
          currentX,
          30,
          heartWidth,
          heartWidth
        );
      } else if (
        !isAlive &&
        heartDeadImgRef.current &&
        heartDeadImgRef.current.complete &&
        heartDeadImgRef.current.naturalWidth > 0
      ) {
        drawImageUnflipped(
          heartDeadImgRef.current,
          currentX,
          30,
          heartWidth,
          heartWidth
        );
      }
    }

    drawTextUnflipped(`Punteggio: ${scoreRef.current}`, 30, 115, "bold 24px Arial", "white");
    drawTextUnflipped(`Vite rimaste: ${livesRef.current}`, 30, 180, "bold 20px Arial", "#FF6666");
    drawTextUnflipped(`Record: ${highScoreRef.current}`, 30, 150, "bold 20px Arial", "#FFD700");

    if ( isGameOverRef.current || isVictoryRef.current ) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

      drawTextUnflipped("HAI PERSO", VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 - 30, "bold 55px Arial", "#FF3333", "center");
      drawTextUnflipped(`Picconi Presi: ${scoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 20, "bold 24px Arial", "white", "center");
      drawTextUnflipped(`Record Massimo: ${highScoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 55, "bold 22px Arial", "#FFD700", "center");
      const remainingTime = Math.max(
        0,
        difficultyRef.current.gameDuration -
          gameTimerRef.current
      );

      drawTextUnflipped(
        `Tempo: ${Math.ceil(remainingTime)}s`,
        VIDEO_WIDTH / 2,
        40,
        "bold 32px Arial",
        "#00FFFF",
        "center"
      );

      drawTextUnflipped(
        difficultyRef.current.name.toUpperCase(),
        VIDEO_WIDTH - 120,
        40,
        "bold 18px Arial",
        "#FFFFFF"
      );

      const percentage =
        remainingTime /
        difficultyRef.current.gameDuration;

      ctx.fillStyle = "#222";
      ctx.fillRect(
        VIDEO_WIDTH / 2 - 120,
        55,
        240,
        18
      );

      ctx.fillStyle =
        percentage > 0.3
          ? "#00FF66"
          : "#FF4444";

      ctx.fillRect(
        VIDEO_WIDTH / 2 - 120,
        55,
        240 * percentage,
        18
      );
    }

    if ( !isGameOverRef.current && !isVictoryRef.current ) {
      animationRef.current = requestAnimationFrame(loop);
    }
  }

  useEffect(() => {
    const img = new Image();
    img.src = 'src/assets/png/piccione-volante.png'; 
    PigeonImageRef.current = img;

    const imgFull = new Image();
    imgFull.src = 'src/assets/png/cuoricino-vite.png';
    heartFullImgRef.current = imgFull;

    const imgDead = new Image();
    imgDead.src = 'src/assets/png//vita.png';
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
    <div className="game-container">
      {/* Header con Titoli e Piccioni Specchiati */}
      <div className="header-container">
        <img 
          src={piccioneTerra} 
          alt="Piccione decorativo sinistro" 
          className="pigeon-decor pigeon-left" 
        />
        
        <div className="title-wrapper">
          <h1 className="main-title">Pigeon Blaster</h1>
          <span className="subtitle">rehab edition</span>
        </div>

        <img 
          src={piccioneTerra} 
          alt="Piccione decorativo destro" 
          className="pigeon-decor pigeon-right" 
        />
      </div>
      
      {/* Info di Gioco */}
      <div className="stats-bar">
        <span>Punteggio Corrente: <strong>{score}</strong></span>
        <span>Vite:<strong>{lives}/{DIFFICULTIES[difficulty].maxLives}</strong></span>
        <span>Record Attuale: <strong>{highScore}</strong></span>
      </div>

      {/* Controlli di Configurazione */}
      <div className="controls-bar">
        <div>
          <label htmlFor="camera-select" className="camera-label">
            Fotocamera: 
          </label>
          <select
            id="camera-select"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            className="camera-select"
          >
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Camera ${device.deviceId.slice(0, 5)}`}
              </option>
            ))}
          </select>
        </div>

        {(isGameOver || isVictory) && (
          <button onClick={restartAndResume} className="btn-retry">
            Riprova
          </button>
        )}
      </div>

      <div className='selettore-difficoulty'>
        <label
          style={{
            marginRight: "10px",
            fontWeight: "bold",
          }}
        >
          Difficoltà:
        </label>
        
        <select
          value={difficulty}
          onChange={(e) =>
            setDifficulty(
              e.target.value as keyof typeof DIFFICULTIES
            )
          }
          className="camera-select"
        >
        
          <option value="easy">
            Facile
          </option>
        
          <option value="medium">
            Media
          </option>
        
          <option value="hard">
            Difficile
          </option>
        
        </select>
      </div>
      {/* Schermo di gioco */}
      <canvas
        ref={canvasRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        className="game-canvas"
      />

      <video ref={videoRef} style={{ display: "none" }} />
    </div>
  );
}

export default PigeonBlaster;