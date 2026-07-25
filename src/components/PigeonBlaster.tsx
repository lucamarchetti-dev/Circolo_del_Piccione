import React, { useEffect, useRef, useState } from 'react';
import './PigeonBlaster.css'; 
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { drawKeypoints, drawArm, getKeypoint } from '../pose-utils'; 
import piccioneTerra from '../assets/png/piccione-terra.png';
import piccione_volante from '../assets/png/PIGGEONNN.gif';
import cuoricino_vite from '../assets/png/cuoricino-vite.png';
import vita from '../assets/png/vita.png';
import golderBuzz from '../assets/png/angel-pigeon-tr.gif';
import pigna from '../assets/png/pignetta.gif';

// Importazione globale per aggirare le restrizioni sui moduli di Vite
import 'gifler';
const gifler = (window as any).gifler;

// Estrae ogni frame di una GIF come canvas indipendente e già "pronto per il disegno".
// NOTA IMPORTANTE: l'oggetto Animator restituito da gifler(...).get() NON popola
// automaticamente frame.buffer (viene creato solo durante animateInCanvas/start).
// Qui replichiamo manualmente la stessa logica di compositing della libreria
// (inclusa la gestione del "disposal method") per ottenere frame completi e
// indipendenti, utilizzabili in un ordine qualsiasi (necessario per l'animazione
// dei piccioni, che scelgono il frame in base alla loro età e non in sequenza).
function extractGifFrames(anim: any): HTMLCanvasElement[] {
  const width = anim.width;
  const height = anim.height;
  const frames = anim._frames || anim.frames || [];

  const workCanvas = document.createElement('canvas');
  workCanvas.width = width;
  workCanvas.height = height;
  const workCtx = workCanvas.getContext('2d');
  if (!workCtx) return [];

  const output: HTMLCanvasElement[] = [];
  let disposePrevious: (() => void) | null = null;

  for (const frame of frames) {
    // Crea il buffer del singolo frame se non esiste ancora (lazy nella libreria)
    if (!frame.buffer) {
      frame.buffer = gifler.createBufferCanvas(frame, width, height);
    }

    // Applica l'eventuale "pulizia" richiesta dal frame precedente PRIMA di disegnare questo
    if (disposePrevious) {
      disposePrevious();
      disposePrevious = null;
    }

    workCtx.drawImage(frame.buffer, frame.x, frame.y);

    // Salva uno snapshot completo e indipendente del canvas di lavoro
    const snapshot = document.createElement('canvas');
    snapshot.width = width;
    snapshot.height = height;
    snapshot.getContext('2d')?.drawImage(workCanvas, 0, 0);
    output.push(snapshot);

    // Prepara la funzione di "disposal" da eseguire prima del prossimo frame
    switch (frame.disposal) {
      case 2: // Ripristina lo sfondo (trasparente)
        disposePrevious = () => workCtx.clearRect(0, 0, width, height);
        break;
      case 3: { // Ripristina il contenuto precedente
        const saved = workCtx.getImageData(0, 0, width, height);
        disposePrevious = () => workCtx.putImageData(saved, 0, 0);
        break;
      }
      default:
        disposePrevious = null;
    }
  }

  return output;
}

// immagine sfondo in caso game over
import gameOverBg from "../assets/png/game-over.png";
const gameOverImage = new Image();
gameOverImage.src = gameOverBg;

// immagine sfondo in caso di vittoria
import cieloDoratoBg from "../assets/png/cielo-dorato.png";
const victoryImage = new Image();
victoryImage.src = cieloDoratoBg;

// dichiarazione dimensione canvas
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

const PIGEON_SIZE_RATIO = 120 / VIDEO_WIDTH;
const PIGEON_SIZE = Math.round(VIDEO_WIDTH * PIGEON_SIZE_RATIO);

const CALIBRATION_BOX_RATIO = 300 / VIDEO_HEIGHT; 
const CALIBRATION_SECONDS = 3;
const CALIBRATION_MOVEMENT_TOLERANCE = 45; 

const PIGNA_SPAWN_CHANCE = 0.4;   
const GOLDEN_SPAWN_CHANCE = 0.12; 

const PIGNA_LIFE_PENALTY = 1;
const GOLDEN_LIFE_BONUS = 1;
const GOLDEN_SCORE_BONUS = 5;

interface PigeonBlasterProps {
  onBack: () => void;
}

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
    pigeonLifetime: 3,
    maxLives: 5,
    maxPigeons: 2,
  },
  medium: {
    name: "Media",
    gameDuration: 120,
    pigeonLifetime: 2,
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

type PigeonKind = 'normal' | 'pigna' | 'golden';

type Pigeon = {
  id: number;
  x: number;
  y: number;
  age: number;
  kind: PigeonKind;
};

type FloatingEffect = {
  id: number;
  x: number;
  y: number;
  text: string;
  color: string;
  age: number;
  life: number;
};

function PigeonBlaster({ onBack }: PigeonBlasterProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef(performance.now());
  const lastPoseRef = useRef(0);
  
  const currentPoseRef = useRef<poseDetection.Pose | null>(null);

  // Array di buffer per salvare i frame estratti dai file GIF
  const normalGifFrames = useRef<HTMLCanvasElement[]>([]);
  const pignaGifFrames = useRef<HTMLCanvasElement[]>([]);
  const goldenGifFrames = useRef<HTMLCanvasElement[]>([]);
  
  // Riferimenti per le immagini dei cuori PNG
  const heartFullImgRef = useRef<HTMLImageElement | null>(null);
  const heartDeadImgRef = useRef<HTMLImageElement | null>(null);

  const wristLeftRef = useRef({ x: 0, y: 0 });
  const wristRightRef = useRef({ x: 0, y: 0 });
  const scoreRef = useRef(0);
  const livesRef = useRef(DIFFICULTIES.easy.maxLives);
  const highScoreRef = useRef(0);
  const isGameOverRef = useRef(false);
  const gameTimerRef = useRef(0);
  const isVictoryRef = useRef(false);
  const endScreenRenderedRef = useRef(false);

  // Contatori statistiche di fine partita
  const normalCaughtRef = useRef(0);
  const goldenCaughtRef = useRef(0);
  const pignaCaughtRef = useRef(0);
  const missedRef = useRef(0);

  const difficultyRef = useRef(DIFFICULTIES.easy);

  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(DIFFICULTIES.easy.maxLives);
  const [highScore, setHighScore] = useState(0);
  const [isGameOver, setIsGameOver] = useState(false);
  const [isVictory, setIsVictory] = useState(false);
  const [difficulty, setDifficulty] = useState<keyof typeof DIFFICULTIES>("easy");
  const [timeRemaining, setTimeRemaining] = useState(DIFFICULTIES.easy.gameDuration);

  useEffect(() => {
    difficultyRef.current = DIFFICULTIES[difficulty];
  }, [difficulty]);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  const isCalibratedRef = useRef(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const calibrationTimerRef = useRef(0);
  const previousWristsRef = useRef({
    left: { x: 0, y: 0 },
    right: { x: 0, y: 0 },
  });
    
  const pigeonsRef = useRef<Pigeon[]>([]);
  const pigeonIdRef = useRef(0);
  const spawnTimerRef = useRef(0);

  const effectsRef = useRef<FloatingEffect[]>([]);
  const effectIdRef = useRef(0);

  function spawnEffect(x: number, y: number, text: string, color: string, life = 1.1) {
    effectsRef.current.push({
      id: effectIdRef.current++,
      x,
      y,
      text,
      color,
      age: 0,
      life,
    });
  }

  function pickSpawnKind(): PigeonKind {
    const roll = Math.random();
    if (roll < PIGNA_SPAWN_CHANCE) return 'pigna';
    if (roll < PIGNA_SPAWN_CHANCE + GOLDEN_SPAWN_CHANCE) return 'golden';
    return 'normal';
  }

  function isTooCloseToWrist(x: number, y: number): boolean {
    const margin = 25; 
    const wrists = [wristLeftRef.current, wristRightRef.current];

    return wrists.some((wrist) => {
      if (wrist.x === 0 && wrist.y === 0) return false; 
      return (
        wrist.x >= x - margin &&
        wrist.x <= x + PIGEON_SIZE + margin &&
        wrist.y >= y - margin &&
        wrist.y <= y + PIGEON_SIZE + margin
      );
    });
  }

  function getSpawnPosition(): { x: number; y: number } {
    const maxAttempts = 15;
    let x = Math.random() * (VIDEO_WIDTH - PIGEON_SIZE);
    let y = Math.random() * (VIDEO_HEIGHT - PIGEON_SIZE);
    let attempts = 0;

    while (isTooCloseToWrist(x, y) && attempts < maxAttempts) {
      x = Math.random() * (VIDEO_WIDTH - PIGEON_SIZE);
      y = Math.random() * (VIDEO_HEIGHT - PIGEON_SIZE);
      attempts++;
    }

    return { x, y };
  }

  function spawnPigeon() {
    const { x, y } = getSpawnPosition();
    pigeonsRef.current.push({
      id: pigeonIdRef.current++,
      x,
      y,
      age: 0,
      kind: pickSpawnKind(),
    });
  }

  async function setupPoseDetector() {
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
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
        width: { ideal: VIDEO_WIDTH }, 
        height: { ideal: VIDEO_HEIGHT },
        deviceId: deviceId ? { exact: deviceId } : undefined
      },
      audio: false,
    });

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.width = VIDEO_WIDTH;
      video.height = VIDEO_HEIGHT;
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
    const config = difficultyRef.current;
    scoreRef.current = 0;
    livesRef.current = config.maxLives;
    isGameOverRef.current = false;
    isVictoryRef.current = false;
    gameTimerRef.current = 0;
    pigeonsRef.current = [];
    spawnTimerRef.current = 0;
    effectsRef.current = [];

    normalCaughtRef.current = 0;
    goldenCaughtRef.current = 0;
    pignaCaughtRef.current = 0;
    missedRef.current = 0;

    // Rimanda il giocatore alla fase di calibrazione: senza questo reset "Riprova"
    // faceva ripartire la partita a occhi chiusi, saltando il controllo che verifica
    // che i polsi siano di nuovo posizionati correttamente nel riquadro.
    isCalibratedRef.current = false;
    calibrationTimerRef.current = 0;
    endScreenRenderedRef.current = false;
    previousWristsRef.current = {
      left: { x: 0, y: 0 },
      right: { x: 0, y: 0 },
    };
    wristLeftRef.current = { x: 0, y: 0 };
    wristRightRef.current = { x: 0, y: 0 };

    setScore(0);
    setLives(config.maxLives);
    setIsGameOver(false);
    setIsVictory(false);
    setIsCalibrated(false);
    setTimeRemaining(config.gameDuration);
  }

  function restartAndResume() {
    restartGame();
    if (!animationRef.current) {
      loop();
    }
  }

  async function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;

    if (!video || !canvas || !detector) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    try {
      await runFrame(video, canvas, ctx, detector);
    } catch (err) {
      console.error("Errore nel game loop:", err);
    }

    const isOver = isGameOverRef.current || isVictoryRef.current;

    if (!isOver) {
      animationRef.current = requestAnimationFrame(loop);
    } else if (!endScreenRenderedRef.current) {
      // Il flag isGameOver/isVictory scatta a metà del frame appena eseguito, quindi
      // quel frame ha ancora disegnato la webcam come sfondo. Serve un frame in più,
      // stavolta con il flag già attivo, per far comparire davvero l'immagine tematica.
      endScreenRenderedRef.current = true;
      animationRef.current = requestAnimationFrame(loop);
    } else {
      animationRef.current = null;
    }
  }

  async function runFrame(
    video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    detector: poseDetection.PoseDetector
  ) {
    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

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

    // Come drawTextUnflipped ma per le immagini: essendo la canvas specchiata via CSS
    // (per l'effetto "specchio" della webcam), qualsiasi immagine con testo o dettagli
    // non simmetrici (cuori, schermate di fine partita) va disegnata con questo
    // contro-specchiamento, altrimenti apparirebbe rovesciata sullo schermo.
    const drawImageUnflipped = (img: HTMLImageElement, x: number, y: number, width: number, height: number) => {
      ctx.save();
      const canvasX = VIDEO_WIDTH - x - width;
      ctx.translate(canvasX, y);
      ctx.scale(-1, 1);
      ctx.drawImage(img, 0, 0, width, height);
      ctx.restore();
    };

    // A partita finita, niente più webcam sullo sfondo: si proietta l'immagine
    // tematica a schermo intero (contro-specchiata per non farla apparire rovesciata,
    // visto che la canvas è specchiata via CSS per l'effetto "specchio" della webcam).
    if (isGameOverRef.current) {
      drawImageUnflipped(gameOverImage, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    } else if (isVictoryRef.current) {
      drawImageUnflipped(victoryImage, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    } else {
      ctx.filter = 'none';
      ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    }

    const boxW = Math.round(Math.min(VIDEO_WIDTH, VIDEO_HEIGHT) * CALIBRATION_BOX_RATIO);
    const boxH = boxW;  
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
    if (activePose && !isGameOverRef.current && !isVictoryRef.current) {
      const leftWrist = getKeypoint(activePose, "left_wrist", 0.3);
      const rightWrist = getKeypoint(activePose, "right_wrist", 0.3);

      if (leftWrist) wristLeftRef.current = { x: leftWrist.x, y: leftWrist.y };
      if (rightWrist) wristRightRef.current = { x: rightWrist.x, y: rightWrist.y };
    }

    if (!isCalibratedRef.current) {
      let isPlayerReady = false;
      if (activePose) {
        const leftWrist = getKeypoint(activePose, "left_wrist", 0.4);
        const rightWrist = getKeypoint(activePose, "right_wrist", 0.4);
      
        if (leftWrist && rightWrist) {
          const wristsInsideBox =
            leftWrist.x >= boxX && leftWrist.x <= boxX + boxW &&
            leftWrist.y >= boxY && leftWrist.y <= boxY + boxH &&
            rightWrist.x >= boxX && rightWrist.x <= boxX + boxW &&
            rightWrist.y >= boxY && rightWrist.y <= boxY + boxH;
        
          const movement =
            Math.abs(leftWrist.x - previousWristsRef.current.left.x) +
            Math.abs(leftWrist.y - previousWristsRef.current.left.y) +
            Math.abs(rightWrist.x - previousWristsRef.current.right.x) +
            Math.abs(rightWrist.y - previousWristsRef.current.right.y);
        
          const isStill = movement < CALIBRATION_MOVEMENT_TOLERANCE;
        
          previousWristsRef.current.left = { x: leftWrist.x, y: leftWrist.y };
          previousWristsRef.current.right = { x: rightWrist.x, y: rightWrist.y };
        
          if (wristsInsideBox && isStill) {
            calibrationTimerRef.current = Math.min(CALIBRATION_SECONDS, calibrationTimerRef.current + dt);
            isPlayerReady = true;
          } else {
            calibrationTimerRef.current = Math.max(0, calibrationTimerRef.current - dt * 1.5);
          }
        }
      }

      if (isPlayerReady && calibrationTimerRef.current >= CALIBRATION_SECONDS) {
        isCalibratedRef.current = true;
        setIsCalibrated(true);
        calibrationTimerRef.current = CALIBRATION_SECONDS;
        spawnPigeon();
      }
    }
    else if (!isGameOverRef.current && !isVictoryRef.current) {
      gameTimerRef.current += dt;
      const gameDuration = difficultyRef.current.gameDuration;

      setTimeRemaining(Math.max(0, gameDuration - gameTimerRef.current));

      if (gameTimerRef.current >= gameDuration) {
        isVictoryRef.current = true;
        setIsVictory(true);

        if (scoreRef.current > highScoreRef.current) {
          highScoreRef.current = scoreRef.current;
          setHighScore(highScoreRef.current);
        }
      } else {
        spawnTimerRef.current += dt;
        const maxPigeons = difficultyRef.current.maxPigeons;

        if (pigeonsRef.current.length < maxPigeons && spawnTimerRef.current > 0.6) {
          spawnPigeon();
          spawnTimerRef.current = 0;
        }

        for (const pigeon of pigeonsRef.current) {
          pigeon.age += dt;
        }

        const lifetime = difficultyRef.current.pigeonLifetime;

        const expiredNormals = pigeonsRef.current.filter(
          (pigeon) => pigeon.age >= lifetime && pigeon.kind === 'normal'
        );
        pigeonsRef.current = pigeonsRef.current.filter((pigeon) => pigeon.age < lifetime);

        const removed = expiredNormals.length;

        if (removed > 0) {
          livesRef.current -= removed;
          setLives(livesRef.current);
          missedRef.current += removed;
          ctx.fillStyle = "rgba(255,0,0,0.2)";
          ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

          for (const missed of expiredNormals) {
            spawnEffect(
              missed.x + PIGEON_SIZE / 2,
              missed.y + PIGEON_SIZE / 2,
              "-1",
              "#FF4444"
            );
          }
        }

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

    const lwx = wristLeftRef.current.x;
    const lwy = wristLeftRef.current.y;
    const rwx = wristRightRef.current.x;
    const rwy = wristRightRef.current.y;

    if (!isGameOverRef.current && !isCalibratedRef.current) {
      const isHolding = calibrationTimerRef.current > 0;
      ctx.strokeStyle = isHolding ? "#00FF00" : "#FFCC00";
      ctx.lineWidth = 5;
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = "rgba(255, 204, 0, 0.15)";
      ctx.fillRect(boxX, boxY, boxW, boxH);
      drawTextUnflipped("POSIZIONA ENTRAMBI I POLSI NEL RIQUADRO", VIDEO_WIDTH / 2, boxY - 20, "bold 20px PPNeueBit", "#FFD700", "center");

      if (isHolding) {
        const remaining = Math.max(0, CALIBRATION_SECONDS - calibrationTimerRef.current);
        const calibProgress = Math.min(1, calibrationTimerRef.current / CALIBRATION_SECONDS);

        drawTextUnflipped(
          `INIZIO TRA ${remaining.toFixed(1)}s`,
          VIDEO_WIDTH / 2,
          boxY + boxH + 34,
          "bold 22px PPNeueBit",
          "#00FF66",
          "center"
        );

        const miniBarW = 180;
        const miniBarH = 10;
        const miniBarX = (VIDEO_WIDTH - miniBarW) / 2;
        const miniBarY = boxY + boxH + 44;

        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.fillRect(miniBarX, miniBarY, miniBarW, miniBarH);
        ctx.fillStyle = "#00FF66";
        ctx.fillRect(miniBarX, miniBarY, miniBarW * calibProgress, miniBarH);
      } else {
        drawTextUnflipped(
          "RESTA FERMO PER AVVIARE",
          VIDEO_WIDTH / 2,
          boxY + boxH + 34,
          "bold 18px PPNeueBit",
          "#FFCC00",
          "center"
        );
      }
    }

    // === RENDERING DEI FRAME ESTRATTI DALLE GIF ===
    if (isCalibratedRef.current && !isGameOverRef.current && !isVictoryRef.current) {
      const fallbackColors: Record<PigeonKind, string> = {
        normal: "green",
        pigna: "#8B4513",
        golden: "#FFD700",
      };

      for (const pigeon of pigeonsRef.current) {
        const frames = 
          pigeon.kind === 'pigna' ? pignaGifFrames.current :
          pigeon.kind === 'golden' ? goldenGifFrames.current :
          normalGifFrames.current;

        if (frames && frames.length > 0) {
          // Calcola il frame corrente correndo a circa 12 FPS
          const frameIndex = Math.floor((pigeon.age * 12) % frames.length);
          const activeFrame = frames[frameIndex];

          ctx.drawImage(activeFrame, pigeon.x, pigeon.y, PIGEON_SIZE, PIGEON_SIZE);
        } else {
          // Se la GIF è ancora in fase di caricamento ed elaborazione, mostra la tinta unita temporanea
          ctx.fillStyle = fallbackColors[pigeon.kind];
          ctx.fillRect(pigeon.x, pigeon.y, PIGEON_SIZE, PIGEON_SIZE);
        }
      }
    }

    if (!isGameOverRef.current && !isVictoryRef.current) {
      pigeonsRef.current = pigeonsRef.current.filter((pigeon) => {
        const hitLeft = lwx >= pigeon.x && lwx <= pigeon.x + PIGEON_SIZE && lwy >= pigeon.y && lwy <= pigeon.y + PIGEON_SIZE;
        const hitRight = rwx >= pigeon.x && rwx <= pigeon.x + PIGEON_SIZE && rwy >= pigeon.y && rwy <= pigeon.y + PIGEON_SIZE;
        
        if (hitLeft || hitRight) {
          const centerX = pigeon.x + PIGEON_SIZE / 2;
          const centerY = pigeon.y + PIGEON_SIZE / 2;

          if (pigeon.kind === 'pigna') {
            pignaCaughtRef.current++;
            livesRef.current = Math.max(0, livesRef.current - PIGNA_LIFE_PENALTY);
            setLives(livesRef.current);
            spawnEffect(centerX, centerY, `-${PIGNA_LIFE_PENALTY}`, "#FF4444");
          } else if (pigeon.kind === 'golden') {
            goldenCaughtRef.current++;
            scoreRef.current += GOLDEN_SCORE_BONUS;
            setScore(scoreRef.current);
            livesRef.current = Math.min(difficultyRef.current.maxLives, livesRef.current + GOLDEN_LIFE_BONUS);
            setLives(livesRef.current);
            spawnEffect(centerX, centerY - 14, `+${GOLDEN_SCORE_BONUS}`, "#FFD700");
            spawnEffect(centerX, centerY + 14, `+${GOLDEN_LIFE_BONUS} VITA`, "#00FF66", 1.3);
          } else {
            normalCaughtRef.current++;
            scoreRef.current++;
            setScore(scoreRef.current);
            spawnEffect(centerX, centerY, "+1", "#00FF66");
          }
          return false;
        }
        return true;
      });

      if (livesRef.current <= 0 && !isGameOverRef.current) {
        isGameOverRef.current = true;
        setIsGameOver(true);

        if (scoreRef.current > highScoreRef.current) {
          highScoreRef.current = scoreRef.current;
          setHighScore(highScoreRef.current);
        }
      }
    }

    effectsRef.current.forEach((effect) => {
      effect.age += dt;
    });
    effectsRef.current = effectsRef.current.filter((effect) => effect.age < effect.life);

    for (const effect of effectsRef.current) {
      const progress = effect.age / effect.life;
      const floatY = effect.y - progress * 45;
      const alpha = 1 - progress;

      ctx.globalAlpha = Math.max(0, alpha);
      drawTextUnflipped(effect.text, effect.x, floatY, "bold 24px PPNeueBit", effect.color, "center");
    }
    ctx.globalAlpha = 1;

    const startX = 30;
    const heartWidth = 35;
    const gap = 8;

    for (let i = 0; i < difficultyRef.current.maxLives; i++) {
      const isAlive = i < livesRef.current;
      const currentX = startX + i * (heartWidth + gap);
    
      if (isAlive && heartFullImgRef.current && heartFullImgRef.current.complete) {
        drawImageUnflipped(heartFullImgRef.current, currentX, 20, heartWidth, heartWidth);
      } else if (!isAlive && heartDeadImgRef.current && heartDeadImgRef.current.complete) {
        drawImageUnflipped(heartDeadImgRef.current, currentX, 20, heartWidth, heartWidth);
      }
    }

    const remainingTime = Math.max(0, difficultyRef.current.gameDuration - gameTimerRef.current);
    const percentage = remainingTime / difficultyRef.current.gameDuration;
    const barY = VIDEO_HEIGHT - 12;
    const barW = VIDEO_WIDTH - 20;
    
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(10, barY, barW, 8);
    
    const grad = ctx.createLinearGradient(10, 0, 10 + barW * percentage, 0);
    grad.addColorStop(0, percentage > 0.3 ? "#00FF66" : "#FF4444");
    grad.addColorStop(1, percentage > 0.3 ? "#00CCAA" : "#FF8800");
    ctx.fillStyle = grad;
    ctx.fillRect(10, barY, barW * percentage, 8);
  }

  useEffect(() => {
    const imgFull = new Image();
    imgFull.src = cuoricino_vite;
    heartFullImgRef.current = imgFull;

    const imgDead = new Image();
    imgDead.src = vita;
    heartDeadImgRef.current = imgDead;

    // Estrazione asincrona e sicura dei fotogrammi tramite l'oggetto interno _frames di gifler.
    // Usiamo extractGifFrames invece del vecchio ".map(f => f.buffer)" perché gifler non crea
    // f.buffer finché la GIF non viene animata: senza questo fix ogni frame risultava
    // "undefined" e ctx.drawImage falliva (canvas nera + spam di errori in console).
    if (gifler) {
      gifler(piccione_volante).get((anim: any) => {
        normalGifFrames.current = extractGifFrames(anim);
      });

      gifler(pigna).get((anim: any) => {
        pignaGifFrames.current = extractGifFrames(anim);
      });

      gifler(golderBuzz).get((anim: any) => {
        goldenGifFrames.current = extractGifFrames(anim);
      });
    }

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

  const timePct = timeRemaining / difficultyRef.current.gameDuration;
  // La difficoltà può essere cambiata solo prima che la partita cominci (fase di
  // calibrazione) o dopo che è finita (game over / vittoria), mai mentre si gioca:
  // cambiarla a metà partita altera durata, vite massime e velocità di spawn senza
  // che i valori correnti (livesRef, gameTimerRef, ecc.) vengano risincronizzati.
  const isPlaying = isCalibrated && !isGameOver && !isVictory;

  return (
    <div className="game-container">
      <div className="header-container">
        <img src={piccioneTerra} alt="Piccione decorativo sinistro" className="pigeon-decor pigeon-left" />
        <div className="title-wrapper">
          <h1 className="main-title">Pigeon Blaster</h1>
          <span className="subtitle">rehab edition</span>
        </div>
        <img src={piccioneTerra} alt="Piccione decorativo destro" className="pigeon-decor pigeon-right" />
      </div>

      <div className="hud-bar">
        <div className="hud-section hud-score">
          <span className="hud-label">Score</span>
          <span className="hud-value">{score}</span>
        </div>

        <div className="hud-section hud-timer">
          <span className="hud-label">Tempo</span>
          <span className="hud-value hud-timer-value" style={{ color: timePct < 0.3 ? '#FF4444' : '#00FFCC' }}>
            {Math.ceil(timeRemaining)}s
          </span>
          <div className="timer-bar-track">
            <div
              className="timer-bar-fill"
              style={{
                width: `${timePct * 100}%`,
                background: timePct > 0.3
                  ? 'linear-gradient(90deg, #00FF66, #00CCAA)'
                  : 'linear-gradient(90deg, #FF4444, #FF8800)',
              }}
            />
          </div>
        </div>

        <div className="hud-section hud-highscore">
          <span className="hud-label">Record</span>
          <span className="hud-value hud-highscore-value">{highScore}</span>
        </div>
      </div>

      <div className="controls-bar">
        <div className="control-group">
          <label htmlFor="camera-select" className="camera-label"> Fotocamera:</label>
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

        <div className="control-group">
          <label className="camera-label"> Difficoltà:</label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value as keyof typeof DIFFICULTIES)}
            className="camera-select"
            disabled={isPlaying}
            title={isPlaying ? "Non puoi cambiare la difficoltà mentre stai giocando" : undefined}
          >
            <option value="easy">Facile</option>
            <option value="medium">Media</option>
            <option value="hard">Difficile</option>
          </select>
        </div>

        {(isGameOver || isVictory) && (
          <button onClick={restartAndResume} className="btn-retry">
            Riprova
          </button>
        )}
      </div>

      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          className="game-canvas"
        />

        {/* Testo di fine partita come overlay HTML vero (non disegnato sulla canvas):
            resta nitido invece di apparire sfocato/pixelato come un canvas.fillText,
            e non essendo dentro l'elemento con transform: scaleX(-1) non viene specchiato. */}
        {(isGameOver || isVictory) && (
          <div className={`end-screen-overlay ${isVictory ? "is-victory" : "is-defeat"}`}>
            <h2 className="end-screen-title">{isVictory ? "HAI VINTO!" : "HAI PERSO"}</h2>
            <p className="end-screen-subtitle">Riepilogo Partita</p>

            <ul className="end-screen-stats">
              <li><span>Piccioni Presi</span><b>{normalCaughtRef.current}</b></li>
              <li className="stat-golden"><span>Piccioni Dorati Presi</span><b>{goldenCaughtRef.current}</b></li>
              <li className="stat-pigna"><span>Pigne Prese</span><b>{pignaCaughtRef.current}</b></li>
              <li className="stat-missed"><span>Piccioni Mancati</span><b>{missedRef.current}</b></li>
              <li className="stat-lives"><span>Vite Rimaste</span><b>{livesRef.current}/{difficultyRef.current.maxLives}</b></li>
            </ul>

            <p className="end-screen-score">Punteggio: <b>{scoreRef.current}</b></p>
            <p className="end-screen-record">Record: <b>{highScoreRef.current}</b></p>
            <p className="end-screen-hint">Premi RIPROVA per giocare ancora</p>
          </div>
        )}
      </div>

      <video ref={videoRef} width={VIDEO_WIDTH} height={VIDEO_HEIGHT} style={{ display: "none" }}/>

      <div className="bottom-bar">
        <button className="btn-back-pigeon" onClick={onBack}>
          ⬅ Torna al Menu
        </button>
      </div>
    </div>
  );
}

export default PigeonBlaster;