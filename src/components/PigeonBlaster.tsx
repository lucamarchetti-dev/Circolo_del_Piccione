import React, { useEffect, useRef, useState } from 'react';
import './PigeonBlaster.css'; 
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { drawKeypoints, drawArm, getKeypoint } from '../pose-utils'; 
import piccioneTerra from '../assets/png/piccione-terra.png';
import piccione_volante from '../assets/png/piccione-volante.png';
import cuoricino_vite from '../assets/png/cuoricino-vite.png';
import vita from '../assets/png/vita.png';

//immagine sfondo in caso game over
import gameOverBg from "../assets/png/game-over.png";
const gameOverImage = new Image();
gameOverImage.src = gameOverBg;

// dichiarazione dimensione canva
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const PIGEON_SIZE = 60;

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

type Pigeon = {
  id: number;
  x: number;
  y: number;
  age: number;
};

function PigeonBlaster({ onBack }: PigeonBlasterProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef(performance.now());
  const lastPoseRef = useRef(0);
  
  const currentPoseRef = useRef<poseDetection.Pose | null>(null);
  const PigeonImageRef = useRef<HTMLImageElement | null>(null);
  
  // Riferimenti per le immagini dei cuori PNG
  const heartFullImgRef = useRef<HTMLImageElement | null>(null);
  const heartDeadImgRef = useRef<HTMLImageElement | null>(null);

  // Riferimenti per i calcoli interni al ciclo di animazione
  const wristLeftRef = useRef({ x: 0, y: 0 });
  const wristRightRef = useRef({ x: 0, y: 0 });
  const scoreRef = useRef(0);
  const livesRef = useRef(DIFFICULTIES.easy.maxLives);
  const highScoreRef = useRef(0);
  const isGameOverRef = useRef(false);
  const gameTimerRef = useRef(0);
  const isVictoryRef = useRef(false);

  // FIX: Spostato difficultyRef al livello base del componente
  const difficultyRef = useRef(DIFFICULTIES.easy);

  // Stati di React per l'interfaccia
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(DIFFICULTIES.easy.maxLives);
  const [highScore, setHighScore] = useState(0);
  const [isGameOver, setIsGameOver] = useState(false);
  const [isVictory, setIsVictory] = useState(false);
  const [difficulty, setDifficulty] = useState<keyof typeof DIFFICULTIES>("easy");

  // FIX: Questo useEffect va qui, non dentro la funzione loop()! 
  // Aggiorna la referenza quando cambia lo stato della difficoltà
  useEffect(() => {
    difficultyRef.current = DIFFICULTIES[difficulty];
  }, [difficulty]);

  // Gestione fotocamere multiple
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Riferimenti per la zona di calibrazione iniziale
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

  function spawnPigeon() {
    pigeonsRef.current.push({
      id: pigeonIdRef.current++,
      x: Math.random() * (VIDEO_WIDTH - PIGEON_SIZE),
      y: Math.random() * (VIDEO_HEIGHT - PIGEON_SIZE),
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
  //#endregion

  // FIX: Ristrutturate le funzioni di riavvio in modo che non siano nidificate
  function restartGame() {
    const config = difficultyRef.current;
    
    // Reset Refs (Logica di gioco)
    scoreRef.current = 0;
    livesRef.current = config.maxLives;
    isGameOverRef.current = false;
    isVictoryRef.current = false;
    gameTimerRef.current = 0;
    pigeonsRef.current = [];
    spawnTimerRef.current = 0;

    // Reset State (Interfaccia React)
    setScore(0);
    setLives(config.maxLives);
    setIsGameOver(false);
    setIsVictory(false);
  }

  function restartAndResume() {
    restartGame();
    // Fa ripartire il loop solo se era fermo
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

    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

    if (isGameOverRef.current || isVictoryRef.current) {
      ctx.drawImage(
        gameOverImage,
        0,
        0,
        VIDEO_WIDTH,
        VIDEO_HEIGHT
      );
    } else {
      ctx.filter = 'none';
    }

    ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    // Helper funzioni rendering
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

    // Overlay Vittoria
    if (isVictoryRef.current) {
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
      drawTextUnflipped("HAI VINTO!", VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 - 40, "bold 20px PPNeueBit", "#00FF66", "center");
      drawTextUnflipped(`Piccioni presi: ${scoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 20, "bold 20px PPNeueBit", "white", "center");
    }

    const boxW = 300;
    const boxH = boxW;  
    const boxX = (VIDEO_WIDTH - boxW) / 2;
    const boxY = (VIDEO_HEIGHT - boxH) / 2;

    // Aggiornamento posizioni Pose (limitato a ~30fps per performance)
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

      // drawKeypoints è disabilitato: commentare la riga seguente per nascondere
      // i pallini sulle articolazioni/faccia della persona rilevata dalla posa.
      // drawKeypoints(ctx, activePose, 0.3);
      drawArm(ctx, activePose, 'left', 0.3);
      drawArm(ctx, activePose, 'right', 0.3);
    }

    // DINAMICHE DI GIOCO

    //#region CONTROLLO POSIZIONE (Calibrazione)
    if (!isCalibratedRef.current) {
      let isPlayerReady = false;
      if (activePose) {
        const leftWrist = getKeypoint(activePose, "left_wrist", 0.5);
        const rightWrist = getKeypoint(activePose, "right_wrist", 0.5);
      
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
        
          const isStill = movement < 25;
        
          previousWristsRef.current.left = { x: leftWrist.x, y: leftWrist.y };
          previousWristsRef.current.right = { x: rightWrist.x, y: rightWrist.y };
        
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
    else if (!isGameOverRef.current && !isVictoryRef.current) {
      gameTimerRef.current += dt;
      const gameDuration = difficultyRef.current.gameDuration;

      // Controllo Vittoria per scadenza tempo
      if (gameTimerRef.current >= gameDuration) {
        isVictoryRef.current = true;
        setIsVictory(true);
        // Non facciamo `return` qui così l'interfaccia si aggiorna al prossimo frame
      } else {
        spawnTimerRef.current += dt;
        const maxPigeons = difficultyRef.current.maxPigeons;

        // Spawn nuovi piccioni
        if (pigeonsRef.current.length < maxPigeons && spawnTimerRef.current > 0.6) {
          spawnPigeon();
          spawnTimerRef.current = 0;
        }

        // Invecchiamento piccioni
        for (const pigeon of pigeonsRef.current) {
          pigeon.age += dt;
        }

        // Rimozione piccioni scaduti
        const lifetime = difficultyRef.current.pigeonLifetime;
        const beforeCount = pigeonsRef.current.length;

        pigeonsRef.current = pigeonsRef.current.filter((pigeon) => pigeon.age < lifetime);

        const removed = beforeCount - pigeonsRef.current.length;

        // Perdita di vite
        if (removed > 0) {
          livesRef.current -= removed;
          setLives(livesRef.current);
          ctx.fillStyle = "rgba(255,0,0,0.2)";
          ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
        }

        // Controllo Sconfitta
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
    //#endregion

    const lwx = wristLeftRef.current.x;
    const lwy = wristLeftRef.current.y;
    const rwx = wristRightRef.current.x;
    const rwy = wristRightRef.current.y;

    const drawImageUnflipped = (img: HTMLImageElement, x: number, y: number, width: number, height: number) => {
      ctx.save();
      const canvasX = VIDEO_WIDTH - x - width;
      ctx.translate(canvasX, y);
      ctx.scale(-1, 1);
      ctx.drawImage(img, 0, 0, width, height);
      ctx.restore();
    };

    // Rendering box di calibrazione
    if (!isGameOverRef.current && !isCalibratedRef.current) {
      ctx.strokeStyle = calibrationTimerRef.current > 0 ? "#00FF00" : "#FFCC00";
      ctx.lineWidth = 5;
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = "rgba(255, 204, 0, 0.15)";
      ctx.fillRect(boxX, boxY, boxW, boxH);
      drawTextUnflipped("POSIZIONA ENTRAMBI I POLSI NEL RIQUADRO", VIDEO_WIDTH / 2, boxY - 20, "bold 20px PPNeueBit", "#FFD700", "center");
    }

    // Rendering piccioni
    if (isCalibratedRef.current && !isGameOverRef.current && !isVictoryRef.current) {
      const imgValid = PigeonImageRef.current && PigeonImageRef.current.complete && PigeonImageRef.current.naturalWidth > 0;
      
      for (const pigeon of pigeonsRef.current) {
        if (imgValid) {
          try {
            ctx.drawImage(PigeonImageRef.current!, pigeon.x, pigeon.y, PIGEON_SIZE, PIGEON_SIZE);
          } catch (e) {
            ctx.fillStyle = "green";
            ctx.fillRect(pigeon.x, pigeon.y, PIGEON_SIZE, PIGEON_SIZE);
          }
        } else {
          ctx.fillStyle = "green";
          ctx.fillRect(pigeon.x, pigeon.y, PIGEON_SIZE, PIGEON_SIZE);
        }
      }
    }

    // Controllo Collisioni (Hit) - Ottimizzato
    if (!isGameOverRef.current && !isVictoryRef.current) {
      pigeonsRef.current = pigeonsRef.current.filter((pigeon) => {
        const hitLeft = lwx >= pigeon.x && lwx <= pigeon.x + PIGEON_SIZE && lwy >= pigeon.y && lwy <= pigeon.y + PIGEON_SIZE;
        const hitRight = rwx >= pigeon.x && rwx <= pigeon.x + PIGEON_SIZE && rwy >= pigeon.y && rwy <= pigeon.y + PIGEON_SIZE;
        
        if (hitLeft || hitRight) {
          scoreRef.current++;
          setScore(scoreRef.current);
          return false; // Rimuove il piccione colpito
        }
        return true;
      });
    }

    // Rendering HUD Vite
    const startX = 30;
    const heartWidth = 35;
    const gap = 8;

    for (let i = 0; i < difficultyRef.current.maxLives; i++) {
      const isAlive = i < livesRef.current;
      const currentX = startX + i * (heartWidth + gap);
    
      if (isAlive && heartFullImgRef.current && heartFullImgRef.current.complete) {
        drawImageUnflipped(heartFullImgRef.current, currentX, 30, heartWidth, heartWidth);
      } else if (!isAlive && heartDeadImgRef.current && heartDeadImgRef.current.complete) {
        drawImageUnflipped(heartDeadImgRef.current, currentX, 30, heartWidth, heartWidth);
      }
    }

    // Testi UI
    drawTextUnflipped(`Punteggio: ${scoreRef.current}`, 30, 115, "bold 20px PPNeueBit", "white");
    drawTextUnflipped(`Record: ${highScoreRef.current}`, 30, 150, "bold 20px PPNeueBit", "#FFD700");

    // Rendering Schermata Game Over
    if (isGameOverRef.current) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

      drawTextUnflipped("HAI PERSO", VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 - 30, "bold 65px PPNeueBit", "#ffffff", "center");
      drawTextUnflipped(`Piccioni Presi: ${scoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 20, "bold 24px PPNeueBit", "white", "center");
      drawTextUnflipped(`Record Massimo: ${highScoreRef.current}`, VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2 + 55, "bold 22px PPNeueBit", "#FFD700", "center");
    }

    // Disegno della barra del tempo (condiviso tra game over, in gioco e vittoria)
    const remainingTime = Math.max(0, difficultyRef.current.gameDuration - gameTimerRef.current);
    drawTextUnflipped(`Tempo: ${Math.ceil(remainingTime)}s`, VIDEO_WIDTH / 2, 40, "bold 20px PPNeueBit", "#00FFFF", "center");
    drawTextUnflipped(difficultyRef.current.name.toUpperCase(), VIDEO_WIDTH - 120, 40, "bold 20px PPNeueBit", "#FFFFFF");

    const percentage = remainingTime / difficultyRef.current.gameDuration;
    ctx.fillStyle = "#222";
    ctx.fillRect(VIDEO_WIDTH / 2 - 120, 55, 240, 18);
    ctx.fillStyle = percentage > 0.3 ? "#00FF66" : "#FF4444";
    ctx.fillRect(VIDEO_WIDTH / 2 - 120, 55, 240 * percentage, 18);

    // Gestione Loop (si ferma solo se l'utente cambia camera o il componente smonta)
    if (!isGameOverRef.current && !isVictoryRef.current) {
      animationRef.current = requestAnimationFrame(loop);
    } else {
      // In caso di game over / vittoria, azzeriamo il ref così sappiamo che è fermo
      animationRef.current = null;
    }
  }

  // Inizializzazione assets e Tensorflow
  useEffect(() => {
    const img = new Image();
    img.src = piccione_volante;
    PigeonImageRef.current = img;

    const imgFull = new Image();
    imgFull.src = cuoricino_vite;
    heartFullImgRef.current = imgFull;

    const imgDead = new Image();
    imgDead.src = vita;
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

  // Avvia/Cambia fotocamera
  useEffect(() => {
    if (selectedDeviceId) {
      startCamera(selectedDeviceId);
    }
  }, [selectedDeviceId]);

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
      
      <div className="stats-bar">
        <span>Punteggio Corrente: <strong>{score}</strong></span>
        <span>Vite:<strong>{lives}/{DIFFICULTIES[difficulty].maxLives}</strong></span>
        <span>Record Attuale: <strong>{highScore}</strong></span>
      </div>

      <div className="controls-bar">
        <div>
          <label htmlFor="camera-select" className="camera-label">Fotocamera: </label>
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
          // Ora questo bottone troverà la funzione al giusto livello!
          <button onClick={restartAndResume} className="btn-retry">
            Riprova
          </button>
        )}
      </div>

      <div className='selettore-difficoulty'>
        <label style={{ marginRight: "10px", fontWeight: "bold" }}>Difficoltà:</label>
        <select
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value as keyof typeof DIFFICULTIES)}
          className="camera-select"
        >
          <option value="easy">Facile</option>
          <option value="medium">Media</option>
          <option value="hard">Difficile</option>
        </select>
      </div>

      <canvas
        ref={canvasRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        className="game-canvas"
      />

      <video ref={videoRef} width={VIDEO_WIDTH} height={VIDEO_HEIGHT} style={{ display: "none" }}/>

      <div>
        <button onClick={onBack}>⬅ Torna al Menu</button>
        <h1>Pigeon Blaster Attivo!</h1>
      </div>

    </div>
  );
}

export default PigeonBlaster;