import { useState, useCallback } from 'react';
import { usePoseDetector } from './hooks/usePoseDetector';
import { useGameLoop } from './hooks/useGameLoop';
import { useGameState } from './hooks/useGameState';
import { detectLane } from './game/lane-detector';
import { updateObstacles } from './game/obstacle-spawner';
import { checkCollisions } from './game/collision';
import { GameCanvas } from './components/GameCanvas';
import type { Obstacle } from './game/obstacle-spawner';
import type { Lane } from './game/lane-detector';
import type { Pose } from '@tensorflow-models/pose-detection/dist/types';
import './App.css';

// Carica le immagini degli ostacoli una volta sola
const OBSTACLE_KEYS = ['roccia.png', 'cactus.png', 'barile.png'];
const obstacleImages: Record<string, HTMLImageElement> = {};
OBSTACLE_KEYS.forEach(key => {
  const img = new Image();
  img.src = `/obstacles/${key}`;
  obstacleImages[key] = img;
});

export default function App() {
  const { videoRef, detectPose, VIDEO_WIDTH, VIDEO_HEIGHT } = usePoseDetector();
  const { state, start, addScore, loseLife, reset } = useGameState();

  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [playerLane, setPlayerLane] = useState<Lane>('center');
  const [pose, setPose] = useState<Pose | null>(null);

  // Tasto SPAZIO per start / restart
  useState(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      if (state.status === 'idle' || state.status === 'gameover') {
        reset();
        start();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  });

  const gameLoop = useCallback(async (deltaMs: number, now: number) => {
    // 1. Rileva posa
    const currentPose = await detectPose();
    setPose(currentPose);

    if (state.status !== 'playing') return;

    // 2. Corsia giocatore
    if (currentPose) {
      const lane = detectLane(currentPose, VIDEO_WIDTH);
      setPlayerLane(lane);
    }

    // 3. Aggiorna ostacoli
    const { obstacles: updated, missed } = updateObstacles(
      obstacles, deltaMs, now, state.score, VIDEO_HEIGHT
    );

    // 4. Collisioni
    const { hit, safeObstacles } = checkCollisions(updated, playerLane);

    setObstacles(safeObstacles);

    if (missed > 0) addScore(missed * 10);
    if (hit) loseLife();

  }, [state, obstacles, playerLane, detectPose, addScore, loseLife, VIDEO_WIDTH, VIDEO_HEIGHT]);

  useGameLoop(gameLoop, true);

  return (
     <div className="app-container">
      {/* HEADER */}
      <header className="app-header">
        <h1>Outlaw Run</h1>
      </header>

      <p>Chi l'ha detto che la riabilitazione è solo noia? Metti in moto le gambe, schiva cactus, barili e rocce come un vero cowboy e trasforma la tua terapia nella sfida pixelata più divertente del West!</p>

      <div className="game-stats">
          <span>Punteggio: {state.score}</span>
      </div>

      {/* CONTENITORE CENTRALE DEL GIOCO */}
      <main className="game-main">
        <div className="stage">
          <video ref={videoRef} className="hidden-video" playsInline />
          <GameCanvas
            videoRef={videoRef}
            pose={pose}
            obstacles={obstacles}
            playerLane={playerLane}
            gameState={state}
            width={VIDEO_WIDTH}
            height={VIDEO_HEIGHT}
            obstacleImages={obstacleImages}
          />
        </div>
      </main>

      

      {/* FOOTER */}
      <footer className="app-footer">
        <h3>Usa i movimenti del corpo per giocare | Alimentato da TensorFlow.js</h3>
      </footer>
    </div>
  );


}