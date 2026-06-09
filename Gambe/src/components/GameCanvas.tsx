import { useRef, useEffect } from 'react';
import type { Pose } from '@tensorflow-models/pose-detection/dist/types';
import type { Obstacle } from '../game/obstacle-spawner';
import type { Lane } from '../game/lane-detector';
import type { GameState } from '../hooks/useGameState';

const LANE_COLORS = {
  left:   'rgba(255,255,255,0.08)',
  center: 'rgba(255,255,255,0.08)',
  right:  'rgba(255,255,255,0.08)',
};
const LANE_ACTIVE_COLOR = 'rgba(99,202,255,0.18)';
const HIT_ZONE_Y = 380;
const OBSTACLE_SIZE = 64;

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  pose: Pose | null;
  obstacles: Obstacle[];
  playerLane: Lane;
  gameState: GameState;
  width: number;
  height: number;
  obstacleImages: Record<string, HTMLImageElement>;
}

export function GameCanvas({
  videoRef, pose: _pose, obstacles, playerLane,
  gameState, width, height, obstacleImages,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Pulisci
    ctx.clearRect(0, 0, width, height);

    // Disegna il feed video (specchiato)
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -width, 0, width, height);
      ctx.restore();
    }

    // Larghezza corsia
    const laneW = width / 3;
    const lanes: Lane[] = ['left', 'center', 'right'];

    // Disegna le tre corsie
    lanes.forEach((lane, i) => {
      ctx.fillStyle = lane === playerLane ? LANE_ACTIVE_COLOR : LANE_COLORS[lane];
      ctx.fillRect(i * laneW, 0, laneW, height);

      // Bordi corsia
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(i * laneW, 0, laneW, height);
    });

    // Hit zone
    ctx.strokeStyle = 'rgba(255,80,80,0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(0, HIT_ZONE_Y, width, 80);
    ctx.setLineDash([]);

    // Ostacoli
    obstacles.forEach(obstacle => {
      const laneIndex = lanes.indexOf(obstacle.lane);
      const x = laneIndex * laneW + laneW / 2 - OBSTACLE_SIZE / 2;
      const img = obstacleImages[obstacle.imageKey];
      if (img) {
        ctx.drawImage(img, x, obstacle.y, OBSTACLE_SIZE, OBSTACLE_SIZE);
      } else {
        // Fallback se l'immagine non è caricata
        ctx.fillStyle = 'red';
        ctx.fillRect(x, obstacle.y, OBSTACLE_SIZE, OBSTACLE_SIZE);
      }
    });

    // Indicatore corsia giocatore (cerchio sopra la hit zone)
    const playerLaneIndex = lanes.indexOf(playerLane);
    const px = playerLaneIndex * laneW + laneW / 2;
    ctx.beginPath();
    ctx.arc(px, HIT_ZONE_Y - 20, 18, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(99,202,255,0.9)';
    ctx.fill();

    // HUD — punteggio e vite
    ctx.fillStyle = 'white';
    ctx.font = 'bold 20px monospace';
    ctx.fillText(`Score: ${gameState.score}`, 12, 28);
    ctx.fillText(`${'❤️'.repeat(gameState.lives)}`, width - 90, 28);

    // Schermate idle / gameover
    if (gameState.status === 'idle' || gameState.status === 'gameover') {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = 'white';
      ctx.font = 'bold 36px monospace';
      ctx.textAlign = 'center';

      if (gameState.status === 'idle') {
        ctx.fillText('GAMBE 🦵', width / 2, height / 2 - 20);
        ctx.font = '18px monospace';
        ctx.fillText('Premi SPAZIO per iniziare', width / 2, height / 2 + 20);
      } else {
        ctx.fillText('GAME OVER', width / 2, height / 2 - 20);
        ctx.font = '18px monospace';
        ctx.fillText(`Score finale: ${gameState.score}`, width / 2, height / 2 + 20);
        ctx.fillText('Premi SPAZIO per riprovare', width / 2, height / 2 + 50);
      }

      ctx.textAlign = 'left';
    }


  }); // nessuna dependency: ridisegna ad ogni render

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="canvas"
      style={{ 
        border: "2px solid black", 
        maxWidth: "100%", 
        height: "auto"
      }}
    />
  );
}