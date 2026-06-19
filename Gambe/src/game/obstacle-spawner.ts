export type ObstacleLane = 'left' | 'center' | 'right';

export interface Obstacle {
  id: number;
  lane: ObstacleLane;
  y: number;
  speed: number;
  imageKey: string;
}

const OBSTACLE_IMAGES = ['cespuglio.png', 'cactus.png', 'barile.png'];
const LANES: ObstacleLane[] = ['left', 'center', 'right'];

let nextId = 0;
let lastSpawnTime = 0;

// Restituisce lo spawn interval in ms in base al punteggio
function spawnInterval(score: number): number {
  return Math.max(1100, 2800 - score * 8);
}

// Restituisce la velocità in px/ms in base al punteggio
function obstacleSpeed(score: number): number {
  return 0.20 + score * 0.0001;
}

export function updateObstacles(
  obstacles: Obstacle[],
  deltaMs: number,
  now: number,
  score: number,
  frameHeight: number
): { obstacles: Obstacle[]; missed: number } {
  // Muovi tutti gli ostacoli verso il basso
  const moved = obstacles.map(o => ({ ...o, y: o.y + o.speed * deltaMs }));

  // Separa quelli ancora in campo da quelli usciti
  const alive = moved.filter(o => o.y < frameHeight + 100);
  const missed = moved.length - alive.length;

  // Spawn nuovo ostacolo se è passato abbastanza tempo
  let updated = alive;
  if (now - lastSpawnTime > spawnInterval(score)) {
    const lane = LANES[Math.floor(Math.random() * LANES.length)];
    const imageKey = OBSTACLE_IMAGES[Math.floor(Math.random() * OBSTACLE_IMAGES.length)];
    updated = [
      ...alive,
      { id: nextId++, lane, y: -100, speed: obstacleSpeed(score), imageKey },
    ];
    lastSpawnTime = now;
  }

  return { obstacles: updated, missed };
}

export function resetSpawner() {
  nextId = 0;
  lastSpawnTime = 0;
}