import type { Obstacle } from './obstacle-spawner';
import type { Lane } from './lane-detector';

// Soglia in px: se un ostacolo scende oltre questo Y viene considerato nella zona pericolosa
const HIT_ZONE_Y = 380;
const HIT_ZONE_HEIGHT = 80;

export function checkCollisions(
  obstacles: Obstacle[],
  playerLane: Lane
): { hit: boolean; safeObstacles: Obstacle[] } {
  const safeObstacles: Obstacle[] = [];
  let hit = false;

  for (const obstacle of obstacles) {
    const inHitZone =
      obstacle.y + 80 >= HIT_ZONE_Y &&
      obstacle.y <= HIT_ZONE_Y + HIT_ZONE_HEIGHT;

    if (inHitZone && obstacle.lane === playerLane) {
      hit = true;
    } else {
      safeObstacles.push(obstacle);
    }
  }

  return { hit, safeObstacles };
}