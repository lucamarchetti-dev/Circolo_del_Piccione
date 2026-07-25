import { useState, useCallback } from 'react';
import { resetSpawner } from '../game/obstacle-spawner';
import { resetLaneHistory } from '../game/lane-detector';

export type GameStatus = 'idle' | 'playing' | 'gameover';

export interface GameState {
  status: GameStatus;
  score: number;
  lives: number;
}

const INITIAL_STATE: GameState = {
  status: 'idle',
  score: 0,
  lives: 3,
};

export function useGameState() {
  const [state, setState] = useState<GameState>(INITIAL_STATE);

  const start = useCallback(() => {
    resetSpawner();
    resetLaneHistory();
    setState({ status: 'playing', score: 0, lives: 3 });
  }, []);

  const addScore = useCallback((points: number) => {
    setState(prev => ({ ...prev, score: prev.score + points }));
  }, []);

  const loseLife = useCallback(() => {
    setState(prev => {
      const lives = prev.lives - 1;
      return lives <= 0
        ? { ...prev, lives: 0, status: 'gameover' }
        : { ...prev, lives };
      });
  }, []);

  const reset = useCallback(() => {
    resetSpawner();
    resetLaneHistory();
    setState(INITIAL_STATE);
  }, []);

  return { state, start, addScore, loseLife, reset };
}