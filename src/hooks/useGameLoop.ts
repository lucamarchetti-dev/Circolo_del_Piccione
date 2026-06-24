import { useEffect, useRef } from 'react';

export function useGameLoop(callback: (deltaMs: number, now: number) => void, active: boolean) {
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const callbackRef = useRef(callback);

  // Tieni sempre aggiornato il ref senza re-registrare il loop
  callbackRef.current = callback;

  useEffect(() => {
    if (!active) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTimeRef.current = null;
      return;
    }

    function loop(now: number) {
      const delta = lastTimeRef.current !== null ? now - lastTimeRef.current : 16;
      lastTimeRef.current = now;
      callbackRef.current(delta, now);
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTimeRef.current = null;
    };
  }, [active]);
}