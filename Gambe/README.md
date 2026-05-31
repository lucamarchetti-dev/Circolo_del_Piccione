# Gambe 🦵

> ⚠️ **Work in Progress** — progetto in sviluppo attivo. La struttura e i file possono cambiare.

Minigioco browser a tre corsie controllato dal movimento del corpo, sviluppato per il **Body Motion & AI Hackathon** di ITS Academy Lanciano.

Il giocatore usa le proprie gambe e anche come controller: spostandosi fisicamente a sinistra, al centro o a destra cambia corsia per evitare gli ostacoli in arrivo.

---

## Stack

- React 19 + TypeScript
- Vite
- TensorFlow.js (backend WebGL)
- MoveNet SINGLEPOSE_LIGHTNING

---

## Come funziona

La webcam riprende il giocatore. MoveNet rileva i keypoint delle anche (`left_hip`, `right_hip`) in tempo reale. La posizione media X delle anche viene tradotta in una delle tre corsie. Gli ostacoli scendono dall'alto: bisogna spostarsi nella corsia libera per evitarli.

---

## Stato attuale

| Modulo | Stato |
|---|---|
| Setup TF.js + MoveNet + webcam | ✅ Fatto |
| `pose-utils.ts` — keypoint e disegno | ✅ Fatto |
| `usePoseDetector.ts` — hook camera + rilevamento | ✅ Fatto |
| `lane-detector.ts` — traduzione posa → corsia | ✅ Fatto |
| `obstacle-spawner.ts` — generazione ostacoli | ⏳ In corso |
| `collision.ts` — rilevamento collisioni | ⏳ In corso |
| `useGameLoop.ts` — loop rAF + delta time | 🔲 Da fare |
| `useGameState.ts` — punteggio, vite, stato | 🔲 Da fare |
| `GameCanvas.tsx` — rendering canvas | 🔲 Da fare |
| `App.tsx` — composizione finale | 🔲 Da fare |

---

## Installazione

```bash
npm install
```

## Avvio

```bash
npm run dev
```

Aprire `http://localhost:5173` con un browser moderno. Concedere l'accesso alla webcam quando richiesto. Posizionarsi a circa 1-2 metri dalla camera in modo che le anche siano visibili nel frame.

---

## Struttura del progetto

```
src/
├── hooks/
│   ├── usePoseDetector.ts   # Camera + MoveNet
│   ├── useGameLoop.ts       # requestAnimationFrame + delta time
│   └── useGameState.ts      # Punteggio, vite, stato partita
├── game/
│   ├── lane-detector.ts     # Posa → corsia (left/center/right)
│   ├── obstacle-spawner.ts  # Generazione e movimento ostacoli
│   └── collision.ts         # Rilevamento collisioni
├── components/
│   └── GameCanvas.tsx       # Rendering canvas
├── pose-utils.ts            # Utility keypoint (dal repo base)
├── App.tsx                  # Composizione
└── App.css
```

---

## Riferimenti

- [Body Motion & AI Hackathon](https://johnnypax.github.io/its-ict-body-detection-challenge/)
- [Repo base — react-motion-tensorflow-yt](https://github.com/johnnypax/react-motion-tensorflow-yt)
- [Playlist video setup](https://www.youtube.com/playlist?list=PLoZNHBEyxFQFmAZm3-vnU1v28VKcuQDQ5)