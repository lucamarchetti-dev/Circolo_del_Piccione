# Pose Browser Demo

Demo React + TypeScript per il riconoscimento della posa umana direttamente nel browser.

Il progetto usa TensorFlow.js e il modello MoveNet per leggere il flusso webcam, disegnare i keypoint rilevati su canvas e contare quante volte il polso viene portato sopra il naso.

## Video tutorial

Playlist YouTube del canale **Archety.dev - Giovanni Pace**:

[Apri la playlist su YouTube](https://www.youtube.com/playlist?list=PLoZNHBEyxFQFmAZm3-vnU1v28VKcuQDQ5)

## Funzionalita

- acquisizione video da webcam tramite `getUserMedia`;
- selezione della camera disponibile dal browser;
- inferenza pose in tempo reale con `@tensorflow-models/pose-detection`;
- backend TensorFlow.js configurato su WebGL;
- rendering su `<canvas>` del frame video;
- disegno dei keypoint della posa;
- disegno delle braccia tramite collegamento spalla-gomito-polso;
- conteggio del gesto "polso sopra il naso" con cooldown per evitare conteggi duplicati.

## Stack

- React 19
- TypeScript
- Vite
- TensorFlow.js
- TensorFlow Pose Detection / MoveNet

## Struttura principale

```text
src/
  App.tsx          Logica principale: camera, detector, canvas e conteggio gesto
  pose-utils.ts   Utility per keypoint, punti e linee sul canvas
  App.css         Stili base per video e canvas
  main.tsx        Bootstrap React
```

## Requisiti

- Node.js installato
- Browser moderno con supporto a webcam e WebGL
- Permesso di accesso alla camera

## Installazione

```bash
npm install
```

## Avvio in sviluppo

```bash
npm run dev
```

Aprire l'URL mostrato da Vite, di solito:

```text
http://localhost:5173/
```

oppure:

```text
http://127.0.0.1:5173/
```

## Build

```bash
npm run build
```

## Lint

```bash
npm run lint
```

## Licenza

Questo progetto e distribuito con licenza [MIT + Coffeeware](./LICENSE.md).

Se il progetto ti e utile e un giorno ci incontriamo, puoi offrirmi un caffe. Non e obbligatorio, ma e apprezzato.

## Nota su Pose Detection e Vite

Per evitare l'errore di build legato a MediaPipe:

```text
"Pose" is not exported by "@mediapipe/pose"
```

l'import di Pose Detection usa l'entrypoint:

```ts
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js';
```

Questo progetto usa MoveNet con runtime TensorFlow.js, quindi non richiede l'uso diretto di MediaPipe Pose.