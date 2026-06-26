<<<<<<< HEAD
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
=======
# Circolo dei Piccioni — Rehab Games

Piattaforma web per la riabilitazione fisica che trasforma gli esercizi terapeutici in due minigiochi divertenti ed interattivi. Il riconoscimento della posa corporea avviene interamente nel browser, senza backend, tramite TensorFlow.js e il modello MoveNet.

---

## Indice

- [Panoramica del progetto](#panoramica-del-progetto)
- [Come avviare il progetto] (#come-avviare-il-progetto)
- [Stack tecnologico](#stack-tecnologico)
- [Struttura del codice](#struttura-del-codice)
- [Pigeon Blaster](#pigeon-blaster)
- [Outlaw Run](#outlaw-run)
- [Installazione e avvio](#installazione-e-avvio)
- [Build e lint](#build-e-lint)
- [Note tecniche](#note-tecniche)

---

## Panoramica del progetto

**Circolo dei Piccioni** è una web-app React pensata per contesti di fisioterapia e riabilitazione motoria. Il paziente usa una webcam: il modello AI rileva la postura del corpo in tempo reale e la traduce in input di gioco. In questo modo gli esercizi di mobilità diventano un modo per giocare e fare eservizio.

Il progetto è composto da:
- un **Menù principale** con selezione del gioco,
- **Pigeon Blaster** — gioco di colpire bersagli con i polsi,
- **Outlaw Run** — runner con schivata di ostacoli tramite movimento del corpo.

---

## Come avviare il progetto

1. **Clona** il repository oppure scaricalo come archivio ZIP da GitHub.
2. **Apri** un terminale e spostati nella cartella del progetto:

```bash
cd Circolo_del_piccione_Main
```

3. **Installa** tutte le dipendenze con:
>>>>>>> 9ceff38 (Demo progetto conclusa)

```bash
npm install
```

<<<<<<< HEAD
## Avvio in sviluppo
=======
Per **avviare** l'applicazione, esegui:
>>>>>>> 9ceff38 (Demo progetto conclusa)

```bash
npm run dev
```

<<<<<<< HEAD
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

---------------------------------------
## DA FIXARE
Ho unito i due progetti insieme, aggiunto un menù principale (va aggiunto un css per il bottone di torna al menù per entrambi i giochi).

## CAMBIAMENTI AD OUTLAW RUN
La vostra funzione si chiama ora OutlawRun al posto di 'App' (dava problemi con il fatto che ora non avete più App.tsx)
'App.tsx' e 'App.css' sono ora dentro la cartella components con i relativi nomi di 'OutlawRun.tsx' e 'OulawRun.css'.
Le immagini sono rimaste all'interno di Public.
useState era stato usato come useEffect — il listener della tastiera era dentro uno useState che non esegue cleanup (avevate cleanappato l'intero codice tranne questa cosa, cattivi), non re-registra alla svariata delle dipendenze, ed è semanticamente sbagliato. Sostituito con useEffect.

## CAMBIAMENTI A PIGEONBLASTER
Commentate le parti di debug che erano state lasciate all'interno del codice.
Aggiustati qualche import per le immagini che per PigeonBlaster sono rimaste all'interno di assets.

=======
Al termine dell'avvio, apri il browser e visita:

```
http://localhost:5173
```

Quando richiesto, concedi al browser l'accesso alla webcam. Per un corretto funzionamento del rilevamento, posizionati a circa **1-2 metri** dalla telecamera, assicurandoti che il corpo sia visibile all'interno dell'inquadratura.

---

## Stack tecnologico

| Tecnologia | Ruolo |
|---|---|
| **React 19 + TypeScript** | UI e logica dei componenti |
| **Vite** | Build tool e dev server |
| **TensorFlow.js** (backend WebGL) | Inferenza modello pose sul client |
| **@tensorflow-models/pose-detection** | API MoveNet per rilevamento 17 keypoint |
| **Canvas API** | Rendering in tempo reale di video, keypoint e oggetti di gioco |
| **WebRTC getUserMedia** | Accesso alla webcam con selezione del device |

---

## Struttura del codice

```text
src/
├── App.tsx                      # Router principale: menu → selezione gioco
├── App.css                      # Stili globali minimi
├── main.tsx                     # Bootstrap React (StrictMode)
├── pose-utils.ts                # Utility canvas: drawKeypoints, drawArm, getKeypoint
│
├── components/
│   ├── MenuPrincipale.tsx       # Schermata di selezione gioco (card-based)
│   ├── MenuPrincipale.css
│   ├── PigeonBlaster.tsx        # Gioco 1 — logica completa (camera + AI + canvas)
│   ├── PigeonBlaster.css
│   ├── OutlawRun.tsx            # Gioco 2 — orchestrazione loop e stato
│   ├── OutlawRun.css
│   └── GameCanvas.tsx           # Canvas renderer per OutlawRun
│
├── hooks/
│   ├── usePoseDetector.ts       # Setup webcam + rilevamento pose (OutlawRun)
│   ├── useGameLoop.ts           # requestAnimationFrame con deltaTime
│   └── useGameState.ts          # Stato partita (score, vite, status) con reducer
│
├── game/
│   ├── lane-detector.ts         # Determina corsia giocatore da keypoint spalla/fianchi
│   ├── obstacle-spawner.ts      # Genera e aggiorna ostacoli con difficoltà progressiva
│   └── collision.ts             # Collision detection ostacoli vs corsia giocatore
│
├── mocks/
│   └── mediapipe-pose.js        # Mock MediaPipe per evitare errori di build con Vite
│
└── assets/                      # Font, immagini, sfondi
```

---

## Pigeon Blaster

### Concept

Il giocatore usa i **polsi** come puntatori. Piccioni compaiono casualmente sulla canvas; il giocatore deve colpirli, muovendo le braccia, prima che scadano. Ogni piccione mancato costa una vita.

### Come funziona

1. **Inizializzazione** — TensorFlow.js carica il backend WebGL e inizializza MoveNet (`SINGLEPOSE_LIGHTNING`). Viene enumerata la lista delle webcam disponibili.
2. **Calibrazione** — Il giocatore deve tenere entrambi i polsi fermi all'interno di un riquadro centrale per 3 secondi. Questo garantisce che la camera rilevi correttamente la persona prima di avviare la partita.
3. **Game loop** — `requestAnimationFrame` esegue `loop()` ad ogni frame:
   - Disegna il frame webcam sulla canvas (specchiato via CSS `scaleX(-1)`).
   - Ogni ~33ms stima la posa con `detector.estimatePoses(video)`.
   - Aggiorna l'età di ogni piccione; quelli scaduti tolgono una vita.
   - Controlla hit-test tra coordinate polso e bounding box piccioni.
   - Aggiorna score, vite e timer.
4. **Difficoltà** — Tre preset (`Facile / Media / Difficile`) controllano durata partita, tempo vita piccione, numero massimo contemporaneo e vite iniziali.
5. **Game over / Vittoria** — La partita finisce per azzeramento vite o scadenza del timer. Il record viene mantenuto in memoria per la sessione.

### Struttura interna

Tutto il codice risiede in `PigeonBlaster.tsx`. Lo stato React (`useState`) gestisce la UI; i `useRef` portano i valori nel loop di animazione senza causare re-render. I `useEffect` gestiscono inizializzazione e cambio fotocamera.

**Rendering**: la canvas è internamente 640×480 (per mantenere la corrispondenza con le coordinate dei keypoint). La dimensione visiva è aumentata via CSS — in questo modo i dot di debug appaiono nel posto corretto.

**Debug keypoints**: il flag `drawKeypoints(ctx, activePose, 0.3)` è attivo. Per disabilitarlo in produzione, commentare quella riga in `PigeonBlaster.tsx`.

---

## Outlaw Run

### Concept

Un runner laterale ispirato al Far West. Il giocatore **sposta il peso del corpo** (rilevato dalla posizione delle spalle/fianchi) per cambiare corsia e schivare ostacoli (cactus, barili, cespugli) che scorrono da destra a sinistra.

### Come funziona

1. **usePoseDetector** — hook che inizializza la webcam e espone `detectPose()`, che restituisce la `Pose` corrente. Gestisce il ciclo di vita del video element.
2. **useGameLoop** — hook che esegue la callback del gioco a ogni frame con `requestAnimationFrame`, calcolando il `deltaMs` dall'ultimo frame.
3. **useGameState** — hook con un reducer interno che gestisce i transizioni di stato: `idle → playing → gameover`.
4. **detectLane** — data la posa, calcola su quale delle tre corsie (sinistra/centro/destra) si trova il giocatore in base alla posizione orizzontale del centro spalle rispetto alla larghezza del video.
5. **updateObstacles** — genera nuovi ostacoli con timer e difficoltà progressiva legata al punteggio; sposta quelli esistenti verso sinistra; rimuove quelli usciti dallo schermo (aggiungendo punti).
6. **checkCollisions** — confronta la corsia degli ostacoli con quella del giocatore; restituisce `hit: true` in caso di collisione.
7. **GameCanvas** — componente puro che riceve video, posa, ostacoli e stato di gioco, e si occupa esclusivamente del rendering su canvas.

### Struttura interna

OutlawRun separa nettamente **logica di gioco** (hooks + game/) da **presentazione** (GameCanvas). Il loop principale in `OutlawRun.tsx` coordina i dati, delegando ogni responsabilità al modulo appropriato.

Il tasto **Spazio** avvia o riavvia la partita.

---

## Installazione e avvio

**Requisiti**: Node.js 18+, browser moderno con supporto WebGL e webcam.

```bash
npm install
npm run dev
```

Aprire `http://localhost:5173` nel browser. Concedere il permesso alla webcam quando richiesto.

---

## Build e lint

```bash
npm run build   # Build di produzione in /dist
npm run lint    # ESLint su tutto il progetto
```

---

## Note tecniche

### Keypoint e specchiamento canvas

La canvas viene specchiata via CSS (`transform: scaleX(-1)`) per dare un effetto "specchio" naturale all'utente. Le coordinate dei keypoint restituite da MoveNet sono nello spazio del video originale (non specchiato), quindi le funzioni di rendering — `drawTextUnflipped`, `drawImageUnflipped` — applicano una trasformazione inversa prima di disegnare, garantendo il corretto allineamento.

### Mock MediaPipe

Per evitare l'errore di build Vite:

```
"Pose" is not exported by "@mediapipe/pose"
```

il file `src/mocks/mediapipe-pose.js` fornisce un modulo stub, e `vite.config.ts` lo aliasa. Il progetto usa esclusivamente il runtime TensorFlow.js (WebGL), senza dipendere direttamente da MediaPipe.

### Performance

MoveNet SINGLEPOSE_LIGHTNING è ottimizzato per bassa latenza. Il rilevamento in PigeonBlaster è limitato a 30fps (`now - lastPoseRef.current > 33ms`) per non saturare la GPU su macchine meno potenti.

---

## Licenza

Questo progetto è distribuito sotto la licenza GNU General Public License v3.0 (GPL-3.0).

In breve:
- Sei libero di usare, studiare, modificare e distribuire questo software.
- Se distribuisci una versione modificata, devi renderla disponibile con la stessa licenza GPL.
- Il software è fornito "così com'è", senza alcuna garanzia.

Per il testo completo e legalmente valido della licenza, consulta il file `LICENSE`.
>>>>>>> 9ceff38 (Demo progetto conclusa)
