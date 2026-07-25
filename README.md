# Circolo dei Piccioni — Rehab Games

Piattaforma web per la riabilitazione fisica che trasforma gli esercizi terapeutici in due minigiochi divertenti ed interattivi. Il riconoscimento della posa corporea avviene interamente nel browser, senza backend, tramite TensorFlow.js e il modello MoveNet.

---

## Indice

- [Panoramica del progetto](#panoramica-del-progetto)
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