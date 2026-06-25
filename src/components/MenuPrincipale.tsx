import { useState } from 'react';
import './MenuPrincipale.css';

// Assets – Vite import
import farwestRoad from '../assets/farwest_road.jpeg';
import sfondoCielo from '../assets/png/sfondo-cielo.png';
import piccione from '../assets/png/piccione-volante-2.png';
import pigeonLogo from '../assets/Pigeon.png';
import menuBg from '../assets/images/sfondo2.jpg';

/* ── Types ── */
export type GameId = 'outlaw' | 'pigeon';

interface MenuPrincipaleProps {
  onSelectGame: (game: GameId) => void;
}

/* ── Game data ── */
const GAMES = [
  {
    id: 'outlaw' as GameId,
    title: 'OUTLAW RUN',
    tag: 'SCHIVA & CORRI',
    description:
      "Metti in moto le gambe e schiva cactus, barili e cespugli come un vero cowboy del West! Trasforma la tua terapia nella sfida più epica del Far West.",
    cover: farwestRoad,
    overlayImg: null,
    accent: '#f8aa03',
    icon: '🤠',
  },
  {
    id: 'pigeon' as GameId,
    title: 'PIGEON BLASTER',
    tag: 'PUNTA & SPARA',
    description:
      "Alza le braccia e abbatti i piccioni che invadono il cielo! Usa tutto il corpo per mirare e sparare. Chi l'ha detto che la fisioterapia è noiosa?",
    cover: sfondoCielo,
    overlayImg: piccione,
    accent: '#60b9f0',
    icon: '🕊️',
  },
] as const;

/* ── Component ── */
export default function MenuPrincipale({ onSelectGame }: MenuPrincipaleProps) {
  const [hovered, setHovered] = useState<GameId | null>(null);

  return (
    <div
      className="menu-wrapper"
      style={{ backgroundImage: `url(${menuBg})` }}
    >
      {/* CRT scanline overlay */}
      <div className="menu-scanlines" aria-hidden="true" />

      {/* ── Header ── */}
      <header className="menu-header">
        <div className="menu-logo-row">
          <img
            src={pigeonLogo}
            alt=""
            className="menu-logo-img"
            aria-hidden="true"
          />

          <h1 className="menu-title">
            CIRCOLO
            <br />
            DEI PICCIONI
          </h1>

          <img
            src={pigeonLogo}
            alt=""
            className="menu-logo-img menu-logo-img--flip"
            aria-hidden="true"
          />
        </div>

        <p className="menu-welcome">
          Benvenuto nel posto dove la riabilitazione diventa un videogioco.
          <br />
          Muovi il corpo, fai punteggio, divertiti davvero.
        </p>

        <p className="menu-choose-label" aria-hidden="true">
          ▼&nbsp;&nbsp;SCEGLI LA TUA SFIDA&nbsp;&nbsp;▼
        </p>
      </header>

      {/* ── Game Cards ── */}
      <main className="menu-cards" role="list" aria-label="Giochi disponibili">
        {GAMES.map((game) => (
          <button
            key={game.id}
            role="listitem"
            className={`game-card${hovered === game.id ? ' game-card--hovered' : ''}`}
            style={{ '--accent': game.accent } as React.CSSProperties}
            onClick={() => onSelectGame(game.id)}
            onMouseEnter={() => setHovered(game.id)}
            onMouseLeave={() => setHovered(null)}
            aria-label={`Avvia ${game.title}`}
          >
            {/* Cover image */}
            <div className="game-card__cover">
              <img
                src={game.cover}
                alt=""
                className="game-card__bg"
                draggable={false}
              />

              {game.overlayImg && (
                <img
                  src={game.overlayImg}
                  alt=""
                  className="game-card__pigeon-overlay"
                  draggable={false}
                />
              )}

              <div className="game-card__vignette" />
              <span className="game-card__tag">{game.tag}</span>
            </div>

            {/* Info */}
            <div className="game-card__body">
              <h2 className="game-card__title">
                <span className="game-card__icon" aria-hidden="true">
                  {game.icon}
                </span>
                {game.title}
              </h2>

              <p className="game-card__desc">{game.description}</p>

              <span className="game-card__play-btn" aria-hidden="true">
                ► GIOCA ORA
              </span>
            </div>
          </button>
        ))}
      </main>

      {/* ── Footer ── */}
      <footer className="menu-footer" aria-label="Info tecnica">
        POWERED BY TENSORFLOW.JS&nbsp;&nbsp;·&nbsp;&nbsp;MUOVITI PER GIOCARE
      </footer>
    </div>
  );
}
