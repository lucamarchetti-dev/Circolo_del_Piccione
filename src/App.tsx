import { useState } from 'react';
import MenuPrincipale from './components/MenuPrincipale';
import type { GameId } from './components/MenuPrincipale';

// Lazy imports are optional — using direct imports keeps the bundle simple
// and avoids flash-of-empty while chunks load for a local app.
import OutlawRun from './components/OutlawRun';
import PigeonBlaster from './components/PigeonBlaster';

type View = 'menu' | GameId;

export default function App() {
  const [view, setView] = useState<View>('menu');

  const goBack = () => setView('menu');

  if (view === 'outlaw') {
    return <OutlawRun onBack={goBack} />;
  }

  if (view === 'pigeon') {
    return <PigeonBlaster onBack={goBack} />;
  }

  return <MenuPrincipale onSelectGame={(game) => setView(game)} />;
}
