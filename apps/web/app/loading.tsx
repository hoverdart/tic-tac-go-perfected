export default function Loading() {
  return (
    <main className="page">
      <section className="game-scene" aria-label="Loading today's solution">
        <header className="loading-header">
          <div className="skeleton loading-title" />
          <div className="skeleton loading-subtitle" />
          <div className="skeleton loading-date" />
        </header>
        <div className="wood-stage">
          <div className="skeleton loading-board" />
        </div>
      </section>
    </main>
  );
}
