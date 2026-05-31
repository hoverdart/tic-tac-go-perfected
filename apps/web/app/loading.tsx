export default function Loading() {
  return (
    <main className="page">
      <section className="scene">
        <header className="hero">
          <div
            className="skeleton"
            style={{ width: 320, height: 76, borderRadius: 12 }}
          />
          <div
            className="skeleton"
            style={{ width: "min(360px, calc(100vw - 48px))", height: 68, borderRadius: 999 }}
          />
          <div
            className="skeleton"
            style={{ width: 154, height: 48, borderRadius: 999 }}
          />
        </header>

        <div className="focus-area" style={{ marginTop: 24 }}>
          <div
            className="board-stage skeleton"
            style={{ minHeight: 380 }}
          />
          <div
            className="skeleton"
            style={{ borderRadius: 20, minHeight: 260 }}
          />
        </div>

        <div
          className="skeleton"
          style={{ marginTop: 20, borderRadius: 20, height: 220 }}
        />
      </section>
    </main>
  );
}
