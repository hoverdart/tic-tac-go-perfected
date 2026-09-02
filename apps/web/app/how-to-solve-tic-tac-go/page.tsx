import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How to Solve Tic Tac Go: Hints, Rules, and Strategy",
  description: "Learn how to solve today’s Tic Tac Go puzzle: line up three Os, avoid three Xs, and plan every push before committing.",
  alternates: { canonical: "/how-to-solve-tic-tac-go" },
};

export default function HowToSolveTicTacGo() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "HowTo",
        name: "How to solve Tic Tac Go",
        description: "A practical method for solving a Tic Tac Go push puzzle.",
        step: [
          { "@type": "HowToStep", name: "Find the target line", text: "Choose a horizontal or vertical line where three Os can finish." },
          { "@type": "HowToStep", name: "Plan the push positions", text: "Check that the square behind every piece you need to push stays reachable." },
          { "@type": "HowToStep", name: "Protect against X lines", text: "Reject any push that creates three Xs in a row." },
          { "@type": "HowToStep", name: "Reveal only the help you need", text: "Use the daily replay one move or strategy phase at a time." },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: [
          { "@type": "Question", name: "Can I solve Tic Tac Go without seeing the full answer?", acceptedAnswer: { "@type": "Answer", text: "Yes. Start with the target line and reveal one move or push checkpoint only when needed." } },
          { "@type": "Question", name: "Do diagonals count?", acceptedAnswer: { "@type": "Answer", text: "No. Three Os must be in a horizontal or vertical row." } },
        ],
      },
      { "@type": "BreadcrumbList", itemListElement: [{ "@type": "ListItem", position: 1, name: "Home", item: "https://tictacgo.shauryav.com/" }, { "@type": "ListItem", position: 2, name: "How to solve Tic Tac Go", item: "https://tictacgo.shauryav.com/how-to-solve-tic-tac-go" }] },
    ],
  };
  return (
    <main className="guide-page">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd).replace(/</g, "\\u003c") }} />
      <p className="guide-kicker">Tic Tac Go strategy</p>
      <h1>How to solve today’s Tic Tac Go</h1>
      <p className="guide-lead">The goal is to make three Os in one horizontal or vertical line. Your character counts as an O, but a line of three Xs loses immediately.</p>
      <ol className="guide-steps">
        <li><strong>Choose the finishing line first.</strong> Look for two Os that can meet your player O, then work backward from that three-cell line.</li>
        <li><strong>Think in pushes, not walks.</strong> Walking only changes your position. A push changes the board, so make sure you can stand on the correct side of a piece before committing.</li>
        <li><strong>Keep escape squares open.</strong> Pushing an O into a wall or corner can make it useless. A piece needs an empty square beyond it and a reachable square behind it.</li>
        <li><strong>Watch every X.</strong> A move that accidentally makes X-X-X ends the attempt, even if it also improves your O line.</li>
        <li><strong>Use hints progressively.</strong> Our daily solution starts with the board and lets you reveal one move or one push phase at a time.</li>
      </ol>
      <div className="guide-actions">
        <Link href="/" className="guide-action">See today’s hint-first solution</Link>
        <Link href="/custom-tic-tac-go-solver" className="guide-action guide-action-secondary">Try a custom board</Link>
      </div>
      <section>
        <h2>How our push solver works</h2>
        <p>Instead of treating every arrow key as a separate strategic decision, the solver groups together all walking positions that can reach the same next push. It searches legal pushes, rejects immediate X-line losses, and independently replays any answer before publishing it.</p>
        <p><a href="https://www.shauryav.com/blog/solving-a-two-minute-puzzle" target="_blank" rel="noreferrer">Read the technical build story</a> or <a href="https://medium.com/@abdullahmindstorm/finding-a-solution-when-every-correct-move-looks-wrong-b4924eec54ae" target="_blank" rel="noreferrer">read Abdullah’s solver journey</a>.</p>
      </section>
      <section>
        <h2>Frequently asked questions</h2>
        <h3>Can I solve Tic Tac Go without seeing the full answer?</h3>
        <p>Yes. Start with the target line, reveal a single next move if you are stuck, and use a push checkpoint only when you need more context.</p>
        <h3>Do diagonals count?</h3>
        <p>No. Only three Os in a horizontal or vertical row win.</p>
      </section>
    </main>
  );
}
