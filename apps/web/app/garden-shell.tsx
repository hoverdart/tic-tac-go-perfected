"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Leaf, Pause } from "lucide-react";
import { createContext, startTransition, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type ComponentProps, type CSSProperties, type ReactNode } from "react";

const NavigationContext = createContext<((href: string) => boolean) | null>(null);

// Keep the real Link behavior for prefetch, modified clicks, and browsers
// without view transitions. Only ordinary same-site navigation is animated.
export function GardenLink({ href, ...props }: Omit<ComponentProps<typeof Link>, "href" | "onNavigate"> & { href: string }) {
  const navigate = useContext(NavigationContext);
  return <Link {...props} href={href} onNavigate={(event) => {
    if (navigate?.(href)) event.preventDefault();
  }} />;
}

const LEAVES = [
  [5, 31, -8, 24], [18, 38, -24, 18], [32, 35, -14, -16],
  [49, 42, -31, 20], [65, 34, -6, -22], [79, 39, -27, 16],
  [91, 44, -19, -12], [98, 36, -3, 22],
];

export function GardenShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const garden = useRef<HTMLDivElement>(null);
  const pending = useRef<{ finish: () => void; timer: ReturnType<typeof setTimeout> } | null>(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const visibility = () => garden.current?.classList.toggle("garden-hidden", document.hidden);
    visibility();
    document.addEventListener("visibilitychange", visibility);
    return () => document.removeEventListener("visibilitychange", visibility);
  }, []);

  useLayoutEffect(() => {
    if (pending.current) {
      clearTimeout(pending.current.timer);
      pending.current.finish();
      pending.current = null;
    }
  }, [pathname]);

  useEffect(() => () => {
    if (pending.current) {
      clearTimeout(pending.current.timer);
      pending.current.finish();
    }
  }, []);

  const navigate = useCallback((href: string) => {
    if (!href.startsWith("/") || href.startsWith("//") || href.includes("#") ||
      href === pathname || !document.startViewTransition ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches || pending.current) return false;

    const transition = document.startViewTransition(() => new Promise<void>((resolve) => {
      // A slow or failed route must never hold the browser's snapshot forever.
      pending.current = { finish: resolve, timer: setTimeout(() => {
        pending.current = null;
        resolve();
      }, 2500) };
      startTransition(() => router.push(href));
    }));
    // An interrupted animation must not interrupt navigation.
    void transition.finished.catch(() => {});
    return true;
  }, [pathname, router]);

  return (
    <NavigationContext.Provider value={navigate}>
      <div ref={garden} className={`garden-backdrop${paused ? " garden-paused" : ""}`} aria-hidden="true">
        <div className="garden-leaves">
          {LEAVES.map(([left, duration, delay, drift], index) => (
            <span key={index} className="garden-leaf" style={{
              "--leaf-left": `${left}%`, "--leaf-duration": `${duration}s`,
              "--leaf-delay": `${delay}s`, "--leaf-drift": `${drift}vw`,
            } as CSSProperties}>
              <svg viewBox="0 0 32 48" fill="none"><path d="M27 2C6 6-3 24 9 40c16-1 27-18 18-38Z" fill="currentColor" /><path d="M8 45 24 9" stroke="currentColor" strokeWidth="1.5" /></svg>
            </span>
          ))}
        </div>
      </div>
      <div key={pathname} className="garden-content">{children}</div>
      <div className="garden-controls">
        <button className="garden-motion-toggle" type="button" onClick={() => setPaused((value) => !value)} aria-label={paused ? "Resume drifting leaves" : "Pause drifting leaves"} aria-pressed={paused}>
          {paused ? <Leaf size={16} aria-hidden="true" /> : <Pause size={16} aria-hidden="true" />}
          <span>{paused ? "Resume leaves" : "Pause leaves"}</span>
        </button>
      </div>
    </NavigationContext.Provider>
  );
}
