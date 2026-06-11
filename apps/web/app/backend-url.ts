// Server-side only — resolves the FastAPI backend base URL from environment
// variables. This module should never be imported on the client; it references
// process.env keys that are only available at build/runtime on the server.
//
// Priority order for the resolved URL:
//   1. API_BASE_URL env var
//   2. BACKEND_URL env var
//   3. Vercel-derived URL using VERCEL_URL + the /api/python mount prefix
//   4. null (caller must handle the unconfigured case)

// On Vercel, the Python FastAPI app is mounted at /api/python by the framework's
// Python runtime adapter. This is used as a fallback when no explicit URL is set.
const BACKEND_ROUTE_PREFIX = "/api/python";

function trimTrailingSlash(value: string): string {
  const trimmed = value.trim();
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

// Guards against env vars that were accidentally stored as `KEY=value` strings
// (e.g. copy-pasting from a .env file). Strips the assignment prefix so we
// get a clean URL.
function normalizeEnvUrl(value: string): string {
  const trimmed = value.trim();
  const assignmentMatch = trimmed.match(/^(?:API_BASE_URL|BACKEND_URL)=(.+)$/);
  return assignmentMatch ? assignmentMatch[1].trim() : trimmed;
}

function isLocalhostUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
  } catch {
    return false;
  }
}

// Turns a raw env value into a fully-qualified URL. Relative paths are promoted
// to absolute using VERCEL_URL when available; bare relative paths with no
// VERCEL_URL resolve to null since they can't be fetched server-to-server.
function resolveBackendUrl(value: string): string | null {
  const trimmed = trimTrailingSlash(normalizeEnvUrl(value));
  if (/^https?:\/\//.test(trimmed)) return trimmed;

  if (trimmed.startsWith("/") && process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}${trimmed}`;
  }

  return trimmed.startsWith("/") ? null : trimmed;
}

export function getBackendBaseUrl(): string | null {
  const configuredUrls = [process.env.API_BASE_URL, process.env.BACKEND_URL];
  for (const configuredUrl of configuredUrls) {
    if (!configuredUrl) continue;

    const resolvedUrl = resolveBackendUrl(configuredUrl);
    if (!resolvedUrl) continue;

    // A Vercel deployment calling localhost would always fail — skip it so we
    // fall through to the Vercel-derived URL below.
    if (process.env.VERCEL_URL && isLocalhostUrl(resolvedUrl)) continue;

    return resolvedUrl;
  }

  // No explicit URL configured — derive one from VERCEL_URL if we're on Vercel.
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}${BACKEND_ROUTE_PREFIX}`;
  }

  return null;
}
