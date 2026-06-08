const BACKEND_ROUTE_PREFIX = "/api/python";

function trimTrailingSlash(value: string): string {
  const trimmed = value.trim();
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

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
    if (process.env.VERCEL_URL && isLocalhostUrl(resolvedUrl)) continue;

    return resolvedUrl;
  }

  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}${BACKEND_ROUTE_PREFIX}`;
  }

  return null;
}
