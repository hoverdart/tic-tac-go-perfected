const BACKEND_ROUTE_PREFIX = "/api/python";

function trimTrailingSlash(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
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
  const trimmed = trimTrailingSlash(value);
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
