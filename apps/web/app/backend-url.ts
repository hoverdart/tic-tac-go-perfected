const BACKEND_ROUTE_PREFIX = "/api/python";

function trimTrailingSlash(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
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
  const configuredUrl = process.env.API_BASE_URL ?? process.env.BACKEND_URL;
  if (configuredUrl) return resolveBackendUrl(configuredUrl);

  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}${BACKEND_ROUTE_PREFIX}`;
  }

  return null;
}
