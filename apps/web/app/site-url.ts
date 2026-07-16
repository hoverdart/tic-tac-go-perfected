const DEFAULT_SITE_URL = "https://tictacgo.shauryav.com";

function normalizeSiteUrl(value: string): URL {
  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("SITE_URL must use http or https.");
  }
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url;
}

export function getSiteUrl(): URL {
  const configuredUrl =
    process.env.SITE_URL ?? process.env.NEXT_PUBLIC_SITE_URL ?? DEFAULT_SITE_URL;

  try {
    return normalizeSiteUrl(configuredUrl);
  } catch {
    return new URL(DEFAULT_SITE_URL);
  }
}
