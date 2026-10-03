// Service worker: (1) adds COOP/COEP so the page is cross-origin isolated (multi-threaded AI on CPU), which GitHub Pages
// cannot do with headers; (2) keeps every file the app has used in a local cache so it opens and works with no network.
// The page itself is network-first (to pick up updates); libraries, fonts and AI models are cache-first.
const CACHE = "clip-editor-v1";
const CDN = /^(cdn\.jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com)$/;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

function isolate(res) {
  if (res.status === 0) return res;
  const h = new Headers(res.headers);
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.method !== "GET" || (r.cache === "only-if-cached" && r.mode !== "same-origin")) return;
  const url = new URL(r.url);
  const own = url.origin === self.location.origin;
  if (!own && !CDN.test(url.hostname)) return;
  const fresh = own && (r.mode === "navigate" || url.pathname.endsWith("/") || url.pathname.endsWith(".html") || url.pathname.endsWith(".webmanifest"));
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (fresh) {
      try {
        const net = await fetch(r);
        if (net.ok) await cache.put(r, net.clone());
        return isolate(net);
      } catch (err) {
        const hit = await cache.match(r, { ignoreSearch: true }) || (r.mode === "navigate" && await cache.match("./"));
        if (hit) return isolate(hit);
        throw err;
      }
    }
    const hit = await cache.match(r);
    if (hit) return isolate(hit);
    const net = await fetch(r);
    if (net.ok) await cache.put(r, net.clone());
    return isolate(net);
  })());
});
