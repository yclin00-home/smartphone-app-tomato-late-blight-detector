const CACHE_NAME = 'tomato-app-v1';
const assetsToCache = [
    './',
    './index.html',
    './app.js',
    './opencv.js',
    './ort.min.js',
    './ort-wasm.wasm',
    './ort-wasm-simd.wasm',
    './m1_tomato.onnx',
    './m2_late_blight.onnx',
    './manifest.json'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(assetsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => response || fetch(event.request))
    );
});