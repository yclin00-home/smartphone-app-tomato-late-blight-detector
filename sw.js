const CACHE_NAME = 'tomato-offline-v6';

// 需要強制快取的清單（確保檔名完全正確）
const ASSETS = [
  './',
  './index.html',
  './app.js',
  './manifest.json',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
  'https://docs.opencv.org/4.8.0/opencv.js',
  'm1_tomato.onnx',
  'm2_late_blight.onnx'
];

// 安裝時快取所有檔案
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(ASSETS);
    }).catch(err => {
      console.error('快取安裝失敗：', err);
    })
  );
});

// 啟動時清理舊快取並接管控制權
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// 攔截請求：採用「快取優先 (Cache-First)」策略
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      if (cachedResponse) {
        return cachedResponse; // 如果快取有，立刻回傳，不經過網路
      }
      return fetch(event.request).then(networkResponse => {
        // 如果是動態請求，順便抓進快取
        return caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
      }).catch(() => {
        // 斷網且快取沒有時的備用方案 (如果是網頁導航，退回首頁)
        if (event.request.mode === 'navigate') {
          return caches.match('./index.html');
        }
      });
    })
  );
});
