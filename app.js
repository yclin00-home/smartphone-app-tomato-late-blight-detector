// ===== 設定（與 Python 版相同）=====
const MAX_INPUT_SIDE = 800, MIN_CONF = 0.60, MAX_LEAVES = 10, MIN_LEAF_FRAC = 0.015;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const M1_TOMATO = 1;   // m1 類別 ['Non_tomato','Tomato']
const M2_BLIGHT = 0;   // m2 類別 ['Late_blight','Not_late_blight']

let sess1, sess2;
const $ = id => document.getElementById(id);

// ===== 載入 OpenCV 與兩個模型（會顯示失敗在哪一步）=====
async function init() {
  let step = '';
  try {
    step = '1/3 Loading OpenCV';
    $('status').textContent = step + '…';
    if (window.cv instanceof Promise) {
      window.cv = await window.cv;
    } else if (!window.cv.Mat) {
      await new Promise(r => { window.cv['onRuntimeInitialized'] = r; });
    }

    ort.env.wasm.wasmPaths = './';
    ort.env.wasm.numThreads = 1;   // 一般網站無法多執行緒，設 1 最穩

    step = '2/3 Loading model 1 (ONNX Runtime)';
    $('status').textContent = step + '…';
    sess1 = await ort.InferenceSession.create('m1_tomato.onnx');

    step = '3/3 Loading model 2';
    $('status').textContent = step + '…';
    sess2 = await ort.InferenceSession.create('m2_late_blight.onnx');

    $('status').textContent = 'Ready. Take or choose a photo.';
    $('file').disabled = false;
  } catch (err) {
    $('status').textContent = 'Error at step ' + step + ': ' + (err.message || err);
    console.error(err);
  }
}

// ===== 工具：inRange（OpenCV.js 要用同大小的 Mat 當上下限）=====
function inRange(hsv, lo, hi) {
  const l = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [...lo, 0]);
  const h = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [...hi, 0]);
  const out = new cv.Mat();
  cv.inRange(hsv, l, h, out);
  l.delete(); h.delete();
  return out;
}
function morph(mask, op, size) {
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
  const out = new cv.Mat();
  cv.morphologyEx(mask, out, op, k, new cv.Point(-1, -1), 1,
                  cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  k.delete();
  return out;
}

// ===== 找葉子（對應 find_leaf_regions）=====
function findLeafRegions(rgb) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const green = inRange(hsv, [35, 20, 20], [85, 255, 255]);
  const brown = inRange(hsv, [5, 15, 15], [35, 255, 255]);
  const mask0 = new cv.Mat();
  cv.bitwise_or(green, brown, mask0);
  const m1 = morph(mask0, cv.MORPH_CLOSE, 15);
  const mask = morph(m1, cv.MORPH_OPEN, 15);

  const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(mask, labels, stats, cents, 8, cv.CV_32S);
  const regions = [];
  for (let i = 1; i < n; i++) {
    const [x, y, w, h, area] = [0, 1, 2, 3, 4].map(j => stats.data32S[i * 5 + j]);
    if (area >= MIN_LEAF_FRAC * rgb.rows * rgb.cols) regions.push({ area, x, y, w, h });
  }
  [hsv, green, brown, mask0, m1, mask, labels, stats, cents].forEach(m => m.delete());
  regions.sort((a, b) => b.area - a.area);
  return regions.slice(0, MAX_LEAVES);
}

// ===== 模型推論（對應 tf + predict）=====
function toTensor(rgbCrop) {
  const r = new cv.Mat();
  cv.resize(rgbCrop, r, new cv.Size(224, 224), 0, 0, cv.INTER_AREA);
  const d = r.data, out = new Float32Array(3 * 224 * 224), P = 224 * 224;
  for (let i = 0; i < P; i++)
    for (let c = 0; c < 3; c++)
      out[c * P + i] = (d[i * 3 + c] / 255 - MEAN[c]) / STD[c];   // CHW + 標準化
  r.delete();
  return new ort.Tensor('float32', out, [1, 3, 224, 224]);
}
function softmax(a) {
  const m = Math.max(...a), e = a.map(v => Math.exp(v - m)), s = e.reduce((x, y) => x + y);
  return e.map(v => v / s);
}
async function predict(sess, tensor) {
  const o = await sess.run({ input: tensor });
  return softmax(Array.from(o.logits.data));
}

// ===== 病斑比例（對應 calc_infection_ratio）=====
function infectionRatio(rgbCrop) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgbCrop, hsv, cv.COLOR_RGB2HSV);
  const g = inRange(hsv, [35, 40, 40], [85, 255, 255]);
  const dis = inRange(hsv, [10, 30, 30], [35, 255, 255]);
  const l0 = new cv.Mat(); cv.bitwise_or(g, dis, l0);
  const l1 = morph(l0, cv.MORPH_CLOSE, 5);
  const leaf = morph(l1, cv.MORPH_OPEN, 5);
  const dark = inRange(hsv, [0, 0, 0], [180, 255, 80]);
  const les0 = new cv.Mat(); cv.bitwise_or(dis, dark, les0);
  const les = new cv.Mat(); cv.bitwise_and(les0, leaf, les);
  const leafPx = cv.countNonZero(leaf);
  const ratio = leafPx === 0 ? 0 : cv.countNonZero(les) / leafPx * 100;
  [hsv, g, dis, l0, l1, leaf, dark, les0, les].forEach(m => m.delete());
  return ratio;
}

// ===== 主流程（對應 diagnose）=====
async function diagnose(file) {
  $('status').textContent = 'Analyzing…';
  try {
    const bmp = await createImageBitmap(file);          // 瀏覽器會自動依 EXIF 轉正
    const s = Math.min(1, MAX_INPUT_SIDE / Math.max(bmp.width, bmp.height));
    const view = $('view');
    view.width = Math.round(bmp.width * s); view.height = Math.round(bmp.height * s);
    const ctx = view.getContext('2d');
    ctx.drawImage(bmp, 0, 0, view.width, view.height);

    const rgba = cv.imread(view), rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    const regions = findLeafRegions(rgb);

    ctx.lineWidth = Math.max(2, view.width / 300);
    ctx.strokeStyle = '#ff0000'; ctx.fillStyle = '#ff0000'; ctx.font = 'bold 28px Arial';
    const results = [];
    let idx = 0;
    for (const r of regions) {
      idx++;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.fillText(String(idx), r.x + 8, r.y + 34);
      const crop = rgb.roi(new cv.Rect(r.x, r.y, r.w, r.h));
      const t = toTensor(crop);
      const p1 = await predict(sess1, t);
      if (p1[M1_TOMATO] >= MIN_CONF) {
        const p2 = await predict(sess2, t);
        const blight = p1[M1_TOMATO] * p2[M2_BLIGHT];
        const not = p1[M1_TOMATO] * (1 - p2[M2_BLIGHT]);
        results.push({ top: blight >= not ? 'blight' : 'not', conf: Math.max(blight, not), crop: crop.clone() });
      } else {
        results.push({ top: 'non', conf: p1[1 - M1_TOMATO] });
      }
      crop.delete();
    }
    rgba.delete(); rgb.delete();

    const tomato = results.filter(r => r.top !== 'non');
    const blighted = tomato.filter(r => r.top === 'blight');
    let html;
    if (!regions.length) {
      html = banner('non-tomato', '🚫 No leaf regions detected. Please photograph a clear leaf.');
    } else if (!tomato.length) {
      html = banner('non-tomato', '🚫 Leaves detected, but none are tomato leaves.');
    } else if (!blighted.length) {
      const c = Math.max(...tomato.map(r => r.conf));
      html = banner('tomato-without-late-blight',
        `✅ Tomato leaf, no late blight detected (confidence ${(c * 100).toFixed(1)}%).`);
    } else {
      const ch = blighted.reduce((a, b) => (b.conf > a.conf ? b : a));
      const ratio = infectionRatio(ch.crop);
      const msg = ratio < 33.3
        ? 'Mild late blight. Remove infected leaves and apply an organic copper fungicide.'
        : ratio < 66.6
          ? 'Moderate late blight. Isolate the plant, remove infected foliage, use a systemic fungicide.'
          : '🚨 Severe late blight! Remove and destroy the plant to prevent spore spread.';
      html = banner('tomato-with-late-blight', msg, ratio);
    }
    results.forEach(r => r.crop && r.crop.delete());
    $('report').innerHTML = html;
    $('report').hidden = false;
    $('status').textContent = `Done (checked ${regions.length} leaf region(s)).`;
  } catch (err) {
    $('status').textContent = 'Error while analyzing: ' + (err.message || err);
    console.error(err);
  }
}

function banner(status, msg, ratio) {
  let bar = '';
  if (ratio !== undefined) {
    const v = Math.max(0, Math.min(100, ratio));
    bar = `<p><b>SEVERITY SCORE: [${v.toFixed(1)}%]</b></p>
      <div style="position:relative"><div style="position:absolute;left:${v}%;top:-18px;transform:translateX(-50%)">▲</div>
      <div class="bar"><div style="background:#27ae60"></div><div style="background:#f39c12"></div><div style="background:#c0392b"></div></div></div>`;
  }
  return `<p><b>DIAGNOSIS STATUS: [${status}]</b></p>${bar}<p><b>RECOMMENDED ACTION:</b><br>${msg}</p>`;
}

// ===== 啟動 =====
$('file').addEventListener('change', e => { if (e.target.files[0]) diagnose(e.target.files[0]); });
init();