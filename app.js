const MAX_INPUT_SIDE = 800, MIN_CONF = 0.60, MAX_LEAVES = 10, MIN_LEAF_FRAC = 0.05;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const M1_TOMATO = 1;   // index of "Tomato" in model 1 output
const M2_BLIGHT = 0;   // index of "Late_blight" in model 2 output

// Status colors: only late blight is red; everything else is neutral
const STATUS_COLORS = {
  'non-tomato': '#555555',
  'tomato-without-late-blight': '#1e8449',
  'tomato-with-late-blight': '#a93226',
};

let sess1, sess2;
const $ = id => document.getElementById(id);

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
    ort.env.wasm.numThreads = 1;
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

// ---------- OpenCV helpers ----------
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

// ---------- Model helpers ----------
function toTensor(rgbCrop) {
  const r = new cv.Mat();
  cv.resize(rgbCrop, r, new cv.Size(224, 224), 0, 0, cv.INTER_AREA);
  const d = r.data, out = new Float32Array(3 * 224 * 224), P = 224 * 224;
  for (let i = 0; i < P; i++)
    for (let c = 0; c < 3; c++)
      out[c * P + i] = (d[i * 3 + c] / 255 - MEAN[c]) / STD[c];
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

// Keep only the big connected pieces of a mask (drops tiny background specks)
function keepBigComponents(mask) {
  const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(mask, labels, stats, cents, 8, cv.CV_32S);
  let maxA = 0;
  for (let i = 1; i < n; i++) maxA = Math.max(maxA, stats.data32S[i * 5 + 4]);
  const keep = new Uint8Array(n);
  for (let i = 1; i < n; i++) keep[i] = stats.data32S[i * 5 + 4] >= 0.1 * maxA ? 1 : 0;
  const out = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1);
  const L = labels.data32S, O = out.data;
  for (let k = 0; k < L.length; k++) if (keep[L[k]]) O[k] = 255;
  labels.delete(); stats.delete(); cents.delete();
  return out;
}

// Infection ratio + leaf outline (red) + lesion overlay (red tint) for one leaf crop.
// Caller must delete leafVis and lesionVis.
function analyzeLeaf(rgbCrop) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgbCrop, hsv, cv.COLOR_RGB2HSV);
  const g = inRange(hsv, [35, 40, 40], [85, 255, 255]);
  const dis = inRange(hsv, [10, 30, 30], [35, 255, 255]);
  const l0 = new cv.Mat(); cv.bitwise_or(g, dis, l0);
  const l1 = morph(l0, cv.MORPH_CLOSE, 5);
  const leaf0 = morph(l1, cv.MORPH_OPEN, 5);
  const leaf = keepBigComponents(leaf0);   // remove tiny specks so they are not outlined
  leaf0.delete();
  const dark = inRange(hsv, [0, 0, 0], [180, 255, 80]);
  const les0 = new cv.Mat(); cv.bitwise_or(dis, dark, les0);
  const les = new cv.Mat(); cv.bitwise_and(les0, leaf, les);
  const leafPx = cv.countNonZero(leaf);
  const ratio = leafPx === 0 ? 0 : cv.countNonZero(les) / leafPx * 100;

  // Leaf outline (red)
  const leafVis = rgbCrop.clone();
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(leaf, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  cv.drawContours(leafVis, contours, -1, new cv.Scalar(255, 0, 0, 255),
                  Math.max(2, Math.floor(rgbCrop.cols / 300)));

  // Lesion overlay (red tint)
  const lesionVis = rgbCrop.clone();
  const d = lesionVis.data, m = les.data;
  for (let i = 0; i < m.length; i++) {
    if (m[i]) {
      d[i * 3] = 0.4 * d[i * 3] + 0.6 * 255;
      d[i * 3 + 1] = 0.4 * d[i * 3 + 1];
      d[i * 3 + 2] = 0.4 * d[i * 3 + 2];
    }
  }

  [hsv, g, dis, l0, l1, leaf, dark, les0, les, hier].forEach(x => x.delete());
  contours.delete();
  return { ratio, leafVis, lesionVis };
}

// ---------- Output panels (created automatically, same as the Python version) ----------
// "Analyzed Image" = the main #view canvas. The two expert panels sit in an open <details> block.
function getCanvas(id, title) {
  let wrap = $('wrap-' + id);
  if (!wrap) {
    let det = $('expert');
    if (!det) {
      det = document.createElement('details');
      det.id = 'expert'; det.open = true; det.style.margin = '12px 0';
      const sum = document.createElement('summary');
      sum.textContent = '🔬 Expert View: Detailed Analysis & Masks'; sum.style.cssText = 'font-weight:bold;cursor:pointer;margin-bottom:8px';
      const row = document.createElement('div');
      row.id = 'expert-row'; row.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px';
      det.append(sum, row);
      $('report').insertAdjacentElement('afterend', det);
    }
    wrap = document.createElement('div');
    wrap.id = 'wrap-' + id; wrap.style.cssText = 'flex:1 1 320px;min-width:0';
    const cap = document.createElement('div');
    cap.textContent = title; cap.style.cssText = 'font-weight:bold;margin-bottom:4px';
    const c = document.createElement('canvas');
    c.id = id; c.style.maxWidth = '100%';
    wrap.append(cap, c);
    $('expert-row').appendChild(wrap);
  }
  wrap.hidden = false;
  return $(id);
}

function showMat(id, title, mat) { cv.imshow(getCanvas(id, title), mat); }

function hideCanvas(id) {
  const wrap = $('wrap-' + id);
  if (wrap) wrap.hidden = true;
}

// ---------- Report ----------
function confLine(labelHtml, conf) {
  return `${labelHtml} (Confidence ${(conf * 100).toFixed(1)}%)`;
}

function banner(status, msg, ratio) {
  const color = STATUS_COLORS[status] || '#2c2c2c';
  let bar = '';
  if (ratio !== undefined) {
    const v = Math.max(0, Math.min(100, ratio));
    bar = `<p><b>SEVERITY SCORE: [${v.toFixed(1)}%]</b></p>
      <div style="position:relative"><div style="position:absolute;left:${v}%;top:-18px;transform:translateX(-50%)">▲</div>
      <div class="bar"><div style="background:#27ae60"></div><div style="background:#f39c12"></div><div style="background:#c0392b"></div></div></div>`;
  }
  return `<p><b>DIAGNOSIS STATUS: <span style="color:${color}">[${status}]</span></b></p>${bar}` +
         `<p><b>RECOMMENDED ACTION:</b><br>${msg}</p>`;
}

// ---------- Main ----------
async function diagnose(file) {
  $('status').textContent = 'Analyzing…';
  try {
    const bmp = await createImageBitmap(file);
    const s = Math.min(1, MAX_INPUT_SIDE / Math.max(bmp.width, bmp.height));
    const view = $('view');
    view.width = Math.round(bmp.width * s); view.height = Math.round(bmp.height * s);
    const ctx = view.getContext('2d');
    ctx.drawImage(bmp, 0, 0, view.width, view.height);
    const rgba = cv.imread(view), rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    rgba.delete();

    const regions = findLeafRegions(rgb);

    // Detected leaf regions (red boxes) get their own panel, like the Python version
    const bc = getCanvas('out-boxes', 'Detected Leaf Regions (Red Boxes)');
    cv.imshow(bc, rgb);
    const bctx = bc.getContext('2d');
    bctx.lineWidth = Math.max(2, bc.width / 300);
    bctx.strokeStyle = '#ff0000'; bctx.fillStyle = '#ff0000'; bctx.font = 'bold 28px Arial';

    const results = [];
    let idx = 0;
    for (const r of regions) {
      idx++;
      bctx.strokeRect(r.x, r.y, r.w, r.h);
      bctx.fillText(String(idx), r.x + 8, r.y + 34);
      const crop = rgb.roi(new cv.Rect(r.x, r.y, r.w, r.h));
      const t = toTensor(crop);
      const p1 = await predict(sess1, t);
      const tomatoConf = p1[M1_TOMATO];
      if (tomatoConf >= MIN_CONF) {
        const p2 = await predict(sess2, t);
        const lb = p2[M2_BLIGHT], nlb = 1 - lb;
        const stage2Conf = Math.max(lb, nlb);
        results.push({
          top: lb >= nlb ? 'blight' : 'not',
          tomatoConf,                    // Stage 1: confidence it is a tomato leaf
          stage2Conf,                    // Stage 2: confidence of late blight / no late blight
          score: tomatoConf * stage2Conf,
          crop: crop.clone(),
        });
      } else {
        const nt = p1[1 - M1_TOMATO];    // Stage 1: confidence it is NOT a tomato leaf
        results.push({ top: 'non', nonTomatoConf: nt, score: nt });
      }
      crop.delete();
    }

    const tomato = results.filter(r => r.top !== 'non');
    const blighted = tomato.filter(r => r.top === 'blight');
    const note = ` (Checked ${regions.length} leaf region(s))`;
    let html;

    if (!regions.length) {
      // 1) No leaf region found -> not a tomato (models were not run, so no confidence)
      html = banner('non-tomato', '🚫 <b>Not a tomato leaf</b> — no leaf region was detected in the image.');
      cv.imshow($('view'), rgb);
      hideCanvas('out-leafmask');
    } else if (!tomato.length) {
      // 2) No region is a tomato leaf -> not a tomato, with confidence
      const ch = results.reduce((a, b) => (b.score > a.score ? b : a));
      html = banner('non-tomato', `🚫 ${confLine('<b>Not a tomato leaf</b>', ch.nonTomatoConf)}.${note}`);
      cv.imshow($('view'), rgb);
      hideCanvas('out-leafmask');
    } else if (!blighted.length) {
      // 3) Tomato, but no late blight: tomato confidence + "no late blight" confidence
      const ch = tomato.reduce((a, b) => (b.score > a.score ? b : a));
      const { leafVis, lesionVis } = analyzeLeaf(ch.crop);
      html = banner('tomato-without-late-blight',
        `${confLine('🍅 <b>Tomato leaf</b>', ch.tomatoConf)}<br>` +
        `${confLine('<b>No late blight detected</b>', ch.stage2Conf)}.${note}`);
      cv.imshow($('view'), rgb);
      showMat('out-leafmask', 'Leaf Mask (Red Outline)', leafVis);
      leafVis.delete(); lesionVis.delete();
    } else {
      // 4) Tomato with late blight: both confidences + severity and advice
      const ch = blighted.reduce((a, b) => (b.score > a.score ? b : a));
      const { ratio, leafVis, lesionVis } = analyzeLeaf(ch.crop);
      const advice = ratio < 33.3
        ? 'Mild late blight. Remove infected leaves promptly and apply an organic copper fungicide.'
        : ratio < 66.6
          ? 'Moderate late blight. Isolate the plant, remove infected foliage, and use a systemic fungicide.'
          : '🚨 Severe late blight! Remove and destroy the plant immediately to prevent spore transmission.';
      html = banner('tomato-with-late-blight',
        `${confLine('🍅 <b>Tomato leaf</b>', ch.tomatoConf)}<br>` +
        `${confLine('⚠️ <b>Late blight detected</b>', ch.stage2Conf)}<br>` +
        `${advice}${note}`, ratio);
      cv.imshow($('view'), lesionVis);
      showMat('out-leafmask', 'Leaf Mask (Red Outline)', leafVis);
      leafVis.delete(); lesionVis.delete();
    }

    results.forEach(r => r.crop && r.crop.delete());
    rgb.delete();
    $('report').innerHTML = html;
    $('report').hidden = false;
    $('status').textContent = `Done (checked ${regions.length} leaf region(s)).`;
  } catch (err) {
    $('status').textContent = 'Error while analyzing: ' + (err.message || err);
    console.error(err);
  }
}

$('file').addEventListener('change', e => { if (e.target.files[0]) diagnose(e.target.files[0]); });
init();
