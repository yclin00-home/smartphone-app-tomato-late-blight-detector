const MAX_INPUT_SIDE = 800, MIN_CONF_INFERENCE = 0.60, MAX_LEAVES = 10, MIN_LEAF_FRAC = 0.05;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const M1_TOMATO = 1;   // index of "Tomato" in model 1 output
const M2_BLIGHT = 0;   // index of "Late_blight" in model 2 output
const RED = [255, 0, 0, 255];
const GREEN = [0, 255, 0, 255];

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

// 升級為 v2 的精準輪廓與獨立區塊提取
function findLeafContoursAndCrops(rgb) {
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
  const leafItems = [];

  for (let i = 1; i < n; i++) {
    const [x, y, w, h, area] = [0, 1, 2, 3, 4].map(j => stats.data32S[i * 5 + j]);
    if (area >= MIN_LEAF_FRAC * rgb.rows * rgb.cols) {
      // 擷取單一葉片精準輪廓
      const singleMask = new cv.Mat.zeros(labels.rows, labels.cols, cv.CV_8U);
      for (let r = 0; r < labels.rows; r++) {
        for (let c = 0; c < labels.cols; c++) {
          if (labels.intAt(r, c) === i) singleMask.ucharPtr(r, c)[0] = 255;
        }
      }
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(singleMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      
      leafItems.push({ area, box: { x, y, w, h }, contours });
      singleMask.delete(); hierarchy.delete();
    }
  }
  [hsv, green, brown, mask0, m1, mask, labels, stats, cents].forEach(m => m.delete());
  leafItems.sort((a, b) => b.area - a.area);
  return leafItems.slice(0, MAX_LEAVES);
}

// 畫出貼合葉緣的純輪廓線（不使用方形框）
function drawPureLeafOutlines(rgb, analyzedResults) {
  const vis = rgb.clone();
  const thickness = Math.max(2, Math.floor(rgb.cols / 300));
  
  analyzedResults.forEach((res, idx) => {
    let color;
    if (res.status === 'tomato-with-late-blight') color = new cv.Scalar(255, 0, 0, 255);
    else if (res.status === 'tomato-without-late-blight') color = new cv.Scalar(0, 255, 0, 255);
    else color = new cv.Scalar(128, 128, 128, 255);

    cv.drawContours(vis, res.contours, -1, color, thickness);
    
    // 計算輪廓中心點以放置編號標籤 (#1, #2...)
    let M = cv.moments(res.contours.get(0));
    let cX = M.m00 > 0 ? Math.round(M.m10 / M.m00) : res.box.x + 8;
    let cY = M.m00 > 0 ? Math.round(M.m01 / M.m00) : res.box.y + 35;
    
    cv.putText(vis, `#${idx + 1}`, new cv.Point(cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv.LINE_AA);
  });
  return vis;
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

// 計算感染比例與病斑高亮
function calcInfectionRatio(rgbCrop) {
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

  const lesionVis = rgbCrop.clone();
  const d = lesionVis.data, m = les.data;
  for (let i = 0; i < m.length; i++) {
    if (m[i]) {
      d[i * 3] = 0.4 * d[i * 3] + 0.6 * 255;
      d[i * 3 + 1] = 0.4 * d[i * 3 + 1];
      d[i * 3 + 2] = 0.4 * d[i * 3 + 2];
    }
  }
  [hsv, g, dis, l0, l1, leaf, dark, les0, les].forEach(x => x.delete());
  return { ratio, lesionVis };
}

// ---------- Extra output canvases ----------
function showMat(id, title, mat) {
  let wrap = $('wrap-' + id);
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.id = 'wrap-' + id;
    wrap.style.margin = '12px 0';
    const cap = document.createElement('div');
    cap.textContent = title;
    cap.style.fontWeight = 'bold';
    cap.style.marginBottom = '4px';
    const c = document.createElement('canvas');
    c.id = id;
    c.style.maxWidth = '100%';
    wrap.append(cap, c);
    $('report').parentNode.appendChild(wrap);
  }
  wrap.hidden = false;
  cv.imshow($(id), mat);
}

function hideCanvas(id) {
  const wrap = $('wrap-' + id);
  if (wrap) wrap.hidden = true;
}

// ---------- Multi-Report Rendering (對應 Python v2 介面卡片與顏色強制修正) ----------
function renderMultiReport(analyzedResults) {
  let htmlCards = '';
  analyzedResults.forEach((res, idx) => {
    const status = res.status;
    const ratio = res.ratio || 0.0;
    const msg = res.msg;
    
    let statusColor = '#555555';
    let sliderHtml = '';

    if (status === 'tomato-with-late-blight') {
      statusColor = '#a93226';
      const rVal = Math.max(0.0, Math.min(100.0, ratio));
      sliderHtml = `
        <div style="margin-top: 10px; font-size: 16px; font-weight: bold; color: #a93226 !important;">
            📉 Severity Score: [${rVal.toFixed(1)}%]
        </div>
        <div style="position: relative; margin-top: 10px; margin-bottom: 20px;">
            <div style="display: flex; height: 16px; border-radius: 4px; overflow: hidden; border: 1px solid #b5b0a5;">
                <div style="background-color: #27ae60; flex: 1;" title="Mild (<33.3%)"></div>
                <div style="background-color: #f39c12; flex: 1;" title="Moderate (33.3% - 66.6%)"></div>
                <div style="background-color: #c0392b; flex: 1;" title="Severe (>66.6%)"></div>
            </div>
            <div style="position: absolute; top: -3px; bottom: -3px; left: ${rVal}%; width: 3px; background-color: #111111; transform: translateX(-50%); z-index: 5; box-shadow: 0 0 3px rgba(255,255,255,0.8);"></div>
            <div style="position: absolute; top: 18px; left: ${rVal}%; transform: translateX(-50%); font-size: 12px; font-weight: bold; color: #111111 !important; background: rgba(255,255,255,0.9); padding: 0 4px; border: 1px solid #b5b0a5; border-radius: 3px; white-space: nowrap;">
                ${rVal.toFixed(1)}%
            </div>
        </div>
      `;
    } else if (status === 'tomato-without-late-blight') {
      statusColor = '#1e8449';
    }

    htmlCards += `
    <div style="background-color: #f4f1ea !important; border: 2px solid #dcd6cd !important; border-radius: 10px !important; padding: 15px !important; margin-bottom: 12px !important; font-family: Arial, sans-serif !important; color: #2c2c2c !important;">
        <div style="font-size: 16px !important; font-weight: bold !important; color: ${statusColor} !important;">
            🌿 Leaf #${idx + 1} — DIAGNOSIS STATUS: [${status.toUpperCase()}]
        </div>
        ${sliderHtml}
        <div style="margin-top: 8px !important; font-size: 15px !important; line-height: 1.6 !important; color: #2c2c2c !important;">
            ${msg}
        </div>
    </div>
    `;
  });
  return htmlCards;
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

    const leafItems = findLeafContoursAndCrops(rgb);

    if (!leafItems.length) {
      const emptyReport = renderMultiReport([{ status: 'non-tomato', msg: '🚫 <b>Non-tomato leaf</b> — No valid leaf regions detected.' }]);
      showMat('out-analyzed', '1. True Leaf Edge Outlines', rgb);
      hideCanvas('out-leafmask');
      $('report').innerHTML = emptyReport;
      $('report').hidden = false;
      rgb.delete();
      $('status').textContent = 'Done.';
      return;
    }

    const analyzedResults = [];
    let firstBlightLesionVis = rgb.clone();
    let hasBlightFound = false;

    for (const item of leafItems) {
      const { x, y, w, h } = item.box;
      const crop = rgb.roi(new cv.Rect(x, y, w, h));
      if (crop.empty()) { crop.delete(); continue; }

      const t = toTensor(crop);
      const p1 = await predict(sess1, t);
      const tomatoConf = p1[M1_TOMATO];

      if (tomatoConf >= MIN_CONF_INFERENCE) {
        const p2 = await predict(sess2, t);
        const lbConf = p2[M2_BLIGHT];
        const isBlight = lbConf >= (1 - lbConf);

        if (isBlight) {
          const { ratio, lesionVis } = calcInfectionRatio(crop);
          let advice = '';
          if (ratio < 33.3) {
            advice = 'Advice: Mild late blight. Recommend removing infected leaves and applying copper fungicide.';
          } else if (ratio < 66.6) {
            advice = 'Advice: Moderate late blight. Isolate plant and use systemic fungicide.';
          } else {
            advice = 'Advice: 🚨 Severe late blight! Strongly recommend immediate removal and destruction of the plant to prevent spread.';
          }

          const msg = `🍅 <b>Tomato Leaf Confidence</b>: ${(tomatoConf * 100).toFixed(1)}%<br>⚠️ <b>Late Blight Confidence</b>: ${(lbConf * 100).toFixed(1)}%<br><b>${advice}</b>`;
          analyzedResults.push({
            box: { x, y, w, h },
            contours: item.contours,
            status: 'tomato-with-late-blight',
            ratio: ratio,
            msg: msg
          });

          if (!hasBlightFound) {
            firstBlightLesionVis.delete();
            firstBlightLesionVis = rgb.clone();
            // 在全圖對應位置填入病斑視覺化
            let roi = firstBlightLesionVis.roi(new cv.Rect(x, y, w, h));
            lesionVis.copyTo(roi);
            roi.delete();
            hasBlightFound = true;
          }
          lesionVis.delete();
        } else {
          const nlbConf = 1 - lbConf;
          const msg = `🍅 <b>Tomato Leaf Confidence</b>: ${(tomatoConf * 100).toFixed(1)}%<br>✅ <b>No Late Blight Confidence</b>: ${(nlbConf * 100).toFixed(1)}%`;
          analyzedResults.push({
            box: { x, y, w, h },
            contours: item.contours,
            status: 'tomato-without-late-blight',
            msg: msg
          });
        }
      } else {
        const nonTomatoConf = p1[1 - M1_TOMATO];
        const msg = `🚫 <b>Non-tomato / Background Noise Confidence</b>: ${(nonTomatoConf * 100).toFixed(1)}%<br>This region is not identified as a tomato leaf.`;
        analyzedResults.push({
          box: { x, y, w, h },
          contours: item.contours,
          status: 'non-tomato',
          msg: msg
        });
      }
      crop.delete();
    }

    const pureOutlinesVis = drawPureLeafOutlines(rgb, analyzedResults);
    const lesionOutputVis = hasBlightFound ? firstBlightLesionVis : rgb;
    const htmlReport = renderMultiReport(analyzedResults);

    showMat('out-analyzed', '1. True Leaf Edge Outlines', pureOutlinesVis);
    showMat('out-leafmask', '2. Lesion Highlight (Late Blight Only)', lesionOutputVis);

    pureOutlinesVis.delete();
    if (!hasBlightFound) firstBlightLesionVis.delete();
    else lesionOutputVis.delete();
    rgb.delete();

    $('report').innerHTML = htmlReport;
    $('report').hidden = false;
    $('status').textContent = `Done (checked ${leafItems.length} leaf region(s)).`;
  } catch (err) {
    $('status').textContent = 'Error while analyzing: ' + (err.message || err);
    console.error(err);
  }
}

$('file').addEventListener('change', e => { if (e.target.files[0]) diagnose(e.target.files[0]); });
init();
