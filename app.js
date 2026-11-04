const MAX_INPUT_SIDE = 800, MIN_CONF = 0.60, MAX_LEAVES = 10, MIN_LEAF_FRAC = 0.05;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const M1_TOMATO = 1;   // index of "Tomato" in model 1 output
const M2_BLIGHT = 0;   // index of "Late_blight" in model 2 output (若你的模型順序為 [Late_blight, Not_late_blight] 請改為 0)

let sess1, sess2;
const $ = id => document.getElementById(id);

async function init() {
  let step = '';
  try {
    step = '1/3 載入 OpenCV';
    $('status').textContent = step + '…';

    await new Promise((resolve, reject) => {
      let checkCount = 0;
      const interval = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
          clearInterval(interval);
          resolve();
        } else if (checkCount > 75) {
          clearInterval(interval);
          reject(new Error('OpenCV.js 載入逾時，請檢查網路連線或 CDN 是否被阻擋。'));
        }
        checkCount++;
      }, 200);

      if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
        cv['onRuntimeInitialized'] = () => {
          clearInterval(interval);
          resolve();
        };
      }
    });

    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    ort.env.wasm.numThreads = 1;

    step = '2/3 載入模型 1 (Tomato)';
    $('status').textContent = step + '…';
    sess1 = await ort.InferenceSession.create('m1_tomato.onnx');

    step = '3/3 載入模型 2 (Late Blight)';
    $('status').textContent = step + '…';
    sess2 = await ort.InferenceSession.create('m2_late_blight.onnx');

    $('status').textContent = '系統就緒，請上傳或拍攝番茄葉片照片。';
    $('file').disabled = false;
  } catch (err) {
    $('status').textContent = '步驟 ' + step + ' 發生錯誤: ' + (err.message || err);
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
  const leafItems = [];

  for (let i = 1; i < n; i++) {
    const x = stats.data32S[i * 5 + 0];
    const y = stats.data32S[i * 5 + 1];
    const bw = stats.data32S[i * 5 + 2];
    const bh = stats.data32S[i * 5 + 3];
    const area = stats.data32S[i * 5 + 4];

    if (area >= MIN_LEAF_FRAC * rgb.rows * rgb.cols) {
      const singleMask = new cv.Mat.zeros(labels.rows, labels.cols, cv.CV_8UC1);
      const lower = new cv.Mat(labels.rows, labels.cols, labels.type(), [i, 0, 0, 0]);
      const upper = new cv.Mat(labels.rows, labels.cols, labels.type(), [i, 0, 0, 0]);
      cv.inRange(labels, lower, upper, singleMask);
      lower.delete(); upper.delete();

      const contours = new cv.MatVector();
      const hier = new cv.Mat();
      cv.findContours(singleMask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      singleMask.delete(); hier.delete();

      leafItems.push({
        area: parseInt(area),
        box: { x: parseInt(x), y: parseInt(y), w: parseInt(bw), h: parseInt(bh) },
        contours: contours
      });
    }
  }

  [hsv, green, brown, mask0, m1, mask, labels, stats, cents].forEach(m => m.delete());
  leafItems.sort((a, b) => b.area - a.area);
  return leafItems.slice(0, MAX_LEAVES);
}

function drawPureLeafOutlines(rgb, analyzedResults) {
  const vis = rgb.clone();
  const thickness = Math.max(2, Math.floor(rgb.cols / 300));
  const RED_SCALAR = new cv.Scalar(255, 0, 0, 255);
  const GREEN_SCALAR = new cv.Scalar(0, 255, 0, 255);
  const GRAY_SCALAR = new cv.Scalar(128, 128, 128, 255);

  analyzedResults.forEach((res, index) => {
    const idx = index + 1;
    let color = GRAY_SCALAR;
    if (res.status === 'tomato-with-late-blight') color = RED_SCALAR;
    else if (res.status === 'tomato-without-late-blight') color = GREEN_SCALAR;

    cv.drawContours(vis, res.contours, -1, color, thickness);

    let cX = res.box.x + 8;
    let cY = res.box.y + 35;
    if (res.contours.size() > 0) {
      const cnt = res.contours.get(0);
      const M = cv.moments(cnt);
      if (M.m00 > 0) {
        cX = Math.round(M.m10 / M.m00);
        cY = Math.round(M.m01 / M.m00);
      }
    }
    cv.putText(vis, `#${idx}`, new cv.Point(cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv.LINE_AA);
  });

  return vis;
}

function calcInfectionRatio(rgbCrop) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgbCrop, hsv, cv.COLOR_RGB2HSV);
  const maskGreen = inRange(hsv, [35, 40, 40], [85, 255, 255]);
  const maskDiseased = inRange(hsv, [10, 30, 30], [35, 255, 255]);
  const leafMask0 = new cv.Mat();
  cv.bitwise_or(maskGreen, maskDiseased, leafMask0);

  const leafMask1 = morph(leafMask0, cv.MORPH_CLOSE, 5);
  const leafMask = morph(leafMask1, cv.MORPH_OPEN, 5);

  const maskDark = inRange(hsv, [0, 0, 0], [180, 255, 80]);
  const lesion0 = new cv.Mat();
  cv.bitwise_or(maskDiseased, maskDark, lesion0);
  const lesion = new cv.Mat();
  cv.bitwise_and(lesion0, lesion0, lesion, leafMask);

  const leafPx = cv.countNonZero(leafMask);
  const ratio = leafPx === 0 ? 0.0 : (cv.countNonZero(lesion) / leafPx) * 100.0;

  const lesionVis = rgbCrop.clone();
  const d = lesionVis.data;
  const m = lesion.data;
  for (let i = 0; i < m.length; i++) {
    if (m[i] > 0) {
      d[i * 3]     = Math.round(0.4 * d[i * 3] + 0.6 * 255);
      d[i * 3 + 1] = Math.round(0.4 * d[i * 3 + 1]);
      d[i * 3 + 2] = Math.round(0.4 * d[i * 3 + 2]);
    }
  }

  [hsv, maskGreen, maskDiseased, leafMask0, leafMask1, leafMask, maskDark, lesion0, lesion].forEach(mat => mat.delete());
  return { ratio, lesionVis };
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
  const inputName = sess.inputNames[0];
  const outputName = sess.outputNames[0];
  const o = await sess.run({ [inputName]: tensor });
  return softmax(Array.from(o[outputName].data));
}

// ---------- Report Rendering (Aligned with Python render_multi_report) ----------
function renderMultiReport(analyzedResults) {
  let htmlCards = '';
  analyzedResults.forEach((res, index) => {
    const idx = index + 1;
    const status = res.status;
    const ratio = res.ratio || 0.0;
    const msg = res.msg;

    let statusColor = '#555555';
    if (status === 'tomato-with-late-blight') statusColor = '#a93226';
    else if (status === 'tomato-without-late-blight') statusColor = '#1e8449';

    let sliderHtml = '';
    if (status === 'tomato-with-late-blight') {
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
    }

    htmlCards += `
      <div class="custom-report-card">
          <div style="font-size: 16px !important; font-weight: bold !important; color: ${statusColor} !important;">
              🌿 Leaf #${idx} — DIAGNOSIS STATUS: [${status.toUpperCase()}]
          </div>
          ${sliderHtml}
          <div style="margin-top: 8px !important; font-size: 15px !important; line-height: 1.6 !important;">
              ${msg}
          </div>
      </div>
    `;
  });
  return htmlCards;
}

// ---------- Main Diagnosis Logic ----------
async function diagnose(file) {
  $('status').textContent = 'Analyzing…';
  let rgb = null;
  let pureOutlinesVis = null;
  let lesionOutputVis = null;
  let firstBlightLesionVis = null;
  let leafItems = [];

  try {
    const bmp = await createImageBitmap(file);
    const s = Math.min(1, MAX_INPUT_SIDE / Math.max(bmp.width, bmp.height));
    const width = Math.round(bmp.width * s);
    const height = Math.round(bmp.height * s);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(bmp, 0, 0, width, height);

    const rgba = cv.imread(tempCanvas);
    rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    rgba.delete();

    leafItems = findLeafRegions(rgb);
    if (leafItems.length === 0) {
      const emptyReport = renderMultiReport([{
        status: 'non-tomato',
        msg: '🚫 <b>Non-tomato leaf</b> — No valid leaf regions detected.'
      }]);
      cv.imshow($('view-outlines'), rgb);
      cv.imshow($('view-lesion'), rgb);
      $('report').innerHTML = emptyReport;
      $('report').hidden = false;
      $('status').textContent = 'Done.';
      rgb.delete();
      return;
    }

    const analyzedResults = [];
    firstBlightLesionVis = rgb.clone();
    let hasBlightFound = false;

    for (const item of leafItems) {
      const { x, y, w: bw, h: bh } = item.box;
      const rect = new cv.Rect(x, y, bw, bh);
      const crop = rgb.roi(rect);
      if (crop.empty() || crop.cols === 0 || crop.rows === 0) {
        crop.delete();
        continue;
      }

      const tensor = toTensor(crop);
      const p1 = await predict(sess1, tensor);
      const tomatoConf = p1[M1_TOMATO];

      if (tomatoConf >= MIN_CONF) {
        const tensor2 = toTensor(crop);
        const p2 = await predict(sess2, tensor2);
        const lbConf = p2[M2_BLIGHT];
        const notLbConf = p2[1 - M2_BLIGHT];
        const isBlight = lbConf >= notLbConf;

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
            box: item.box,
            contours: item.contours,
            status: 'tomato-with-late-blight',
            ratio: ratio,
            msg: msg
          });

          if (!hasBlightFound) {
            const fullLesionCanvas = rgb.clone();
            const roi = fullLesionCanvas.roi(rect);
            lesionVis.copyTo(roi);
            roi.delete();
            firstBlightLesionVis.delete();
            firstBlightLesionVis = fullLesionCanvas;
            hasBlightFound = true;
          }
          lesionVis.delete();
        } else {
          // 對齊 Python 版：無晚疫病時顯示正確的信心分數與狀態
          const msg = `🍅 <b>Tomato Leaf Confidence</b>: ${(tomatoConf * 100).toFixed(1)}%<br>✅ <b>No Late Blight Confidence</b>: ${(notLbConf * 100).toFixed(1)}%`;
          analyzedResults.push({
            box: item.box,
            contours: item.contours,
            status: 'tomato-without-late-blight',
            msg: msg
          });
        }
      } else {
        const nonTomatoConf = p1[1 - M1_TOMATO];
        const msg = `🚫 <b>Non-tomato / Background Noise Confidence</b>: ${(nonTomatoConf * 100).toFixed(1)}%<br>This region is not identified as a tomato leaf.`;
        analyzedResults.push({
          box: item.box,
          contours: item.contours,
          status: 'non-tomato',
          msg: msg
        });
      }
      crop.delete();
    }

    pureOutlinesVis = drawPureLeafOutlines(rgb, analyzedResults);
    lesionOutputVis = hasBlightFound ? firstBlightLesionVis : rgb.clone();
    const htmlReport = renderMultiReport(analyzedResults);

    cv.imshow($('view-outlines'), pureOutlinesVis);
    cv.imshow($('view-lesion'), lesionOutputVis);
    $('report').innerHTML = htmlReport;
    $('report').hidden = false;

    $('status').textContent = 'Done.';
  } catch (err) {
    $('status').textContent = 'Error while analyzing: ' + (err.message || err);
    console.error(err);
  } finally {
    if (pureOutlinesVis && !pureOutlinesVis.isDeleted()) pureOutlinesVis.delete();
    if (lesionOutputVis && !lesionOutputVis.isDeleted()) lesionOutputVis.delete();
    if (firstBlightLesionVis && !firstBlightLesionVis.isDeleted()) firstBlightLesionVis.delete();
    if (rgb && !rgb.isDeleted()) rgb.delete();
    if (leafItems && leafItems.length > 0) {
      leafItems.forEach(item => {
        if (item.contours && !item.contours.isDeleted()) {
          item.contours.delete();
        }
      });
    }
  }
}

$('file').addEventListener('change', e => { if (e.target.files[0]) diagnose(e.target.files[0]); });
init();
