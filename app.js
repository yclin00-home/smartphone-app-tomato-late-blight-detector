// ==========================================
// 參數與模型設定 (對應 Python 版本)
// ==========================================
const MAX_INPUT_SIDE = 800;    // 限制輸入圖片最大邊長以防 OOM
const MIN_CONF = 0.60;         // 最低番茄葉片信心水準
const MAX_LEAVES = 10;         // 最多檢測葉片區域數
const MIN_LEAF_FRAC = 0.05;    // 最小葉片面積佔比

let model1 = null;
let model2 = null;
let names1 = ["Non_tomato", "Tomato"];
let names2 = ["Late_blight", "Not_late_blight"];

// 狀態顏色對應
const STATUS_COLORS = {
    "non-tomato": "#555555",
    "tomato-without-late-blight": "#1e8449",
    "tomato-with-late-blight": "#a93226",
};

// 初始化載入模型
async function loadModels() {
    try {
        console.log("⏳ 正在載入 ONNX 模型...");
        // 確保使用 WebAssembly 執行緒加速 (如果支援)
        ort.env.wasm.numThreads = 4;
        ort.env.wasm.simd = true;

        model1 = await ort.InferenceSession.create('./m1_tomato.onnx', { executionProviders: ['wasm'] });
        model2 = await ort.InferenceSession.create('./m2_late_blight.onnx', { executionProviders: ['wasm'] });
        console.log("✅ 雙模型載入成功！");
    } catch (e) {
        console.error("❌ 模型載入失敗：", e);
        alert("模型載入失敗，請確認 .onnx 檔案是否正確放在根目錄！");
    }
}

// 當 OpenCV.js 準備就緒時觸發
let cvReady = false;
function onOpenCvReady() {
    cvReady = true;
    console.log("✅ OpenCV.js 已就緒");
}

// 頁面載入時初始化
window.addEventListener('DOMContentLoaded', () => {
    loadModels();
    
    const diagnoseBtn = document.getElementById('diag_btn');
    if (diagnoseBtn) {
        diagnoseBtn.addEventListener('click', handleDiagnosis);
    }
});

// ==========================================
// 影像預處理與模型推論輔助函式
// ==========================================
// 將影像轉為 224x224 Tensor，並進行 ImageNet 標準化
function preprocessImage(canvasOrImage) {
    let canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    let ctx = canvas.getContext('2d');
    ctx.drawImage(canvasOrImage, 0, 0, 224, 224);
    
    let imgData = ctx.getImageData(0, 0, 224, 224);
    let pixels = imgData.data;
    
    const floatData = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < 224 * 224; i++) {
        let r = pixels[i * 4] / 255.0;
        let g = pixels[i * 4 + 1] / 255.0;
        let b = pixels[i * 4 + 2] / 255.0;
        
        // NCHW 格式轉換與 ImageNet 標準化
        floatData[i] = (r - mean[0]) / std[0];
        floatData[224 * 224 + i] = (g - mean[1]) / std[1];
        floatData[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
    }
    
    return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

// Softmax 函式
function softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
}

async function predict(session, names, tensor) {
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;
    const probs = softmax(Array.from(outputData));
    
    let res = {};
    names.forEach((name, idx) => {
        res[name] = probs[idx];
    });
    return res;
}

// ==========================================
// OpenCV.js 電腦視覺與葉片分割
// ==========================================
function findLeafRegions(srcMat) {
    if (!cvReady) return [];
    let h = srcMat.rows;
    let w = srcMat.cols;
    
    let hsv = new cv.Mat();
    cv.cvtColor(srcMat, hsv, cv.COLOR_RGB2HSV);
    
    let greenMask = new cv.Mat();
    let brownMask = new cv.Mat
    let lowGreen = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [35, 20, 20, 0]);
    let highGreen = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [85, 255, 255, 0]);
    cv.inRange(hsv, lowGreen, highGreen, greenMask);
    
    let lowBrown = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [5, 15, 15, 0]);
    let highBrown = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [35, 255, 255, 0]);
    cv.inRange(hsv, lowBrown, highBrown, brownMask);
    
    let mask = new cv.Mat();
    cv.bitwise_or(greenMask, brownMask, mask);
    
    let kSize = new cv.Size(15, 15);
    let k = cv.getStructuringElement(cv.MORPH_ELLIPSE, kSize);
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, k);
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
    
    let labels = new cv.Mat();
    let stats = new cv.Mat();
    let centroids = new cv.Mat();
    let n = cv.connectedComponentsWithStats(mask, labels, stats, centroids, 8, cv.CV_32S);
    
    let regions = [];
    let minArea = MIN_LEAF_FRAC * h * w;
    
    for (let i = 1; i < n; i++) {
        let area = stats.intAt(i, cv.CC_STAT_AREA);
        let x = stats.intAt(i, cv.CC_STAT_LEFT);
        let y = stats.intAt(i, cv.CC_STAT_TOP);
        let bw = stats.intAt(i, cv.CC_STAT_WIDTH);
        let bh = stats.intAt(i, cv.CC_STAT_HEIGHT);
        
        if (area >= minArea) {
            regions.push({ area, x, y, bw, bh });
        }
    }
    
    regions.sort((a, b) => b.area - a.area);
    
    // 釋放記憶體
    hsv.delete(); greenMask.delete(); brownMask.delete(); lowGreen.delete(); highGreen.delete();
    lowBrown.delete(); highBrown.delete(); mask.delete(); k.delete(); labels.delete(); stats.delete(); centroids.delete();
    
    return regions.slice(0, MAX_LEAVES).map(r => [r.x, r.y, r.x + r.bw, r.y + r.bh]);
}

function drawBoxes(srcMat, regions) {
    let vis = srcMat.clone();
    let t = Math.max(2, Math.floor(srcMat.cols / 300));
    let color = new cv.Scalar(255, 0, 0, 255); // 紅色
    
    regions.forEach((r, idx) => {
        let p1 = new cv.Point(r[0], r[1]);
        let p2 = new cv.Point(r[2], r[3]);
        cv.rectangle(vis, p1, p2, color, t);
        cv.putText(vis, String(idx + 1), new cv.Point(r[0] + 8, r[1] + 38), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv.LINE_AA);
    });
    return vis;
}

function calcInfectionRatio(cropMat) {
    let h = cropMat.rows;
    let w = cropMat.cols;
    let hsv = new cv.Mat();
    cv.cvtColor(cropMat, hsv, cv.COLOR_RGB2HSV);
    
    let maskGreen = new cv.Mat();
    let maskDiseased = new cv.Mat();
    cv.inRange(hsv, new cv.Scalar(35, 40, 40, 0), new cv.Scalar(85, 255, 255, 0), maskGreen);
    cv.inRange(hsv, new cv.Scalar(10, 30, 30, 0), new cv.Scalar(35, 255, 255, 0), maskDiseased);
    
    let leafMask = new cv.Mat();
    cv.bitwise_or(maskGreen, maskDiseased, leafMask);
    
    let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    cv.morphologyEx(leafMask, leafMask, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(leafMask, leafMask, cv.MORPH_OPEN, kernel);
    
    let maskDark = new cv.Mat();
    cv.inRange(hsv, new cv.Scalar(0, 0, 0, 0), new cv.Scalar(180, 255, 80, 0), maskDark);
    
    let lesion = new cv.Mat();
    cv.bitwise_or(maskDiseased, maskDark, lesion);
    cv.bitwise_and(lesion, lesion, lesion, leafMask);
    
    let leafPx = cv.countNonZero(leafMask);
    let lesionPx = cv.countNonZero(lesion);
    let ratio = leafPx === 0 ? 0.0 : (lesionPx / leafPx) * 100.0;
    
    // 繪製外框
    let leafVis = cropMat.clone();
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(leafMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    cv.drawContours(leafVis, contours, -1, new cv.Scalar(255, 0, 0, 255), Math.max(2, Math.floor(w / 300)));
    
    // 繪製病斑渲染
    let lesionVis = cropMat.clone();
    let redColor = [255, 0, 0];
    let data = lesionVis.data;
    let mData = lesion.data;
    for (let i = 0; i < mData.length; i++) {
        if (mData[i] > 0) {
            data[i * 4] = Math.round(0.4 * data[i * 4] + 0.6 * redColor[0]);
            data[i * 4 + 1] = Math.round(0.4 * data[i * 4 + 1] + 0.6 * redColor[1]);
            data[i * 4 + 2] = Math.round(0.4 * data[i * 4 + 2] + 0.6 * redColor[2]);
        }
    }
    
    // 清理資源
    hsv.delete(); maskGreen.delete(); maskDiseased.delete(); leafMask.delete();
    kernel.delete(); maskDark.delete(); lesion.delete(); contours.delete(); hierarchy.delete();
    
    return { ratio, leafVis, lesionVis };
}

// ==========================================
// 網頁報表排版生成
// ==========================================
function renderCustomStyle(diagnosisStatus, ratio = null, msg = "") {
    let rVal = ratio !== null ? Math.max(0.0, Math.min(100.0, ratio)) : 0.0;
    let statusColor = STATUS_COLORS[diagnosisStatus] || "#2c2c2c";

    let sliderHtml = diagnosisStatus === "tomato-with-late-blight" ? `
        <div class="score-header" style="margin-top: 15px; font-size: 18px;">
            <span class="score-title">SEVERITY SCORE:</span>
            <span class="score-bracket">[</span><span class="score-value">${rVal.toFixed(1)}%</span><span class="score-bracket">]</span>
        </div>
        <div class="slider-container" style="position: relative; margin: 25px 0 15px 0;">
            <div class="slider-labels-grid" style="display: grid; grid-template-columns: 1fr 1fr 1fr; margin-bottom: 5px; font-size: 14px; font-weight: bold;">
                <span style="text-align: left;">Low (0-33%)</span>
                <span style="text-align: center;">Medium (33-66%)</span>
                <span style="text-align: right;">High (66-100%)</span>
            </div>
            <div class="pointer-up" style="position: absolute; top: 18px; font-size: 16px; color: #000000; left: ${rVal}%; transform: translateX(-50%);">▲</div>
            <div class="color-bar" style="display: flex; height: 24px; border-radius: 6px; overflow: hidden; border: 1.5px solid #b5b0a5; margin-top: 5px;">
                <div style="background-color: #27ae60; flex: 1;"></div>
                <div style="background-color: #f39c12; flex: 1;"></div>
                <div style="background-color: #c0392b; flex: 1;"></div>
            </div>
        </div>
    ` : '';

    return `
    <div class="custom-report-card" style="background-color: #f4f1ea; border: 2px solid #dcd6cd; border-radius: 12px; padding: 20px; font-family: Arial, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 15px; color: #2c2c2c;">
        <div class="score-header" style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">
            <span class="score-title">DIAGNOSIS STATUS:</span>
            <span class="score-bracket" style="color: ${statusColor} !important;">[</span><span class="score-value" style="font-size: 20px; color: ${statusColor} !important;">${diagnosisStatus}</span><span class="score-bracket" style="color: ${statusColor} !important;">]</span>
        </div>
        ${sliderHtml}
        <div class="treatment-section">
            <div class="treatment-title" style="font-weight:bold; margin-bottom:5px;">RECOMMENDED ACTION:</div>
            <div class="treatment-text">${msg}</div>
        </div>
    </div>
    `;
}

function confLine(labelHtml, conf) {
    return `${labelHtml} (Confidence ${(conf * 100).toFixed(1)}%)`;
}

// ==========================================
// 核心診斷流程
// ==========================================
async function handleDiagnosis() {
    const fileInput = document.getElementById('image_input');
    if (!fileInput || fileInput.files.length === 0) {
        alert("請先上傳或拍攝一張照片！");
        return;
    }

    if (!model1 || !model2) {
        alert("模型尚未載入完成，請稍候幾秒再試！");
        return;
    }

    if (!cvReady) {
        alert("OpenCV.js 正在初始化，請稍候再試！");
        return;
    }

    let file = fileInput.files[0];
    let imgElement = document.createElement('img');
    imgElement.src = URL.createObjectURL(file);
    
    await new Promise((resolve) => { imgElement.onload = resolve; });

    // 縮放防 OOM
    let canvas = document.createElement('canvas');
    let w = imgElement.width;
    let h = imgElement.height;
    if (w > MAX_INPUT_SIDE || h > MAX_INPUT_SIDE) {
        if (w > h) {
            h = Math.round(h * (MAX_INPUT_SIDE / w));
            w = MAX_INPUT_SIDE;
        } else {
            w = Math.round(w * (MAX_INPUT_SIDE / h));
            h = MAX_INPUT_SIDE;
        }
    }
    canvas.width = w;
    canvas.height = h;
    let ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, w, h);

    let srcMat = cv.imread(canvas);
    let regions = findLeafRegions(srcMat);
    let boxesVisMat = regions.length > 0 ? drawBoxes(srcMat, regions) : srcMat.clone();

    // 顯示紅框圖
    let boxesCanvas = document.createElement('canvas');
    cv.imshow(boxesCanvas, boxesVisMat);
    document.getElementById('boxes_output').src = boxesCanvas.toDataURL();

    if (regions.length === 0) {
        let status = "non-tomato";
        let msg = "🚫 <b>Not a tomato leaf</b> — no leaf region was detected in the image.";
        document.getElementById('report_output').innerHTML = renderCustomStyle(status, null, msg);
        let mainCanvas = document.createElement('canvas');
        cv.imshow(mainCanvas, srcMat);
        document.getElementById('image_output').src = mainCanvas.toDataURL();
        srcMat.delete(); boxesVisMat.delete();
        return;
    }

    let tomatoFound = false;
    let results = [];

    for (let r of regions) {
        let [x0, y0, x1, y1] = r;
        let rect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
        let crop = srcMat.rowRange(rect.y, rect.y + rect.height).colRange(rect.x, rect.x + rect.width);
        
        if (crop.rows === 0 || crop.cols === 0) {
            crop.delete();
            continue;
        }

        let cropCanvas = document.createElement('canvas');
        cv.imshow(cropCanvas, crop);
        
        let tensor = preprocessImage(cropCanvas);
        let p1 = await predict(model1, names1, tensor);
        let tomatoConf = p1["Tomato"];

        if (tomatoConf >= MIN_CONF) {
            tomatoFound = true;
            let p2 = await predict(model2, names2, tensor);
            let lbConf = p2["Late_blight"];
            let nlbConf = p2["Not_late_blight"];
            let top = lbConf >= nlb_conf ? "Late_blight" : "Not_late_blight";
            let stage2Conf = lbConf >= nlb_conf ? lbConf : nlbConf;

            results.push({
                top,
                tomato_conf: tomatoConf,
                stage2_conf: stage2Conf,
                score: tomatoConf * stage2Conf,
                crop: crop.clone()
            });
        } else {
            results.push({
                top: "Non_tomato",
                tomato_conf: tomatoConf,
                stage2_conf: null,
                non_tomato_conf: p1["Non_tomato"],
                score: p1["Non_tomato"],
                crop: crop.clone()
            });
        }
        crop.delete();
    }

    let note = ` (Checked ${regions.length} leaf region(s))`;

    if (!tomatoFound) {
        let chosen = results.reduce((max, res) => res.score > max.score ? res : max, results[0]);
        let status = "non-tomato";
        let msg = `🚫 ${confLine('<b>Not a tomato leaf</b>', chosen.non_tomato_conf)}.${note}`;
        document.getElementById('report_output').innerHTML = renderCustomStyle(status, null, msg);
        
        let mainCanvas = document.createElement('canvas');
        cv.imshow(mainCanvas, srcMat);
        document.getElementById('image_output').src = mainCanvas.toDataURL();
        srcMat.delete(); boxesVisMat.delete();
        return;
    }

    let tomatoResults = results.filter(res => res.top === "Late_blight" || res.top === "Not_late_blight");
    let blighted = tomatoResults.filter(res => res.top === "Late_blight");

    let chosen;
    let label;
    if (blighted.length > 0) {
        chosen = blighted.reduce((max, res) => res.score > max.score ? res : max, blighted[0]);
        label = "Late_blight";
    } else {
        chosen = tomatoResults.reduce((max, res) => res.score > max.score ? res : max, tomatoResults[0]);
        label = "Not_late_blight";
    }

    let tomatoLine = confLine("🍅 <b>Tomato leaf</b>", chosen.tomato_conf);
    let { ratio, leafVis, lesionVis } = calcInfectionRatio(chosen.crop);

    // 顯示葉片外框圖
    let leafCanvas = document.createElement('canvas');
    cv.imshow(leafCanvas, leafVis);
    document.getElementById('leaf_output').src = leafCanvas.toDataURL();

    if (label === "Not_late_blight") {
        let status = "tomato-without-late-blight";
        let lbLine = confLine("<b>No late blight detected</b>", chosen.stage2_conf);
        let msg = `${tomatoLine}<br>${lbLine}.${note}`;
        document.getElementById('report_output').innerHTML = renderCustomStyle(status, null, msg);

        let mainCanvas = document.createElement('canvas');
        cv.imshow(mainCanvas, srcMat);
        document.getElementById('image_output').src = mainCanvas.toDataURL();
    } else {
        let status = "tomato-with-late-blight";
        let lbLine = confLine("⚠️ <b>Late blight detected</b>", chosen.stage2_conf);
        let advice = "";
        if (ratio < 33.3) {
            advice = "Mild late blight. Remove infected leaves promptly and apply an organic copper fungicide.";
        } else if (ratio < 66.6) {
            advice = "Moderate late blight. Isolate the plant, remove infected foliage, and use a systemic fungicide.";
        } else {
            advice = "🚨 Severe late blight! Remove and destroy the plant immediately to prevent spore transmission.";
        }

        let msg = `${tomatoLine}<br>${lbLine}<br>${advice}${note}`;
        document.getElementById('report_output').innerHTML = renderCustomStyle(status, ratio, msg);

        // 顯示病斑渲染圖
        let mainCanvas = document.createElement('canvas');
        cv.imshow(mainCanvas, lesionVis);
        document.getElementById('image_output').src = mainCanvas.toDataURL();
    }

    srcMat.delete(); boxesVisMat.delete();
    results.forEach(r => r.crop.delete());
}
