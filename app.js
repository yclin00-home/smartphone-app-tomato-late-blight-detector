let session1 = null;
let session2 = null;

// 1. 初始化載入 ONNX 模型
async function initModels() {
    try {
        document.getElementById("report").innerText = "⏳ 正在載入 AI 模型...";
        ort.env.wasm.wasmPaths = "./"; // 指向當前資料夾的 wasm 檔案
        
        session1 = await ort.InferenceSession.create('./m1_tomato.onnx');
        session2 = await ort.InferenceSession.create('./m2_late_blight.onnx');
        
        document.getElementById("report").innerText = "✅ 模型就緒，請上傳番茄葉片照片。";
    } catch (e) {
        console.error(e);
        document.getElementById("report").innerText = "❌ 模型載入失敗";
    }
}
initModels();

// 2. 執行診斷
async function runDiagnosis() {
    const fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) {
        alert("請先選擇照片！");
        return;
    }
    if (!session1 || !session2) {
        alert("模型還在載入中...");
        return;
    }

    const file = fileInput.files[0];
    const imgUrl = URL.createObjectURL(file);
    const preview = document.getElementById("previewImg");
    preview.src = imgUrl;
    preview.style.display = "block";

    document.getElementById("report").innerText = "🔍 AI 正在分析中...";

    // 模擬推理結果（實際串接時會將圖片縮放成 224x224 轉成 Tensor 餵給 session1 與 session2）
    setTimeout(() => {
        document.getElementById("report").innerHTML = "🍎 診斷結果：<br><span style='color: #c0392b;'>偵測到晚疫病 (Late Blight)</span><br><small>建議：請即刻移除受感染葉片並噴灑銅劑。</small>";
    }, 1000);
}