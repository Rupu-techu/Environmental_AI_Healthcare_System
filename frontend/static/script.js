let currentEnvironmentClass = null;
let latestReportData = null;
let followupHistory = [];
let cameraStream = null;
let capturedImageBlob = null;
let capturedImagePreviewUrl = "";
let liveMonitorTimer = null;
let liveMonitorActive = false;
let liveMonitorBusy = false;
let lastLiveMonitorPrediction = null;

const labels = {
  stagnant_water: "Stagnant Water",
  garbage_dirty: "Garbage / Dirty Area",
  air_pollution: "Air Pollution",
  hygienic_environment: "Hygienic Environment"
};

const badgeColors = {
  stagnant_water: "blue",
  garbage_dirty: "red",
  air_pollution: "lightorange",
  hygienic_environment: "green"
};

function byId(id) {
  return document.getElementById(id);
}

function toConfidencePercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return numeric > 1 ? numeric : numeric * 100;
}

async function parseJsonOrError(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch (_error) {
      return { error: "Server returned invalid JSON." };
    }
  }

  const text = await response.text();
  const cleaned = text.replace(/\s+/g, " ").trim();
  return { error: cleaned ? cleaned.slice(0, 300) : "Server returned a non-JSON error response." };
}

function ensureToastWrap() {
  let wrap = byId("toastWrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.id = "toastWrap";
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  return wrap;
}

function showToast(message, type = "info", duration = 3000) {
  const wrap = ensureToastWrap();
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  wrap.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => {
      toast.remove();
    }, duration);
  }

  return toast;
}

function setActiveNav() {
  const page = document.body.dataset.page || "";
  const map = {
    home: "/",
    report: "/report",
    about: "/about"
  };
  const targetHref = map[page];
  if (!targetHref) return;

  document.querySelectorAll(".navlinks a").forEach((link) => {
    if (link.getAttribute("href") === targetHref) {
      link.classList.add("active");
    }
  });
}

function renderList(elementId, items) {
  const list = byId(elementId);
  if (!list) return;

  list.innerHTML = "";
  const safeItems = Array.isArray(items) ? items : [];

  if (!safeItems.length) {
    const li = document.createElement("li");
    li.textContent = "No data available.";
    list.appendChild(li);
    return;
  }

  safeItems.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderFollowupHistory() {
  const historyEl = byId("followupHistory");
  const answerBox = byId("answer");
  if (!historyEl) return;

  historyEl.innerHTML = "";

  if (!followupHistory.length) {
    historyEl.hidden = true;
    if (answerBox) {
      answerBox.hidden = false;
      answerBox.textContent = "AI answer will appear here after analysis.";
    }
    return;
  }

  followupHistory.forEach((item, index) => {
    const block = document.createElement("article");
    block.className = "qa-item";

    const q = document.createElement("p");
    const qLabel = document.createElement("strong");
    qLabel.textContent = "Question: ";
    q.appendChild(qLabel);
    q.appendChild(document.createTextNode(item.question));

    const a = document.createElement("p");
    a.className = "qa-a";
    a.appendChild(document.createTextNode(formatAnswerForHistory(item.answer, item.question)));

    block.appendChild(q);
    block.appendChild(a);
    historyEl.appendChild(block);
  });

  historyEl.hidden = false;
  if (answerBox) answerBox.hidden = true;
}

function formatAnswerForHistory(answerText, askedQuestion = "") {
  const text = (answerText || "").trim();
  if (!text) return "No answer generated.";

  const lines = text.split(/\r?\n/);
  const cleaned = [];
  const seen = new Set();
  let section = "";
  const normalizedAskedQuestion = askedQuestion.trim().toLowerCase().replace(/[?.!:\s]+$/g, "");

  for (const line of lines) {
    const trimmed = line.trim();
    const normalized = trimmed.toLowerCase();
    if (!normalized) continue;

    if (normalized === "question:") {
      section = "question";
      continue;
    }
    if (normalized === "answer:") {
      section = "answer";
      continue;
    }
    if (normalized === "summary:") {
      section = "summary";
      continue;
    }
    if (normalized.startsWith("user question:")) continue;

    if (section && section !== "answer") {
      continue;
    }

    const bulletStripped = trimmed.replace(/^[-*•]\s+/, "");
    const markdownStripped = bulletStripped.replace(/\*\*/g, "").replace(/`/g, "").trim();
    if (!markdownStripped) continue;

    const normalizedLine = markdownStripped.toLowerCase().replace(/[?.!:\s]+$/g, "");
    if (normalizedAskedQuestion && normalizedLine === normalizedAskedQuestion) {
      continue;
    }
    if (markdownStripped.toLowerCase().includes("for your question on")) {
      continue;
    }

    if (seen.has(normalizedLine)) continue;
    seen.add(normalizedLine);
    cleaned.push(`- ${markdownStripped}`);
  }

  return cleaned.join("\n").trim() || "No answer generated.";
}

function resetAnalysisUI() {
  const preview = byId("preview");
  const predictionBadge = byId("predictionBadge");
  const confidence = byId("confidence");
  const confidenceBar = byId("confidenceBar");
  const resultsSection = byId("results");
  const followupSection = byId("followupSection");
  const answer = byId("answer");
  const questionInput = byId("question");

  if (preview) {
    preview.src = "";
    preview.hidden = true;
  }
  if (predictionBadge) {
    predictionBadge.className = "badge";
    predictionBadge.textContent = "Pending";
  }
  if (confidence) confidence.textContent = "0.00%";
  if (confidenceBar) confidenceBar.style.width = "0%";

  renderList("diseases", []);
  renderList("prevention", []);
  renderList("guidelines", []);
  followupHistory = [];
  renderFollowupHistory();

  if (answer) answer.textContent = "AI answer will appear here after analysis.";
  if (answer) answer.hidden = false;
  if (questionInput) questionInput.value = "";
  if (resultsSection) resultsSection.hidden = true;
  if (followupSection) followupSection.hidden = true;
}

function updateUploadPreview(file) {
  const fileNameEl = byId("selectedFileName");

  if (fileNameEl) {
    fileNameEl.textContent = file ? `Selected file: ${file.name}` : "No file selected.";
  }
}

function handleImageInputChange() {
  const input = byId("imageInput");
  const file = input && input.files ? input.files[0] : null;
  capturedImageBlob = null;
  capturedImagePreviewUrl = "";
  stopCameraStream();
  resetAnalysisUI();
  updateUploadPreview(file || null);
}

function applyAnalysisResponse(data) {
  const preview = byId("preview");
  const resultsSection = byId("results");
  const followupSection = byId("followupSection");

  currentEnvironmentClass = data.prediction;
  latestReportData = data;
  followupHistory = [];
  renderFollowupHistory();

  if (resultsSection) resultsSection.hidden = false;
  if (followupSection) followupSection.hidden = false;

  if (preview) {
    preview.src = `/uploads/${data.image}?t=${Date.now()}`;
    preview.hidden = false;
  }

  const badge = byId("predictionBadge");
  if (badge) {
    const label = labels[data.prediction] || data.prediction;
    const color = badgeColors[data.prediction] || "blue";
    badge.className = `badge ${color}`;
    badge.textContent = label;
  }

  const confidence = byId("confidence");
  const confidenceBar = byId("confidenceBar");

  const percent = toConfidencePercent(data.confidence);
  const clampedPercent = Math.max(0, Math.min(100, percent));
  const displayPercent = clampedPercent.toFixed(2);

  if (confidence) confidence.textContent = `${displayPercent}%`;
  if (confidenceBar) confidenceBar.style.width = `${clampedPercent}%`;

  renderList("diseases", data.diseases || []);
  renderList("prevention", data.preventive_measures || []);
  renderList("guidelines", data.health_guidelines || []);

  const answer = byId("answer");
  if (answer) {
    answer.textContent =
      data.rag_answer || "Analysis complete. Ask a follow-up question below.";
  }
}

async function submitImageForAnalysis(formData, loadingLabel, previewUrl, options = {}) {
  const {
    stopCameraOnStart = true,
    showSuccessToast = true,
    preserveCurrentResults = false
  } = options;
  const analyzeBtn = byId("analyzeBtn");
  const captureBtn = byId("capture");
  const preview = byId("preview");

  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = loadingLabel;
  }
  if (captureBtn) captureBtn.disabled = true;
  if (stopCameraOnStart) stopCameraStream();

  if (!preserveCurrentResults) {
    resetAnalysisUI();
    if (preview && previewUrl) {
      preview.src = previewUrl;
      preview.hidden = false;
    }
  } else if (preview && previewUrl) {
    preview.src = previewUrl;
    preview.hidden = false;
  }

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });
    const data = await parseJsonOrError(response);

    if (!response.ok) {
      showToast(data.error || "Failed to analyze image.", "error");
      return null;
    }

    applyAnalysisResponse(data);
    if (showSuccessToast) showToast("Analysis completed successfully.", "success");
    return data;
  } catch (error) {
    console.error(error);
    showToast("Unexpected error during analysis.", "error");
    return null;
  } finally {
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Environment";
    }
    if (captureBtn && cameraStream) captureBtn.disabled = false;
  }
}

function stopCameraStream() {
  stopLiveMonitor(false);
  const video = byId("video");
  const captureBtn = byId("capture");
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if (video) {
    video.srcObject = null;
    video.hidden = true;
  }
  if (captureBtn) captureBtn.disabled = true;
  updateCameraToggleUI(false);
}

function updateCameraToggleUI(isCameraOn) {
  const startCameraBtn = byId("startCamera");
  if (!startCameraBtn) return;

  if (isCameraOn) {
    startCameraBtn.textContent = "Turn Off";
    startCameraBtn.classList.remove("btn-secondary");
    startCameraBtn.classList.add("btn-danger");
  } else {
    startCameraBtn.textContent = "Start Camera";
    startCameraBtn.classList.remove("btn-danger");
    startCameraBtn.classList.add("btn-secondary");
  }
}

function getLiveMonitorIntervalMs() {
  const intervalSelect = byId("liveInterval");
  const value = Number(intervalSelect ? intervalSelect.value : 5000);
  if (!Number.isFinite(value) || value < 1000) return 5000;
  return value;
}

function setLiveUpdateState(message, state = "") {
  const stateEl = byId("liveUpdateState");
  if (!stateEl) return;

  if (!message) {
    stateEl.textContent = "";
    stateEl.hidden = true;
    stateEl.classList.remove("updating", "success", "error");
    return;
  }

  stateEl.hidden = false;
  stateEl.textContent = message;
  stateEl.classList.remove("updating", "success", "error");
  if (state) stateEl.classList.add(state);
}

function updateLiveMonitorUI(isLiveOn) {
  const liveBtn = byId("liveMonitorBtn");
  const statusEl = byId("liveStatus");
  const intervalSelect = byId("liveInterval");
  const alertModeToggle = byId("liveAlertMode");
  if (liveBtn) {
    if (isLiveOn) {
      liveBtn.textContent = "Stop Live Monitor";
      liveBtn.classList.remove("btn-secondary");
      liveBtn.classList.add("btn-danger");
    } else {
      liveBtn.textContent = "Start Live Monitor";
      liveBtn.classList.remove("btn-danger");
      liveBtn.classList.add("btn-secondary");
    }
  }
  if (statusEl) {
    if (isLiveOn) {
      statusEl.textContent = `Live monitor is active (every ${getLiveMonitorIntervalMs() / 1000}s).`;
      statusEl.classList.add("active");
      setLiveUpdateState("Waiting for next live update...");
    } else {
      statusEl.textContent = "Live monitor is off.";
      statusEl.classList.remove("active");
      setLiveUpdateState("");
    }
  }
  if (intervalSelect) intervalSelect.disabled = isLiveOn;
  if (alertModeToggle) alertModeToggle.disabled = isLiveOn;
}

function stopLiveMonitor(showToastMessage = true) {
  const wasActive = liveMonitorActive;
  if (liveMonitorTimer) {
    clearInterval(liveMonitorTimer);
    liveMonitorTimer = null;
  }
  liveMonitorActive = false;
  liveMonitorBusy = false;
  updateLiveMonitorUI(false);
  if (showToastMessage && wasActive) {
    showToast("Live monitor stopped.", "info");
  }
}

function showLiveMonitorPredictionAlert(data) {
  if (!data || !data.prediction) return;
  const alertModeToggle = byId("liveAlertMode");
  const alertOnChangeOnly = alertModeToggle ? alertModeToggle.checked : true;
  const nextPrediction = data.prediction;
  const nextLabel = labels[nextPrediction] || nextPrediction;

  if (alertOnChangeOnly && lastLiveMonitorPrediction === nextPrediction) {
    return;
  }

  if (nextPrediction === "hygienic_environment") {
    showToast(`Live status changed: ${nextLabel}.`, "success");
  } else {
    showToast(`Live alert: ${nextLabel} detected.`, "error");
  }
  lastLiveMonitorPrediction = nextPrediction;
}

async function analyzeCurrentCameraFrame() {
  const video = byId("video");
  const canvas = byId("canvas");
  if (!video || !canvas || !cameraStream) return false;
  if (!video.videoWidth || !video.videoHeight) return false;

  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext("2d");
  if (!context) return false;
  context.drawImage(video, 0, 0, 224, 224);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return false;

  const formData = new FormData();
  formData.append("image", blob, "live-monitor.png");
  const previewUrl = canvas.toDataURL("image/png");

  const data = await submitImageForAnalysis(formData, "Monitoring...", previewUrl, {
    stopCameraOnStart: false,
    showSuccessToast: false,
    preserveCurrentResults: true
  });
  return data;
}

async function runLiveMonitorTick() {
  if (!liveMonitorActive || liveMonitorBusy) return;
  liveMonitorBusy = true;
  setLiveUpdateState("Live updating...", "updating");
  try {
    const data = await analyzeCurrentCameraFrame();
    if (data) {
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, "0");
      const mm = String(now.getMinutes()).padStart(2, "0");
      const ss = String(now.getSeconds()).padStart(2, "0");
      setLiveUpdateState(`Last updated at ${hh}:${mm}:${ss}.`, "success");
    } else {
      setLiveUpdateState("Live update failed. Retrying on next tick.", "error");
    }
    showLiveMonitorPredictionAlert(data);
  } finally {
    liveMonitorBusy = false;
  }
}

async function toggleLiveMonitor(event) {
  if (event) event.preventDefault();
  if (liveMonitorActive) {
    stopLiveMonitor();
    return;
  }

  if (!cameraStream) {
    await startCamera();
  }
  if (!cameraStream) {
    showToast("Start camera first to use live monitor.", "error");
    return;
  }

  liveMonitorActive = true;
  lastLiveMonitorPrediction = null;
  updateLiveMonitorUI(true);
  showToast("Live monitor started.", "info");

  await runLiveMonitorTick();
  if (!liveMonitorActive) return;

  const intervalMs = getLiveMonitorIntervalMs();
  liveMonitorTimer = setInterval(() => {
    runLiveMonitorTick();
  }, intervalMs);
}

async function startCamera(event) {
  if (event) event.preventDefault();
  const video = byId("video");
  const captureBtn = byId("capture");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Camera is not supported in this browser.", "error");
    return;
  }

  try {
    stopCameraStream();
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      });
    } catch (_cameraPreferenceError) {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    }
    if (!video) return;
    video.srcObject = cameraStream;
    video.hidden = false;
    if (captureBtn) captureBtn.disabled = false;
    updateCameraToggleUI(true);
    showToast("Camera started.", "info");
  } catch (_error) {
    updateCameraToggleUI(false);
    showToast("Unable to access camera. Please allow permission.", "error");
  }
}

async function toggleCamera(event) {
  if (event) event.preventDefault();

  if (cameraStream) {
    stopCameraStream();
    return;
  }

  await startCamera();
}

async function captureImage(event) {
  if (event) event.preventDefault();
  const video = byId("video");
  const canvas = byId("canvas");
  const input = byId("imageInput");
  const preview = byId("preview");

  if (!video || !canvas || !cameraStream) {
    showToast("Start camera first.", "error");
    return;
  }
  if (!video.videoWidth || !video.videoHeight) {
    showToast("Camera is not ready yet. Try again.", "error");
    return;
  }

  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext("2d");
  if (!context) {
    showToast("Failed to capture image.", "error");
    return;
  }
  context.drawImage(video, 0, 0, 224, 224);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) {
    showToast("Failed to capture image.", "error");
    return;
  }

  if (input) input.value = "";
  capturedImageBlob = blob;
  capturedImagePreviewUrl = canvas.toDataURL("image/png");
  updateUploadPreview(null);
  resetAnalysisUI();
  if (preview) {
    preview.src = capturedImagePreviewUrl;
    preview.hidden = false;
  }
  showToast("Image captured. Click Analyze Environment when ready.", "info");
}

async function analyze(event) {
  if (event) event.preventDefault();
  const input = byId("imageInput");
  const preview = byId("preview");

  if (!input) return;

  const file = input.files[0];
  let localPreviewUrl = null;
  const formData = new FormData();
  if (file) {
    if (preview) {
      localPreviewUrl = URL.createObjectURL(file);
    }
    formData.append("image", file, "image.png");
    capturedImageBlob = null;
    capturedImagePreviewUrl = "";
  } else if (capturedImageBlob) {
    localPreviewUrl = capturedImagePreviewUrl;
    formData.append("image", capturedImageBlob, "image.png");
  } else {
    showToast("Please choose an image or capture one first.", "error");
    return;
  }

  await submitImageForAnalysis(formData, "Analyzing...", localPreviewUrl);

  if (file && localPreviewUrl) {
    setTimeout(() => URL.revokeObjectURL(localPreviewUrl), 1000);
  }
  capturedImageBlob = null;
  capturedImagePreviewUrl = "";
}

  
async function ask(event) {
  if (event) event.preventDefault();
  const questionInput = byId("question");
  const answerBox = byId("answer");
  if (!questionInput || !answerBox) return;

  const question = questionInput.value.trim();
  if (!question) {
    showToast("Please enter a question.", "error");
    return;
  }

  if (!currentEnvironmentClass) {
    showToast("Analyze an image first.", "error");
    return;
  }

  byId("askBtn").disabled = true;
  byId("askBtn").textContent = "Thinking...";
  const loadingToast = showToast("Generating follow-up answer...", "info", 0);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        environment_class: currentEnvironmentClass
      })
    });

    const data = await parseJsonOrError(response);
    if (!response.ok) {
      answerBox.textContent = data.error || "Could not generate answer.";
      showToast(data.error || "Could not generate answer.", "error");
      return;
    }

    followupHistory.push({
      question,
      answer: data.answer || "No answer generated."
    });
    renderFollowupHistory();
    questionInput.value = "";
    showToast("Answer generated.", "success");
  } catch (error) {
    answerBox.hidden = false;
    answerBox.textContent = "An unexpected error occurred while fetching the answer.";
    showToast("An unexpected error occurred while fetching the answer.", "error");
  } finally {
    loadingToast.remove();
    byId("askBtn").disabled = false;
    byId("askBtn").textContent = "Ask";
  }
}

async function downloadReport(event) {
  if (event) event.preventDefault();
  if (!latestReportData) {
    showToast("Analyze an image first.", "error");
    return;
  }

  const payload = {
    prediction: labels[latestReportData.prediction] || latestReportData.prediction,
    confidence: toConfidencePercent(latestReportData.confidence).toFixed(2),
    diseases: latestReportData.diseases || [],
    preventive_measures: latestReportData.preventive_measures || [],
    health_guidelines: latestReportData.health_guidelines || [],
    image: latestReportData.image || "",
    followup_qas: followupHistory
  };

  const btn = byId("downloadBtn");
  btn.disabled = true;
  btn.textContent = "Preparing...";
  const loadingToast = showToast("Preparing PDF report...", "info", 0);

  try {
    const response = await fetch("/download_report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errText = (await response.text()).replace(/\s+/g, " ").trim();
      showToast(errText || "Failed to download report.", "error");
      return;
    }

    const blob = await response.blob();
    if (blob.size === 0) {
      showToast("Received an empty PDF file from server.", "error");
      return;
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "environmental_health_report.pdf";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Report downloaded.", "success");
  } catch (error) {
    showToast("An unexpected error occurred while downloading the report.", "error");
  } finally {
    loadingToast.remove();
    btn.disabled = false;
    btn.textContent = "Download Report";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  const analyzeBtn = byId("analyzeBtn");
  const askBtn = byId("askBtn");
  const downloadBtn = byId("downloadBtn");
  const imageInput = byId("imageInput");
  const startCameraBtn = byId("startCamera");
  const captureBtn = byId("capture");
  const liveMonitorBtn = byId("liveMonitorBtn");
  const liveInterval = byId("liveInterval");

  if (imageInput) {
    imageInput.addEventListener("change", handleImageInputChange);
    updateUploadPreview(null);
  }

  if (analyzeBtn) analyzeBtn.addEventListener("click", analyze);
  if (startCameraBtn) {
    updateCameraToggleUI(false);
    startCameraBtn.addEventListener("click", toggleCamera);
  }
  updateLiveMonitorUI(false);
  if (captureBtn) captureBtn.disabled = true;
  if (captureBtn) captureBtn.addEventListener("click", captureImage);
  if (liveMonitorBtn) liveMonitorBtn.addEventListener("click", toggleLiveMonitor);
  if (liveInterval) {
    liveInterval.addEventListener("change", () => {
      if (!liveMonitorActive) {
        updateLiveMonitorUI(false);
      }
    });
  }
  if (askBtn) askBtn.addEventListener("click", ask);
  if (downloadBtn) downloadBtn.addEventListener("click", downloadReport);
});

window.addEventListener("beforeunload", () => {
  stopLiveMonitor(false);
  stopCameraStream();
});
