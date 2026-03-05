let currentEnvironmentClass = null;
let latestReportData = null;
let followupHistory = [];

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
  resetAnalysisUI();
  updateUploadPreview(file || null);
}

async function analyze(event) {
  if (event) event.preventDefault();
  const input = byId("imageInput");
  const analyzeBtn = byId("analyzeBtn");
  const preview = byId("preview");
  const resultsSection = byId("results");
  const followupSection = byId("followupSection");

  if (!input) return;

  const file = input.files[0];
  if (!file) {
    showToast("Please choose an image first.", "error");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  resetAnalysisUI();

  const formData = new FormData();
  formData.append("image", file);

  let localPreviewUrl = null;
  if (preview) {
    localPreviewUrl = URL.createObjectURL(file);
    preview.src = localPreviewUrl;
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
      return;
    }

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

    showToast("Analysis completed successfully.", "success");

  } catch (error) {
    console.error(error);
    showToast("Unexpected error during analysis.", "error");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Environment";

    if (localPreviewUrl) {
      setTimeout(() => URL.revokeObjectURL(localPreviewUrl), 1000);
    }
  }
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

  if (imageInput) {
    imageInput.addEventListener("change", handleImageInputChange);
    updateUploadPreview(null);
  }

  if (analyzeBtn) analyzeBtn.addEventListener("click", analyze);
  if (askBtn) askBtn.addEventListener("click", ask);
  if (downloadBtn) downloadBtn.addEventListener("click", downloadReport);
});
