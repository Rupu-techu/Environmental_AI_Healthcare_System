from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import html
import hashlib
import io
import logging
import os
import sys
import time
import uuid
import warnings
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_utils import predict_environment, warmup_model
from rag_utils import generate_health_advisory, answer_followup_question, load_rag_resources
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Suppress unnecessary logs/warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

API_ENDPOINTS = {"/analyze", "/ask", "/download_report"}


@app.errorhandler(Exception)
def handle_api_exceptions(exc):
    # Keep regular page routes as HTML, but force JSON for API routes.
    if request.path not in API_ENDPOINTS:
        if isinstance(exc, HTTPException):
            return exc
        raise exc

    if isinstance(exc, HTTPException):
        app.logger.warning("API HTTP error on %s: %s", request.path, exc)
        message = exc.description or "Request failed."
        return jsonify({"error": message}), exc.code

    app.logger.exception("Unhandled API error on %s", request.path)
    return jsonify({"error": "Internal server error."}), 500

UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["LAST_ENV_CLASS"] = None

# In-memory analysis cache keyed by image SHA-256.
ANALYSIS_CACHE = {}
CACHE_TTL_SECONDS = int(os.environ.get("ANALYSIS_CACHE_TTL_SECONDS", "1800"))


# ------------------------------
# Utility helpers
# ------------------------------
def cleanup_old_uploads(folder, max_age_seconds=7200):
    now = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            if now - os.path.getmtime(file_path) > max_age_seconds:
                try:
                    os.remove(file_path)
                except OSError as exc:
                    app.logger.warning("Failed to remove old upload %s: %s", file_path, exc)


def cleanup_analysis_cache(max_age_seconds=CACHE_TTL_SECONDS):
    now = time.time()
    stale_keys = [
        key for key, value in ANALYSIS_CACHE.items()
        if now - value.get("timestamp", 0) > max_age_seconds
    ]
    for key in stale_keys:
        ANALYSIS_CACHE.pop(key, None)


def file_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def warmup_services():
    # Warmup removes first-request latency spikes for model and RAG.
    try:
        warmup_model()
        app.logger.info("Model warmup complete.")
    except Exception as exc:
        app.logger.warning("Model warmup skipped/failed: %s", exc)
    try:
        load_rag_resources()
        app.logger.info("RAG warmup complete.")
    except Exception as exc:
        app.logger.warning("RAG warmup skipped/failed: %s", exc)


# ------------------------------
# Routes: Pages
# ------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/report")
def report_page():
    return render_template("report.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


# ------------------------------
# Route: Analyze image
# ------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    req_start = time.perf_counter()
    cleanup_old_uploads(app.config["UPLOAD_FOLDER"])
    cleanup_analysis_cache()

    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded image first so it can be served and hashed.
    safe_name = secure_filename(image_file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid file name"}), 400
    filename = f"{uuid.uuid4()}_{safe_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)
    upload_ms = (time.perf_counter() - req_start) * 1000

    digest = file_sha256(filepath)
    cached = ANALYSIS_CACHE.get(digest)
    if cached and (time.time() - cached["timestamp"] <= CACHE_TTL_SECONDS):
        app.config["LAST_ENV_CLASS"] = cached["prediction"]
        total_ms = (time.perf_counter() - req_start) * 1000
        app.logger.info(
            "analyze cache_hit upload_ms=%.1f total_ms=%.1f image_hash=%s",
            upload_ms,
            total_ms,
            digest[:12]
        )
        return jsonify({
            "prediction": cached["prediction"],
            "confidence": cached["confidence"],
            "diseases": cached["diseases"],
            "preventive_measures": cached["preventive_measures"],
            "health_guidelines": cached["health_guidelines"],
            "image": filename,
            "rag_answer": cached.get("rag_answer", "Analysis loaded from cache.")
        })

    # 1) Predict environment
    t_predict_start = time.perf_counter()
    label, confidence = predict_environment(filepath)
    predict_ms = (time.perf_counter() - t_predict_start) * 1000

    # 2) Generate RAG info
    t_rag_start = time.perf_counter()
    advisory = generate_health_advisory(label)
    rag_ms = (time.perf_counter() - t_rag_start) * 1000

    # Handle both dict (current) and tuple (legacy) advisory responses.
    if isinstance(advisory, dict):
        diseases = advisory.get("diseases", []) or []
        preventive_measures = advisory.get("preventive_measures", []) or []
        health_guidelines = advisory.get("health_guidelines", []) or []
        rag_answer = advisory.get("rag_answer", "Advisory generated.")
    else:
        diseases, preventive_measures, health_guidelines = advisory
        rag_answer = "Advisory generated."

    # Store last detected class for follow-up questions
    app.config["LAST_ENV_CLASS"] = label

    ANALYSIS_CACHE[digest] = {
        "timestamp": time.time(),
        "prediction": label,
        "confidence": confidence,
        "diseases": diseases,
        "preventive_measures": preventive_measures,
        "health_guidelines": health_guidelines,
        "rag_answer": rag_answer,
    }

    total_ms = (time.perf_counter() - req_start) * 1000
    app.logger.info(
        "analyze cache_miss upload_ms=%.1f predict_ms=%.1f rag_ms=%.1f total_ms=%.1f image_hash=%s",
        upload_ms,
        predict_ms,
        rag_ms,
        total_ms,
        digest[:12]
    )

    # 3) Return JSON
    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "diseases": diseases,
        "preventive_measures": preventive_measures,
        "health_guidelines": health_guidelines,
        "image": filename,
        "rag_answer": rag_answer
    })


# ------------------------------
# Route: Follow-up question
# ------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    environment_class = app.config.get("LAST_ENV_CLASS", "")

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not environment_class:
        return jsonify({"error": "Analyze an image first to establish context."}), 400

    answer = answer_followup_question(environment_class, question)
    return jsonify({"answer": answer})


# ------------------------------
# Route: Serve uploaded files
# ------------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ------------------------------
# Route: Download PDF report
# ------------------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.get_json(silent=True) or {}
    image_filename = os.path.basename((data.get("image") or "").strip())
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename) if image_filename else None

    pdf_buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    section_style = styles["Heading2"]
    text_style = styles["BodyText"]
    elements = []

    # Title and header
    elements.append(Paragraph("Environmental AI Healthcare System", title_style))
    elements.append(Paragraph("Environmental Health Assessment Report", section_style))
    elements.append(Spacer(1, 20))

    # Image
    if image_path and os.path.exists(image_path):
        try:
            image_reader = ImageReader(image_path)
            img_width, img_height = image_reader.getSize()
            max_width, max_height = 6.2 * inch, 3.4 * inch
            scale = min(max_width / img_width, max_height / img_height, 1)
            elements.append(RLImage(image_path, width=img_width * scale, height=img_height * scale))
            elements.append(Spacer(1, 14))
        except Exception:
            elements.append(Paragraph("Uploaded image could not be embedded.", styles["Normal"]))
            elements.append(Spacer(1, 10))

    # Data sections
    prediction = data.get("prediction", "N/A")
    confidence = data.get("confidence", "N/A")
    diseases = data.get("diseases", []) or []
    preventive_measures = data.get("preventive_measures", []) or []
    health_guidelines = data.get("health_guidelines", []) or []
    followup_qas = data.get("followup_qas", []) or []

    elements.append(Paragraph(f"<b>Detected Environment:</b> {html.escape(str(prediction))}", text_style))
    elements.append(Paragraph(f"<b>Model Confidence:</b> {html.escape(str(confidence))}%", text_style))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Health Risks", section_style))
    for item in diseases:
        elements.append(Paragraph(f"- {html.escape(str(item))}", text_style))

    elements.append(Paragraph("Preventive Measures", section_style))
    for item in preventive_measures:
        elements.append(Paragraph(f"- {html.escape(str(item))}", text_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Health Guidelines", section_style))
    for item in health_guidelines:
        elements.append(Paragraph(f"- {html.escape(str(item))}", text_style))

    if isinstance(followup_qas, list) and followup_qas:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Follow-up Questions and RAG Answers", section_style))
        for idx, qa in enumerate(followup_qas, start=1):
            if not isinstance(qa, dict):
                continue
            question = html.escape(str((qa.get("question") or "").strip()))
            answer = str((qa.get("answer") or "").strip())
            if not question and not answer:
                continue

            if question:
                elements.append(Paragraph(f"<b>Q{idx}:</b> {question}", text_style))

            if answer:
                lines = [line.strip() for line in answer.splitlines() if line.strip()]
                if not lines:
                    lines = [answer.strip()]
                elements.append(Paragraph(f"<b>A{idx}:</b>", text_style))
                for line in lines:
                    safe_line = html.escape(line)
                    if safe_line.startswith("- ") or safe_line.startswith("* ") or safe_line.startswith("• "):
                        safe_line = safe_line[2:].strip()
                        elements.append(Paragraph(f"- {safe_line}", text_style))
                    else:
                        elements.append(Paragraph(safe_line, text_style))
                elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 25))
    elements.append(Paragraph(
        "This report was automatically generated by the Environmental AI Healthcare System "
        "using CNN-based environmental detection and Retrieval-Augmented Generation.",
        styles["Italic"]
    ))

    # Build PDF
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    doc.build(elements)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="environmental_health_report.pdf"
    )


# Startup warmup for smoother first request.
warmup_services()


# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
