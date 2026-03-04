# Environmental AI Healthcare System

## Our Website Demo

<p align="center">
  <a href="https://drive.google.com/file/d/1VmDExKHBxS2pdVi3bIv3-bacg2WeTGnq/view?usp=drive_link" target="_blank">
    <img alt="Watch Demo Video" src="https://img.shields.io/badge/%E2%96%B6-Watch%20Demo%20Video‑ff69b4?style=for-the-badge&logo=github&logoColor=white">
  </a>
</p>

🎬 Click the button above to watch the video!
## Project Overview
The **Environmental AI Healthcare System** is an AI-powered platform that analyzes environmental images and generates context-aware public health advisories. It combines computer vision and retrieval-augmented generation (RAG) to classify environmental conditions and provide actionable healthcare guidance through a web interface.

The system is designed for local deployment and integrates:
- EfficientNet-based CNN for image classification
- FAISS + SentenceTransformers for semantic document retrieval
- LLaMA 3.2 (via Ollama) for advisory generation
- Flask backend APIs and a Vanilla JavaScript frontend
- ReportLab for downloadable PDF report generation

## Features
- Environmental image upload and processing
- CNN-based classification of:
  - Air pollution
  - Garbage/dirty area
  - Stagnant water
  - Hygienic environment
- RAG pipeline for grounded health recommendations
- AI-generated advisory sections:
  - Possible diseases
  - Preventive measures
  - Health guidelines
- Browser-based results dashboard
- PDF report export including image and advisory summary

## System Architecture
1. **Input Layer**: User uploads an environmental image.
2. **Vision Inference Layer**: EfficientNet model predicts environment category.
3. **Retrieval Layer (RAG)**: FAISS retrieves relevant public health documents using SentenceTransformers embeddings.
4. **Generation Layer**: LLaMA 3.2 (Ollama local runtime) generates structured health advisory content.
5. **Application Layer**: Flask serves APIs and frontend pages.
6. **Reporting Layer**: ReportLab generates downloadable PDF reports.

## Folder Structure
```text
Environmental_AI_Healthcare_System/
|-- backend/
|   |-- app.py
|   |-- model_utils.py
|   `-- rag_utils.py
|-- frontend/
|   |-- static/
|   |   |-- script.js
|   |   `-- style.css
|   `-- templates/
|       |-- home.html
|       |-- report.html
|       `-- about.html
|-- model/
|   `-- efficientnet_model.keras
|-- rag/
|   |-- documents.pkl
|   `-- faiss.index
|-- uploads/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Installation
### Prerequisites
- Python 3.10+
- Ollama installed locally
- LLaMA 3.2 model pulled in Ollama

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/logitechsoumili/Environmental_AI_Healthcare_System.git
   cd Environmental_AI_Healthcare_System
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Ollama is running and LLaMA 3.2 is available:
   ```bash
   ollama pull llama3.2:1b
   ollama run llama3.2:1b
   ```

## Run Locally
1. Start the Flask backend:
   ```bash
   cd backend
   python app.py
   ```
2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
## Future Improvements
- Add geolocation-aware public health alerts
- Integrate real-time environmental sensor feeds (AQI, humidity, water quality)
- Add multilingual advisory generation
- Improve explainability for CNN predictions (Grad-CAM/attention maps)
- Introduce user authentication and report history dashboard
- Extend deployment to cloud and edge environments

## Academic Relevance
This project demonstrates a multidisciplinary AI workflow spanning computer vision, information retrieval, and large language model inference for healthcare-oriented environmental decision support.
