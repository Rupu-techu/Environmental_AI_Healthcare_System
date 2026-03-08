# Environmental AI Healthcare System

## Project Overview
The **Environmental AI Healthcare System** is a Flask-based AI application that analyzes environmental scene images and generates health-risk guidance for users. It addresses a real problem in preventive healthcare: people often observe unsafe environmental conditions (such as stagnant water or polluted areas) but lack immediate, contextual health advice.

The system combines:
- **Computer Vision** using an EfficientNet CNN to classify environmental conditions from uploaded or captured images
- **Retrieval-Augmented Generation (RAG)** using FAISS + SentenceTransformers + LLaMA 3.2 (via Ollama) to generate grounded, context-aware advisories

This architecture delivers both a predicted environmental class and practical healthcare recommendations in an interactive web interface, with downloadable PDF reporting.

## Features

### Home Page
<img width="1919" height="865" alt="home_page" src="https://github.com/user-attachments/assets/68ee9d36-882e-4557-b146-167f982faafc" />

The home page introduces the Environmental AI Healthcare System and provides navigation to the report generation and system information pages.

---

### Report Page
<img width="1919" height="867" alt="report" src="https://github.com/user-attachments/assets/2b3db2b9-f162-47b0-a31b-f661a2d0ff33" />

The report page is the main analysis interface where users can upload or capture environmental images and receive AI-generated health advisories.

Key capabilities:

<img width="1910" height="868" alt="Screenshot 2026-03-08 193050" src="https://github.com/user-attachments/assets/bfcc6889-8002-4eac-92f2-1ac82ff016da" />


- Environmental image upload and processing
- ### Image Upload Analysis
- Upload environmental images
- AI-based risk detection

### Live Monitoring and Image Capture
- Start / Stop live capture
- Interval control (default 5s)
- Alert only when class changes
- Live status updates
- Persistent disease/prevention information

- CNN-based classification of:
  - Air pollution
  - Garbage / dirty areas
  - Stagnant water
  - Hygienic environments
- Retrieval-Augmented Generation (RAG) for health recommendations
- AI-generated advisory sections:
  - Possible diseases
  - Preventive measures
  - Health guidelines
- Interactive results dashboard
- PDF report export including image and advisory summary

<img width="1916" height="872" alt="home" src="https://github.com/user-attachments/assets/213891d9-1580-482f-a40a-babd971c3795" />

---

### About Page
<img width="1919" height="869" alt="about" src="https://github.com/user-attachments/assets/28e897a4-dd45-4a52-b492-096688ce9950" />

The about page provides an overview of the project, its motivation, system architecture, and the technologies used to build the platform.


## System Architecture
The platform follows a pipeline architecture:

1. **Input Layer**: Image upload or camera capture from the report interface
2. **Vision Layer**: EfficientNet CNN performs environmental class prediction
3. **Retrieval Layer**: FAISS searches relevant health knowledge using SentenceTransformers embeddings
4. **Generation Layer**: LLaMA 3.2 (via Ollama) generates structured, context-grounded health advisories
5. **Application Layer**: Flask APIs serve analysis, follow-up Q&A, and frontend interactions
6. **Reporting Layer**: ReportLab compiles predictions and advisories into downloadable PDF reports

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
|-- notebooks/
|    |-- 01_image_data_preprocessing.ipynb
|    |-- 02_transfer_learning.ipynb
|    `-- 03_rag_implementation.ipynb
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
1. Clone the repository and enter the project directory:
```bash
git clone https://github.com/logitechsoumili/Environmental_AI_Healthcare_System.git
cd Environmental_AI_Healthcare_System
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
```bash
# macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Pull the Ollama model (first-time setup):
```bash
ollama pull llama3.2:1b
```

## Run Locally
1. Ensure Ollama is running locally (separate terminal):
```bash
ollama run llama3.2:1b
```

2. Start the Flask application:
```bash
cd backend
python app.py
```

3. Open your browser:
```text
http://127.0.0.1:5000
```

## Usage
1. Upload an environmental image or capture one using the device camera.
2. Click **Analyze Environment**.
3. View the predicted environmental class and confidence score.
4. Review the AI-generated advisory (possible diseases, preventive measures, and health guidelines).
5. Download the generated PDF report.

## Future Work
- Real-time environmental monitoring with sensor and streaming integrations
- Waste type classification for more granular sanitation risk detection
- Air pollution subtype detection (e.g., smoke, dust, industrial emissions)
- Geolocation-based public health alerts
- Multilingual health advisories for broader accessibility
- Grad-CAM explainability overlays for CNN prediction transparency

## Academic Relevance
This project demonstrates a multidisciplinary AI workflow spanning computer vision, information retrieval, and large language model inference for healthcare-oriented environmental decision support.
