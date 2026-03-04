import json
import os
import pickle
import re
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "..", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
DOCS_PATH = os.path.join(RAG_DIR, "documents.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
TOP_K = 3
SEARCH_CANDIDATES = 10
MAX_CONTEXT_CHARS = 450

RETRIEVAL_CACHE_SIZE = 300
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RAG_RETRIEVAL_CACHE_TTL_SECONDS", "1200"))
ANSWER_CACHE_SIZE = 200
ANSWER_CACHE_TTL_SECONDS = int(os.getenv("RAG_ANSWER_CACHE_TTL_SECONDS", "1800"))


_faiss_index = None
_documents = []
_embedder = None
_retrieval_cache: "OrderedDict[str, Dict]" = OrderedDict()
_answer_cache: "OrderedDict[str, Dict]" = OrderedDict()


CLASS_QUERY_HINTS = {
    "stagnant_water": "stagnant water, mosquito breeding, dengue, malaria, water-borne disease prevention",
    "garbage_dirty": "solid waste, garbage exposure, sanitation, vector control, diarrhea prevention",
    "air_pollution": "air quality, respiratory disease, asthma prevention, particulate matter, public health guidelines",
    "hygienic_environment": "healthy environment maintenance, sanitation best practices, preventive public health",
}

CLASS_DISEASE_PRIORITIES = {
    "stagnant_water": ["Dengue fever", "Malaria", "Typhoid fever"],
    "air_pollution": ["Asthma exacerbation", "Chronic bronchitis", "COPD exacerbation"],
    "garbage_dirty": ["Acute diarrheal disease", "Typhoid fever", "Leptospirosis"],
    "hygienic_environment": ["No disease risk detected"],
}


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_document(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ["text", "content", "page_content", "body", "chunk"]:
            if key in doc and _safe_text(doc[key]):
                return _safe_text(doc[key])
        return _safe_text(doc)
    return _safe_text(doc)


def _cache_get(cache: OrderedDict, key: str, ttl_seconds: int):
    item = cache.get(key)
    if not item:
        return None
    if time.time() - item.get("ts", 0) > ttl_seconds:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return item.get("value")


def _cache_set(cache: OrderedDict, key: str, value, max_size: int):
    cache[key] = {"value": value, "ts": time.time()}
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _safe_text(text).lower())


def _keyword_score(chunk: str, env_class: str, question: str) -> float:
    chunk_tokens = set(_tokenize(chunk))
    if not chunk_tokens:
        return 0.0

    env_tokens = set(_tokenize(env_class.replace("_", " ")))
    hint_tokens = set(_tokenize(CLASS_QUERY_HINTS.get(env_class, "")))
    question_tokens = set(_tokenize(question))
    query_tokens = env_tokens | hint_tokens | question_tokens

    if not query_tokens:
        return 0.0

    overlap = len(chunk_tokens.intersection(query_tokens))
    # Weighted slightly toward explicit question terms for follow-up precision.
    question_overlap = len(chunk_tokens.intersection(question_tokens)) if question_tokens else 0
    return (overlap / len(query_tokens)) + (0.4 * question_overlap / max(len(question_tokens), 1))


def _prepare_context(chunks: List[str], top_k: int = TOP_K) -> str:
    selected = []
    for chunk in chunks[:top_k]:
        compact = " ".join(_safe_text(chunk).split())
        if compact:
            selected.append(compact[:MAX_CONTEXT_CHARS])
    return "\n\n".join(selected)


def load_rag_resources() -> None:
    global _faiss_index, _documents, _embedder

    if _documents and _faiss_index is not None and _embedder is not None:
        return

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"RAG documents not found: {DOCS_PATH}")
    with open(DOCS_PATH, "rb") as file:
        raw_docs = pickle.load(file)
    _documents = [_normalize_document(doc) for doc in raw_docs]

    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu.")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    _faiss_index = faiss.read_index(INDEX_PATH)

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)


def _build_query(environment_class: str, question: str = "") -> str:
    hint = CLASS_QUERY_HINTS.get(environment_class, environment_class.replace("_", " "))
    return f"{environment_class} environmental health risks and prevention. {hint}. {question}".strip()


def _retrieve_context(environment_class: str, question: str = "", top_k: int = TOP_K) -> List[str]:
    load_rag_resources()

    question = _safe_text(question)
    cache_key = f"{environment_class}|{question}|{top_k}"
    cached = _cache_get(_retrieval_cache, cache_key, RETRIEVAL_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    query = _build_query(environment_class, question)
    try:
        embedding = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        embedding = _embedder.encode([query], convert_to_numpy=True)

    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)

    candidate_k = min(max(top_k * 3, SEARCH_CANDIDATES), max(len(_documents), 1))
    _, indices = _faiss_index.search(embedding, candidate_k)

    candidates: List[Tuple[float, str]] = []
    seen = set()
    for idx in indices[0]:
        if 0 <= idx < len(_documents):
            chunk = _safe_text(_documents[idx])
            if not chunk:
                continue
            norm_key = chunk.lower()
            if norm_key in seen:
                continue
            seen.add(norm_key)
            score = _keyword_score(chunk=chunk, env_class=environment_class, question=question)
            candidates.append((score, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)
    context_chunks = [text for _, text in candidates[:top_k]]
    _cache_set(_retrieval_cache, cache_key, context_chunks, RETRIEVAL_CACHE_SIZE)
    return context_chunks


def _call_ollama(system_prompt: str, user_prompt: str, temperature: float = 0.2, num_predict: int = 220) -> str:
    if ollama is None:
        raise RuntimeError("ollama python package is not installed.")

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    return _safe_text(response.get("message", {}).get("content"))


def _extract_json_block(text: str) -> Dict:
    text = _safe_text(text)
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _fallback_advisory(environment_class: str) -> Dict[str, List[str]]:
    label = environment_class.replace("_", " ").title()
    disease_fallback = CLASS_DISEASE_PRIORITIES.get(
        environment_class,
        [f"General environment-related health risks linked to {label}"],
    )
    return {
        "diseases": disease_fallback,
        "preventive_measures": [
            "Maintain clean surroundings and remove contamination sources",
            "Use protective equipment when exposure risk is high",
            "Seek local public health guidance for current outbreaks",
        ],
        "health_guidelines": [
            "Monitor symptoms such as fever, cough, breathing issues, or skin irritation",
            "Consult a qualified healthcare professional for diagnosis",
            "Use official local health department advisories for final decisions",
        ],
        "rag_answer": "Advisory generated from available public-health guidance.",
    }


def _ensure_environment_specific_diseases(environment_class: str, diseases: List[str]) -> List[str]:
    priorities = CLASS_DISEASE_PRIORITIES.get(environment_class, [])
    cleaned: List[str] = []
    seen = set()

    for item in diseases:
        text = _safe_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    if environment_class == "hygienic_environment":
        return ["No disease risk detected"]

    if environment_class == "stagnant_water":
        must_terms = ["dengue", "malaria"]
        for term in must_terms:
            if not any(term in item.lower() for item in cleaned):
                for candidate in priorities:
                    if term in candidate.lower():
                        cleaned.insert(0, candidate)
                        break

    existing_lower = {item.lower() for item in cleaned}
    for candidate in priorities:
        if len(cleaned) >= 3:
            break
        if candidate.lower() not in existing_lower:
            cleaned.append(candidate)
            existing_lower.add(candidate.lower())

    if not cleaned:
        cleaned = priorities[:]

    return cleaned[:3]


def _fallback_followup_answer(environment_class: str, question: str, context_chunks: List[str]) -> str:
    label = environment_class.replace("_", " ")
    tip_map = {
        "stagnant_water": "Eliminate standing water weekly and use mosquito protection.",
        "air_pollution": "Limit outdoor exposure during poor AQI and use a well-fitted mask.",
        "garbage_dirty": "Dispose waste in closed bins and maintain hand hygiene.",
        "hygienic_environment": "Keep sanitation routines consistent and monitor water cleanliness.",
    }
    tip = tip_map.get(environment_class, "Follow local public-health guidance and reduce exposure risks.")
    safe_question = _safe_text(question) or f"health risk in {label}"
    q_lower = safe_question.lower()

    question_specific = "- Prioritize exposure reduction, hygiene, and timely symptom monitoring."
    if any(word in q_lower for word in ["symptom", "sign", "fever", "cough", "breath", "rash"]):
        question_specific = "- Monitor warning signs: fever, cough, breathing issues, and skin irritation."
    elif any(word in q_lower for word in ["prevent", "prevention", "avoid", "protect", "stop"]):
        question_specific = "- Focus on prevention first: remove exposure source and improve sanitation."
    elif any(word in q_lower for word in ["drink", "water", "boil", "safe water"]):
        question_specific = "- Use only safe drinking water: boil/filter water and store it hygienically."
    elif any(word in q_lower for word in ["child", "elderly", "pregnan", "asthma", "high risk"]):
        question_specific = "- Protect vulnerable groups first with lower exposure and early medical consultation."

    if any(word in q_lower for word in ["clean", "sanitize", "disinfect", "waste", "garbage", "area"]):
        answer_points = [
            "- Remove visible waste using gloves and closed collection bags.",
            "- Disinfect hard surfaces and nearby drains with approved disinfectant.",
            "- Segregate wet and dry waste; use covered bins only.",
            "- Arrange regular municipal pickup to prevent re-accumulation.",
        ]
        summary_line = f"For {label}, keep the area clean through safe waste removal, disinfection, and regular disposal."
    elif any(word in q_lower for word in ["effect", "adverse", "impact", "harm", "health risk", "danger"]):
        answer_points = [
            "- Prolonged exposure can raise risk of respiratory and gastrointestinal illness.",
            "- Contaminated surroundings increase vector-borne and skin infection risk.",
            "- Children, elderly people, and chronically ill individuals face higher risk.",
            "- Early symptom monitoring helps prevent severe complications.",
        ]
        summary_line = f"For {label}, long exposure increases health risk, especially for vulnerable people."
    elif any(word in q_lower for word in ["symptom", "sign", "fever", "cough", "breath", "rash"]):
        answer_points = [
            "- Track fever, cough, breathing discomfort, vomiting, and diarrhea.",
            "- Watch for skin rashes, eye irritation, and unusual fatigue.",
            "- Seek medical care early if symptoms persist beyond 24-48 hours.",
            "- Escalate urgently for breathing difficulty or dehydration signs.",
        ]
        summary_line = f"For {label}, monitor early warning symptoms daily and escalate care promptly when severe."
    elif any(word in q_lower for word in ["prevent", "prevention", "avoid", "protect", "stop"]):
        answer_points = [
            "- Eliminate exposure source and maintain strict sanitation routines.",
            "- Use personal protection such as masks/gloves during cleanup.",
            "- Ensure safe water, clean storage, and proper waste disposal.",
            "- Follow local public-health advisories during outbreaks.",
        ]
        summary_line = f"For {label}, prevention depends on sanitation, exposure control, and local guidance."
    else:
        answer_points = [
            "- Apply condition-specific public-health precautions consistently.",
            "- Prioritize exposure reduction, hygiene, and timely symptom monitoring.",
            "- Use local authority advisories for outbreak-specific updates.",
            f"- Practical prevention: {tip}",
        ]
        summary_line = f"For {label}, reduce exposure risk and follow practical prevention steps consistently."

    return "\n".join([
        "Question:",
        safe_question,
        "Answer:",
        *answer_points,
        "Summary:",
        summary_line,
    ])


def _normalize_followup_answer(raw_text: str, environment_class: str, question: str, context_chunks: List[str]) -> str:
    text = _safe_text(raw_text)
    if not text:
        return _fallback_followup_answer(environment_class, question, context_chunks)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets: List[str] = []
    summary_candidate = ""
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("quick summary") or lowered.startswith("action tip"):
            continue
        if lowered.startswith("question:") or lowered.startswith("answer:") or lowered.startswith("summary:"):
            continue
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            bullets.append(line[2:].strip())
            continue
        if re.match(r"^\d+[\).\s]+", line):
            cleaned = re.sub(r"^\d+[\).\s]+", "", line).strip()
            if cleaned:
                bullets.append(cleaned)
                continue
        if not summary_candidate and len(line.split()) >= 6 and not line.endswith(":"):
            summary_candidate = line

    deduped: List[str] = []
    seen = set()
    for bullet in bullets:
        if bullet.lower().startswith("for your question on"):
            continue
        key = bullet.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bullet)

    if not deduped:
        return _fallback_followup_answer(environment_class, question, context_chunks)

    deduped = deduped[:6]
    safe_question = _safe_text(question) or f"health risk in {environment_class.replace('_', ' ')}"
    if not summary_candidate:
        summary_candidate = deduped[0]
    if len(summary_candidate) > 160:
        summary_candidate = summary_candidate[:157].rstrip() + "..."

    return "\n".join([
        "Question:",
        safe_question,
        "Answer:",
        *[f"- {item}" for item in deduped],
        "Summary:",
        summary_candidate,
    ])


def generate_health_advisory(environment_class: str) -> Dict:
    try:
        context_chunks = _retrieve_context(environment_class=environment_class, question="")
    except Exception:
        return _fallback_advisory(environment_class)

    context = _prepare_context(chunks=context_chunks, top_k=TOP_K)

    if environment_class == "stagnant_water":
        task = """
    List exactly 3 diseases linked to stagnant or unsafe contaminated water.
    Must include dengue and malaria from mosquito breeding.
    Include one water-borne disease from drinking unsafe water (e.g., typhoid or cholera).
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """
    elif environment_class == "air_pollution":
        task = """
    List ONLY 3 diseases caused by air pollution.
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """
    elif environment_class == "garbage_dirty":
        task = """
    List ONLY 3 common diseases caused by garbage accumulation.
    Focus only on realistic infectious diseases.
    Do not include cancers or unrelated illnesses.
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """
    elif environment_class == "hygienic_environment":
        task = """
    Display that "No disease risk detected" in disease list.
    Then list exactly 3 preventive practices.
    Then list exactly 2 health promotion guidelines.
    Each item must be under 10 words.
    """
    else:
        task = "Explain the health impact concisely."

    system_prompt = """
    You are an environmental health expert providing evidence-based public health advice.
    Return structured information only and avoid unnecessary commentary.
    Never invent facts outside the provided context.
    """

    user_prompt = f"""
    Detected environmental condition: {environment_class}

    Retrieved context:
    {context}

    Task instructions:
    {task}

    Return ONLY valid JSON in this format:

    {{
    "diseases": ["item1", "item2", "item3"],
    "preventive_measures": ["item1", "item2", "item3"],
    "health_guidelines": ["item1", "item2"]
    }}

    Rules:
    - Do not include any extra text outside JSON.
    - Do not add explanations outside the lists.
    - Follow the exact number of items requested.
    - Each item must be concise.
    - If context is weak, still return conservative, general public-health-safe items.
    """.strip()

    try:
        raw = _call_ollama(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1, num_predict=180)
        parsed = _extract_json_block(raw)
        if not parsed:
            return _fallback_advisory(environment_class)

        diseases = parsed.get("diseases", [])
        prevention = parsed.get("preventive_measures", [])
        guidelines = parsed.get("health_guidelines", [])
        rag_answer = _safe_text(parsed.get("rag_answer"))

        if not isinstance(diseases, list):
            diseases = []
        if not isinstance(prevention, list):
            prevention = []
        if not isinstance(guidelines, list):
            guidelines = []

        diseases = _ensure_environment_specific_diseases(
            environment_class=environment_class,
            diseases=[str(x) for x in diseases],
        )

        return {
            "diseases": diseases,
            "preventive_measures": [str(x) for x in prevention][:8],
            "health_guidelines": [str(x) for x in guidelines][:8],
            "rag_answer": rag_answer or "Advisory generated from retrieved public health context.",
        }
    except Exception:
        return _fallback_advisory(environment_class)


def answer_followup_question(environment_class: str, question: str) -> str:
    question = _safe_text(question)
    if not question:
        return "Please enter a valid question."

    answer_cache_key = f"{environment_class}|{question.lower()}"
    cached_answer = _cache_get(_answer_cache, answer_cache_key, ANSWER_CACHE_TTL_SECONDS)
    if cached_answer is not None:
        return cached_answer

    try:
        context_chunks = _retrieve_context(environment_class=environment_class, question=question)
    except Exception:
        return "Follow-up advisory is unavailable right now. Please try again after enabling RAG dependencies."

    context = _prepare_context(chunks=context_chunks, top_k=TOP_K)

    system_prompt = (
        "You are an environmental health expert answering follow-up questions. "
        "Use only the provided context and stay focused on the detected condition. "
        "If context is insufficient, state uncertainty briefly instead of guessing."
    )

    user_prompt = f"""
    Detected environmental condition: {environment_class}

    User question:
    {question}

    Retrieved context:
    {context}

    Instructions:
    - Answer the exact user question first, staying specific to this detected condition.
    - Use a professional tone and provide practical, evidence-aligned guidance only.
    - Keep content relevant; do not mention unrelated environmental hazards.
    - If context is limited, say uncertainty briefly and provide safe conservative advice.
    - Use this exact structure and nothing else:
      Question:
      <repeat the user question in one line>
      Answer:
      - bullet 1
      - bullet 2
      - bullet 3
      - bullet 4 (optional)
      - bullet 5 (optional)
      Summary:
      <one-line summary of the answer>
    - Bullets must directly address the specific user question.
    - Do not include sections named Quick Summary, Action Tip, Conclusion, or anything else.
    """.strip()

    try:
        answer = _call_ollama(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1, num_predict=260)
        answer = _normalize_followup_answer(answer, environment_class, question, context_chunks)
        _cache_set(_answer_cache, answer_cache_key, answer, ANSWER_CACHE_SIZE)
        return answer
    except Exception:
        fallback = _fallback_followup_answer(environment_class, question, context_chunks)
        _cache_set(_answer_cache, answer_cache_key, fallback, ANSWER_CACHE_SIZE)
        return fallback

