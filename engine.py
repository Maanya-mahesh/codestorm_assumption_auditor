import json
import re
import fitz
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL,
    HF_EMBED_MODEL, CHROMA_PATH, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS
)
from prompts import (
    CONCLUSION_EXTRACTION_PROMPT,
    EXPLICIT_ASSUMPTION_PROMPT,
    IMPLICIT_ASSUMPTION_PROMPT,
    CRITICALITY_PROMPT,
    LAYMAN_PROMPT
)


def get_llm():
    if LLM_PROVIDER == "groq":
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(
            model=GROQ_MODEL,
            api_base="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            is_chat_model=True,
            context_window=32000,
            max_tokens=4096,
        )
    elif LLM_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama #noqa
        return Ollama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=300.0,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def get_embed_model():
    return HuggingFaceEmbedding(model_name=HF_EMBED_MODEL)


def init_settings():
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


class Assumption(BaseModel):
    assumption: str
    category: str
    quote: Optional[str] = ""
    evidence: Optional[str] = ""
    detection_reasoning: Optional[str] = ""
    explicit: bool
    criticality: Optional[str] = None
    criticality_score: Optional[int] = None
    criticality_reasoning: Optional[str] = None
    real_world_bridge: Optional[str] = None
    layman_explanation: Optional[str] = None


def load_pdf(pdf_path: str) -> List[Document]:
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if len(text) > 100:
                docs.append(Document(
                    text=text,
                    metadata={
                        "source": Path(pdf_path).name,
                        "page": page_num + 1
                    }
                ))
    return docs


def build_index(pdf_path: str) -> VectorStoreIndex:
    init_settings()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = load_pdf(pdf_path)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    return index


def load_existing_index() -> VectorStoreIndex:
    init_settings()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def parse_json_safe(raw: str):
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def llm_call(prompt: str) -> str:
    return Settings.llm.complete(prompt).text.strip()


def extract_conclusion(index: VectorStoreIndex) -> str:
    query_engine = index.as_query_engine(similarity_top_k=6)
    result = query_engine.query(
        "What is the main conclusion and contribution of this paper?"
    )
    full_text = " ".join([n.get_content()[:800] for n in result.source_nodes])
    raw = llm_call(CONCLUSION_EXTRACTION_PROMPT.format(text=full_text))
    try:
        parsed = parse_json_safe(raw)
        conclusion = parsed.get("main_conclusion", "")
        secondary = parsed.get("secondary_conclusion", "")
        return f"{conclusion} {secondary}".strip()
    except Exception:
        return str(result)


def extract_assumptions_from_chunks(
    index: VectorStoreIndex,
    conclusion: str
) -> List[Assumption]:
    query_engine = index.as_query_engine(similarity_top_k=TOP_K_CHUNKS)
    result = query_engine.query(
        "assumptions, methods, experimental setup, mathematical formulation"
    )
    chunks = [n.get_content() for n in result.source_nodes]
    all_assumptions = []

    for chunk in chunks:
        try:
            raw = llm_call(EXPLICIT_ASSUMPTION_PROMPT.format(text=chunk))
            for item in parse_json_safe(raw):
                item["explicit"] = True
                all_assumptions.append(Assumption(**item))
        except Exception:
            pass

        try:
            raw = llm_call(IMPLICIT_ASSUMPTION_PROMPT.format(text=chunk))
            for item in parse_json_safe(raw):
                item["explicit"] = False
                all_assumptions.append(Assumption(**item))
        except Exception:
            pass

    seen, deduped = [], []
    for a in all_assumptions:
        key = a.assumption[:60].lower()
        if key not in seen:
            seen.append(key)
            deduped.append(a)
    return deduped


def score_assumptions(
    assumptions: List[Assumption],
    conclusion: str
) -> List[Assumption]:
    scored = []
    for a in assumptions:
        try:
            raw = llm_call(CRITICALITY_PROMPT.format(
                assumption=a.assumption,
                category=a.category,
                conclusion=conclusion
            ))
            parsed = parse_json_safe(raw)
            a.criticality           = parsed.get("criticality", "weaken")
            a.criticality_score     = parsed.get("criticality_score", 2)
            a.criticality_reasoning = parsed.get("criticality_reasoning", "")
            a.real_world_bridge     = parsed.get("real_world_bridge", "")
        except Exception:
            a.criticality = "weaken"
            a.criticality_score = 2
        scored.append(a)
    scored.sort(key=lambda x: x.criticality_score or 0, reverse=True)
    return scored


def add_layman_explanations(assumptions: List[Assumption]) -> List[Assumption]:
    for a in assumptions:
        try:
            a.layman_explanation = llm_call(LAYMAN_PROMPT.format(
                assumption=a.assumption,
                category=a.category
            ))
        except Exception:
            a.layman_explanation = a.assumption
    return assumptions


def run_audit(index: VectorStoreIndex, progress_callback=None) -> dict:
    def update(msg):
        if progress_callback:
            progress_callback(msg)

    update("Extracting paper conclusion...")
    conclusion = extract_conclusion(index)

    update("Scanning for explicit and implicit assumptions...")
    assumptions = extract_assumptions_from_chunks(index, conclusion)

    update(f"Scoring criticality for {len(assumptions)} assumptions...")
    assumptions = score_assumptions(assumptions, conclusion)

    update("Generating plain-English explanations...")
    assumptions = add_layman_explanations(assumptions)

    return {
        "conclusion": conclusion,
        "assumptions": assumptions,
        "total": len(assumptions),
        "explicit_count":  sum(1 for a in assumptions if a.explicit),
        "implicit_count":  sum(1 for a in assumptions if not a.explicit),
        "collapse_count":  sum(1 for a in assumptions if a.criticality == "collapse"),
        "weaken_count":    sum(1 for a in assumptions if a.criticality == "weaken"),
        "survive_count":   sum(1 for a in assumptions if a.criticality == "survive"),
    }