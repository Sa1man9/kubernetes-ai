import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.config import settings

BATCH_SIZE=50
_GEMINI_DIM=3072
_FALLBACK_DIM=763

_active_model = None
_model_type = str| None=None

def _probe_gemini() -> GoogleGenerativeAIEmbeddings | None:
    """Probe the Gemini API to check if it's available and working."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info(f"Gemini API probe successful. model: {settings.EMBEDDING_MODEL}, dim: {_GEMINI_DIM}")
        return model
    except Exception as e:
        logfire.warning(f"Gemini API probe failed: {e}. we will use sentence-transformers as fallback.")
        return None

def _load_fallback()->SentenceTransformer | None:
    """Load the fallback embedding model (sentence-transformers)."""
    try:
        model = SentenceTransformer(settings.FALLBACK_EMBEDDING_MODEL)
        logfire.info(f"Fallback embedding model loaded: {settings.FALLBACK_EMBEDDING_MODEL}, dim: {_FALLBACK_DIM}")
        return model
    except Exception as e:
        logfire.error(f"Failed to load fallback embedding model: {e}")
        return None

def _init():
    """Initialize the embedding model, trying Gemini first, then fallback."""
    global _active_model, _model_type
    if _active_model is not None:
        return  
    gemini=_probe_gemini()
    if gemini:
        _active_model=gemini
        _model_type="gemini"
    else:
        fallback=_load_fallback()
        if fallback:
            _active_model=fallback
            _model_type="fallback"
        else:
            raise RuntimeError("No embedding model available. Both Gemini and fallback failed to load.")

def get_embedding_dim() -> int:
    """Return the dimension of the active embedding model."""
    if _active_model is None:
        _init()
    if _model_type=="gemini":
        return _GEMINI_DIM
    elif _model_type=="fallback":
        return _FALLBACK_DIM
    else:
        raise RuntimeError("Unknown embedding model type.")

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _active_model is None:
        _init()
    if _model_type=="gemini":
        for attempts in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err=str(e).lower()
                is_rate_limit=any(keyword in err for keyword in ["rate", "quota", "429","resource_exhausted"])
                if is_rate_limit and attempts < 3:
                    wait=2**attempts
                    logfire.warning(f"Gemini API rate limit hit. Retrying in {wait} seconds... (attempt {attempts+1}/4)")
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini API embedding failed: {e}")
                    raise 
        raise RuntimeError("Gemini API embedding failed after 4 attempts.")
    elif _model_type=="fallback":
        return _active_model.encode(batch).tolist()
    else:
        raise RuntimeError("Unknown embedding model type.")

def embed_query(query: str) -> list[float]:
    if _active_model is None:
        _init()
    if _model_type=="gemini":
        return _active_model.embed_query(query)
    elif _model_type=="fallback":
        return _active_model.encode([query])[0].tolist()
    else:
        raise RuntimeError("Unknown embedding model type.")

def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings:list[list[float]]=[]
    for i in range(0,len(texts),BATCH_SIZE):
        batch=texts[i:i+BATCH_SIZE]
        with logfire.span("Embed BATCH", model=_model_type, start=i,size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings