"""LLM post-processor for cleaning up transcribed text."""

from mlx_lm import load, generate

# Cached model and tokenizer
_llm_model = None
_llm_tokenizer = None
_loaded_llm_repo = None

# Available LLM models (repo, display name, approximate size)
LLM_MODELS = [
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "Qwen 2.5 0.5B (4-bit)", "~350 MB"),
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "Qwen 2.5 1.5B (4-bit)", "~900 MB"),
]

DEFAULT_LLM_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# System prompt for text cleanup
CLEANUP_PROMPT = """You are a text cleanup assistant. Your task is to clean up transcribed speech by:
1. Fixing punctuation and capitalization
2. Removing filler words (um, uh, like, you know, basically, actually, literally, I mean, sort of, kind of)
3. Fixing obvious transcription errors while preserving the original meaning

Rules:
- Output ONLY the cleaned text, nothing else
- Do not add any explanations or commentary
- Preserve the speaker's intent and meaning
- Keep contractions natural (don't expand them)
- Do not change technical terms or proper nouns"""


def load_llm(model_repo: str = DEFAULT_LLM_MODEL, on_status=None):
    """Load the LLM model."""
    global _llm_model, _llm_tokenizer, _loaded_llm_repo

    if _loaded_llm_repo == model_repo and _llm_model is not None:
        return True

    try:
        if on_status:
            on_status("Loading LLM...", "⏳")

        _llm_model, _llm_tokenizer = load(model_repo)
        _loaded_llm_repo = model_repo

        if on_status:
            on_status("LLM ready", "🎤")

        print(f"LLM loaded: {model_repo}")
        return True
    except Exception as e:
        print(f"Error loading LLM: {e}")
        if on_status:
            on_status("LLM load failed", "❌")
        return False


def process_text(text: str, model_repo: str = DEFAULT_LLM_MODEL) -> str:
    """Process transcribed text through the LLM to clean it up."""
    global _llm_model, _llm_tokenizer, _loaded_llm_repo

    if not text or not text.strip():
        return text

    # Load model if needed
    if _loaded_llm_repo != model_repo or _llm_model is None:
        if not load_llm(model_repo):
            return text  # Return original if loading fails

    try:
        # Build the prompt
        messages = [
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": text}
        ]

        prompt = _llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate with constrained output length
        response = generate(
            _llm_model,
            _llm_tokenizer,
            prompt=prompt,
            max_tokens=len(text) * 2,  # Allow some expansion for punctuation
            verbose=False
        )

        cleaned = response.strip()
        if cleaned:
            return cleaned
        return text

    except Exception as e:
        print(f"LLM processing error: {e}")
        return text  # Return original on error
