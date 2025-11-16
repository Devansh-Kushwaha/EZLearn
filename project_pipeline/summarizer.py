# # nlp_pipeline/summarizer.py
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# def load_summarizer():
#     """
#     Load a reliable summarization model for technical content.
#     BART-large-CNN is stable, non-hallucinatory, and handles academic text well.
#     """
#     model_name = "facebook/bart-large-cnn"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     return pipeline(
#         "summarization",
#         model=model,
#         tokenizer=tokenizer,
#         framework="pt",
#         device=-1  # CPU to avoid CUDA VRAM issues
#     )


# def summarize_text(summarizer, text: str):
#     """
#     Accurate chunked summarization WITHOUT hallucinations.
#     No prompts. Feed raw text directly since BART is not instruction-tuned.
#     """
#     if len(text.strip()) == 0:
#         return ""

#     chunk_size = 900  # BART sweet spot (900 chars ≈ 400–500 tokens)
#     overlap = 100

#     summaries = []
#     start = 0

#     while start < len(text):
#         end = min(start + chunk_size, len(text))
#         chunk = text[start:end]

#         # Summarize raw text — NO instructions
#         result = summarizer(
#             chunk,
#             max_length=180,
#             min_length=60,
#             do_sample=False,
#             truncation=True
#         )

#         summaries.append(result[0]["summary_text"].strip())
#         start = end - overlap

#     # Simply join summaries — DO NOT re-summarize a second time
#     return "\n".join(summaries)


# nlp_pipeline/summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )


def summarize_text(summarizer, text: str):
    """
    Summarize while automatically adjusting max_length/min_length
    based on ACTUAL tokenizer length, preventing HF warnings FOREVER.
    """
    if not text.strip():
        return ""

    tokenizer = summarizer.tokenizer

    chunk_size = 2500
    overlap = 200

    chunks = []
    start = 0
    length = len(text)

    # --- FIXED-SIZE CHUNKING ---
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        if len(chunk) > 0:
            chunks.append(chunk)

        if end == length:
            break
        start = end - overlap

    summaries = []

    for chunk in chunks:
        # Tokenize chunk EXACTLY as BART sees it
        tokenized = tokenizer(chunk, truncation=False, return_tensors=None)
        input_len = len(tokenized["input_ids"])

        # AUTO-ADJUST SUMMARY LENGTH
        # Avoid warning by keeping max_length < input_length
        if input_len < 50:
            max_len = max(15, int(input_len * 0.6))
            min_len = max(5, int(input_len * 0.3))
        else:
            max_len = min(180, int(input_len * 0.6))
            min_len = min(120, int(input_len * 0.3))

        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )

        summaries.append(result[0]["summary_text"].strip())

    return "\n".join(summaries)
