# nlp_pipeline/rag.py
import ollama
from .config import LLM_MODEL

def load_llm():
    return LLM_MODEL

def answer_question(model_name, retrieved_chunks, question: str):
    context = "\n\n".join(retrieved_chunks)
    prompt = (
        "You are an educational assistant. Answer the question using only the context provided.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def refine_summary_with_ollama(summary: str):
    prompt = (
        "You are a subject expert. Refine and expand the following summary so it helps students revise for exams. "
        "Add key definitions, simple explanations, and conceptual clarity without adding unrelated content.\n\n"
        f"Original Summary:\n{summary}\n\nRefined Educational Summary:"
    )

    try:
        response = ollama.chat(model="mistral:instruct", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Refinement failed: {e}]"
