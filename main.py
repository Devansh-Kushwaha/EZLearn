# nlp_pipeline/main.py
from project_pipeline.extractors import get_text_from_file
from project_pipeline.preprocess import clean_text, chunk_text
from project_pipeline.embeddings import LocalVectorStore
from project_pipeline.summarizer import load_summarizer, summarize_text
from project_pipeline.rag import load_llm, answer_question, refine_summary_with_ollama

import os
import json
from project_pipeline.extractors import get_text_from_file
from project_pipeline.preprocess import clean_text, chunk_text
from project_pipeline.embeddings import LocalVectorStore
from project_pipeline.summarizer import load_summarizer, summarize_text
from project_pipeline.rag import load_llm, answer_question, refine_summary_with_ollama


def run_pipeline(file_path: str):
    store = LocalVectorStore()
    file_hash = store._hash_file(file_path)
    summary_path = f"{file_hash}_summary.json"

    # Try loading cached vectorstore
    if store.load_index(file_hash):
        print(f"✅ Loaded existing vectorstore for {file_path}")

        # ----------------------------------------------------
        # LOAD SUMMARY IF IT EXISTS
        # ----------------------------------------------------
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            print("\n===== SUMMARY =====\n", saved["summary"])
            print("\n===== EDUCATIONAL SUMMARY =====\n", saved["refined"])
        else:
            print("⚠️ Summary file missing, generating again...")

            raw_text = get_text_from_file(file_path)
            clean = clean_text(raw_text)

            summarizer = load_summarizer()
            summary = summarize_text(summarizer, clean)
            refined = refine_summary_with_ollama(summary)

            # Save summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "refined": refined}, f, ensure_ascii=False, indent=2)

            print("\n===== SUMMARY =====\n", summary)
            print("\n===== EDUCATIONAL SUMMARY =====\n", refined)

    else:
        print(f"🔍 Building new vectorstore for {file_path}")

        raw_text = get_text_from_file(file_path)
        clean = clean_text(raw_text)
        print(f"Total text length: {len(clean)} characters")

        chunks = chunk_text(clean, max_chars=500, overlap=50)
        print(f"Extracted {len(chunks)} chunks")

        summarizer = load_summarizer()
        summary = summarize_text(summarizer, clean)
        refined = refine_summary_with_ollama(summary)

        print("\n===== SUMMARY =====\n", summary)
        print("\n===== EDUCATIONAL SUMMARY =====\n", refined)

        # Save summary file
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "refined": refined}, f, ensure_ascii=False, indent=2)

        store.build_index(chunks)
        store.save_index(file_hash)
        print("💾 Vectorstore saved for reuse")

    # LLM load once
    llm = load_llm()

    # Question loop
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()

        if question.lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting. Goodbye!")
            break

        retrieved = store.retrieve(question, top_k=3)
        answer = answer_question(llm, retrieved, question)

        print("\n===== ANSWER =====\n", answer)



if __name__ == "__main__":
    path = input("Enter PDF or PPTX path: ").strip()
    run_pipeline(path)
