# EZLearn – Summarization + RAG Pipeline 🚀

EZLearn is an end-to-end **Retrieval-Augmented Generation (RAG)** and **Summarization pipeline** designed to process PDFs and PPTX files, extract knowledge efficiently, and generate high‑quality summaries or answers to user queries. It automates text extraction, chunking, vector embeddings, summarization, and interactive Q&A using LLMs.

---

## 🔍 Features

- 📄 Extracts text from **PDF** and **PPTX** files  
- ✨ Cleans and preprocesses raw text  
- ✂️ Chunks content for optimal retrieval  
- 🧠 Creates and stores **local vector embeddings**  
- 🗂️ Saves vectorstores for fast reuse  
- 📝 Generates summaries using a local or remote LLM  
- 🎓 Creates **educational summaries** using Ollama refinement  
- 🔎 Allows interactive Q&A through a RAG loop  
- ⚡ Caches results to avoid reprocessing  

---

## 🧱 Pipeline Architecture

1. **Extraction** – Convert PDF/PPTX files into raw text  
2. **Preprocessing** – Clean, normalize, and chunk the text  
3. **Embedding** – Generate embeddings using SentenceTransformers  
4. **Vectorstore** – Save/reload embeddings for efficient retrieval  
5. **Summarization** – Generate a concise summary of the entire document  
6. **Ollama Refinement** – Produce an educational-style summary  
7. **RAG Q&A** – Retrieve relevant chunks + answer user questions  
8. **Interactive Loop** – Chat with your document indefinitely  

---

## 📁 Project Structure

```
nlp_pipeline/
│── main.py
│── config.py
│── project_pipeline/
│     ├── extractors.py
│     ├── preprocess.py
│     ├── embeddings.py
│     ├── summarizer.py
│     ├── rag.py
```

---

## ▶️ How the Pipeline Works

1. User enters file path  
2. Pipeline checks if a cached vectorstore exists  
3. If yes → loads vectorstore + summary  
4. If no → processes file, builds embeddings, creates summaries  
5. Loads the LLM once  
6. Enters an infinite question‑answer loop using RAG

---

## 🛠️ Tech Stack

- Python  
- PyPDF / python-pptx  
- SentenceTransformers  
- Ollama (Mistral model)  
- Local vector store (FAISS-like)  
- Torch  
- JSON for caching summaries  

---

## 🔧 Configuration (config.py)

- **DEVICE** – CPU/GPU auto-detection  
- **EMBEDDING_MODEL** – MiniLM-L6-v2 (fast + accurate)  
- **LLM_MODEL** – Local Mistral via Ollama  
- **CHUNK_SIZE** – Characters per chunk  
- **CHUNK_OVERLAP** – Overlap between chunks  
- **TOP_K** – Retrieval depth  

---

## 🚀 Running the Pipeline

```
python main.py
Enter PDF or PPTX path: myfile.pdf
```

Then ask:

```
Ask a question (or type 'exit' to quit):
```

Example:

```
What is the main idea of chapter 2?
```

---

## 🧠 Example Output

### **Summary**
A concise explanation of the entire document.

### **Educational Summary**
A simplified explanation refined using a secondary LLM (Ollama Mistral).

### **RAG Answer**
Retrieves 3 most relevant chunks → produces contextual answer.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a PR or create an issue.

---

## 📜 License

MIT License  

---

Enjoy using **EZLearn** to turn heavy documents into clean summaries and interactive knowledge! 🚀
