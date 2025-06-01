# 🧠 Retrieval-Augmented Generation (RAG) Pipeline on Open-Source Nutrition Textbook

This repository implements a complete **Retrieval-Augmented Generation (RAG)** system using a publicly available *Human Nutrition* textbook. The pipeline extracts knowledge from the textbook, performs semantic retrieval using embeddings, and generates context-aware answers using a local LLM.

---

## 📌 Key Features

- ✅ Extract and clean text from a structured PDF textbook.
- ✅ Chunk and embed content using `SentenceTransformers`.
- ✅ Retrieve relevant information via semantic similarity search.
- ✅ Augment queries with contextual knowledge before passing to an LLM.
- ✅ Generate informative, grounded responses using a local Gemma 2B model.

---

## 📂 Project Structure

### 🔍 1. Retrieval

This component builds a semantic search engine over the textbook content:

- **Embedding Generation**
  - Uses [`sentence-transformers`](https://www.sbert.net/) (`all-mpnet-base-v2`) to generate vector embeddings for sentence-level chunks.
  - Accelerated via GPU (`torch.device`) when available.
- **Data Storage**
  - Embeddings and associated text chunks are stored in a CSV file.
  - CSV is reloaded with proper tensor conversion using NumPy and PyTorch.
- **Semantic Search**
  - Uses cosine similarity to match a query to the most relevant chunks.
  - Retrieval operates on the same embedding space used during encoding.

### 🧩 2. Augmentation

This step prepares a prompt for the language model:

- **Prompt Formatting**
  - The `prompt_formatter` function takes a user query and top-k retrieved chunks.
  - The final prompt includes a task instruction, a few-shot style answer format, and the retrieved context.
- **Few-Shot Learning Structure**
  - Incorporates well-structured Q&A examples to guide generation.
  - Encourages the model to extract, reason from, and ground its answer in the provided context.

### 🧠 3. Generation

This module runs an LLM to generate the final response:

- **Model Used**
  - [`unsloth/gemma-2-2b-it`](https://huggingface.co/unsloth/gemma-2-2b-it) from Hugging Face.
  - Uses Flash Attention 2 when possible for fast inference.
- **Tokenization and Inference**
  - Prompts are encoded with `AutoTokenizer`, and responses generated using `generate()`.
  - Outputs are decoded and printed cleanly.
- **Hardware-Aware Utilities**
  - Includes tools to inspect GPU memory usage and model size.
- **Example Questions**
  - A set of realistic, domain-relevant questions (e.g., "What are the macronutrients?") are used to test the pipeline.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Make sure you have:
- A CUDA-enabled GPU (for embedding and inference speed)
- `bfloat16` and Flash Attention 2 support for optimal performance

---

## 🚀 Running the Pipeline

1. **Download and parse textbook**  
   The notebook automatically downloads the textbook if not present.

2. **Embed content**  
   Use SentenceTransformers to embed all text chunks and save to CSV.

3. **Perform retrieval**  
   Provide a query, embed it, and compute cosine similarity with saved chunks.

4. **Format prompt**  
   Create a final LLM-ready prompt using the retrieved content.

5. **Generate output**  
   Use the Gemma 2B model to answer your query in an informed, explanatory manner.

---

## 📘 Dataset

- **Source**: [Human Nutrition 2e – University of Hawaii](https://pressbooks.oer.hawaii.edu/humannutrition2/)
- **License**: Open access, Creative Commons Attribution (CC BY)

---

## 📌 Example Query

```python
query = "What are the macronutrients, and what roles do they play in the human body?"
formatted_prompt = prompt_formatter(query, top_k_chunks)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:1")
outputs = llm_model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Future Improvements

- Add FAISS-based retrieval for larger corpora
- Integrate OpenAI/GPT-4 for comparative evaluation
- Build a simple UI with Streamlit or Gradio

