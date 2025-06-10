# 🚀 SHL Test Assessment Recommender

This Streamlit application uses a **Retrieval-Augmented Generation (RAG)** pipeline to recommend relevant SHL individual test solutions based on a user's natural language query or job description. It combines **Hugging Face embeddings**, **FAISS indexing**, and **Google’s Gemini 1.5 Flash** for efficient retrieval and intelligent response generation.

The repository includes:
- ✅ A working **Streamlit app**
- 📒 A **Jupyter Notebook** for experimentation in **Google Colab**
- 📂 A scraped **dataset** of SHL assessments
- 📦 Required libraries listed for setup

---

## 🚀 Features

- 🧾 Accepts natural language input (job description or requirement)
- 🔍 Uses **SentenceTransformer** + **FAISS** for semantic vector search
- 🧠 Uses **Gemini 1.5 Flash** for natural language response generation
- 📊 Displays top **10 recommended SHL assessments** in a **Markdown table**
- ⚡ Fast and efficient due to on-the-fly chunking and caching

---

## 🔧 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **Embeddings**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Hugging Face
- **Indexing**: [FAISS](https://github.com/facebookresearch/faiss)
- **Text Splitting**: `RecursiveCharacterTextSplitter` from LangChain
- **LLM**: [Gemini 1.5 Flash](https://ai.google.dev/)
- **Data**: `SHL_Scraped_Data1.csv`

---

## ✅ Example Query

> `"Test for entry-level role in sales"`

The app returns a **Markdown-formatted table** with the most suitable assessments including:

- 📝 Name (clickable link)
- 🧪 Test Types
- ⏱ Completion Time
- ✅ Remote Testing support
- 🧠 Adaptive/IRT status

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shl-test-recommender.git
cd shl-test-recommender

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
