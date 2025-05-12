# Chat with PDFs using LLMs - Beta

## Overview
This is a simple web-based chat application built using **Streamlit** and **Langchain**. The app backend follows the **Retrieval Augmented Generation (RAG)** framework.

The app allows users to provide a list of PDFs and ask questions that are answered by these documents. The system uses OpenAI's GPT for querying the PDFs and retrieving the most relevant information.

* **OpenAI API Key Required**: The user needs to provide their own OpenAI API key for the application to function.

---

## Features

* **Upload PDFs**: Users can upload PDFs (e.g., research papers, product manuals, policies).
* **Ask Questions**: Users can ask questions, and the system will return answers based on the uploaded PDFs.
* **FAISS Indexing**: The provided PDFs are parsed, chunked, and indexed using FAISS for quick retrieval.
* **LLM-Powered Query Interface**: Uses OpenAI's GPT to answer questions based on the document snippets.
* **Session Memory**: The app remembers previous interactions, including the PDFs and user messages, to provide context to the LLM for more accurate responses.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/umeshyadav7988/Chat-with-PDFs-using-LLMs---Beta.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run Locally

1. Navigate to the project directory:

```bash
cd Chat-with-PDFs-using-LLMs---Beta
```

2. Run the Streamlit app:

```bash
streamlit run chat_app.py
```

3. Open your browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Add the URLs of the PDFs you want to use in the sidebar, and start chatting with the bot.

---

## How It Works

1. **PDF Parsing & Chunking**:

   * The uploaded PDFs are parsed, and the content is chunked into smaller pieces for easier processing.

2. **Embedding & Storage in Vector DB**:

   * Each chunk is converted into an embedding vector using OpenAI's embeddings API.
   * These embeddings are indexed using **FAISS**, which allows fast and efficient retrieval.

3. **LLM-Powered Query Interface**:

   * When a user asks a question, the relevant chunks from the PDF are retrieved.
   * These chunks, along with the user's previous messages, are sent to the OpenAI GPT model as context.
   * The model uses this context to provide accurate answers based on the documents.

4. **Session Memory**:

   * The app keeps track of previous conversations and the documents being used.
   * This allows the LLM to generate more contextually aware responses, improving accuracy over time.

---

## API Integration

* **OpenAI API**: Users need to provide their own API key for OpenAI.
* **FAISS**: Used for fast vector-based document retrieval.

---

## Known Issues

* **Rate Limits**: If you exceed OpenAI's rate limit or quota, the app will show a relevant error message. Ensure your API key has sufficient usage limits.
* **PDF Size**: Extremely large PDFs may cause performance issues due to the chunking process.


## Contact

For any questions or further clarifications, feel free to reach out to me.

