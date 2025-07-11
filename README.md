# LLM-Powered Legal Research Chatbot

This project builds a RAG-based chatbot that helps answer legal research questions using dense legal PDFs.

## UI screenshot

<img src="ui_ss.png" alt="UI screenshot" width="600"/>

## Features

**Contextual Legal Question Answering**  
Users can ask complex legal questions, and the chatbot answers them using information pulled directly from uploaded PDFs. The answers are generated using Mistral-7B and include citations to the relevant source PDFs. Multiple PDFs may be added.

**PDF Embedding with Nomic**  
Legal documents are split into manageable chunks and embedded using Nomic Embed. These embeddings help the system understand and retrieve semantically similar text during a query.

**Vector Search with Chroma**  
All document embeddings are stored in a local Chroma vector database. At runtime, the most relevant chunks are retrieved using similarity search to ground the model’s response.

**LLM-Powered Generation via Ollama**  
Responses are generated using the open-weight Mistral-7B model running locally via Ollama. This setup keeps everything offline and avoids API calls to external services.

**Interactive UI with Streamlit**  
The chatbot is accessible through a simple Streamlit interface. Users can type in questions, view model responses, and see the source documents that were used.

