import streamlit as st
import runpy
import sys
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Page configuration
st.set_page_config(
    page_title="Legal Research Assistant",
    page_icon="⚖️",
    layout="centered"
)

PROMPT_TEMPLATE = """Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def populate_db(reset=False):
    argv_backup = sys.argv
    try:
        sys.argv = ['populate_database.py'] + (['--reset'] if reset else [])
        runpy.run_path('populate_database.py', run_name='__main__')
    finally:
        sys.argv = argv_backup


def query_rag_ui(text):
    embedder = get_embedding_function()
    db = Chroma(persist_directory='chroma', embedding_function=embedder)
    results = db.similarity_search_with_score(text, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(context=context, question=text)
    model = Ollama(model='mistral')
    answer = model.invoke(prompt)
    # extract source metadata
    raw_sources = [doc.metadata.get('source', '') for doc, _ in results]
    # remove duplicates while preserving order
    seen = set()
    sources = []
    for src in raw_sources:
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return answer, sources

# App header
st.title("⚖️ Legal Research Assistant")
st.markdown(
    "Use this tool to analyze your legal documents"
)

# Sidebar controls
with st.sidebar:
    st.header("Database")
    if st.button('Populate'):
        with st.spinner('Populating database...'):
            populate_db(False)
        st.success('Database populated.')

    if st.button('Reset'):
        with st.spinner('Resetting database...'):
            populate_db(True)
        st.success('Database reset and rebuilt.')

st.divider()

# User query
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if not query:
        st.error("Please enter a question.")
    else:
        with st.spinner('Searching..'):
            answer, sources = query_rag_ui(query)
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources"):
            if sources:
                for src in sources:
                    st.write(f"• {src}")
            else:
                st.write("No source found.")
