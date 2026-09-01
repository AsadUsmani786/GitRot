import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="GitRot", layout="wide")
st.title("GitRot")
st.caption("AI-powered code analysis and documentation platform")

# --- Sidebar ---
st.sidebar.header("Step 1 — Ingest your codebase")
folder_path = st.sidebar.text_input("Folder path", placeholder="C:/Users/you/your-project")

if st.sidebar.button("Ingest"):
    if folder_path:
        with st.spinner("Reading and embedding your codebase..."):
            try:
                res = requests.post(f"{API_URL}/ingest", json={"folder_path": folder_path})
                if res.status_code == 200:
                    data = res.json()
                    st.sidebar.success(f"✅ {data['files_ingested']} files, {data['chunks_stored']} chunks stored")
                else:
                    st.sidebar.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.sidebar.error(f"Could not connect to API: {e}")
    else:
        st.sidebar.warning("Please enter a folder path")

# --- Main area tabs ---
tab1, tab2 = st.tabs(["💬 Ask", "📄 Generate Docs"])

with tab1:
    st.subheader("Ask anything about your CodeBase")
    question = st.text_input("Your question", placeholder="How does the authentication work?")
    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(f"{API_URL}/ask", json={"question": question})
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown(data["answer"])
                        st.divider()
                        st.caption("Sources: " + ", ".join(data["sources"]))
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            st.warning("Please enter a question")

with tab2:
    st.subheader("Generate documentation for a file")
    filename = st.text_input("Filename", placeholder="ingestion.py")
    if st.button("Generate"):
        if filename:
            with st.spinner("Generating docs..."):
                try:
                    res = requests.post(f"{API_URL}/generate-docs", json={"filename": filename})
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown(data["documentation"])
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
        else:
            st.warning("Please enter a filename")