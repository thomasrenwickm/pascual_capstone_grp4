import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Configure Gemini API ===
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual key
model = genai.GenerativeModel("models/gemini-1.5-flash")

# === Load embedding model ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Streamlit UI ===
st.set_page_config(page_title="RAG Chatbot with Gemini", layout="wide")
st.title("💬 CSV Chatbot using Gemini + RAG")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Global state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.chunk_embeddings = None
    st.session_state.chat_history = []  # Full conversation memory

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    # Show data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Convert rows to chunks
    chunks = [", ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    st.session_state.chunks = chunks

    # Embed chunks
    st.info("Embedding data... please wait")
    embeddings = embed_model.encode(chunks)
    st.session_state.chunk_embeddings = embeddings

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.session_state.index = index
    st.success("Index ready! Start chatting below.")

# Chatbot UI
st.subheader("💬 Ask your data anything")
user_input = st.chat_input("Ask a question about the data...")

if user_input and st.session_state.index is not None:
    # Embed and retrieve relevant chunks
    q_embedding = embed_model.encode([user_input])
    _, indices = st.session_state.index.search(np.array(q_embedding), k=5)
    relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]

    # Construct full conversation context
    chat_context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['bot']}" for entry in st.session_state.chat_history[-5:]])

    prompt = f"You are a helpful assistant answering questions about a CSV dataset.\n{chat_context}\n\nData rows:\n{chr(10).join(relevant_chunks)}\n\nUser: {user_input}\nAssistant:"

    with st.spinner("Thinking..."):
        try:
            response = model.generate_content(prompt)
            bot_reply = response.text.strip()

            # Save to chat history
            st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})

        except Exception as e:
            bot_reply = f"❌ Error: {e}"

# Display chat history
if st.session_state.chat_history:
    for entry in st.session_state.chat_history[-10:][::-1]:
        with st.chat_message("user"):
            st.markdown(entry['user'])
        with st.chat_message("assistant"):
            st.markdown(entry['bot'])