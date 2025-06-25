import streamlit as st
import google.generativeai as genai

# === Configure API ===
genai.configure(api_key="AIzaSyDoapR8ta82slxJSdnmJlVroMGS5JwlVK4")  # Replace with your actual key

# === Initialize the Gemini model ===
model = genai.GenerativeModel("models/gemini-2.5-flash")

# === Streamlit UI ===
st.set_page_config(page_title="Gemini CSV Assistant", layout="wide")
st.title("📊 Gemini CSV Assistant")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Define a session state for processed context
if 'context_chunks' not in st.session_state:
    st.session_state.context_chunks = []

if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)

    # Show dataframe preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    # Convert rows to natural language context (limit to first 20 for tokens)
    chunks = []
    for i, row in df.iterrows():
        text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(text)
        if i >= 19:  # limit to 20 rows for performance
            break

    st.session_state.context_chunks = chunks

# Ask a question
question = st.text_input("Ask a question about the uploaded data:")

if st.button("Submit") and question and st.session_state.context_chunks:
    context = "\n".join(st.session_state.context_chunks)
    prompt = f"Based on the following data:\n{context}\n\nAnswer this question: {question}"

    with st.spinner("Thinking..."):
        try:
            response = model.generate_content(prompt)
            st.subheader("Answer")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error: {e}")