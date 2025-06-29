import streamlit as st
import os
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load API Key ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Add it to .streamlit/secrets.toml or .env.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- UI Setup ---
st.image("assets/LOGO-CALIDAD-PASCUAL.png", width=200)
st.title("📦 Pascual GPT - Client Insights")
st.markdown("---")

# --- Load CSV and prepare client lookup ---
@st.cache_data
def load_client_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.astype(str)  # Make sure all data is text for prompting
    return df.set_index("client_id").to_dict(orient="index")

csv_path = "yearly_df.csv"  # Replace with your optimized table path when ready
client_lookup = load_client_data(csv_path)

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- Explanation function ---
def explain_client_data(client_id: str, question: str) -> str:
    client_data = client_lookup.get(client_id)

    if not client_data:
        return f"❌ Client {client_id} not found."

    # Format the row into natural language context
    context = "\n".join([f"{key}: {value}" for key, value in client_data.items()])

    # Prompt: descriptive only, no suggestions
    prompt = f"""
You are an assistant helping Pascual understand its clients' past behavior.

You will receive structured data for one client. Use this data to answer questions such as:
- What is the city or channel of the client?
- What was the yearly income, volume, median ticket, or efficiency?
- How many contacts or visits they had?

🟢 Only use the data provided.
❌ Do not suggest changes or make assumptions.
❌ If a field is missing, say "That information is not available."

Client data:
{context}

User question:
{question}
"""

    return llm.invoke(prompt).content

# --- Streamlit Input ---
st.markdown("### 💬 Ask a question about a client")
user_input = st.text_input("Example: What is the yearly income of client 100229632?")

if st.button("Run Query"):
    if user_input:
        with st.spinner("🔍 Thinking..."):
            match = re.search(r"([0-9]{6,})", user_input)
            if match:
                client_id = match.group(1)
                response = explain_client_data(client_id, user_input)
                st.markdown("### ✅ Answer:")
                st.write(response)
            else:
                st.warning("Please include a valid client ID in your question.")
    else:
        st.warning("Please enter a question.")
