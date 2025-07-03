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
st.title("📦 Pascual GPT - Client Optimization Insights")
st.markdown("---")

# --- Load CSV and create client lookup ---
@st.cache_data
def load_client_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.astype(str)  # Ensure all fields are string for clean formatting
    return df.set_index("client_id").to_dict(orient="index")

csv_path = "mock_data.csv"  # <-- Update with your latest full dataset
client_lookup = load_client_data(csv_path)

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# --- Explanation Function ---
def explain_client_data(client_id: str, question: str) -> str:
    client_data = client_lookup.get(client_id)
    if not client_data:
        return f"❌ Client {client_id} not found."

    try:
        total_cost_old = float(client_data.get("total_cost_old", 0))
        total_cost_new = float(client_data.get("total_cost_new", 0))
        savings = float(client_data.get("savings", 0))
        monthly_savings = total_cost_old - total_cost_new
        recovery_months = round(savings / monthly_savings) if monthly_savings > 0 else "N/A"
    except Exception:
        recovery_months = "N/A"

    # Format context string
    context = "\n".join([f"{key}: {value}€" if "cost" in key or "savings" in key else f"{key}: {value}" for key, value in client_data.items()])
    context += f"\nrecovery_months: {recovery_months}"

    # Updated prompt template
    prompt = f"""
You are a data analyst assistant helping Pascual understand client visit frequency optimizations.

You will receive structured data about a single client.

🎯 Your task:
- Summarize what the client's values were before optimization.
- Describe what the new values are after optimization.
- Mention the savings and improvements in efficiency.
- Estimate how many months it would take to recover the cost, based on the savings and the monthly cost difference.
- Only use the data provided. Never make up or assume values.

❌ Do not hallucinate.
❌ Do not guess if a value is missing.
✅ If a field is missing, say: "This information is not available."

🔍 Example answer format:
- The client 653025 had a median ticket of 80€.
- The old frequency was 3.0 and they had 4 contacts/month.
- The new model suggests 2.0 frequency and 3 contacts/month.
- This leads to estimated savings of 180€ and improves efficiency from 0.3 to 0.5.
- At this rate, the estimated time to recover the savings is 2 months.

---

Client data:
{context}

Question:
{question}
"""


    return llm.invoke(prompt).content

# --- Streamlit Input ---
st.markdown("### 💬 Ask a question about a client")
user_input = st.text_input("Example: Explain the changes made to the frequency and efficiency of client 100229632.")

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

