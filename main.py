import streamlit as st
import os
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from typing import Dict, Optional, Union

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION AND SETUP ---
def setup_api_key() -> str:
    """
    Load and validate the Google API key from Streamlit secrets or environment variables.
    
    Returns:
        str: The API key if found and valid
        
    Raises:
        SystemExit: If API key is not found
    """
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("❌ Google API key not found. Add it to .streamlit/secrets.toml or set GOOGLE_API_KEY environment variable.")
        st.stop()
    
    # Set the API key in environment for LangChain
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    return GOOGLE_API_KEY

# --- UI SETUP AND BRANDING ---
def setup_ui():
    """
    Configure the Streamlit UI with branding and layout.
    """
    # Try to load the Pascual logo, fallback gracefully if not found
    try:
        st.image("assets/LOGO-CALIDAD-PASCUAL.png", width=200)
    except:
        st.markdown("### 🥛 PASCUAL")
        st.caption("*Logo not found - add assets/LOGO-CALIDAD-PASCUAL.png*")
    
    st.title("📦 Pascual GPT - Client Optimization Insights")
    st.markdown("""
    This tool helps explain the optimization changes made to client visit frequencies and contact strategies.
    
    **How to use:**
    - Enter a question about a specific client (include the client ID)
    - The system will analyze the optimization changes and provide insights
    - Example: "Explain the changes made to client 100229632"
    """)
    st.markdown("---")

# --- DATA LOADING AND PROCESSING ---
@st.cache_data
def load_client_data(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load client data from CSV and create a lookup dictionary for fast access.
    
    Args:
        csv_path (str): Path to the CSV file containing client data
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary with client_id as key and client data as value
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        Exception: For other data loading errors
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} clients from {csv_path}")
        
        # Validate required columns exist
        required_columns = ['client_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert all fields to strings for consistent formatting
        df = df.astype(str)
        
        # Create lookup dictionary with client_id as key
        client_lookup = df.set_index("client_id").to_dict(orient="index")
        
        return client_lookup
        
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {csv_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# --- CLIENT ID EXTRACTION ---
def extract_client_id(user_input: str) -> Optional[str]:
    """
    Extract client ID from user input using regex pattern.
    Looks for numeric sequences of 6 or more digits as specified in requirements.
    
    Args:
        user_input (str): The user's input question
        
    Returns:
        Optional[str]: The extracted client ID or None if not found
    """
    # Search for numeric sequences of 6 or more digits
    match = re.search(r'([0-9]{6,})', user_input)
    if match:
        client_id = match.group(1)
        logger.info(f"Extracted client ID: {client_id}")
        return client_id
    return None

# --- FINANCIAL CALCULATIONS ---
def calculate_derived_metrics(client_data: Dict[str, str]) -> Dict[str, Union[str, float]]:
    """
    Calculate derived metrics like monthly savings and recovery time.
    
    Args:
        client_data (Dict[str, str]): Raw client data from CSV
        
    Returns:
        Dict[str, Union[str, float]]: Dictionary with calculated metrics
    """
    try:
        # Extract and convert financial values
        total_cost_old = float(client_data.get("total_cost_old", 0))
        total_cost_new = float(client_data.get("total_cost_new", 0))
        savings = float(client_data.get("savings", 0))
        
        # Calculate monthly cost difference
        monthly_cost_difference = total_cost_old - total_cost_new
        
        # Calculate recovery time (how long to recover the optimization investment)
        if monthly_cost_difference > 0 and savings > 0:
            recovery_months = round(savings / monthly_cost_difference, 1)
        else:
            recovery_months = "N/A"
        
        # Calculate efficiency improvements
        try:
            efficiency_old = float(client_data.get("efficiency_old", 0))
            efficiency_new = float(client_data.get("efficiency_new", 0))
            efficiency_improvement = round((efficiency_new - efficiency_old) / efficiency_old * 100, 1) if efficiency_old > 0 else "N/A"
        except:
            efficiency_improvement = "N/A"
        
        return {
            "monthly_savings": monthly_cost_difference,
            "recovery_months": recovery_months,
            "efficiency_improvement": efficiency_improvement
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            "monthly_savings": "N/A",
            "recovery_months": "N/A", 
            "efficiency_improvement": "N/A"
        }

# --- CONTEXT FORMATTING ---
def format_client_context(client_data: Dict[str, str], derived_metrics: Dict[str, Union[str, float]]) -> str:
    """
    Format client data into a structured context string for the LLM.
    
    Args:
        client_data (Dict[str, str]): Raw client data
        derived_metrics (Dict[str, Union[str, float]]): Calculated metrics
        
    Returns:
        str: Formatted context string
    """
    context_lines = []
    
    # Add all client data with proper formatting
    for key, value in client_data.items():
        if "cost" in key.lower() or "savings" in key.lower() or "income" in key.lower():
            # Format monetary values
            context_lines.append(f"{key}: {value}€")
        else:
            # Format other values
            context_lines.append(f"{key}: {value}")
    
    # Add derived metrics
    for key, value in derived_metrics.items():
        if key == "efficiency_improvement" and value != "N/A":
            context_lines.append(f"{key}: {value}%")
        else:
            context_lines.append(f"{key}: {value}")
    
    return "\n".join(context_lines)

# --- LLM INTEGRATION ---
def create_llm_prompt(context: str, user_question: str) -> str:
    """
    Create a comprehensive prompt for the LLM based on business requirements.
    
    Args:
        context (str): Formatted client data context
        user_question (str): Original user question
        
    Returns:
        str: Complete prompt for the LLM
    """
    prompt = f"""You are a data analyst assistant helping Pascual (a Spanish CPG company) understand client visit frequency optimizations and cost savings.

🎯 YOUR TASK:
Analyze the client data provided and explain the optimization analysis in a business-friendly format.

📋 CRITICAL FIRST STEP - CHECK FOR NO CHANGES:
⚠️ BEFORE writing your response, carefully examine these values:
- Compare prom_contacts_month_old with prom_contacts_month_new (round to 1 decimal)
- Compare frequency_old with frequency_new (round to 1 decimal)  
- Check if savings equals 0 or 0.0

🔍 IF ALL THREE CONDITIONS ARE TRUE (contacts same, frequency same, savings = 0):
- This is a NO CHANGES scenario
- ONLY mention: median_ticket, income, volume (if available)
- State that no optimization changes were suggested
- DO NOT mention before/after optimization, efficiency, or business impact

🔍 IF ANY CONDITION IS FALSE (there are actual changes):
- This is a WITH CHANGES scenario
- Include full analysis with before/after comparison

📋 WHAT TO INCLUDE IN YOUR RESPONSE (for cases WITH changes):
1. **Client Background**: Summarize key client characteristics (median ticket, volume, income)
2. **Pre-Optimization State**: Describe original frequency, contacts, and efficiency
3. **Post-Optimization Changes**: Explain new suggested frequency and contacts
4. **Business Impact**: Quantify savings, efficiency improvements, and recovery time

📐 BUSINESS CONTEXT:
- Logistics cost per order: 10€
- Promotor visit cost: 15€
- Target: Optimize promotor visits while maintaining service quality
- Goal: Align frequency with contacts to reduce inefficiencies

⚠️ CRITICAL REQUIREMENTS:
❌ NEVER hallucinate or guess values
❌ NEVER assume missing information
✅ Only use data provided in the context
✅ If information is missing, explicitly state "This information is not available"
✅ Be precise with numbers and calculations
✅ Always check for no-changes scenario first

💡 EXAMPLE FORMAT FOR NO CHANGES:
"Client [ID] operates with a median ticket of [amount]€, generates [income]€ in annual income, and has an annual volume of [volume]. Based on the optimization analysis, the model suggests no changes to the current visit frequency and contact strategy for this client, indicating that the current approach is already optimal."

💡 EXAMPLE FORMAT FOR WITH CHANGES:
"Client [ID] operates with a median ticket of [amount]€ and generates [income]€ in annual income. 

Before optimization: The client made [frequency_old] median orders per month with [contacts_old] promotor visits. The client had an efficiency of [efficiency_old].

After optimization: The model suggests [frequency_new] orders per month with [contacts_new] visits, improving efficiency to [efficiency_new].

Business Impact: This optimization generates [savings]€ in total savings with an estimated recovery time of [recovery_months] months. The changes reduce unnecessary visits while maintaining order volume."

---

CLIENT DATA:
{context}

USER QUESTION:
{user_question}

RESPONSE:"""

    return prompt

# 5. **Strategic Rationale**: Explain why these changes make business sense
def get_llm_explanation(client_id: str, user_question: str, client_lookup: Dict[str, Dict[str, str]]) -> str:
    """
    Generate LLM explanation for a specific client's optimization changes.
    
    Args:
        client_id (str): The client ID to analyze
        user_question (str): User's original question
        client_lookup (Dict[str, Dict[str, str]]): Client data lookup dictionary
        
    Returns:
        str: LLM generated explanation
    """
    # Check if client exists in our data
    client_data = client_lookup.get(client_id)
    if not client_data:
        return f"❌ Client {client_id} not found in the database. Please verify the client ID and try again."
    
    try:
        # Calculate derived metrics
        derived_metrics = calculate_derived_metrics(client_data)
        
        # Format context for LLM
        context = format_client_context(client_data, derived_metrics)
        
        # Create prompt
        prompt = create_llm_prompt(context, user_question)
        
        # Initialize LLM with temperature 0 for deterministic responses
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error getting LLM explanation: {str(e)}")
        return f"❌ Error generating explanation: {str(e)}. Please try again."

# --- MAIN APPLICATION ---
def main():
    """
    Main application function that orchestrates the Streamlit app.
    """
    # Setup API key
    setup_api_key()
    
    # Setup UI
    setup_ui()
    
    # Load client data
    csv_path = "optimized_df.csv"  # Update this path as needed
    client_lookup = load_client_data(csv_path)
    
    # Display data statistics
    st.sidebar.markdown("### 📊 Data Statistics")
    st.sidebar.info(f"Total clients loaded: {len(client_lookup):,}")
    
    # Main input section
    st.markdown("### 💬 Ask about a client optimization")
    
    # Example questions for user guidance
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - "Explain the changes made to client 100229632"
        - "Show me the efficiency improvements for client 123456789"
        """)
    
    # User input
    user_input = st.text_input(
        "Enter your question about a specific client:",
        placeholder="Example: Explain the changes made to client 100229632",
        help="Make sure to include the client ID (6+ digits) in your question"
    )
    
    # Process query
    if st.button("🔍 Analyze Client", type="primary"):
        if not user_input:
            st.warning("⚠️ Please enter a question.")
            return
        
        # Extract client ID
        client_id = extract_client_id(user_input)
        
        if not client_id:
            st.error("❌ No valid client ID found. Please include a client ID (6+ digits) in your question.")
            return
        
        # Show processing
        with st.spinner(f"🔍 Analyzing client {client_id}..."):
            # Get LLM explanation
            response = get_llm_explanation(client_id, user_input, client_lookup)
            
            # Display results
            st.markdown("### ✅ Analysis Results")
            st.markdown(f"**Client ID:** {client_id}")
            st.markdown("**Explanation:**")
            st.write(response)
            
            # Add raw data option for transparency
            with st.expander("📋 View Raw Client Data"):
                client_data = client_lookup.get(client_id, {})
                if client_data:
                    df_display = pd.DataFrame([client_data]).T
                    df_display.columns = ['Value']
                    st.dataframe(df_display)
    
    # Footer
    st.markdown("---")
    st.markdown("*Pascual Client Optimization Tool - Powered by Gemini 2.0 Flash*")

# --- RUN APPLICATION ---
if __name__ == "__main__":
    main()