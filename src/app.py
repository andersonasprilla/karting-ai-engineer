import streamlit as st
import pandas as pd
import tempfile
import os
import shutil
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from stitcher import load_and_stitch_from_folder

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="LO206 Virtual Engineer",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ LO206 Virtual Race Engineer")
st.markdown("---")

# --- 2. SIDEBAR: SETUP CONTEXT ---
with st.sidebar:
    st.header("🛠️ Kart Setup")
    st.info("Update this for each session!")
    
    gearing = st.text_input("Gearing (Front/Rear)", "15/58")
    tire_psi = st.text_input("Tire Pressure (Hot)", "12.5 psi")
    track_cond = st.selectbox("Track Condition", ["Green/Cold", "Good/Grippy", "Greasy/Hot"])
    notes = st.text_area("Driver Notes", "Car feels tight in the hairpin.")
    
    # This context string is injected into the AI's brain
    setup_context = f"""
    CURRENT SESSION CONTEXT:
    - Gearing: {gearing}
    - Tire Pressure: {tire_psi}
    - Track Condition: {track_cond}
    - Driver Notes: {notes}
    """

# --- 3. FILE UPLOADER ---
st.subheader("1. Upload Session Data")
st.caption("Select all CSV files from your RaceStudio export (RPM, _GPS, Steering, etc.)")
uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type="csv")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. MAIN LOGIC ---
if uploaded_files:
    # Create a temporary directory to process these files
    # This mimics the folder structure your stitcher expects
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Save uploaded files to the temp dir
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Run the Stitcher
        with st.spinner("⚙️ Stitching telemetry files..."):
            master_df, status_msg = load_and_stitch_from_folder(temp_dir)
        
        if master_df is not None:
            st.success(f"✅ Data Processed! ({len(master_df)} samples)")
            
            # Save master dataframe to a temp CSV for the Agent to read
            master_csv_path = os.path.join(temp_dir, "master_telemetry.csv")
            master_df.to_csv(master_csv_path, index=False)
            
            # --- 5. CHAT INTERFACE ---
            st.subheader("2. Ask the Engineer")

            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User Input
            if prompt := st.chat_input("Ex: 'Where is my minimum RPM in Turn 4?'"):
                # 1. Show User Message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 2. Generate AI Response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Initialize the Brain
                            llm = ChatOpenAI(temperature=0, model="gpt-4o")
                            
                            agent = create_csv_agent(
                                llm,
                                master_csv_path,
                                verbose=True,
                                allow_dangerous_code=True,
                                number_of_head_rows=5,
                                # ADD THIS LINE BELOW:
                                agent_executor_kwargs={"handle_parsing_errors": True}
                            )

                            # Combine Setup Context + User Question
                            full_prompt = f"""
                            You are an expert LO206 Karting Race Engineer.
                            
                            {setup_context}
                            
                            USER QUESTION: {prompt}
                            
                            When answering:
                            1. Use the data in the dataframe.
                            2. Look for 'lat_g' spikes to detect handling issues.
                            3. Focus on 'rpm' drops to detect momentum loss.
                            """
                            
                            response = agent.run(full_prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        except Exception as e:
                            st.error(f"AI Error: {e}")
        else:
            st.error(f"Could not stitch files: {status_msg}")