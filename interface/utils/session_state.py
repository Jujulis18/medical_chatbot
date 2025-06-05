import streamlit as st

def initialize_session_state():
    """Initialise l'état de session Streamlit"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if "settings" not in st.session_state:
        st.session_state.settings = {}