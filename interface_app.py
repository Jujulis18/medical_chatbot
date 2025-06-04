import streamlit as st
from src.chatbot.rag_pipeline import RAGPipeline
from interface.components.chat_interface import ChatInterface
from interface.components.settings_panel import SettingsPanel
from interface.utils.session_state import initialize_session_state

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialiser l'état de session
    initialize_session_state()
    
    # Sidebar pour les paramètres
    with st.sidebar:
        settings_panel = SettingsPanel()
        settings = settings_panel.render()
    
    # Interface principale de chat
    chat_interface = ChatInterface(settings)
    chat_interface.render()

if __name__ == "__main__":
    main()
