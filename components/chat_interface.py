import streamlit as st
from typing import Dict, List
from src.chatbot.rag_pipeline import RAGPipeline

class ChatInterface:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.rag_pipeline = self._initialize_rag_pipeline()
    
    def _initialize_rag_pipeline(self) -> RAGPipeline:
        # Initialiser le pipeline RAG avec les paramètres
        return RAGPipeline.from_config(self.settings)
    
    def render(self):
        st.title("🤖 RAG Chatbot")
        
        # Container pour les messages
        messages_container = st.container()
        
        # Afficher l'historique des messages
        with messages_container:
            self._display_chat_history()
        
        # Input pour la nouvelle question
        self._render_chat_input()
        
        # Sidebar avec informations sur la réponse
        self._render_response_info()
    
    def _display_chat_history(self):
        """Affiche l'historique des conversations"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Afficher les sources si disponibles
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources utilisées"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:** {source['title']}")
                            st.write(f"*Score:* {source['score']:.3f}")
                            st.write(f"*Extrait:* {source['content'][:200]}...")
    
    def _render_chat_input(self):
        """Render l'input de chat"""
        if prompt := st.chat_input("Posez votre question..."):
            # Ajouter la question de l'utilisateur
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Générer la réponse
            with st.chat_message("assistant"):
                with st.spinner("Recherche en cours..."):
                    response_data = self.rag_pipeline.query_with_sources(prompt)
                
                st.write(response_data["answer"])
                
                # Afficher les sources
                if response_data["sources"]:
                    with st.expander("📚 Sources utilisées"):
                        for i, source in enumerate(response_data["sources"]):
                            st.write(f"**Source {i+1}:** {source['title']}")
                            st.write(f"*Score:* {source['score']:.3f}")
            
            # Sauvegarder la réponse
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "sources": response_data["sources"]
            })
    
    def _render_response_info(self):
        """Affiche des informations sur la dernière réponse"""
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.sidebar.subheader("📊 Informations sur la réponse")
            
            last_response = st.session_state.messages[-1]
            if "sources" in last_response:
                st.sidebar.metric("Sources trouvées", len(last_response["sources"]))
                
                # Graphique des scores de similarité
                if last_response["sources"]:
                    scores = [s["score"] for s in last_response["sources"]]
                    st.sidebar.bar_chart(scores)
