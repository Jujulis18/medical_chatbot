import streamlit as st
from typing import Dict

class SettingsPanel:
    def render(self) -> Dict:
        st.header("⚙️ Configuration")
        
        # Paramètres du modèle d'embedding
        st.subheader("Embedding")
        embedding_model = st.selectbox(
            "Modèle d'embedding",
            ["sentence-transformers/all-MiniLM-L6-v2", 
             "sentence-transformers/all-mpnet-base-v2",
             "text-embedding-ada-002"]
        )
        
        # Paramètres de récupération
        st.subheader("Récupération")
        top_k = st.slider("Nombre de documents à récupérer", 1, 20, 5)
        similarity_threshold = st.slider("Seuil de similarité", 0.0, 1.0, 0.7)
        
        # Paramètres de génération
        st.subheader("Génération")
        temperature = st.slider("Température", 0.0, 2.0, 0.7)
        max_tokens = st.slider("Tokens max", 50, 2000, 500)
        
        # Bouton pour reconstruire l'index
        st.subheader("Gestion des documents")
        if st.button("🔄 Reconstruire l'index"):
            self._rebuild_index()
        
        return {
            "embedding_model": embedding_model,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    def _rebuild_index(self):
        with st.spinner("Reconstruction de l'index..."):
            # Logique pour reconstruire l'index
            st.success("Index reconstruit avec succès !")
