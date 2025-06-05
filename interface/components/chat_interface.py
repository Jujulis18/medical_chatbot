import streamlit as st
from typing import Dict, List
import time
import sys
import os

# Ajouter le chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot.rag_pipeline import RAGPipeline
from interface.utils.session_state import update_metrics, add_query_to_history

class ChatInterface:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.rag_pipeline = self._initialize_rag_pipeline()
    
    def _initialize_rag_pipeline(self) -> RAGPipeline:
        """Initialiser le pipeline RAG avec les paramètres"""
        try:
            print("Initialisation du pipeline RAG...")
            pipeline = RAGPipeline.from_config(self.settings)
            print("Pipeline RAG initialisé avec succès")
            return pipeline
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation du pipeline RAG: {str(e)}")
            print(f"Erreur pipeline RAG: {e}")
            return None
    
    def render(self):
        """Afficher l'interface de chat"""
        st.title("🏥 Assistant Médical RAG")
        st.markdown("*Assistant d'information médicale - À des fins éducatives uniquement*")
        
        # Vérifier si le pipeline est initialisé
        if self.rag_pipeline is None:
            st.error("❌ Le système n'est pas correctement initialisé. Veuillez vérifier la configuration.")
            return
        
        # Container pour les messages
        messages_container = st.container()
        
        # Afficher l'historique des messages
        with messages_container:
            self._display_chat_history()
        
        # Input pour la nouvelle question
        self._render_chat_input()
        
        # Sidebar avec informations
        with st.sidebar:
            self._render_response_info()
            self._render_chat_controls()
    
    def _display_chat_history(self):
        """Affiche l'historique des conversations"""
        if not st.session_state.messages:
            return
        
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Afficher les sources si disponibles et activées
                if (message["role"] == "assistant" and 
                    "sources" in message and 
                    message["sources"] and
                    st.session_state.get("show_sources", True)):
                    
                    with st.expander(f"📚 Sources utilisées ({len(message['sources'])})"):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {j+1}: {source.get('title', 'Sans titre')}**")
                            st.markdown(f"*Score de pertinence: {source.get('score', 0):.3f}*")
                            
                            # Afficher un extrait du contenu
                            content = source.get('content', '')
                            if len(content) > 200:
                                content = content[:200] + "..."
                            st.markdown(f"*Extrait:* {content}")
                            st.divider()
    
    def _render_chat_input(self):
        """Gérer l'input de chat et les réponses"""
        # Input de chat
        prompt = st.chat_input("💬 Posez votre question médicale...")
        
        if prompt:
            self._process_user_question(prompt)
    
    def _process_user_question(self, prompt: str):
        """Traiter une question utilisateur"""
        # Ajouter la question de l'utilisateur à l'historique
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Afficher la question utilisateur immédiatement
        with st.chat_message("user"):
            st.write(prompt)
        
        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Afficher un spinner pendant le traitement
            with st.spinner("🔍 Recherche d'informations médicales..."):
                start_time = time.time()
                
                try:
                    # Appeler le pipeline RAG
                    response_data = self.rag_pipeline.query(prompt)
                    response_time = time.time() - start_time
                    
                    # Vérifier la structure de la réponse
                    if isinstance(response_data, dict):
                        answer = response_data.get("answer", "Désolé, je n'ai pas pu générer une réponse.")
                        sources = response_data.get("sources", [])
                    else:
                        # Fallback si la réponse n'est pas un dictionnaire
                        answer = str(response_data)
                        sources = []
                    
                    success = True
                    
                except Exception as e:
                    answer = f"❌ Une erreur s'est produite: {str(e)}\n\nPour des informations médicales fiables, veuillez consulter un professionnel de santé."
                    sources = []
                    response_time = time.time() - start_time
                    success = False
                    
                    # Log l'erreur pour le débogage
                    print(f"Erreur lors du traitement de la question: {e}")
            
            # Afficher la réponse
            ##response_container.write(answer)
            
            # Afficher les sources si disponibles
            if sources and st.session_state.get("show_sources", True):
                with st.expander(f"📚 Sources consultées ({len(sources)})"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**📄 {source.get('title', f'Document {i+1}')}**")
                        st.markdown(f"*Pertinence: {source.get('score', 0):.1%}*")
                        
                        # Extrait du contenu
                        content = source.get('content', '')
                        if content:
                            if len(content) > 300:
                                content = content[:300] + "..."
                            st.markdown(f"```\n{content}\n```")
                        
                        if i < len(sources) - 1:
                            st.divider()
        
        # Sauvegarder la réponse dans l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        
        # Mettre à jour les métriques
        update_metrics(success, response_time)
        add_query_to_history(prompt, answer, sources, success)
        
        # Forcer le rafraîchissement pour afficher les nouveaux messages
        st.rerun()
    
    def _render_response_info(self):
        """Afficher les informations sur la dernière réponse"""
        st.subheader("📊 Informations")
        
        # Métriques générales
        metrics = st.session_state.get("metrics", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions posées", metrics.get("total_queries", 0))
        with col2:
            st.metric("Réponses réussies", metrics.get("successful_queries", 0))
        
        if metrics.get("avg_response_time", 0) > 0:
            st.metric("Temps moyen", f"{metrics['avg_response_time']:.1f}s")
        
        # Informations sur la dernière réponse
        if st.session_state.messages and len(st.session_state.messages) > 1:
            last_response = st.session_state.messages[-1]
            if last_response["role"] == "assistant" and "sources" in last_response:
                sources = last_response["sources"]
                if sources:
                    st.markdown("### 📚 Dernière recherche")
                    st.metric("Sources trouvées", len(sources))
                    
                    # Graphique des scores de pertinence
                    if len(sources) > 1:
                        scores = [s.get("score", 0) for s in sources]
                        if any(score > 0 for score in scores):
                            st.bar_chart({"Pertinence": scores})
    
    def _render_chat_controls(self):
        """Afficher les contrôles du chat"""
        st.subheader("⚙️ Contrôles")
        
        # Toggle pour afficher les sources
        show_sources = st.checkbox(
            "Afficher les sources", 
            value=st.session_state.get("show_sources", True),
            help="Afficher ou masquer les sources utilisées pour générer les réponses"
        )
        st.session_state.show_sources = show_sources
        
        # Toggle pour le mode debug
        show_debug = st.checkbox(
            "Mode debug", 
            value=st.session_state.get("show_debug", False),
            help="Afficher des informations de débogage"
        )
        st.session_state.show_debug = show_debug
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Effacer l'historique", help="Supprimer tous les messages de la conversation"):
            self._clear_chat_history()
        
        # Bouton pour exporter l'historique
        if st.button("💾 Exporter la conversation", help="Télécharger l'historique des messages"):
            self._export_chat_history()
        
        # Afficher les informations de debug si activé
        if show_debug:
            self._render_debug_info()
    
    def _clear_chat_history(self):
        """Effacer l'historique des messages"""
        from interface.utils.session_state import clear_chat_history
        clear_chat_history()
        st.success("✅ Historique effacé!")
        st.rerun()
    
    def _export_chat_history(self):
        """Exporter l'historique des messages"""
        try:
            from interface.utils.session_state import export_chat_history
            export_data = export_chat_history()
            
            st.download_button(
                label="📥 Télécharger la conversation",
                data=export_data,
                file_name=f"conversation_medicale_{int(time.time())}.json",
                mime="application/json",
                help="Télécharger la conversation au format JSON"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'export: {str(e)}")
    
    def _render_debug_info(self):
        """Afficher les informations de débogage"""
        with st.expander("🔧 Informations de débogage"):
            debug_info = {
                "Pipeline initialisé": self.rag_pipeline is not None,
                "Nombre de messages": len(st.session_state.messages),
                "Paramètres actuels": self.settings,
                "État du système": "Opérationnel" if self.rag_pipeline else "Erreur"
            }
            
            for key, value in debug_info.items():
                st.text(f"{key}: {value}")
    
    def render_welcome_message(self):
        """Afficher un message d'accueil si c'est la première visite"""
        if len(st.session_state.messages) <= 1:  # Seulement le message initial
            st.info("""
            👋 **Bienvenue dans votre Assistant Médical RAG !**
            
            Je peux vous aider avec des informations médicales générales sur :
            - Les symptômes de diverses pathologies
            - Les traitements généraux
            - Les mesures de prévention
            - Les informations de santé publique
            
            ⚠️ **Important**: Ces informations sont fournies à titre éducatif uniquement. 
            Pour un diagnostic ou un traitement personnalisé, consultez toujours un professionnel de santé qualifié.
            """)

# Test de l'interface
if __name__ == "__main__":
    print("Test de ChatInterface...")
    
    # Configuration de test
    test_settings = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 3,
        "similarity_threshold": 0.5,
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Créer l'interface (nécessite Streamlit pour fonctionner complètement)
    try:
        interface = ChatInterface(test_settings)
        print("✓ Interface créée avec succès")
        
        if interface.rag_pipeline:
            print("✓ Pipeline RAG initialisé")
        else:
            print("❌ Problème avec le pipeline RAG")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        
    print("Test terminé.")