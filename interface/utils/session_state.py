import streamlit as st
from typing import Dict, List, Any

def initialize_session_state():
    """Initialise l'état de session pour le chatbot médical"""
    
    # Messages de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis votre assistant médical virtuel. Je peux vous aider à comprendre des informations médicales générales. Comment puis-je vous aider aujourd'hui ?",
                "sources": []
            }
        ]
    
    # Pipeline RAG
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    # Paramètres utilisateur
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "temperature": 0.7,
            "max_tokens": 500
        }
    
    # Historique des requêtes pour analytics
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # État de l'interface
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    # Métriques de performance
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0
        }

def update_metrics(success: bool, response_time: float = 0.0):
    """Mettre à jour les métriques de performance"""
    st.session_state.metrics["total_queries"] += 1
    
    if success:
        st.session_state.metrics["successful_queries"] += 1
    else:
        st.session_state.metrics["failed_queries"] += 1
    
    # Calculer le temps de réponse moyen
    current_avg = st.session_state.metrics["avg_response_time"]
    total_queries = st.session_state.metrics["total_queries"]
    
    new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
    st.session_state.metrics["avg_response_time"] = new_avg

def add_query_to_history(query: str, response: str, sources: List[Dict], success: bool):
    """Ajouter une requête à l'historique"""
    import datetime
    
    query_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "response": response,
        "sources_count": len(sources),
        "success": success
    }
    
    st.session_state.query_history.append(query_data)
    
    # Limiter l'historique aux 100 dernières requêtes
    if len(st.session_state.query_history) > 100:
        st.session_state.query_history = st.session_state.query_history[-100:]

def clear_chat_history():
    """Effacer l'historique des messages"""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre assistant médical virtuel. Comment puis-je vous aider aujourd'hui ?",
            "sources": []
        }
    ]

def get_session_stats() -> Dict[str, Any]:
    """Obtenir les statistiques de la session"""
    return {
        "messages_count": len(st.session_state.messages),
        "queries_history_count": len(st.session_state.query_history),
        "metrics": st.session_state.metrics.copy(),
        "settings": st.session_state.user_settings.copy()
    }

def export_chat_history() -> str:
    """Exporter l'historique des messages en format texte"""
    import json
    from datetime import datetime
    
    export_data = {
        "export_date": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "query_history": st.session_state.query_history,
        "metrics": st.session_state.metrics,
        "settings": st.session_state.user_settings
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

# Fonctions utilitaires pour la persistance
def save_user_preferences(preferences: Dict[str, Any]):
    """Sauvegarder les préférences utilisateur"""
    st.session_state.user_settings.update(preferences)

def get_user_preferences() -> Dict[str, Any]:
    """Récupérer les préférences utilisateur"""
    return st.session_state.user_settings.copy()

# Fonction de débogage
def debug_session_state():
    """Afficher l'état de session pour le débogage"""
    if st.session_state.get("show_debug", False):
        with st.expander("🔧 Debug - État de session"):
            st.json({
                "messages_count": len(st.session_state.get("messages", [])),
                "rag_pipeline_loaded": st.session_state.get("rag_pipeline") is not None,
                "settings": st.session_state.get("user_settings", {}),
                "metrics": st.session_state.get("metrics", {}),
                "query_history_count": len(st.session_state.get("query_history", []))
            })

# Test de la fonction
if __name__ == "__main__":
    print("Test du module session_state...")
    
    # Simuler l'initialisation
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def __contains__(self, key):
            return key in self.data
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    # Mock st.session_state
    import sys
    mock_st = type('MockStreamlit', (), {'session_state': MockSessionState()})()
    sys.modules['streamlit'] = mock_st
    
    # Test d'initialisation
    initialize_session_state()
    print("✓ Initialisation réussie")
    
    # Test des métriques
    update_metrics(True, 1.5)
    update_metrics(False, 0.8)
    print("✓ Métriques mises à jour")
    
    # Test de l'historique
    add_query_to_history("Test query", "Test response", [], True)
    print("✓ Historique mis à jour")
    
    # Afficher les stats
    stats = get_session_stats()
    print(f"✓ Stats: {stats}")
    
    print("Tous les tests passés !")