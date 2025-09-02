import streamlit as st
from src.pipeline.rag_index import create_index, query_llm

st.title("🤖 Chatbot Simple")

# Initialisation du query engine
@st.cache_resource
def init_chatbot():
    return create_index()

query_engine = init_chatbot()

# Interface de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input utilisateur
if user_input := st.chat_input("Votre question..."):
    # Affichage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Réponse du bot
    response = query_llm(query_engine, user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)