
# modules/chatbot.py
import streamlit as st
import pandas as pd

def show_chatbot(pipeline):

	st.set_page_config(page_title="Chatbot Médical RAG")
	st.title("Chatbot Médical basé sur RAG")
	st.warning("⚠️ This chatbot is available in English only.")


	# Historique de conversation
	if "messages" not in st.session_state:
	    st.session_state.messages = []


	# Messages du chatbot (user + assistant)
	for msg in st.session_state.messages:
	    st.chat_message(msg["role"]).markdown(msg["content"])

	# Message d'accueil assistant
	with st.chat_message("assistant"):
	    st.write("Hello I'm your medical assistant, How can I help you today?")


	# Champ de saisie utilisateur
	user_input = st.chat_input("Posez votre question médicale...")


	if user_input:
	    # Ajout à l'historique
	    st.session_state.messages.append({"role": "user", "content": user_input})
	    st.chat_message("user").markdown(user_input)

	    with st.chat_message("assistant"):
	        with st.spinner("Recherche des informations..."):
	            try:
	                response, debug_info = pipeline.run(user_input)
	                
	                # Affichage du debug
	                with st.expander("Informations de debug"):
	                    for info in debug_info:
	                        st.text(info)
	                
	                st.markdown(response)
	                st.session_state.messages.append({"role": "assistant", "content": response})
	                
	            except Exception as e:
	                st.error(f"ERREUR: {str(e)}")
	                import traceback
	                st.code(traceback.format_exc())



