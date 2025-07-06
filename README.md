# Chatbot Médical

## Contexte
Dans le cadre du développement d'un assistant virtuel pour le domaine médical, j'ai été chargé de créer un chatbot capable de fournir des informations de santé fiables et précises. L'objectif était de développer un système utilisant des techniques avancées de traitement du langage naturel (NLP) et de recherche par similarité pour offrir des réponses basées sur des connaissances métiers.

## Objectifs
- **Objectifs principaux** :
  - Répondre aux questions médicales courantes
  - Fournir des informations fiables basées sur des sources médicales vérifiées
- **Objectifs secondaires** :
  - Maintenir un niveau de sécurité élevé avec des garde-fous appropriés
  - Offrir une interface utilisateur intuitive et accessible

## Méthodologie
- **Outils et technologies utilisés** :
  - Pandas, NumPy, Scikit-learn : Manipulation et analyse des données
  - Transformers (Hugging Face), BERT, Sentence-Transformers : NLP et embeddings
  - FAISS, Vector Database : Recherche et indexation
  - Large Language Models (LLM), Mistral via OpenAI API : Génération de texte
  - Streamlit : Interface utilisateur web

- **Processus** :
  1. Collecte et préparation des données médicales depuis Kaggle
  2. Traitement et nettoyage des données pour assurer la qualité
  3. Création d'embeddings pour la recherche sémantique
  4. Implémentation RAG (Retrieval-Augmented Generation) avec BERT
  5. Génération de réponses via des modèles de langage avancés
  6. Mise en place de garde-fous pour la sécurité médicale
  7. Interface utilisateur accessible via Streamlit
 
<img src="https://github.com/user-attachments/assets/c24aed73-b023-4e14-adc0-c967cfd74418" width="500">

## Analyse et Résultats
- **Analyse des données** :
  - Statistiques descriptives : Distribution des données, valeurs manquantes
  - Analyse de qualité : Détection des doublons
  - Visualisations : Graphiques de distribution, nuages de mots

- **Résultats obtenus** :
  - Segmentation intelligente : Division du contenu en chunks cohérents
  - Génération d'embeddings avec BERT
  - Importance des garde-fous : Détection et gestion des contenus sensibles, Disclaimer obligatoire, Gestion des urgences médicales, Interdiction de diagnostic et de prescription pour assurer un environnement sûr et approprié.

## Impact Business
- **Valeur ajoutée** :
  - Fourniture d'informations médicales fiables et précises
  - Amélioration de l'accessibilité des informations de santé
  - Réduction de la charge de travail des professionnels de santé pour les questions courantes

- **Recommandations** :
  - Utilisation de modèles plus avancés pour la génération de réponses
  - Personnalisation des réponses selon le profil utilisateur
  - Ajout de bases de données sur d'autres spécialités médicales

## Conclusion
- **Résumé** :
  - Développement d'un chatbot médical utilisant des techniques avancées de NLP et de recherche par similarité
  - Implémentation réussie d'un système RAG avec BERT et génération de réponses via des LLM
  - Mise en place de garde-fous pour assurer la sécurité médicale

- **Leçons apprises** :
  - Importance de la qualité des données pour la génération de réponses précises
  - Nécessité de garde-fous stricts pour les applications médicales
  - Efficacité des modèles BERT pour la recherche sémantique

## Références et Liens
- **Sources de données** :
  - [Mental health faq for chatbot](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)

