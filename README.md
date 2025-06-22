# Chatbot Médical 
**Datasets** : [Liens vers les données Kaggle utilisées](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)

## Introduction

Ce projet vise à développer un chatbot médical capable de fournir des informations de santé fiables et précises. 
Le système utilise des techniques avancées de traitement du langage naturel (NLP) et de recherche par similarité pour offrir des réponses basées sur des connaissances métiers.

L'objectif principal est de créer un assistant virtuel qui peut :
- Répondre aux questions médicales courantes
- Fournir des informations fiables basées sur des sources médicales vérifiées
- Maintenir un niveau de sécurité élevé avec des garde-fous appropriés
- Offrir une interface utilisateur intuitive et accessible

Limites des réponses: informations présentes dans la base de données.

## Librairies et Outils

### Traitement des Données
- **Pandas** : Manipulation et analyse des données
- **NumPy** : Calculs numériques et opérations matricielles
- **Scikit-learn** : Préprocessing et outils d'apprentissage automatique

### NLP et Embeddings
- **Transformers (Hugging Face)** : Modèles BERT pour les embeddings
- **BERT** : Modèle de langage pour la compréhension contextuelle
- **Sentence-Transformers** : Génération d'embeddings de phrases

### Recherche et Indexation
- **FAISS** : Recherche de similarité vectorielle rapide
- **Vector Database** : Stockage et récupération d'embeddings

### Génération de Texte
- **Large Language Models (LLM)** : Génération de réponses
- **Mistral via OpenAI API** : Modèles de génération de texte

### Interface et Déploiement
- **Streamlit** : Interface utilisateur web

## Résumé du Projet

Le pipeline comprend :

1. [**Collecte et préparation des données**](#analyse) médicales depuis Kaggle
2. [**Traitement et nettoyage**](#nettoyage) des données pour assurer la qualité
3. [**Création d'embeddings**](#embedding) pour la recherche sémantique
4. [**Implémentation RAG**](#RAG) (Retrieval-Augmented Generation) avec BERT
5. [**Génération de réponses**](#generation) via des modèles de langage avancés
6. [**Mise en place de garde-fous**](#gardefou) pour la sécurité médicale
7. [**Interface utilisateur**](#interface) accessible via Streamlit

## Collecte des Données

### Source : Kaggle
Les données médicales proviennent de datasets Kaggle soigneusement sélectionnés :
- **Datasets utilisés** : Mental health faq for chatbot
- **Volume de données** : 98 entrées
- **Types de données** : date, question, answer about mental health
- **Format** : CSV

## Analyse et Nettoyage des Données <a name="analyse">

### Analyse Exploratoire
- **Statistiques descriptives** : Distribution des données, valeurs manquantes
- **Analyse de qualité** : Détection des doublons
- **Visualisations** : Graphiques de distribution, nuages de mots

### Processus de Nettoyage
- **Suppression des doublons** et des entrées corrompues
- **Normalisation du texte** : Suppression des caractères spéciaux, uniformisation  

## Chunking et Embeddings <a name="embedding">

### Stratégie de Chunking
- **Segmentation intelligente** : Division du contenu en chunks cohérents

### Génération d'Embeddings
- **Modèle utilisé** : BERT 
- **Dimension des vecteurs** : 

## Base de Données Vectorielle - FAISS

### Configuration FAISS
- **Type d'index** : IndexFlatL2 
- **Métriques de distance** : Similarité cosinus 

## RAG avec BERT <a name="RAG">

### Architecture RAG
- **Retriever** : BERT pour la récupération de documents pertinents

### Processus de Récupération
1. **Requête utilisateur** → Embedding BERT
2. **Recherche FAISS** → Top-k documents similaires
3. **Re-ranking** → Affinement de la pertinence
4. **Sélection finale** → Documents les plus pertinents

### Prochaine Étape : LLM
- **Utilisation** de modèles plus avancés 
- **Personnalisation** des réponses selon le profil utilisateur
- **Ajout** de database sur d'autres specialités médicales

## Génération de Messages par LLM <a name="generation">

### Modèle de Génération
- **Architecture** : mistral-small-latest
- **Prompting** : Techniques de prompt engineering 

## Garde-Fous <a name="gardefou">

### Liste des Garde-Fous Implémentés

#### Sécurité Médicale
- **Disclaimer obligatoire** : Rappel que le chatbot ne remplace pas un médecin
- **Urgences médicales** : Redirection vers les services d'urgence
- **Diagnostic interdit** : Refus de poser des diagnostics définitifs
- **Prescription interdite** : Aucune recommandation de médicaments spécifiques

#### Filtrage de Contenu
- **Détection de contenus sensibles** : Suicide, automutilation, substances illégales

## Interface Utilisateur <a name="interface">
<img src="https://github.com/user-attachments/assets/c24aed73-b023-4e14-adc0-c967cfd74418" width="500">

🚀 **Application Streamlit** : [Lien vers l'application déployée](https://firstdraft-qoslojm4n4xesvknrpsc3r.streamlit.app/)

## Prochaines Étapes

### Métriques et Évaluation
- **Benchmarking** : Comparaison avec d'autres solutions
- **Précision des réponses** : Pourcentage de réponses correctes
- **Temps de réponse** : Latence moyenne du système



