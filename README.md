# Chatbot Médical

## Contexte
Dans un contexte où les patients cherchent de plus en plus des réponses rapides à leurs questions de santé, j’ai cherché à concevoir un chatbot médical capable de fournir des informations fiables, compréhensibles, et surtout sécurisées. 
**Le défi ?** Allier la puissance des modèles de langage actuels avec les exigences du domaine médical.

## Objectifs
L’idée n’est pas de remplacer un professionnel, mais d’aider les utilisateurs à mieux comprendre certains symptômes ou traitements, tout en posant des garde-fous stricts pour éviter tout dérapage.

Objectifs principaux :
- Répondre aux questions médicales fréquentes
- Garantir la fiabilité des réponses via des sources vérifiées

Objectifs secondaires :
- Assurer un haut niveau de sécurité (disclaimers, gestion des urgences…)
- Offrir une interface simple et accessible à tous

## Méthodologie
Pour cela, j’ai mis en place un système hybride basé sur le NLP :
  1. Collecte et préparation des données médicales depuis Kaggle
  2. Traitement et nettoyage des données pour assurer la qualité
  3. Création d'embeddings pour la recherche sémantique
  4. Implémentation RAG (Retrieval-Augmented Generation) avec BERT
  5. Génération de réponses via des modèles de langage avancés
  6. Mise en place de garde-fous pour la sécurité médicale
  7. Interface utilisateur accessible via Streamlit

- **Outils et technologies utilisés** :
  - Pandas, NumPy, Scikit-learn : Manipulation et analyse des données
  - Transformers (Hugging Face), BERT, Sentence-Transformers : NLP et embeddings
  - FAISS, Vector Database : Recherche et indexation
  - Large Language Models (LLM), Mistral via OpenAI API : Génération de texte
  - Streamlit : Interface utilisateur web [Streamlit](https://medicalchatbot-dr8qsucyifmvyhshnk5erh.streamlit.app/)
 
<img src="https://github.com/user-attachments/assets/c24aed73-b023-4e14-adc0-c967cfd74418" width="500">

## Analyse et Résultats
- **Analyse des données médicales de référence** :
  - Statistiques descriptives : Distribution des données, valeurs manquantes
  - Analyse de qualité : Détection des doublons
  - Visualisations : Graphiques de distribution, nuages de mots

- **Résultats obtenus** :
  - Segmentation intelligente : Division du contenu en chunks cohérents
  - Génération d'embeddings avec BERT
  - Importance des garde-fous : Détection et gestion des contenus sensibles, Disclaimer obligatoire, Gestion des urgences médicales, Interdiction de diagnostic et de prescription pour assurer un environnement sûr et approprié.

## Impact Business
- **Valeur ajoutée** :
  - Amélioration de l'accessibilité des informations de santé
  - Réduction de la charge de travail des professionnels de santé pour les questions courantes

- **Prochaines pistes** :
  - Utilisation de modèles plus avancés pour la génération de réponses
  - Personnalisation des réponses selon le profil utilisateur (age, contexte...)
  - Ajout de bases de données sur d'autres spécialités médicales

## Conclusion
Ce projet m’a permis de combiner mes compétences en NLP, data science et éthique pour répondre à un vrai besoin. Il m’a aussi rappelé à quel point la qualité des données et la confiance utilisateur sont au cœur de toute solution dans le domaine médical.

- **Leçons apprises** :
  - Importance de la qualité des données pour la génération de réponses précises
  - Nécessité de garde-fous stricts pour les applications médicales
  - Efficacité des modèles BERT pour la recherche sémantique

## 🚀 Comment démarrer

1. Clonez le dépôt 
2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

3. Lancez le dashboard :

```bash
streamlit run app.py
```

3. Créer un fichier .env à la racine 
```bash
MISTRAL_API_KEY=YOUR_API_KEY
```

---

## Références et Liens
- **Sources de données** :
  - [Mental health faq for chatbot](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot)

