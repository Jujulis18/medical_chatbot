from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json
import re

class LLMService(ABC):
    """Service abstrait pour la génération de texte"""
    
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Générer une réponse basée sur la question et le contexte"""
        pass

class SimpleLLMService(LLMService):
    """Service LLM simple pour les tests - génère des réponses basées sur des templates"""
    
    def __init__(self, temperature: float = 0.7, max_tokens: int = 500):
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Templates de réponses pour différents types de questions médicales
        self.response_templates = {
            "symptômes": [
                "Selon les informations médicales disponibles, les symptômes typiques incluent : {symptoms}. Il est important de consulter un professionnel de santé pour un diagnostic précis.",
                "Les manifestations cliniques couramment observées sont : {symptoms}. Ces symptômes peuvent varier d'une personne à l'autre.",
                "D'après la littérature médicale, on observe généralement : {symptoms}. Si vous présentez ces symptômes, consultez votre médecin."
            ],
            "traitement": [
                "Le traitement recommandé comprend généralement : {treatment}. Il est essentiel de suivre les recommandations de votre médecin.",
                "Les approches thérapeutiques incluent : {treatment}. Le traitement doit être adapté à chaque patient.",
                "La prise en charge médicale consiste en : {treatment}. Consultez un professionnel de santé pour un traitement personnalisé."
            ],
            "diagnostic": [
                "Le diagnostic repose sur : {diagnosis}. Seul un médecin peut établir un diagnostic précis.",
                "Les éléments diagnostiques comprennent : {diagnosis}. Une évaluation médicale complète est nécessaire.",
                "Pour poser le diagnostic, on s'appuie sur : {diagnosis}. Consultez votre médecin pour une évaluation appropriée."
            ],
            "général": [
                "D'après les informations médicales disponibles : {info}. Pour des conseils personnalisés, consultez un professionnel de santé.",
                "Selon la documentation médicale : {info}. Il est recommandé de consulter votre médecin pour des conseils adaptés à votre situation.",
                "Les informations médicales indiquent que : {info}. N'hésitez pas à consulter un professionnel de santé pour plus de détails."
            ]
        }
        
        # Mots-clés pour identifier le type de question
        self.keywords = {
            "symptômes": ["symptôme", "symptômes", "signes", "manifestation", "comment reconnaître", "quels sont"],
            "traitement": ["traitement", "traiter", "soigner", "médicament", "thérapie", "comment guérir"],
            "diagnostic": ["diagnostic", "diagnostiquer", "comment savoir", "test", "examen", "dépistage"],
            "prévention": ["prévention", "prévenir", "éviter", "protection", "précaution"]
        }
    
    def _identify_question_type(self, question: str) -> str:
        """Identifier le type de question médicale"""
        question_lower = question.lower()
        
        for category, keywords in self.keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return "général"
    
    def _extract_key_information(self, context: str) -> Dict[str, str]:
        """Extraire les informations clés du contexte"""
        if not context or context == "Aucun document pertinent trouvé.":
            return {"info": "Aucune information spécifique trouvée dans ma base de connaissances"}
        
        # Rechercher des symptômes
        symptoms_patterns = [
            r"[Ss]ymptômes?\s*:?\s*([^.]*)",
            r"[Ss]ignes?\s*:?\s*([^.]*)",
            r"[Ii]ncluent?\s*:?\s*([^.]*symptômes?[^.]*)",
            r"comprennent?\s*:?\s*([^.]*)"
        ]
        
        symptoms = []
        for pattern in symptoms_patterns:
            matches = re.findall(pattern, context)
            symptoms.extend(matches)
        
        # Rechercher des traitements
        treatment_patterns = [
            r"[Tt]raitement\s*:?\s*([^.]*)",
            r"[Pp]rise en charge\s*:?\s*([^.]*)",
            r"[Tt]hérapie\s*:?\s*([^.]*)",
            r"[Mm]édicaments?\s*:?\s*([^.]*)"
        ]
        
        treatments = []
        for pattern in treatment_patterns:
            matches = re.findall(pattern, context)
            treatments.extend(matches)
        
        # Rechercher des éléments diagnostiques
        diagnosis_patterns = [
            r"[Dd]iagnostic\s*:?\s*([^.]*)",
            r"[Dd]iagnostiquer\s*:?\s*([^.]*)",
            r"se fait par\s*:?\s*([^.]*)",
            r"repose sur\s*:?\s*([^.]*)"
        ]
        
        diagnoses = []
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, context)
            diagnoses.extend(matches)
        
        # Nettoyer et formater les résultats
        def clean_text(text_list):
            cleaned = []
            for text in text_list:
                text = text.strip()
                if text and len(text) > 10:  # Éviter les fragments trop courts
                    cleaned.append(text)
            return cleaned
        
        result = {
            "symptoms": ", ".join(clean_text(symptoms)[:3]),  # Max 3 éléments
            "treatment": ", ".join(clean_text(treatments)[:3]),
            "diagnosis": ", ".join(clean_text(diagnoses)[:3]),
            "info": context[:200] + "..." if len(context) > 200 else context
        }
        
        return result
    
    def generate(self, question: str, context: str) -> str:
        """Générer une réponse basée sur la question et le contexte"""
        try:
            # Identifier le type de question
            question_type = self._identify_question_type(question)
            
            # Extraire les informations du contexte
            info = self._extract_key_information(context)
            
            # Choisir le template approprié
            templates = self.response_templates.get(question_type, self.response_templates["général"])
            
            # Sélectionner un template (simple rotation)
            template_index = hash(question) % len(templates)
            template = templates[template_index]
            
            # Formater la réponse
            if question_type == "symptômes" and info["symptoms"]:
                response = template.format(symptoms=info["symptoms"])
            elif question_type == "traitement" and info["treatment"]:
                response = template.format(treatment=info["treatment"])
            elif question_type == "diagnostic" and info["diagnosis"]:
                response = template.format(diagnosis=info["diagnosis"])
            else:
                response = template.format(info=info["info"])
            
            # Ajouter une note de sécurité
            safety_note = "\n\n⚠️ Cette information est fournie à titre éducatif uniquement. Pour un diagnostic ou un traitement personnalisé, consultez toujours un professionnel de santé qualifié."
            
            # Limiter la longueur de la réponse
            max_length = min(self.max_tokens * 4, 1000)  # Approximation 4 chars par token
            if len(response) > max_length:
                response = response[:max_length] + "..."
            
            return response + safety_note
            
        except Exception as e:
            return f"Je suis désolé, mais je n'ai pas pu traiter votre question correctement. Erreur: {str(e)}\n\nPour obtenir des informations médicales fiables, veuillez consulter un professionnel de santé."

class OpenAILLMService(LLMService):
    """Service LLM utilisant l'API OpenAI (exemple d'implémentation)"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 500):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Note: Vous devez installer openai avec: pip install openai
        # self.client = OpenAI(api_key=api_key)
        
    def generate(self, question: str, context: str) -> str:
        """Générer une réponse via l'API OpenAI"""
        # Implémentation exemple (nécessite la clé API OpenAI)
        prompt = f"""Tu es un assistant médical expert. Réponds à la question suivante .

Contexte médical:
{context}

Question: {question}

Réponds de manière précise et professionnelle. Rappelle toujours que tes réponses sont à titre informatif et qu'il faut consulter un professionnel de santé pour un diagnostic ou traitement personnalisé."""

        try:
            # Code d'exemple - à décommenter si vous avez une clé API OpenAI
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "Tu es un assistant médical expert."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens
            # )
            # return response.choices[0].message.content
            
            return "Service OpenAI non configuré. Veuillez configurer votre clé API."
            
        except Exception as e:
            return f"Erreur lors de l'appel à l'API OpenAI: {str(e)}"

# Factory function pour créer le bon service LLM
def create_llm_service(service_type: str = "simple", **kwargs) -> LLMService:
    """Créer un service LLM approprié"""
    if service_type == "simple":
        return SimpleLLMService(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 500)
        )
    elif service_type == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Clé API OpenAI requise pour le service OpenAI")
        return OpenAILLMService(
            api_key=api_key,
            model=kwargs.get("model", "gpt-3.5-turbo"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 500)
        )
    else:
        raise ValueError(f"Type de service LLM non supporté: {service_type}")

# Test du service
if __name__ == "__main__":
    print("Test du service LLM...")
    
    # Créer le service
    llm_service = create_llm_service("simple")
    
    # Contexte de test
    context = """La grippe (influenza) est une infection virale qui affecte principalement le système respiratoire. 
    Les symptômes typiques incluent: fièvre élevée (38-40°C), maux de tête sévères, courbatures et douleurs musculaires, 
    fatigue intense, toux sèche, mal de gorge, écoulement nasal. Le traitement comprend repos, hydratation, 
    antalgiques et antipyrétiques."""
    
    # Questions de test
    questions = [
        "Quels sont les symptômes de la grippe ?",
        "Comment traiter la grippe ?",
        "Comment diagnostiquer la grippe ?",
        "Qu'est-ce que la grippe ?"
    ]
    
    for question in questions:
        print(f"\n--- Question: {question} ---")
        response = llm_service.generate(question, context)
        print(f"Réponse: {response}")
        print("-" * 50)