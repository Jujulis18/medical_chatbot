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