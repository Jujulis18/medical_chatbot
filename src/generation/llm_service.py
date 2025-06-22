from openai import OpenAI

class MISTRALChatGenerator:
    from openai import OpenAI
import logging
from typing import List, Optional

class MISTRALChatGenerator:
    def __init__(self, api_key: str, model: str = "mistral-small-latest", debug: bool = False):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
        self.model = model
        self.debug = debug
        
        # Messages prédéfinis
        self.greetings = ["bonjour", "salut", "hello", "bonsoir", "hey", "coucou"]
        self.intro_message = """Bonjour ! Je suis votre assistant médical.

Que puis-je faire pour vous ?
• Répondre à vos questions médicales
• Vous aider avec des symptômes
• Expliquer des termes médicaux
• Fournir des informations sur les traitements

Exemples de questions :
- "Quels sont les symptômes de l'hypertension ?"
- "Comment traiter une migraine ?"
- "Qu'est-ce que le diabète de type 2 ?"

Important : Je ne remplace pas un avis médical professionnel. En cas d'urgence, contactez le 15 (SAMU)."""

        self.help_message = """ Mes capacités :

• Questions médicales : symptômes, maladies, traitements
• Médicaments : posologie, effets secondaires, interactions  
• Prévention : conseils santé, dépistage
• Terminologie : explication de termes médicaux

Comment bien me poser une question :
- Soyez précis dans vos symptômes
- Mentionnez la durée des symptômes
- Indiquez votre âge si pertinent

Rappel : Consultez toujours un professionnel pour un diagnostic ou traitement."""

        # Configuration des logs
        if self.debug:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def generate(self, question: str, context_chunks: List[str]) -> str:
        try:
            # Détecter les salutations
            if self._is_greeting(question):
                return self.intro_message
            
            # Détecter les questions d'aide
            if self._is_general_help_question(question):
                return self.help_message
            
            # Vérifier les sujets sensibles
            sensitive_response = self._check_sensitive_topics(question)
            if sensitive_response:
                return sensitive_response
            
            # Générer avec le contexte RAG
            return self._generate_with_context(question, context_chunks)
            
        except Exception as e:
            self._log_error(f"Erreur lors de la génération: {str(e)}")
            return "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?"
    
    def _generate_with_context(self, question: str, context_chunks: List[str]) -> str:
        """Génère une réponse avec le contexte RAG"""
        context = "\n\n".join(context_chunks)
        
        # Debug: afficher le contexte récupéré
        if self.debug:
            self._log_context_info(context_chunks, question)
        
        # Vérifier si on a du contexte 
        if not context.strip():
            return """Information non trouvée

Je ne trouve pas d'information pertinente dans ma base de données médicale pour répondre à votre question.

Suggestions :
• Reformulez votre question avec d'autres termes
• Vérifiez l'orthographe
• Posez une question plus générale
• Consultez un professionnel de santé"""

        # Construire le prompt
        prompt = f"""Tu es un assistant médical expert. RÈGLES IMPORTANTES :

1. UTILISE UNIQUEMENT les informations du contexte fourni
2. Si l'information n'est PAS dans le contexte, dis clairement "Cette information n'est pas disponible dans ma base de données"
3. Sois précis, clair et structuré dans tes réponses
4. Toujours rappeler de consulter un professionnel pour un diagnostic
5. En cas de doute, recommande une consultation médicale

**CONTEXTE MÉDICAL :**
{context}

**QUESTION :**
{question}

**RÉPONSE :**"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Tu es un assistant médical prudent et précis. Tu ne donnes que des informations basées sur le contexte fourni."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Ajouter un disclaimer si pas déjà présent
            if "consulter" not in answer.lower() and "professionnel" not in answer.lower():
                answer += "\n\nRappel : Consultez un professionnel de santé pour tout diagnostic ou traitement."
            
            return answer
            
        except Exception as e:
            self._log_error(f"Erreur API Mistral: {str(e)}")
            return "Erreur de communication avec l'IA. Veuillez réessayer."
    
    def _is_greeting(self, question: str) -> bool:
        """Détecte si c'est une salutation"""
        question_lower = question.lower().strip()
        # Salutation simple ou avec ponctuation
        for greeting in self.greetings:
            if question_lower == greeting or question_lower == f"{greeting}!" or question_lower == f"{greeting}?":
                return True
        return False
    
    def _is_general_help_question(self, question: str) -> bool:
        """Détecte les questions d'aide générale"""
        help_patterns = [
            "que peux-tu faire", "que peux tu faire", "aide moi", "aide-moi",
            "comment ça marche", "qu'est-ce que tu fais", "tes capacités",
            "que proposes-tu", "que proposes tu", "comment tu fonctionnes",
            "quelles sont tes fonctions", "aide", "help"
        ]
        question_lower = question.lower()
        return any(pattern in question_lower for pattern in help_patterns)
    
    def _check_sensitive_topics(self, question: str) -> Optional[str]:
        """Vérifie et gère les sujets sensibles"""
        question_lower = question.lower()
        
        # Urgences/suicide
        suicide_keywords = [
            "suicide", "me tuer", "en finir", "plus envie de vivre", 
            "mourir", "mettre fin", "pas envie de continuer"
        ]
        if any(keyword in question_lower for keyword in suicide_keywords):
            return """Aide immédiate disponible

Si vous traversez une crise :
• Suicide Écoute : 01 45 39 40 00 (24h/24, gratuit)
• SOS Amitié : 09 72 39 40 50 (24h/24)
• 3114 : Numéro national de prévention du suicide (gratuit, 24h/24)

Urgences médicales : 15 (SAMU) ou 112

Vous n'êtes pas seul(e). Des professionnels sont là pour vous aider."""

        # Urgences médicales
        emergency_keywords = [
            "urgent", "urgence", "grave", "douleur intense", "ne peux plus respirer",
            "perte de conscience", "malaise", "accident", "empoisonnement",
            "overdose", "surdose", "hémorragie"
        ]
        if any(keyword in question_lower for keyword in emergency_keywords):
            return """Situation d'urgence médicale ?

Contactez immédiatement :
• 15 (SAMU) 
• 112 (Numéros d'urgence européen)
• Rendez-vous aux urgences les plus proches

Je ne peux pas évaluer les urgences médicales. N'attendez pas, contactez les secours maintenant."""

        # Médicaments contrôlés
        controlled_meds = [
            "antibiotique", "morphine", "anxiolytique", "antidépresseur",
            "benzodiazépine", "opiacé", "cortisone", "psychotrope"
        ]
        if ("comment obtenir" in question_lower or "où acheter" in question_lower) and \
           any(med in question_lower for med in controlled_meds):
            return """Médicaments sur ordonnance

Ces médicaments nécessitent une **prescription médicale** pour votre sécurité :
• Consultation avec un médecin
• Évaluation de votre état de santé
• Prescription adaptée à votre cas

Pourquoi ? Pour éviter :
- Interactions dangereuses
- Surdosage
- Effets secondaires graves
- Dépendance

Consultez votre médecin ou pharmacien."""

        # Autodiagnostic dangereux
        dangerous_self_diagnosis = [
            "j'ai un cancer", "c'est un cancer", "tumeur maligne",
            "crise cardiaque", "infarctus", "avc", "accident vasculaire"
        ]
        if any(phrase in question_lower for phrase in dangerous_self_diagnosis):
            return """Diagnostic médical requis

Les symptômes que vous décrivez nécessitent impérativement :
• Un examen médical professionnel  
• Des analyses complémentaires
• Un diagnostic posé par un médecin

Ne vous autodiagnostiquez pas pour des conditions graves.
Consultez rapidement un médecin ou rendez-vous aux urgences si nécessaire."""

        return None
    
    def _log_context_info(self, context_chunks: List[str], question: str):
        """Log les informations de contexte pour debugging"""
        if not self.debug:
            return
            
        self.logger.info(f"QUESTION: {question}")
        self.logger.info(f"CONTEXTE: {len(context_chunks)} chunks récupérés")
        
        if not context_chunks:
            self.logger.warning("AUCUN CONTEXTE RÉCUPÉRÉ!")
        else:
            for i, chunk in enumerate(context_chunks[:3]):  # Afficher max 3 chunks
                preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                self.logger.info(f"  Chunk {i+1}: {preview}")
    
    def _log_error(self, message: str):
        """Log les erreurs"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    
