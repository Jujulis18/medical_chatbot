from openai import OpenAI
import logging
from typing import List, Optional

class MISTRALChatGenerator:
    def __init__(self, api_key: str, model: str = "mistral-small-latest", debug: bool = False):
        print(api_key, flush=True)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1"
        )
        self.model = model
        self.debug = debug
        
        # Messages prédéfinis
        self.greetings = ["Hello", "Hi", "hi", "hello", "good morning"]
        self.intro_message = """Hello! I’m your medical assistant.
How can I help you today?
• Answer your medical questions
• Help you understand your symptoms
• Explain medical terms
• Provide information about treatments

Examples of questions:
• “I have bruises—what can I do?”
• “How can I treat back pain?”
• “What is a cyst or gallstones?”

Important: I do not replace professional medical advice. In case of emergency, please call 911."""

        self.help_message = """ My capabilities:

Medical questions: symptoms, diseases, treatments
Prevention: health advice, screening
Terminology: explanation of medical terms
Reminder: Always consult a professional for a diagnosis or treatment."""

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
            print(f"Erreur lors de la génération: {str(e)}", flush=True)
            return "Sorry, an error occurred. Could you please rephrase your question?"
    
    def _generate_with_context(self, question: str, context_chunks: List[str]) -> str:
        """Génère une réponse avec le contexte RAG"""
        context = "\n\n".join(context_chunks)
        
        # Debug: afficher le contexte récupéré
        if self.debug:
            self._log_context_info(context_chunks, question)
        
        # Vérifier si on a du contexte 
        if not context.strip():
            return """Information Not Found

I cannot find relevant information in my medical database to answer your question.

Suggestions:

Rephrase your question with different terms
Check the spelling
Ask a more general question
Consult a healthcare professional"""

        # Construire le prompt
        prompt = f"""You are an expert medical assistant. Answer the patient's question based solely on the context. 
        Follow all system rules. Ensure that all information is present in the context.
**MEDICAL CONTEXTE :**
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
                        "content": """I'm an expert medical assistant. IMPORTANT RULES:

USE ONLY the information from the provided context.
If the information is NOT in the context, clearly state, "This information is not available in my database."
Be precise, clear, and structured in your responses.
Always remind to consult a professional for a diagnosis.
In case of doubt, recommend a medical consultation.
Mention only the medications listed in the context.
"""
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
            if "consult" not in answer.lower() and "professional" not in answer.lower():
                answer += "\n\nReminder: Consult a healthcare professional for any diagnosis or treatment."
            
            return answer
        
        except Exception as e:            
            print(f"[ERREUR API Mistral] {e}", flush=True)
            #print(f"clé API utilisée : {os.getenv("MISTRAL_API_KEY")}")
            return "Communication error with the AI. Please try again."
    
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
            "what can you do", "help me", "how does it work?", "what do you do",
            "your capabilities", "what do you offer", "how do you work",
            "what are your functions", "help"
        ]
        question_lower = question.lower()
        return any(pattern in question_lower for pattern in help_patterns)
    
    def _check_sensitive_topics(self, question: str) -> Optional[str]:
        """Vérifie et gère les sujets sensibles"""
        question_lower = question.lower()
        
        # Urgences/suicide
        suicide_keywords = [
            "suicide", "kill myself", "end it all", "don't want to live anymore", 
            "die", "put an end", "don't want to continue", "stop everything"
        ]
        if any(keyword in question_lower for keyword in suicide_keywords):
            return """Immediate Help Available

If you are going through a crisis:

Suicide Écoute: 01 45 39 40 00 (24/7, free)
SOS Amitié: 09 72 39 40 50 (24/7)
3114: National suicide prevention hotline (free, 24/7)
Medical emergencies: 911

You are not alone. Professionals are there to help you."""

        # Urgences médicales
        emergency_keywords = [
            "urgent", "emergency", "serious", "severe pain", "can't breathe",
            "loss of consciousness", "malaise", "accident", "poisoning",
            "overdose", "hemorrhage"
        ]
        if any(keyword in question_lower for keyword in emergency_keywords):
            return """Medical Emergency Situation?

Contact immediately:

911 (USA emergency number)
Go to the nearest emergency room

I cannot assess medical emergencies. Do not wait, contact emergency services now."""

        # Médicaments contrôlés
        controlled_meds = [
            "antibiotique", "morphine", "anxiolytique", "antidépresseur",
            "benzodiazépine", "opiacé", "cortisone", "psychotrope"
        ]
        if ("comment obtenir" in question_lower or "où acheter" in question_lower) and \
           any(med in question_lower for med in controlled_meds):
            return """Prescription Medications

These medications require a medical prescription for your safety:

Consultation with a doctor
Evaluation of your health condition
Prescription tailored to your case
Why? To avoid:

Dangerous interactions
Overdose
Serious side effects
Dependence
Consult your doctor or pharmacist."""

        # Autodiagnostic dangereux
        dangerous_self_diagnosis = [
           "I have cancer", "it's a cancer", "tumor", "cancer", "heart attack", "stroke"]
        if any(phrase in question_lower for phrase in dangerous_self_diagnosis):
            return """Medical diagnosis required

The symptoms you describe absolutely require:

A professional medical examination
Additional tests
A diagnosis made by a doctor
Do not self-diagnose for serious conditions. Consult a doctor promptly or go to the emergency room if necessary."""

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
    
    
