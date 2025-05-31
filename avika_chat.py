import os
import re
import chromadb
import numpy as np
import google.generativeai as genai
from docx import Document
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_avika_titles():
    """Load titles and categories directly from Avika_Titles.docx"""
    docx_path = os.getenv("AVIKA_TITLES_PATH", "Avika_Titles.docx")
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Could not find Avika_Titles.docx at {docx_path}")
    doc = Document(docx_path)
    titles_data = []
    current_category = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: # Skip empty paragraphs
            continue

        if text.startswith("Category:"):
            current_category = text.replace("Category:", "").strip()
        elif text and current_category and not text.endswith("Titles"):
            clean_title = text
            # Enhanced cleaning for titles like "1. Title" or "1 Title"
            if text[0].isdigit() and ("." in text[:3] or " " in text[:3]):
                 # Split by first occurrence of '.' or ' '
                parts = re.split(r'[.\s]', text, 1)
                if len(parts) > 1:
                    clean_title = parts[1].strip()
            
            embedding_text = f"{clean_title} - {current_category if current_category else 'General Support'} - helps with {current_category.lower() if current_category else 'general'} related challenges and emotional support"
            
            titles_data.append({
                "title": clean_title,
                "category": current_category if current_category else "Uncategorized", # Default category
                "embedding_text": embedding_text
            })
    if not titles_data:
        print("Warning: No titles were loaded from the DOCX file. Check the file format and content.")
    return titles_data

def gemini_generate(prompt):
    """Generate response using Google Gemini API"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "API key not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.6,
                top_p=0.9,
                stop_sequences=["User:", "Avika:"]
            )
        )
        
        if not response.text:
            print("Empty response from Gemini API")
            return "I received an empty response. Please try again later."
            
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "I'm currently unable to connect to my services. Please check your internet connection and try again."

class AvikaChat:
    RECOMMENDATION_KEYWORDS = [
        "recommend", "suggest", "book", "video", "song", "audio", "music",
        "read", "watch", "listen", "resource", "tool", "help me find", "article"
    ]
    MIN_EMPATHY_TURNS = 1 # Ensure at least one turn of empathy by default
    SAFETY_CLASSIFIER_THRESHOLD = 0.7 # Threshold for offensive content detection

    def __init__(self, model, avika_titles, title_embeddings, chroma_collection):
        self.model = model
        self.avika_titles = avika_titles
        self.title_embeddings = title_embeddings
        self.chroma_collection = chroma_collection
        self.chat_history = []
        self.turn_number = 0
        # Constants for persona
        self.USER_PREFIX = "User: "
        self.AVIKA_PREFIX = "Avika: "
        self.INITIAL_GREETING = (
            "Hello! I'm Avika, a supportive chatbot here to listen and help you find helpful resources. "
            "Please know that I'm not a licensed therapist or a replacement for professional care. "
            "If you're in crisis or experiencing severe distress, please reach out to a mental health professional or a crisis hotline. "
            "What's on your mind today?"
        )
        self.chat_history.append(f"{self.AVIKA_PREFIX}{self.INITIAL_GREETING}")

        # Initialize safety classifier
        try:
            self.safety_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
            self.safety_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
            print("✅ Safety classifier model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load safety classifier model. Falling back to keyword-only detection. Error: {e}")
            self.safety_tokenizer = None
            self.safety_model = None

    def _construct_empathy_prompt(self, user_input, context_str, recent_history, 
                                is_short_input: bool, no_strong_themes: bool, is_stuck_early: bool):
        guidance_on_ambiguity = ""
        if (is_short_input and no_strong_themes) or is_stuck_early:
            guidance_on_ambiguity = (
                "\n\nIMPORTANT GUIDANCE FOR THIS TURN:\n"
                "The user's input is brief or the conversation needs more substance to be truly helpful. "
                "Your priority is to gently encourage the user to share more. You might say something like: \n"
                "- \"I'd love to understand a bit more — can you tell me what's been weighing on you the most lately?\"\n"
                "- \"To help me understand better, could you tell me a bit more about what's on your mind?\"\n"
                "- \"I'm here to listen. Sometimes it helps to just start talking about what's on your mind, no matter how small it seems.\"\n"
                "Make sure your response is a question that prompts them to elaborate."
            )
        else:
            guidance_on_ambiguity = (
                "\n\nGUIDANCE FOR THIS TURN:\n"
                "- Reflect on their feelings.\n"
                "- Offer a supportive statement.\n"
                "- If it feels natural, ask a gentle follow-up question to understand better."
            )

        return f\"\"\"
        You are Avika, a mobile mental health chatbot. Your primary goal is to listen, understand, and respond with empathy.
        Keep responses concise (around 3-4 sentences).
        I am not a human, nor a replacement for professional care. If the user mentions serious harm to themselves or others, gently guide them to seek professional help.

        CONVERSATION SO FAR:
        {chr(10).join(recent_history)}

        CURRENT USER MESSAGE:
        {user_input}

        [EMOTIONAL THEMES DETECTED FROM KNOWLEDGE BASE]
        {context_str if context_str else "No specific themes detected yet."}
        {guidance_on_ambiguity}

        Focus on making the user feel heard and understood. DO NOT recommend any resources or titles yet.
        Avika (respond according to the guidance for this turn):
        \"\"\".strip()

    def _construct_recommendation_prompt(self, user_input, title_list_str, emotional_context_for_titles, reflection_summary):
        # Added reflection_summary and improved "no titles" message
        no_titles_guidance = (
            "I wasn't able to find a specific resource that perfectly matches what you've described right now. "
            "Sometimes, telling me a bit more about what you're looking for or how you're feeling can help me find something more suitable. "
            "Would you like to try describing your needs differently, or perhaps explore a general topic?"
        )
        
        return f"""
        You are Avika, a mobile mental health chatbot.
        I am not a human, nor a replacement for professional care.

        REFLECTION ON OUR CONVERSATION:
        {reflection_summary}

        USER'S CURRENT MESSAGE:
        {user_input}
        
        TASK:
        Based on the user's needs and the emotional context, if there are relevant titles from the list below, recommend ONE.
        If no titles seem appropriate or the list is empty, use the 'NO TITLES GUIDANCE'.

        CRITICAL RULES FOR RECOMMENDATIONS:
        - You may ONLY recommend titles from the [AVAILABLE TITLES] list.
        - Start with 1-2 sentences of continued empathy related to their situation.
        - Then, if recommending, say "I recommend trying out '[EXACT TITLE]' from our '[EXACT CATEGORY]' resources."
        - If no titles are suitable, use the 'NO TITLES GUIDANCE' text.
        - Keep the entire response to 4-5 sentences maximum.
        - Always refer to yourself in the first person ("I recommend...").
        - If you detect any concerning content about harm at any point, defer to crisis resources.

        [AVAILABLE TITLES FROM AVIKA_TITLES.DOCX]
        {title_list_str if title_list_str else "No titles were found based on the current context."}

        [EMOTIONAL CONTEXT FROM FULL CONVERSATION]
        {emotional_context_for_titles}
        
        [NO TITLES GUIDANCE]
        {no_titles_guidance}

        Avika (respond naturally, recommend ONE title if suitable, otherwise use the guidance):
        """.strip()

    def check_safety_concerns(self, text_to_check, conversation_history=None):
        """Check for safety concerns in user input, considering recent history and a classifier.
        
        Note: Distinguishing humor from genuine distress is complex and this model may not always be accurate.
        """
        # Combine current text with recent user messages for better context
        full_context_text = text_to_check
        if conversation_history:
            user_messages = [msg.replace(self.USER_PREFIX, "") for msg in conversation_history if msg.startswith(self.USER_PREFIX)]
            # Consider last 3 user messages plus current input
            context_window = " ".join(user_messages[-3:] + [text_to_check])
            full_context_text = context_window # Keep case for classifier, convert to lower for keywords
        else:
            full_context_text = text_to_check

        full_context_text_lower = full_context_text.lower()

        # 1. Keyword-based detection (existing logic)
        self_harm_indicators = [
            "kill myself", "suicide", "end my life", "hurt myself", "self harm",
            "don't want to live", "want to die", "cut myself", "overdose",
            "take my own life", "self-harm", "harm myself", "no point in anything"
        ]
        harm_others_indicators = [
            "kill them", "hurt them", "want to hurt", "want to kill",
            "harm others", "murder", "revenge", "make them pay", "shoot",
            "violence", "attack", "destroy them", "hurt people"
        ]
        
        found_self_harm_keyword = any(indicator in full_context_text_lower for indicator in self_harm_indicators)
        found_harm_others_keyword = any(indicator in full_context_text_lower for indicator in harm_others_indicators)

        # 2. Classifier-based detection
        classifier_triggered_concern = False
        if self.safety_model and self.safety_tokenizer:
            try:
                inputs = self.safety_tokenizer(full_context_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.safety_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                # Model: cardiffnlp/twitter-roberta-base-offensive
                # Labels: 0 -> not offensive, 1 -> offensive
                offensive_score = scores[0][1].item() 
                if offensive_score > self.SAFETY_CLASSIFIER_THRESHOLD:
                    classifier_triggered_concern = True
                    print(f"Classifier detected potential concern. Offensive score: {offensive_score:.2f}")
            except Exception as e:
                print(f"⚠️ Error during safety classification: {e}. Relying on keyword detection for this turn.")
        
        # Combine keyword and classifier results
        # For determining the *type* of crisis (self-harm vs. harm-others), we'll rely on keywords for now,
        # as the 'offensive' classifier is general.
        is_critical_concern = (found_self_harm_keyword or found_harm_others_keyword or classifier_triggered_concern)
        
        if is_critical_concern:
            # Log escalation for human review (simulated)
            alert_reason = []
            if found_self_harm_keyword: alert_reason.append("self-harm keywords")
            if found_harm_others_keyword: alert_reason.append("harm-others keywords")
            if classifier_triggered_concern and not (found_self_harm_keyword or found_harm_others_keyword):
                 alert_reason.append(f"classifier (score: {offensive_score:.2f})") # type: ignore
            
            print(f"SAFETY ALERT: Potential crisis detected (Reason: {', '.join(alert_reason)}). User history hint: {full_context_text[-100:]}. Flag for human review.")
            # Determine response type based on keywords primarily for specific crisis lines
            return True, self._get_crisis_response(found_self_harm_keyword, found_harm_others_keyword)
            
        return False, None

    def _get_crisis_response(self, is_self_harm, is_harm_others):
        # Rewritten to be shorter, more empathetic, and context-aware
        if is_self_harm:
            response = (
                "It sounds like you're going through something incredibly painful, and I want you to know you're not alone. "
                "For the kind of support you need right now, it's best to talk with a professional. "
                "Can I share some resources that can help immediately?"
            )
            # For now, we'll append resources directly. Ideally, the bot would ask first.
            response += (
                "\n\nPlease reach out: \n"
                "• 988 Suicide & Crisis Lifeline (US): Call or text 988\n"
                "• Crisis Text Line: Text HOME to 741741\n"
                "Your life matters."
            )
        elif is_harm_others:
            response = (
                "I'm concerned by what you're expressing. When thoughts of harming others come up, it's important to talk to someone who can help. "
                "Can I provide you with some resources for professional support?"
            )
            response += (
                 "\n\nPlease reach out for support: \n"
                "• iCall Helpline (India): Call 1800-599-0019\n"
                "• Contact your local mental health services."
            )
        else: # Generic fallback, though one flag should always be true if this is called
            response = (
                "It sounds like you are in a lot of distress. It's important to talk to a mental health professional. "
                "Please reach out to a crisis line or mental health service."
            )
        return response

    def _get_emotional_context(self, user_input):
        """Get relevant emotional context using semantic search (internal helper method)"""
        # Combine recent context with current input
        recent_messages = [msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)]
        context_query = " ".join(recent_messages + [user_input])
        
        # Get emotional context from main corpus
        results = self.chroma_collection.query(
            query_texts=[context_query],
            n_results=3  # Get top 3 relevant snippets
        )
        
        # Extract emotional themes and key points
        themes = []
        if results and results['documents'] and results['metadatas']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                source = meta.get('source', 'Unknown Source').replace(".docx", "").replace("_", " ")
                # Take first 150 chars of content for key points
                snippet = doc[:150] + "..." if len(doc) > 150 else doc
                themes.append(f"Theme from {source}: {snippet}")
        
        return themes
        
    def _get_relevant_titles(self, emotional_context, content_type=None, top_k=5):
        """Get semantically relevant titles based on emotional context (internal helper method)"""
        if not self.title_embeddings: # Handle case with no titles/embeddings
            return []

        context_embedding = self.model.encode(emotional_context)
        
        # Get all title embeddings as a matrix
        title_embeddings_matrix = np.array(list(self.title_embeddings.values()))
        
        # Calculate cosine similarity (vectorized)
        similarities = np.dot(title_embeddings_matrix, context_embedding.T) / (
            np.linalg.norm(title_embeddings_matrix, axis=1) * np.linalg.norm(context_embedding)
        )
        
        # Get indices sorted by similarity (descending)
        original_indices = list(self.title_embeddings.keys())
        sorted_indices_of_similarities = np.argsort(similarities)[::-1]
        
        matching_titles = []
        for i in sorted_indices_of_similarities:
            original_title_idx = original_indices[i]
            title_info = self.avika_titles[original_title_idx]
            
            if content_type:
                if (content_type.lower() in title_info["category"].lower() or 
                    content_type.lower() in title_info["title"].lower()):
                    matching_titles.append(title_info)
            else:
                matching_titles.append(title_info)
            
            if len(matching_titles) >= top_k:
                break
        
        return matching_titles

    def _is_requesting_recommendation(self, user_input_lower: str) -> bool:
        """Check if user input explicitly asks for a recommendation."""
        return any(keyword in user_input_lower for keyword in self.RECOMMENDATION_KEYWORDS)

    def _has_sufficient_context_for_proactive_recommendation(self) -> bool:
        """Heuristic to check if conversation context might be rich enough for a recommendation."""
        if not self.chat_history:
            return False
        
        full_user_context = " ".join([
            msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)
        ])
        if not full_user_context.strip(): # Ensure there's actual user input
            return False

        # Perform a light check for any relevant title
        potential_titles = self._get_relevant_titles(
            emotional_context=full_user_context,
            top_k=1 # We only need to know if at least one potentially relevant title exists
        )
        return bool(potential_titles)

    def _get_content_preference(self, user_input_lower: str) -> Optional[str]:
        """Detect content type preference from user input."""
        if any(word in user_input_lower for word in ["video", "watch", "youtube"]):
            return "video"
        elif any(word in user_input_lower for word in ["book", "read", "reading", "article"]):
            return "book"
        elif any(word in user_input_lower for word in ["music", "song", "listen", "audio"]):
            return "music"
        return None

    def _generate_reflection_summary(self) -> str:
        """Generate a reflective summary based on the conversation so far."""
        full_user_context = " ".join([
            msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)
        ])
        emotional_themes = self._get_emotional_context(full_user_context) # _get_emotional_context uses history + its input argument effectively
        primary_theme_info = emotional_themes[0] if emotional_themes else "what you are currently experiencing"
        # Refine theme extraction to be more concise if it's a long snippet
        if "Theme from" in primary_theme_info and ": " in primary_theme_info:
            theme_content = primary_theme_info.split(": ", 1)[1]
            if len(theme_content) > 100:
                theme_content = theme_content[:100] + "..."
            reflection_text = f"you're dealing with themes like '{theme_content}'"
        else:
            reflection_text = primary_theme_info

        return f"From our conversation, it seems like {reflection_text}. I'd like to offer something that might be helpful."

    def chat(self, user_input, max_turns_for_history=4):
        user_input_lower = user_input.lower()

        # Initial safety check on current input + recent history
        has_concerns, crisis_response = self.check_safety_concerns(user_input, self.chat_history[-5:])
        if has_concerns:
            self.chat_history.append(f"{self.USER_PREFIX}{user_input}")
            self.chat_history.append(f"{self.AVIKA_PREFIX}{crisis_response}")
            return crisis_response

        self.chat_history.append(f"{self.USER_PREFIX}{user_input}")
        recent_history_for_prompt = self.chat_history[-(max_turns_for_history*2):]

        attempt_recommendation = False
        if self.turn_number >= self.MIN_EMPATHY_TURNS:
            if self._is_requesting_recommendation(user_input_lower):
                attempt_recommendation = True
            elif self._has_sufficient_context_for_proactive_recommendation():
                # Check if we *just* had an ambiguity prompt, if so, maybe wait one more turn
                # This is to avoid asking for more info and then immediately recommending
                if not (self.chat_history[-2].startswith(self.AVIKA_PREFIX) and 
                        ("understand a bit more" in self.chat_history[-2] or "tell me a bit more" in self.chat_history[-2])):
                    attempt_recommendation = True
        
        response_text = ""

        if attempt_recommendation:
            # --- Recommendation Phase ---
            # Safety check on full user context before making recommendations
            full_user_dialogue_for_safety = " ".join([msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)])
            has_concerns, crisis_response = self.check_safety_concerns(full_user_dialogue_for_safety, self.chat_history)
            if has_concerns:
                if not self.chat_history[-1].startswith(self.AVIKA_PREFIX) or crisis_response not in self.chat_history[-1]:
                    self.chat_history.append(f"{self.AVIKA_PREFIX}{crisis_response}")
                self.turn_number += 1
                return crisis_response

            reflection_summary = self._generate_reflection_summary()
            content_preference = self._get_content_preference(user_input_lower)
            
            # Use the full user dialogue for title matching context
            emotional_context_for_titles = " ".join([
                msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)
            ])

            matching_titles = self._get_relevant_titles(
                emotional_context=emotional_context_for_titles,
                content_type=content_preference,
                top_k=5
            )
            
            title_list_str = "\n".join([
                f"- {t['title']} (Category: {t['category']})"
                for t in matching_titles
            ]) if matching_titles else "" 

            prompt = self._construct_recommendation_prompt(user_input, title_list_str, emotional_context_for_titles, reflection_summary)
            llm_response = gemini_generate(prompt)

            has_concerns_llm, crisis_response_llm = self.check_safety_concerns(llm_response, self.chat_history)
            response_text = crisis_response_llm if has_concerns_llm else llm_response
        
        else:
            # --- Empathy Phase ---
            emotional_themes = self._get_emotional_context(user_input)
            context_str = "\n\n".join(emotional_themes)
            
            # Assess ambiguity for empathy prompt
            is_short_input = len(user_input.split()) < 4
            no_strong_themes = not emotional_themes
            # is_stuck_early: True if it's turn 1 (second user interaction) and we don't have enough for proactive rec
            is_stuck_early = (self.turn_number == 1 and not self._has_sufficient_context_for_proactive_recommendation())
            
            prompt = self._construct_empathy_prompt(
                user_input, 
                context_str, 
                recent_history_for_prompt, 
                is_short_input,
                no_strong_themes,
                is_stuck_early
            )
            llm_response = gemini_generate(prompt)

            has_concerns_llm, crisis_response_llm = self.check_safety_concerns(llm_response, self.chat_history)
            response_text = crisis_response_llm if has_concerns_llm else llm_response
        
        self.chat_history.append(f"{self.AVIKA_PREFIX}{response_text}")
        self.turn_number += 1
        return response_text

def main():
    """Main function for running the chat application"""
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set. Please set it and try again.")
        return
    
    try:
        s_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        avika_titles_data = load_avika_titles()
        title_embeddings_data = {
            idx: s_model.encode(title["embedding_text"]) 
            for idx, title in enumerate(avika_titles_data)
        }
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_storage")
        persistent_client = chromadb.PersistentClient(path=chroma_db_path)
        chroma_collection_instance = persistent_client.get_or_create_collection(name="docx_chunks")

        if not avika_titles_data:
            print("❌ Error: No titles loaded. Please check Avika_Titles.docx and its path.")
            return

        avika_chatbot = AvikaChat(
            model=s_model,
            avika_titles=avika_titles_data,
            title_embeddings=title_embeddings_data,
            chroma_collection=chroma_collection_instance
        )
        
        # Use the initial greeting from the AvikaChat class instance
        print(f"\n🧠 {avika_chatbot.INITIAL_GREETING}")
        print("(Type 'bye' to stop the chat.)\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit", "bye", "ok thanks", "okay thank you", "thanks bye"]:
                    print("🧠 Avika: Take care! I'm always here if you need to talk.")
                    break

                response = avika_chatbot.chat(user_input)
                print(f"\n🧠 Avika: {response}\n")
            except KeyboardInterrupt:
                print("\n🧠 Avika: Take care! I'm always here if you need to talk.")
                break
            except Exception as e:
                import traceback
                print(f"\n❌ Unexpected Error: {str(e)}")
                traceback.print_exc() 
                print("🧠 Avika: I'm having a problem right now. Please try again, or restart if it continues.\n")

    except FileNotFoundError as fnf_error:
        print(f"❌ Error: {str(fnf_error)}")
    except ValueError as val_error:
        print(f"❌ Configuration Error: {str(val_error)}")
    except Exception as e:
        import traceback
        print(f"\n❌ Critical Initialization Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 