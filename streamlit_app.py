import streamlit as st

# --- Page Configuration (MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Avika Chat", page_icon="��", layout="wide")

import os
import re
# import requests # No longer needed for HTTP requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import numpy as np
import google.generativeai as genai
from docx import Document
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import traceback # For better error logging

load_dotenv() # Load environment variables

# --- Global Initialization of Avika Components ---
# This section will run once when the Streamlit app starts.

INITIALIZATION_ERROR = None
S_MODEL = None
AVIKA_TITLES_DATA = []
TITLE_EMBEDDINGS_DATA = {}
QDRANT_CLIENT = None
SAFETY_TOKENIZER = None
SAFETY_MODEL = None

# --- Utility Functions (Moved from avika_chat.py) ---

@st.cache_data
def load_avika_titles_cached(): # Renamed from load_avika_titles to avoid conflict for now, will adjust later
    """Load titles and categories directly from Avika_Titles.docx (Cached)"""
    # Use a default path if the environment variable is not set
    st.info("Loading Avika titles...") # Moved info here to show once on cache compute
    docx_path = os.getenv("AVIKA_TITLES_PATH", "Avika_Titles.docx")
    if not os.path.exists(docx_path):
        st.error(f"CRITICAL ERROR: Could not find Avika_Titles.docx at {docx_path}. Title recommendations will not work.")
        return [] 
    
    try:
        doc = Document(docx_path)
        titles_data = []
        current_category = None
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text: 
                continue

            if text.startswith("Category:"):
                current_category = text.replace("Category:", "").strip()
            elif text and current_category and not text.endswith("Titles"):
                clean_title = text
                if text[0].isdigit() and ("." in text[:3] or " " in text[:3]):
                    parts = re.split(r'[.\\s]', text, 1)
                    if len(parts) > 1:
                        clean_title = parts[1].strip()
                
                embedding_text = f"{clean_title} - {current_category if current_category else 'General Support'} - helps with {current_category.lower() if current_category else 'general'} related challenges and emotional support"
                
                titles_data.append({
                    "title": clean_title,
                    "category": current_category if current_category else "Uncategorized",
                    "embedding_text": embedding_text
                })
        if not titles_data:
            st.warning("Warning: No titles were loaded from the DOCX file. Check the file format and content. Recommendations might be limited.")
        else:
            st.success(f"✅ Loaded {len(titles_data)} Avika titles.")
        return titles_data
    except Exception as e:
        st.error(f"Error loading Avika titles from {docx_path}: {e}")
        return []

def gemini_generate(prompt: str) -> str:
    """Generate response using Google Gemini API"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. Avika cannot generate responses.")
        return "API key not configured. I cannot process your request right now."
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=400, # Increased slightly
                temperature=0.6, # Maintained
                top_p=0.9,       # Maintained
                stop_sequences=["User:", "Avika:"] # Maintained
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.text.strip()
        else:
            st.warning("Gemini API returned no content or an unexpected response structure.")
            if response.prompt_feedback:
                 st.warning(f"Prompt feedback: {response.prompt_feedback}")
                 if response.prompt_feedback.block_reason:
                     return f"My ability to respond was limited due to: {response.prompt_feedback.block_reason_message}. Please try rephrasing."

            return "I'm having a bit of trouble formulating a response right now. Could you try rephrasing or asking again in a moment?"
        
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return "I'm currently unable to connect to my core services. Please check your internet connection and try again later."

@st.cache_resource
def load_sentence_transformer_model():
    """Loads the SentenceTransformer model (Cached Resource)."""
    st.info("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        st.success("✅ SentenceTransformer model initialized.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load SentenceTransformer model: {e}")
        return None

@st.cache_data
def generate_title_embeddings_cached(_sentence_model, _avika_titles_data):
    """Generates title embeddings using a cached sentence model and title data (Cached Data)."""
    if not _avika_titles_data or not _sentence_model:
        if not _avika_titles_data and _sentence_model:
             st.warning("⚠️ No Avika titles loaded, cannot generate title embeddings.")
        elif not _sentence_model:
             st.error("❌ Sentence model not loaded. Cannot generate title embeddings.")
        return {}
    
    st.info("Generating title embeddings...")
    try:
        embeddings = {
            idx: _sentence_model.encode(title["embedding_text"])
            for idx, title in enumerate(_avika_titles_data)
        }
        st.success(f"✅ Title embeddings generated for {len(embeddings)} titles.")
        return embeddings
    except Exception as e:
        st.error(f"❌ Failed to generate title embeddings: {e}")
        return {}

@st.cache_resource
def load_safety_classifier_model():
    """Loads the safety classifier model and tokenizer (Cached Resource)."""
    st.info("Loading safety classifier model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
        st.success("✅ Safety classifier model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        st.warning(f"⚠️ Warning: Could not load safety classifier model. Offensive content detection might be impaired. Error: {e}")
        return None, None

@st.cache_resource
def get_qdrant_client_cached():
    """Initializes and returns a Qdrant client (Cached Resource)."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        st.warning("QDRANT_URL not set. Document context search will be unavailable.")
        return None
    
    st.info(f"Connecting to Qdrant at {qdrant_url}...")
    try:
        client = None
        if qdrant_api_key:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=20)
        else:
            client = QdrantClient(url=qdrant_url, timeout=20)
        
        client.get_collections() # Reverted to get_collections() for connection verification
        st.success(f"Successfully connected to Qdrant at {qdrant_url}.")
        return client
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}. Document context search will be unavailable.")
        return None

# --- AvikaChat Class (Moved and adapted from avika_chat.py) ---
class AvikaChat:
    MIN_EMPATHY_TURNS = 1
    SAFETY_CLASSIFIER_THRESHOLD = 0.7
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_DOC_COLLECTION_NAME", "avika_doc_chunks") # Use env var
    VECTOR_SIZE = 384 # For all-MiniLM-L6-v2

    def __init__(self, sentence_model, avika_titles, title_embeddings, qdrant_client, safety_tokenizer, safety_model):
        self.s_model = sentence_model
        self.avika_titles = avika_titles
        self.title_embeddings = title_embeddings # This is a dict {idx: embedding_vector}
        self.chat_history = []
        self.turn_number = 0
        self.recommendation_offered_in_last_turn = False
        self.avika_just_asked_for_clarification = False
        self.avika_just_said_no_titles_found = False
        self.USER_PREFIX = "User: "
        self.AVIKA_PREFIX = "Avika: "
        self.INITIAL_GREETING = (
            "Hello! I'm Avika, a supportive chatbot here to listen and help you find helpful resources. "
            "Please know that I'm not a licensed therapist or a replacement for professional care. "
            "If you're in crisis or experiencing severe distress, please reach out to a mental health professional or a crisis hotline. "
            "What's on your mind today?"
        )

        self.qdrant_client = qdrant_client
        self.safety_tokenizer = safety_tokenizer
        self.safety_model = safety_model

        # Verify Qdrant collection existence if client is available
        if self.qdrant_client:
            try:
                # Check if the document collection exists
                self.qdrant_client.get_collection(collection_name=self.QDRANT_COLLECTION_NAME)
                st.info(f"AvikaChat: Qdrant collection '{self.QDRANT_COLLECTION_NAME}' found.")
            except Exception as e_get_coll: 
                # Broader exception for collection not found or other Qdrant issues during this check
                st.warning(f"AvikaChat: Qdrant collection '{self.QDRANT_COLLECTION_NAME}' not found or error accessing it: {e_get_coll}. Attempting to create/recreate...")
                try:
                    self.qdrant_client.recreate_collection(
                        collection_name=self.QDRANT_COLLECTION_NAME,
                        vectors_config=models.VectorParams(size=self.VECTOR_SIZE, distance=models.Distance.COSINE)
                    )
                    st.success(f"AvikaChat: Qdrant collection '{self.QDRANT_COLLECTION_NAME}' created/recreated.")
                except Exception as e_create_coll:
                    st.error(f"AvikaChat: Failed to create/recreate Qdrant collection '{self.QDRANT_COLLECTION_NAME}': {e_create_coll}. Document context search will be impaired.")
        else:
            st.warning("AvikaChat: Qdrant client not provided during initialization. Document context search will be unavailable.")

        if not self.safety_model or not self.safety_tokenizer:
            st.warning("AvikaChat: Safety model/tokenizer not provided or failed to load. Offensive content detection might be impaired.")

    def _construct_empathy_prompt(self, user_input, context_str, recent_history,
                                is_short_input: bool, no_strong_themes: bool, is_stuck_early: bool,
                                user_is_resistant: bool = False):
        guidance_on_ambiguity = ""
        prompt_variation_guidance = "\n- Try to vary your empathetic statements. Avoid starting with the exact same phrase in consecutive turns."

        if user_is_resistant:
            avika_last_response_was_recommendation = False
            if len(self.chat_history) >= 2 and self.chat_history[-2].startswith(self.AVIKA_PREFIX): 
                if ("I recommend trying out" in self.chat_history[-2] or "from our" in self.chat_history[-2]):
                    avika_last_response_was_recommendation = True
            
            specific_resistance_guidance = ""
            if avika_last_response_was_recommendation:
                specific_resistance_guidance = (
                    "The user seems to be questioning or skeptical about the recommendation you just made. "
                    "Acknowledge their question/skepticism directly. You might say something like:\n"
                    "- \"That's a fair question. You're wondering why this particular resource might be helpful for what you're going through, is that right?\"\n"
                    "- \"I hear you. It makes sense to want to know if a suggestion is really a good fit before spending time on it. What are your initial thoughts or concerns about it?\"\n"
                    "- \"It sounds like you're not quite convinced that suggestion is for you. Help me understand a bit more – what feels off, or what kind of support were you hoping for at this moment?\"\n"
                    "Your goal is to understand their reservations about the specific suggestion and validate their inquiry."
                )
            else:
                specific_resistance_guidance = (
                    "The user is expressing feelings of hopelessness or broader resistance to help. "
                    "Your priority is to validate these feelings deeply. Acknowledge that things feel tough or pointless for them right now. "
                    "DO NOT offer solutions or try to cheer them up. Instead, you might say something like:\n"
                    "- \"It sounds incredibly frustrating when it feels like nothing is working.\"\n"
                    "- \"I understand why you might feel skeptical or that things are pointless right now.\"\n"
                    "- \"It's okay to feel that way. Can you tell me more about what makes it feel so overwhelming?\"\n"
                    "Let them know their feelings are heard without trying to change them immediately. Focus on gentle exploration if appropriate."
                )
            
            guidance_on_ambiguity = (
                "\n\nIMPORTANT GUIDANCE FOR THIS TURN (USER IS RESISTANT):\n"
                f"{specific_resistance_guidance}"
                f"{prompt_variation_guidance}"
            )
        elif (is_short_input and no_strong_themes) or is_stuck_early:
            guidance_on_ambiguity = (
                "\n\nIMPORTANT GUIDANCE FOR THIS TURN (NEEDS MORE INFO):\n"
                "The user's input is brief or the conversation needs more substance to be truly helpful. "
                "Your priority is to gently encourage the user to share more. You might say something like: \n"
                "- \"I'd love to understand a bit more — can you tell me what's been weighing on you the most lately?\n"
                "- \"To help me understand better, could you tell me a bit more about what's on your mind?\"\n"
                "- \"I'm here to listen. Sometimes it helps to just start talking about what's on your mind, no matter how small it seems.\"\n"
                "Make sure your response is a question that prompts them to elaborate."
                f"{prompt_variation_guidance}"
            )
        else:
            guidance_on_ambiguity = (
                "\n\nGUIDANCE FOR THIS TURN (GENERAL EMPATHY):\n"
                "- Reflect on their feelings.\n"
                "- Offer a supportive statement that validates their experience.\n"
                "- If it feels natural, ask a gentle follow-up question to understand better or invite them to share more."
                f"{prompt_variation_guidance}"
            )

        return f"""
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
        """

    def _construct_recommendation_prompt(self, user_input, title_list_str, emotional_context_for_titles, reflection_summary):
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

    def check_safety_concerns(self, text_to_check: str, conversation_history: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
        full_context_text = text_to_check
        if conversation_history:
            user_messages = [msg.replace(self.USER_PREFIX, "") for msg in conversation_history if msg.startswith(self.USER_PREFIX)]
            context_window = " ".join(user_messages[-3:] + [text_to_check])
            full_context_text = context_window
        
        full_context_text_lower = full_context_text.lower()

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

        classifier_triggered_concern = False
        offensive_score_val = 0.0 # Define it here for wider scope
        if self.safety_model and self.safety_tokenizer:
            try:
                inputs = self.safety_tokenizer(full_context_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.safety_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                offensive_score_val = scores[0][1].item() 
                if offensive_score_val > self.SAFETY_CLASSIFIER_THRESHOLD:
                    classifier_triggered_concern = True
                    st.sidebar.warning(f"Safety: Classifier detected potential concern. Score: {offensive_score_val:.2f}")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error during safety classification: {e}. Relying on keyword detection.")
        
        is_critical_concern = (found_self_harm_keyword or found_harm_others_keyword or classifier_triggered_concern)
        
        if is_critical_concern:
            alert_reason = []
            if found_self_harm_keyword: alert_reason.append("self-harm keywords")
            if found_harm_others_keyword: alert_reason.append("harm-others keywords")
            if classifier_triggered_concern and not (found_self_harm_keyword or found_harm_others_keyword):
                 alert_reason.append(f"classifier (score: {offensive_score_val:.2f})")
            
            st.sidebar.error(f"SAFETY ALERT: Potential crisis (Reason: {', '.join(alert_reason)}). User: {full_context_text[-100:]}")
            return True, self._get_crisis_response(found_self_harm_keyword, found_harm_others_keyword)
            
        return False, None

    def _get_crisis_response(self, is_self_harm: bool, is_harm_others: bool) -> str:
        if is_self_harm:
            response = (
                "It sounds like you're going through something incredibly painful, and I want you to know you're not alone. "
                "For the kind of support you need right now, it's best to talk with a professional. "
                "Can I share some resources that can help immediately?"
                "\n\nPlease reach out: \n"
                "• 988 Suicide & Crisis Lifeline (US): Call or text 988\n"
                "• Crisis Text Line: Text HOME to 741741\n"
                "Your life matters."
            )
        elif is_harm_others:
            response = (
                "I'm concerned by what you're expressing. When thoughts of harming others come up, it's important to talk to someone who can help. "
                "Can I provide you with some resources for professional support?"
                "\n\nPlease reach out for support: \n"
                "• iCall Helpline (India): Call 1800-599-0019\n" # Example, adjust as needed
                "• Contact your local mental health services."
            )
        else: # Generic fallback for classifier-only trigger
            response = (
                "It sounds like you are in a lot of distress, and I'm concerned. It's important to talk to a mental health professional. "
                "Please reach out to a crisis line or mental health service like 988 (US) or Crisis Text Line (text HOME to 741741)."
            )
        return response

    def _get_emotional_context(self, current_user_input_for_query: str) -> List[str]:
        if not self.qdrant_client:
            st.sidebar.warning("Qdrant client not available. Cannot get emotional context.")
            return []

        # Use the full chat history of the user for context query, plus current input
        user_messages_from_history = [msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)]
        context_query = " ".join(user_messages_from_history + [current_user_input_for_query])
        
        if not context_query.strip():
            return []

        try:
            query_embedding = self.s_model.encode(context_query).tolist()
            hits = self.qdrant_client.search(
                collection_name=self.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=3,
                with_payload=True 
            )
            themes = []
            if hits:
                for hit in hits:
                    payload = hit.payload
                    if payload:
                        source = payload.get('source', 'Unknown Source').replace(".docx", "").replace("_", " ")
                        text_snippet = payload.get('text_chunk', '') 
                        snippet = text_snippet[:150] + "..." if len(text_snippet) > 150 else text_snippet
                        # Include score in a consistent way
                        themes.append(f"Theme from {source}: {snippet} (Score: {hit.score:.2f})") 
            return themes
        except Exception as e:
            st.sidebar.error(f"Error during Qdrant search for context: {e}")
            return []

    def _get_relevant_titles(self, emotional_context: str, content_type: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.avika_titles or not self.title_embeddings or not self.s_model:
            st.sidebar.warning("Titles, embeddings, or sentence model not available. Cannot get relevant titles.")
            return []
        if not emotional_context.strip(): # Avoid encoding empty strings
             st.sidebar.info("_get_relevant_titles: emotional_context was empty.")
             return []

        try:
            context_embedding = self.s_model.encode(emotional_context)
            
            title_indices = list(self.title_embeddings.keys()) 
            if not title_indices:
                st.sidebar.warning("_get_relevant_titles: No title_embeddings keys found.")
                return [] 
            
            title_embeddings_matrix = np.array([self.title_embeddings[idx] for idx in title_indices])

            # Ensure matrix and embedding are 2D for dot product and norm calculation
            if title_embeddings_matrix.ndim == 1: 
                title_embeddings_matrix = title_embeddings_matrix.reshape(1, -1)
            if context_embedding.ndim == 1:
                context_embedding = context_embedding.reshape(1, -1)

            if title_embeddings_matrix.shape[0] == 0 or context_embedding.shape[0] == 0:
                st.sidebar.warning("_get_relevant_titles: Empty title_embeddings_matrix or context_embedding after reshape.")
                return [] 
            
            # Calculate cosine similarities
            norm_matrix = np.linalg.norm(title_embeddings_matrix, axis=1)
            norm_context = np.linalg.norm(context_embedding.T, axis=0) # Correct norm for context_embedding

            # Denominator for cosine similarity
            denominator = norm_matrix * norm_context
            # Replace zero denominators with a very small number to avoid division by zero, or handle as no similarity
            denominator[denominator == 0] = 1e-9 # Avoid division by zero warning, effectively zero similarity
            
            similarities = np.dot(title_embeddings_matrix, context_embedding.T).flatten() / denominator
            similarities = np.asarray(similarities).flatten() # Ensure it's a flat array
            
            # Get indices sorted by similarity (descending)
            sorted_indices_of_similarities = np.argsort(similarities)[::-1] 
            
            matching_titles = []
            for i in sorted_indices_of_similarities:
                # Check if similarity is meaningful (e.g. > 0.3, depends on your data)
                if similarities[i] < 0.3: # Heuristic threshold to avoid very weak matches
                    continue

                original_title_idx = title_indices[i] 
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
        except Exception as e:
            st.sidebar.error(f"Error in _get_relevant_titles: {e}")
            return []

    def _llm_is_requesting_recommendation(self, user_input: str, conversation_history_str: str) -> bool:
        prompt = f"""
        Analyze the following user message within the broader conversation. The user is interacting with Avika, a mental health chatbot.
        Does the user's LATEST MESSAGE primarily ask for a resource, suggestion, book, video, song, audio, or similar actionable help?
        Respond with only one word: YES or NO.

        CONVERSATION HISTORY (summary):
        {conversation_history_str}

        USER'S LATEST MESSAGE:
        {user_input}

        Analysis (YES or NO):
        """
        response = gemini_generate(prompt)
        return response.strip().upper() == "YES"

    def _llm_is_user_resistant(self, user_input: str, conversation_history_str: str) -> bool:
        prompt = f"""
        Analyze the following user message within the broader conversation. The user is interacting with Avika, a mental health chatbot.
        Does the user's LATEST MESSAGE express strong feelings of hopelessness, skepticism about getting help, dismiss previous suggestions, or directly state that help/suggestions are not working (e.g., saying 'nothing helps', 'it's pointless', 'I don't want to')?
        Respond with only one word: YES or NO.

        CONVERSATION HISTORY (summary):
        {conversation_history_str}

        USER'S LATEST MESSAGE:
        {user_input}

        Analysis (YES or NO):
        """
        response = gemini_generate(prompt)
        return response.strip().upper() == "YES"
    
    def _has_sufficient_context_for_proactive_recommendation(self) -> bool:
        if not self.chat_history:
            return False
        # Consider user messages from the entire history for this check
        full_user_context = " ".join([
            msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)
        ])
        if not full_user_context.strip(): 
            return False
        
        # Check if any relevant title can be found with the current context
        potential_titles = self._get_relevant_titles(emotional_context=full_user_context, top_k=1)
        return bool(potential_titles)

    def _get_content_preference(self, user_input_lower: str) -> Optional[str]:
        if any(word in user_input_lower for word in ["video", "watch", "youtube"]): return "video"
        if any(word in user_input_lower for word in ["book", "read", "reading", "article"]): return "book"
        if any(word in user_input_lower for word in ["music", "song", "listen", "audio"]): return "music"
        return None

    def _generate_reflection_summary(self) -> str:
        # Use all user messages in history to generate reflection
        full_user_context = " ".join([
            msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)
        ])
        if not full_user_context.strip():
            return "We've just started talking, and I'm here to listen."

        # Get emotional themes based on the full user context up to now
        emotional_themes = self._get_emotional_context(full_user_context) 
        if emotional_themes:
            primary_theme_info = emotional_themes[0] # Use the first theme as the most prominent
            # Extract the core theme text before the score part
            if "Theme from" in primary_theme_info and ": " in primary_theme_info:
                theme_content = primary_theme_info.split(": ", 1)[1].split(" (Score:")[0].strip()
                if len(theme_content) > 100: theme_content = theme_content[:100] + "..."
                return f"It sounds like you're navigating through feelings and themes around '{theme_content}'."
            # Fallback if theme format is unexpected, try to use a snippet
            concise_theme = primary_theme_info.split(" (Score:")[0][:100].strip()
            if concise_theme:
                return f"It sounds like you're dealing with {concise_theme.lower()}..."
        return "Based on our conversation, I'm getting a sense of what you're experiencing."

    def chat(self, user_input: str, max_turns_for_history=7) -> str: # Max turns for LLM prompt context
        self.turn_number += 1
        # Append user input to history *before* safety check, so safety check has full context up to current input.
        self.chat_history.append(f"{self.USER_PREFIX}{user_input}")

        # Safety check uses the user_input and the history *before* this current user_input was added for its context window.
        # The `user_input` is the very latest thing said.
        # `self.chat_history[:-1]` provides the conversation leading up to this latest input.
        is_safety_concern, crisis_response = self.check_safety_concerns(user_input, self.chat_history[:-1])
        if is_safety_concern and crisis_response:
            self.chat_history.append(f"{self.AVIKA_PREFIX}{crisis_response}")
            return crisis_response

        # Prepare context for LLM checks and prompts
        # For LLM checks, use a summary of the recent conversation (including current user input)
        recent_history_for_llm_checks = self.chat_history[-(max_turns_for_history*2):] 
        conversation_summary_for_llm_checks = "\n".join(recent_history_for_llm_checks)

        # For empathy/recommendation prompts, use a window of actual messages (including current user input)
        recent_history_for_prompts = self.chat_history[-(max_turns_for_history*2):]

        user_wants_recommendation = self._llm_is_requesting_recommendation(user_input, conversation_summary_for_llm_checks)
        user_is_resistant = self._llm_is_user_resistant(user_input, conversation_summary_for_llm_checks)
        
        # Retrieve state flags from *before* this turn
        prev_recommendation_offered = self.recommendation_offered_in_last_turn
        prev_avika_asked_for_clarification = self.avika_just_asked_for_clarification
        prev_avika_said_no_titles_found = self.avika_just_said_no_titles_found

        # Reset flags: these will be set by logic within this turn if applicable
        self.recommendation_offered_in_last_turn = False
        self.avika_just_asked_for_clarification = False
        self.avika_just_said_no_titles_found = False

        proactive_conditions_met = (
            self.turn_number >= self.MIN_EMPATHY_TURNS + 1 and # Ensure some empathy turns first
            not user_is_resistant and # Don't be proactive if user is pushing back
            not prev_avika_asked_for_clarification and # Don't jump to recommend if Avika just asked for more info
            not prev_recommendation_offered and # Don't recommend back-to-back unless user explicitly asks now
            not prev_avika_said_no_titles_found and # If Avika just said no titles, wait for user to guide
            self._has_sufficient_context_for_proactive_recommendation() # Check if context is rich enough
        )
        
        should_offer_recommendation = user_wants_recommendation or proactive_conditions_met

        avika_response = ""
        if should_offer_recommendation:
            st.sidebar.info("Logic Path: Attempting Recommendation")
            # Context for title retrieval should be all user messages so far including current one
            full_user_context = " ".join([msg.replace(self.USER_PREFIX, "") for msg in self.chat_history if msg.startswith(self.USER_PREFIX)])
            preferred_content_type = self._get_content_preference(user_input.lower())
            
            relevant_titles = self._get_relevant_titles(
                emotional_context=full_user_context, 
                content_type=preferred_content_type,
                top_k=5 # Get a few candidates for the LLM to choose from
            )

            title_list_str = ""
            if relevant_titles:
                title_list_str = "\n".join([f"- '{t['title']}' (Category: {t['category']})" for t in relevant_titles])
            
            reflection_summary = self._generate_reflection_summary() # Based on history up to *before* current user input
            # Emotional context for the prompt should be based on current input and recent history
            emotional_context_for_titles_prompt = self._get_emotional_context(user_input) # Uses current input + history

            prompt = self._construct_recommendation_prompt(
                user_input, # Current user message
                title_list_str, 
                "\n".join(emotional_context_for_titles_prompt),
                reflection_summary
            )
            avika_response = gemini_generate(prompt)
            # Check if the response actually contains a recommendation
            if "I recommend trying out" in avika_response and "from our" in avika_response:
                self.recommendation_offered_in_last_turn = True
            else:
                # LLM decided not to recommend (e.g., used no_titles_guidance)
                self.avika_just_said_no_titles_found = True 
                self.recommendation_offered_in_last_turn = False 
        else: 
            st.sidebar.info("Logic Path: Focusing on Empathy/Clarification")
            is_short_input = len(user_input.split()) < 5
            # Emotional themes for empathy prompt based on current input and recent history
            emotional_context_themes_for_prompt = self._get_emotional_context(user_input)
            no_strong_themes = not emotional_context_themes_for_prompt
            # is_stuck_early: if early in convo & input is brief/no themes, Avika should ask for more.
            is_stuck_early = self.turn_number <= self.MIN_EMPATHY_TURNS and (is_short_input or no_strong_themes)
            
            prompt = self._construct_empathy_prompt(
                user_input, # Current user message
                "\n".join(emotional_context_themes_for_prompt),
                recent_history_for_prompts, # Pass actual recent messages including current user input
                is_short_input,
                no_strong_themes,
                is_stuck_early,
                user_is_resistant
            )
            avika_response = gemini_generate(prompt)
            # Heuristic: if Avika asks a question and response is short, likely asking for clarification
            if "?" in avika_response and len(avika_response) < 150: 
                self.avika_just_asked_for_clarification = True
        
        self.chat_history.append(f"{self.AVIKA_PREFIX}{avika_response}")
        return avika_response

    def reset(self) -> str:
        st.sidebar.info("Resetting AvikaChat session state.")
        self.chat_history = []
        self.turn_number = 0
        self.recommendation_offered_in_last_turn = False
        self.avika_just_asked_for_clarification = False
        self.avika_just_said_no_titles_found = False
        # The initial greeting is now added by the Streamlit UI when messages are reset
        return self.INITIAL_GREETING

# --- Initialize global components ---
try:
    # Load models and data using cached functions
    S_MODEL = load_sentence_transformer_model()
    AVIKA_TITLES_DATA = load_avika_titles_cached()
    TITLE_EMBEDDINGS_DATA = generate_title_embeddings_cached(S_MODEL, AVIKA_TITLES_DATA)
    
    SAFETY_TOKENIZER, SAFETY_MODEL = load_safety_classifier_model()
    if SAFETY_TOKENIZER is None or SAFETY_MODEL is None: # load_safety_classifier_model returns (None, None) on failure
        st.warning("⚠️ Safety classifier failed to load. SAFETY_TOKENIZER or SAFETY_MODEL is None.")
        # No explicit error here as the cached function handles its own st.warning/error

    QDRANT_CLIENT = get_qdrant_client_cached()
    if QDRANT_CLIENT is None:
        st.warning("⚠️ Qdrant client failed to initialize. QDRANT_CLIENT is None.")
        # No explicit error here as the cached function handles its own st.warning/error

    # Check if essential models loaded correctly
    if not S_MODEL:
        INITIALIZATION_ERROR = "CRITICAL Error: SentenceTransformer model (S_MODEL) failed to load. App cannot function optimally."
        st.error(INITIALIZATION_ERROR)
    # Further checks for other components can be added if they are strictly critical for app startup

except Exception as e:
    INITIALIZATION_ERROR = f"CRITICAL Error during global component initialization: {str(e)}"
    st.error(INITIALIZATION_ERROR)
    traceback.print_exc()
    st.error("Critical components failed to initialize. The app might not function correctly. Please check logs and environment variables.")

# --- Helper Functions for Streamlit UI ---
def initialize_chat_session():
    """Initializes or retrieves the AvikaChat instance for the current session."""
    global INITIALIZATION_ERROR, S_MODEL, AVIKA_TITLES_DATA, TITLE_EMBEDDINGS_DATA, QDRANT_CLIENT, SAFETY_TOKENIZER, SAFETY_MODEL

    if INITIALIZATION_ERROR:
        st.error(f"Cannot start chat due to existing initialization error: {INITIALIZATION_ERROR}")
        st.stop()
    if not S_MODEL: 
        st.error("Core component (Sentence Model - S_MODEL) not loaded. Cannot start chat. Check earlier logs.")
        st.stop()

    if "avika_chat_instance" not in st.session_state:
        with st.spinner("Preparing Avika for you..."):
            try:
                # Ensure safety components are correctly unpacked if they were loaded as a tuple
                # and handle the case where they might be None if loading failed.
                loaded_safety_tokenizer = SAFETY_TOKENIZER
                loaded_safety_model = SAFETY_MODEL

                st.session_state.avika_chat_instance = AvikaChat(
                    sentence_model=S_MODEL,
                    avika_titles=AVIKA_TITLES_DATA,
                    title_embeddings=TITLE_EMBEDDINGS_DATA,
                    qdrant_client=QDRANT_CLIENT,
                    safety_tokenizer=loaded_safety_tokenizer,
                    safety_model=loaded_safety_model
                )
                st.session_state.messages = [{"role": "assistant", "content": st.session_state.avika_chat_instance.INITIAL_GREETING}]
                st.success("Avika is ready!")
            except Exception as e:
                st.error(f"Failed to initialize AvikaChat instance: {e}")
                traceback.print_exc()
                st.session_state.avika_chat_instance = None
                st.session_state.messages = [{"role": "assistant", "content": "Sorry, I couldn't start up correctly. Please check the logs or contact support."}]
                st.stop()
    return st.session_state.avika_chat_instance

def get_avika_response(chat_instance: AvikaChat, user_message: str) -> str:
    """Gets Avika's response by calling the chat method directly."""
    if not chat_instance:
        return "I'm not available at the moment due to an earlier initialization issue."
    try:
        return chat_instance.chat(user_message)
    except Exception as e:
        st.error(f"An error occurred while getting Avika's response: {e}")
        traceback.print_exc()
        return "I encountered an unexpected issue processing your message. Please try again."

# --- Main App Logic ---
st.title("🧠 Avika - Your Supportive Chat Companion")

st.markdown("""
Welcome! I'm Avika, here to listen and help you find resources.
**Disclaimer:** I am not a licensed therapist. If you're in crisis, please contact a professional.
""")

# Initialize chat session and messages
if "messages" not in st.session_state: # Should be set by initialize_chat_session
    st.session_state.messages = [] # Will be populated by initialize_chat_session

# Get or initialize the chat instance. This also sets the initial greeting.
chat_instance = initialize_chat_session()

# Display existing messages
# Ensure messages list exists from initialization
if hasattr(st.session_state, 'messages') and st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    # This case should ideally be handled by initialize_chat_session ensuring messages are populated
    st.warning("Message history not found. Attempting to re-initialize.")
    if chat_instance: # if instance exists, populate initial greeting
         st.session_state.messages = [{"role": "assistant", "content": chat_instance.INITIAL_GREETING}]
         st.rerun()
    else: # if instance doesn't exist, it means major init failure
        st.error("Chat cannot start due to initialization issues.")
        st.stop()

# Chat input
if prompt := st.chat_input("What's on your mind today?"):
    if not chat_instance:
        st.error("Chat session not properly initialized. Please refresh the page or check logs.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Avika is thinking..."):
            assistant_response = get_avika_response(chat_instance, prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        # No need to display assistant message here, it's handled by st.rerun() and message loop above
        st.rerun()

# --- Sidebar for Controls and Status ---
st.sidebar.title("Controls & Status")
if st.sidebar.button("Reset Conversation"):
    if "avika_chat_instance" in st.session_state and st.session_state.avika_chat_instance is not None:
        with st.spinner("Resetting conversation..."):
            initial_greeting = st.session_state.avika_chat_instance.reset() # Calls AvikaChat.reset()
            st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
            st.sidebar.success("Conversation reset!")
            st.rerun()
    else: # If instance is None or not in session_state
        st.sidebar.warning("Chat not yet initialized or failed to initialize. Cannot reset.")
        # Attempt to re-initialize if possible
        if not ("avika_chat_instance" in st.session_state and st.session_state.avika_chat_instance):
            st.session_state.pop('avika_chat_instance', None) # Clear potentially bad instance
            st.session_state.pop('messages', None) # Clear messages
            st.rerun() # This will trigger initialize_chat_session() again

st.sidebar.markdown("---")
st.sidebar.markdown("**Environment Variables:**")
def display_env_var_status(var_name):
    value = os.getenv(var_name)
    if value:
        st.sidebar.markdown(f"✅ `{var_name}`: Set") # For sensitive keys, don't show value: ({value[:4]}...)
    else:
        st.sidebar.markdown(f"❌ `{var_name}`: **Not Set**")

display_env_var_status("GEMINI_API_KEY")
display_env_var_status("QDRANT_URL")
display_env_var_status("QDRANT_API_KEY")
display_env_var_status("AVIKA_TITLES_PATH")
display_env_var_status("QDRANT_DOC_COLLECTION_NAME")
# AVIKA_DOCS_PATH is used by populate_db.py, not directly by the app runtime

st.sidebar.markdown("---")
st.sidebar.markdown("**Component Status:**")
if INITIALIZATION_ERROR:
    st.sidebar.error(f"Global Init Error: {INITIALIZATION_ERROR}")
# Individual component status based on global variables
if S_MODEL:
    st.sidebar.success("✅ Sentence Model (S_MODEL): Loaded")
else:
    st.sidebar.error("❌ Sentence Model (S_MODEL): FAILED to load")

if AVIKA_TITLES_DATA:
    st.sidebar.success(f"✅ Avika Titles ({len(AVIKA_TITLES_DATA)}): Loaded")
else:
    st.sidebar.warning("⚠️ Avika Titles: Not loaded or empty")

if TITLE_EMBEDDINGS_DATA:
    st.sidebar.success(f"✅ Title Embeddings ({len(TITLE_EMBEDDINGS_DATA)}): Generated")
elif AVIKA_TITLES_DATA: # Only an issue if titles loaded but embeddings didn't
    st.sidebar.warning("⚠️ Title Embeddings: NOT generated (titles available)")
else:
    st.sidebar.warning("⚠️ Title Embeddings: NOT generated (no titles)")

if QDRANT_CLIENT:
    st.sidebar.success("✅ Qdrant Client (QDRANT_CLIENT): Initialized")
else:
    st.sidebar.error("❌ Qdrant Client (QDRANT_CLIENT): FAILED or not configured")

if SAFETY_MODEL and SAFETY_TOKENIZER:
    st.sidebar.success("✅ Safety Model & Tokenizer: Loaded")
elif not SAFETY_MODEL and not SAFETY_TOKENIZER:
    st.sidebar.warning("⚠️ Safety Model & Tokenizer: Not loaded")
else:
    st.sidebar.warning("⚠️ Safety Model & Tokenizer: Partially loaded (check logs)")


# Display Qdrant client status *within* the AvikaChat instance if it was initialized
if "avika_chat_instance" in st.session_state and st.session_state.avika_chat_instance:
    chat_instance_qdrant_status = st.session_state.avika_chat_instance.qdrant_client
    if chat_instance_qdrant_status:
        st.sidebar.info("ℹ️ Qdrant (in AvikaChat): Connected")
    else:
        st.sidebar.warning("⚠️ Qdrant (in AvikaChat): Not connected/available")
    
    chat_instance_safety_status = st.session_state.avika_chat_instance.safety_model and st.session_state.avika_chat_instance.safety_tokenizer
    if chat_instance_safety_status:
        st.sidebar.info("ℹ️ Safety Model (in AvikaChat): Available")
    else:
        st.sidebar.warning("⚠️ Safety Model (in AvikaChat): Not available")

# The two lines above are removed by ensuring no lines follow this comment block. 