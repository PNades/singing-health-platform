import os
import sys
import time
import logging
import concurrent.futures
from operator import itemgetter
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import numpy as np
import io
import json
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'

# Set timeouts for all network operations
import socket
socket.setdefaulttimeout(30)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Constants
CLASSES = ['Healthy', 'Strained', 'Weak']
SR, DURATION, N_MELS, HOP_LENGTH = 22050, 9, 128, 512

COACHING_QUERIES = {
    'Weak': "My tone feels thin or weak when I sing. I don't want an opening or closing paragraph in your response, just the possible causes and solutions.",
    'Strained': "My high notes feel like shouting. What am I doing wrong and how can I fix it. I don't want an opening or closing paragraph in your response, just the possible causes and solutions.",
}

# Vocabulary definitions
VOCAB_DEFINITIONS = {
    "Anchoring": "Stabilizing and engaging the abdominal, back, head, and neck muscles to support the breath, allowing the larynx to remain stable.",
    "Falsetto": "A vocal quality where the true vocal folds (TVF) are held stiff and only partially come together. This produces a light, airy, or breathy tone.",
    "Twang": "A bright, brassy sound achieved by narrowing the aryepiglottic sphincter (AES). Comes in Oral Twang (bright but not nasal) and Nasalized Twang (bright and nasal).",
    "FVF": "False Vocal Fold (FVF) Retraction: The false vocal folds are drawn apart, opening the throat and reducing strain.",
    "False Vocal Fold": "The upper folds above the true vocal folds that can retract to open the throat.",
    "False Vocal Folds": "The upper folds above the true vocal folds that can retract to open the throat.",
    "Glottal Onset": "TVFs close before air is released, creating a percussive start.",
    "Glottal": "TVFs close before air is released, creating a percussive start.",
    "Aspirate Onset": "Air flows before folds touch, creating breathy start.",
    "Aspirate": "Air flows before folds touch, creating breathy start.",
    "Smooth Onset": "Air and fold vibration begin simultaneously.",
    "Smooth": "Air and fold vibration begin simultaneously.",
    "TVF": "True Vocal Fold (TVF): The main vibrating part of the vocal folds.",
    "True Vocal Fold": "The main vibrating part of the vocal folds.",
    "Thick": "Entire vocal fold vibrates (chest voice, belt).",
    "Thin": "Only top layers vibrate (head voice, falsetto).",
    "Stiff": "Folds held tightly (pressed/tense sound).",
    "Slack": "Folds too loose (creaky/fry sound).",
    "Anchored": "Using large muscles to stabilize the body and reduce vocal tract strain.",
    "Sob Quality": "Low larynx, thin TVF cover, FVF retraction - creates sad, mellow tone.",
    "Opera Quality": "Blend of Sob, Speech, and Twang - rich, resonant, soaring tone.",
    "Larynx": "The voice box, which can be high, mid, or low for different sounds.",
    "Speech Quality": "Natural, conversational sound with mid larynx and thick TVF.",
    "Belt Quality": "Loud, high-energy sound using thick TVF and anchoring.",
    "Velum": "The soft palate that can be raised or lowered to change nasal resonance.",
    "Aryepiglottic Sphincter": "Muscle ring above the vocal folds that can be narrowed to create twang.",
    "AES": "Aryepiglottic Sphincter (AES): Muscle ring above the vocal folds that can be narrowed to create twang.",
    "Subglottal Pressure": "Air pressure below the vocal folds that helps them vibrate.",
}

model = None
rag_chain = None
_mel_filterbank_cache = {}

# ============================================================================
# OPTIMIZED AUDIO PROCESSING (NO LIBROSA)
# ============================================================================

def create_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Create mel filterbank matrix (cached for performance)"""
    cache_key = (sr, n_fft, n_mels, fmin, fmax)
    if cache_key in _mel_filterbank_cache:
        return _mel_filterbank_cache[cache_key]
    
    if fmax is None:
        fmax = sr / 2.0
    
    # Mel scale conversion functions
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    # Create mel scale
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mels)
    
    # Convert to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        if center > left:
            fbank[i, left:center] = np.arange(center - left) / (center - left)
        if right > center:
            fbank[i, center:right] = np.arange(right - center, 0, -1) / (right - center)
    
    _mel_filterbank_cache[cache_key] = fbank
    return fbank

def fast_preprocess_audio(audio_data, sr=SR):
    """Ultra-fast audio preprocessing using scipy (NO LIBROSA)"""
    target_length = sr * DURATION
    
    # Pad or truncate
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
    else:
        audio_data = audio_data[:target_length]
    
    # Compute STFT using scipy
    n_fft = 2048
    f, t, Zxx = signal.stft(
        audio_data, 
        sr, 
        nperseg=n_fft, 
        noverlap=n_fft - HOP_LENGTH
    )
    
    # Power spectrogram
    power_spec = np.abs(Zxx) ** 2
    
    # Apply mel filterbank
    mel_basis = create_mel_filterbank(sr, n_fft, N_MELS)
    mel_spec = np.dot(mel_basis, power_spec)
    
    # Convert to log scale
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    
    # Normalize
    normalized = ((log_mel + 80) / 80).astype(np.float32)
    
    return normalized

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_ml_model():
    """Initialize TensorFlow model with proper error handling"""
    global model
    start_time = time.time()
    
    try:
        import tensorflow as tf
        
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        
        model_path = "singing_health_cnn.keras"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        logger.info(f"📦 Loading ML model from {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Warm up the model with dummy prediction
        logger.info("🔥 Warming up model...")
        dummy_input = np.random.rand(1, 128, 389, 1).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ ML model loaded in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load ML model: {str(e)}", exc_info=True)
        return False

def initialize_rag_system():
    """Initialize RAG with timeout protection"""
    global rag_chain
    start_time = time.time()
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Check environment variables
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not all([qdrant_url, qdrant_api_key, groq_api_key]):
            logger.warning("⚠️  Missing RAG environment variables")
            return False
        
        logger.info("📚 Initializing RAG embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        logger.info("🔗 Connecting to Qdrant...")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=qdrant_url,
            collection_name="singing_chatbot",
            api_key=qdrant_api_key
        )
        
        logger.info("🤖 Initializing Groq LLM...")
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            timeout=25.0,
            max_retries=1
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        )
        
        prompt_template = """You are an intelligent, friendly voice teacher tasked with answering user queries about singing technique. Always provide clear, science-based explanations that any singer can understand, even beginners.

SPECIAL CASE - DEFINITION REQUESTS:
If the user is asking for the definition of a specific vocal term (e.g., "What is falsetto?", "What are false vocal folds?", "Define twang", "Explain velum"), respond with ONLY a concise, 2-3 sentence definition in plain language. Do NOT use the structured format with "Possible Causes" and "Solutions". Just explain what the term means clearly and simply.

Examples of definition requests:
- "What is falsetto?"
- "What are the false vocal folds?"
- "Define twang"
- "Explain what the larynx is"
- "What does anchoring mean?"

For definition requests, respond like this:
"Falsetto is a vocal quality where the true vocal folds are held stiff and only partially come together, producing a light, airy, or breathy tone. It's commonly used for high notes and creates a head voice sound."

GENERAL QUERIES - USE THIS FORMAT:
For all other singing-related questions (not simple definition requests), use the structured format below.

IMPORTANT: If the user's message is unclear, too short (like "how", "what", "help"), or doesn't seem related to singing, politely ask them to clarify or provide a more specific singing-related question. Do NOT make assumptions about what they're asking. Only provide detailed singing advice when the question is clearly about singing technique or vocal health.

Response Structure for Valid Singing Questions:

1. Start with a friendly, empathetic opening (2-3 sentences maximum):
   - Acknowledge the user's issue in plain language
   - Briefly indicate the physiological factors without using technical jargon
   - Give a clear roadmap: you will explain causes first, then provide solutions

2. Under the heading "Possible Causes:", explain the causes in separate paragraphs:
   - Each paragraph focuses on one cause
   - Use technical terms but explain them in plain language the first time
   - Keep explanations concise but informative
   - If a cause involves a technique, note that the technique will be explained in the solutions section

3. Under the heading "Solutions:", provide numbered, actionable steps:
   - What to do physically or vocally
   - What to feel or notice while doing it
   - Specific exercises (e.g., "hee-hee-hee" sirens, "Nya!" sounds, scales, lip trills)
   - Include technical terms naturally and explain them in context

4. End with a supportive closing (1-2 sentences):
   - Encourage consistent practice
   - Remind them that improvement comes with focused effort on the exercises

5. Use the provided context (RAG chunk) only when it logically applies to the user's singing-related question. Integrate details naturally as if they are your own knowledge. Never reference or imply that external context or documents are being used. If the context is irrelevant, ignore it completely.

6. For non-singing questions, small talk, or unclear questions: Respond naturally and politely, then invite them to ask a specific singing-related question. Do not use the RAG context for this.

Vocal Terms & Definitions (Always Available): Use these terms to explain concepts clearly and simply whenever needed:

Falsetto: A light, airy sound where the true vocal folds (TVF), the main vibrating parts of the voice, only partially come together.
Twang: A bright, focused sound made by narrowing a small muscle ring above the vocal folds called the aryepiglottic sphincter (AES). Oral Twang: bright, not nasal; Nasalized Twang: bright and nasal.
False Vocal Fold (FVF) Retraction: When the upper folds above the true vocal folds move apart, opening the throat and reducing strain.
Glottal Onset: When the vocal folds close before air is released, creating a hard or percussive start.
Aspirate Onset: When air comes out before the vocal folds fully connect, creating a breathy start.
Smooth Onset: When airflow and vocal fold vibration begin at the same time, creating a balanced start.
True Vocal Fold (TVF): The main vibrating tissues in the voice that create sound.
Thick Vocal Fold Use: When the entire vocal fold vibrates, creating stronger sounds like chest voice or belting.
Thin Vocal Fold Use: When only the top layers of the vocal folds vibrate, creating lighter sounds like head voice or falsetto.
Stiff Vocal Folds: When the folds are held too tightly, causing a pressed or strained sound.
Slack Vocal Folds: When the folds are too loose, causing vocal fry or a creaky sound.
Anchoring: Using larger muscles of the body to stabilize the torso and reduce tension in the throat.
Sob Quality: A sound made with a slightly lowered larynx, lighter vocal fold contact, and an open throat, creating a mellow or emotional tone.
Opera Quality: A balanced mix of sob quality, natural speech-like sound, and twang that creates a rich, resonant tone.
Larynx: The voice box, which can sit higher, lower, or neutral to change tone color.
Speech Quality: A natural, conversational sound using a neutral larynx and thicker vocal fold contact.
Belt Quality: A loud, energetic sound using thicker vocal fold contact and strong body anchoring.
Velum: The soft palate in the mouth that raises or lowers to control nasal sound.
Aryepiglottic Sphincter (AES): A small muscle ring above the vocal folds that narrows to create twang.
Subglottal Pressure: The air pressure below the vocal folds that helps them vibrate.

FORMATTING RULES:
- Do NOT use asterisks for bold or emphasis
- Do NOT use markdown formatting
- Use plain text only
- Only use section headings "Possible Causes:" and "Solutions:" (for structured responses)
- Do NOT label the opening or closing paragraphs
- Write naturally without special formatting

Always ensure that:
- Definition requests get ONLY a concise definition (2-3 sentences)
- The opening is empathetic and concise (2-3 sentences) with a clear roadmap
- Causes are in separate paragraphs under "Possible Causes:", beginner-friendly and technically accurate
- Solutions are numbered and actionable under "Solutions:", always include exercises
- Technical terms are explained in plain language
- Response ends with supportive encouragement (1-2 sentences)
- Guidance is accurate, practical, and beginner-friendly
- NO markdown formatting or asterisks are used

Conversation History (use only to understand follow-up questions):
{history}

Context: {context}
Question: {query}
Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {
                "context": itemgetter("query") | retriever,
                "query": itemgetter("query"),
                "history": itemgetter("history"),
            }
            | prompt | llm | StrOutputParser()
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ RAG system initialized in {elapsed:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG initialization failed: {str(e)}", exc_info=True)
        return False

def get_rag_response(query, history=None, timeout=20):
    """Get RAG response with timeout protection"""
    if rag_chain is None:
        return "RAG system not available"

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in (history or [])
    )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(rag_chain.invoke, {"query": query, "history": history_text})
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"⏱️  RAG request timed out after {timeout}s")
                return "Coaching temporarily unavailable (timeout)"
    except Exception as e:
        logger.error(f"❌ RAG error: {str(e)}")
        return f"Error: {str(e)}"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    """Analyze voice with comprehensive timing logs"""
    request_start = time.time()
    logger.info("=" * 60)
    logger.info(f"📥 NEW ANALYZE REQUEST")
    
    if model is None:
        logger.error("❌ ML model not loaded")
        return jsonify({'error': 'ML model not loaded'}), 500
    
    if 'audio' not in request.files:
        logger.error("❌ No audio file in request")
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        step_start = time.time()
        audio_content = audio_file.read()
        logger.info(f"⏱️  [1/5] Read audio: {time.time() - step_start:.3f}s ({len(audio_content)} bytes)")
        
        if len(audio_content) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        
        step_start = time.time()
        audio_io = io.BytesIO(audio_content)
        sample_rate, audio_data = wavfile.read(audio_io)
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        logger.info(f"⏱️  [2/5] Decode WAV: {time.time() - step_start:.3f}s (sr={sample_rate}, len={len(audio_data)})")
        
        step_start = time.time()
        if sample_rate != SR:
            num_samples = int(len(audio_data) * SR / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            logger.info(f"⏱️  [3/5] Resample: {time.time() - step_start:.3f}s")
        else:
            logger.info(f"⏱️  [3/5] Resample: skipped (already {SR}Hz)")
        
        step_start = time.time()
        mel_spectrogram = fast_preprocess_audio(audio_data, SR)
        input_data = mel_spectrogram[np.newaxis, ..., np.newaxis]
        logger.info(f"⏱️  [4/5] Preprocess: {time.time() - step_start:.3f}s (shape={input_data.shape})")
        
        step_start = time.time()
        predictions = model.predict(input_data, verbose=0)[0]
        logger.info(f"⏱️  [5/5] Prediction: {time.time() - step_start:.3f}s")
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        all_predictions = [
            {'class': CLASSES[i], 'confidence': float(predictions[i])}
            for i in range(len(CLASSES))
        ]
        
        response = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'all_predictions': all_predictions
        }
        
        if predicted_class in ['Strained', 'Weak'] and rag_chain is not None:
            try:
                step_start = time.time()
                query = COACHING_QUERIES[predicted_class]
                
                coaching = get_rag_response(query, timeout=20)
                response['coaching_response'] = coaching
                logger.info(f"⏱️  [RAG] Coaching: {time.time() - step_start:.3f}s")
            except Exception as e:
                logger.error(f"⚠️  RAG failed (continuing): {str(e)}")
        
        total_time = time.time() - request_start
        logger.info(f"✅ REQUEST COMPLETE: {total_time:.3f}s")
        logger.info("=" * 60)
        
        return jsonify(response)
        
    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"❌ ERROR after {total_time:.3f}s: {str(e)}", exc_info=True)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with RAG"""
    if rag_chain is None:
        return jsonify({'error': 'Chat system not available'}), 500
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        raw_history = data.get('history', [])
        history = [
            h for h in raw_history[:6]
            if isinstance(h, dict) and h.get('role') in ('user', 'assistant')
            and isinstance(h.get('content', ''), str)
        ]

        logger.info(f"💬 Chat request: {message[:50]}...")
        response = get_rag_response(message, history=history, timeout=25)

        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if (model and rag_chain) else 'partial',
        'ml_model': model is not None,
        'rag_system': rag_chain is not None,
        'timestamp': time.time()
    })

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve example audio files"""
    return send_from_directory('static/audio', filename)

@app.route('/trainer')
def trainer():
    """Serve pitch trainer page"""
    return TRAINER_HTML

@app.route('/')
def index():
    """Serve main HTML page"""
    return render_template_string(HTML_TEMPLATE, vocab_json=json.dumps(VOCAB_DEFINITIONS))


# ============================================================================
# HTML TEMPLATE - WORKING VOCAB + LARGE CLASSIFICATION
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-4LX04LNZKT"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-4LX04LNZKT');
</script>

<!-- Meta Pixel Code -->
<script>
!function(f,b,e,v,n,t,s)
{if(f.fbq)return;n=f.fbq=function(){n.callMethod?
n.callMethod.apply(n,arguments):n.queue.push(arguments)};
if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';
n.queue=[];t=b.createElement(e);t.async=!0;
t.src=v;s=b.getElementsByTagName(e)[0];
s.parentNode.insertBefore(t,s)}(window, document,'script',
'https://connect.facebook.net/en_US/fbevents.js');
fbq('init', '1248474623849036');
fbq('track', 'PageView');
</script>
<noscript><img height="1" width="1" style="display:none"
src="https://www.facebook.com/tr?id=1248474623849036&ev=PageView&noscript=1"
/></noscript>
<!-- End Meta Pixel Code -->

<!-- Favicon Links -->
<link rel="icon" type="image/x-icon" href="/static/favicon.ico">
<link rel="apple-touch-icon" sizes="180x180" href="/static/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="/static/android-chrome-192x192.png">
<link rel="icon" type="image/png" sizes="512x512" href="/static/android-chrome-512x512.png">
<link rel="manifest" href="/static/site.webmanifest">

<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strainless AI</title>
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa;
    color: #333;
    line-height: 1.6;
    overflow-x: hidden;
}

/* Desktop Layout */
@media (min-width: 768px) {
    .mobile-header {
        display: none;
    }
    
    .desktop-header {
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 15px 30px;
        border-bottom: 1px solid #e2e8f0;
        z-index: 100;
    }
    
    .desktop-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 60px 20px 20px;
    }
    
    .main-container.has-messages {
        justify-content: flex-start;
        padding-top: 70px;
    }
    
    .centered-content {
        max-width: 900px;
        width: 100%;
        text-align: center;
        margin-top: -30px;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 400;
        color: #2d3748;
        margin-bottom: 40px;
    }
    
    .main-container.has-messages .main-title {
        display: none;
    }
    
    .chat-messages {
        max-width: 900px;
        width: 100%;
        margin: 0 auto 20px;
        padding: 20px;
        padding-bottom: 180px;
    }
    
    .input-container {
        max-width: 900px;
        width: 100%;
        margin: 0 auto;
        position: relative;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .main-container.has-messages .input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
    }
}

/* Mobile Layout */
@media (max-width: 767px) {
    .desktop-header {
        display: none;
    }
    
    .mobile-header {
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 15px 20px;
        border-bottom: 1px solid #e2e8f0;
        text-align: center;
        z-index: 100;
    }
    
    .mobile-header h1 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    .main-container {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        padding-top: 60px;
    }
    
    .centered-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding: 20px;
        padding-top: 240px;
        text-align: center;
    }
    
    .main-container.has-messages .centered-content {
        display: none;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 400;
        color: #2d3748;
        margin-bottom: 40px;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        padding-bottom: 240px;
    }
    
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        right: 20px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 12px 16px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
}

/* Classification Result Styling */
.classification-result {
    font-size: 2.0rem;
    font-weight: 600;
    display: block;
    margin-bottom: 0.5rem;
    line-height: 1.2;
    color: #2d3748;
}

@media (max-width: 767px) {
    .classification-result {
        font-size: 1.8rem;
    }
}

/* Input Container Styles */
.input-wrapper {
    display: flex;
    flex-direction: column;
    gap: 22px;
}

.input-box {
    width: 100%;
    min-height: 24px;
    max-height: 240px;
    padding: 10px 12px;
    border: none;
    outline: none;
    font-size: 1rem;
    font-family: inherit;
    line-height: 1.5;
    overflow-y: auto;
    box-sizing: border-box;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.input-box:empty:before {
    content: attr(data-placeholder);
    color: #a0aec0;
    pointer-events: none;
}

.input-box:focus:empty:before {
    content: attr(data-placeholder);
    color: #a0aec0;
}

.button-group {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding-left: 4px;
}

.record-voice-btn, .back-to-chat-btn, .record-now-btn {
    background: #3182ce;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 12px 20px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s;
}

.record-voice-btn:hover, .back-to-chat-btn:hover, .record-now-btn:hover {
    background: #2c5aa0;
}

.record-now-btn {
    width: 100%;
    margin: 20px 0 0 0;
    padding: 12px;
    font-size: 1rem;
}

.send-btn {
    background: #3182ce;
    color: white;
    border: none;
    border-radius: 10px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1.3rem;
    transition: background 0.2s;
    flex-shrink: 0;
    line-height: 1;
    padding-bottom: 2px;
}

.send-btn:hover {
    background: #2c5aa0;
}

/* Recording Interface */
.recording-interface {
    display: none;
    flex-direction: column;
    gap: 22px;
}

.recording-interface.active {
    display: flex;
}

.recording-status {
    color: #718096;
    font-size: 0.95rem;
    padding: 10px 12px;
    min-height: 24px;
}

.recording-buttons {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding-left: 4px;
}

.recording-buttons-right {
    display: flex;
    gap: 8px;
}

.icon-btn {
    width: 44px;
    height: 44px;
    border: 1px solid #e2e8f0;
    background: #3182ce;
    color: white;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
}

.icon-btn:hover {
    background: #2c5aa0;
}

.icon-btn.info-btn {
    background: white;
    color: #718096;
    border: 1px solid #e2e8f0;
}

.icon-btn.info-btn:hover {
    background: #f7fafc;
    color: #4a5568;
}

.icon-btn svg {
    width: 22px;
    height: 22px;
}

/* Chat Messages - ChatGPT Style */
.chat-message {
    margin-bottom: 24px;
    display: flex;
    font-size: 1rem;
    line-height: 1.6;
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.bot {
    justify-content: flex-start;
}

.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 18px;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.chat-message.user .message-content {
    background: #e5e7eb;
    color: #1f2937;
    border-bottom-right-radius: 4px;
}

.chat-message.bot .message-content {
    background: transparent;
    color: #374151;
    padding-left: 0;
    max-width: 100%;
}

/* Typing Indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #9ca3af;
    animation: typing 1.4s infinite;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: translateY(0);
    }
    30% {
        opacity: 1;
        transform: translateY(-10px);
    }
}

/* Modal Overlay */
.modal-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);
    z-index: 999;
    backdrop-filter: blur(2px);
}

.modal-overlay.active {
    display: block;
}

/* Instructions Modal */
.instructions-modal {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    border-radius: 16px;
    padding: 30px;
    max-width: 500px;
    width: 90%;
    z-index: 1000;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

.instructions-modal.active {
    display: block;
}

.instructions-modal h3 {
    font-size: 1.3rem;
    margin-bottom: 15px;
    color: #2d3748;
}

.instructions-modal p {
    color: #4a5568;
    margin-bottom: 20px;
    line-height: 1.6;
}

.example-buttons {
    display: flex;
    gap: 10px;
    justify-content: space-between;
}

.example-btn {
    flex: 1;
    padding: 10px 20px;
    background: #e2e8f0;
    color: #4a5568;
    border: 1px solid #cbd5e0;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.example-btn:hover {
    background: #cbd5e0;
    color: #2d3748;
}

/* Vocab Highlighting */
.vocab-word {
    color: #2563eb;
    text-decoration: underline;
    text-decoration-style: dotted;
    text-decoration-thickness: 2px;
    text-underline-offset: 2px;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
}

.vocab-word:hover {
    color: #1d4ed8;
    background: rgba(37, 99, 235, 0.1);
    padding: 2px 4px;
    margin: -2px -4px;
    border-radius: 3px;
}

.vocab-tooltip {
    position: absolute;
    background: white;
    color: #2d3748;
    border: 1px solid #cbd5e0;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    z-index: 9999;
    max-width: 300px;
    font-size: 0.875rem;
    line-height: 1.5;
    pointer-events: none;
}

/* Utility */
.hidden {
    display: none !important;
}

/* Scroll to bottom button */
.scroll-to-bottom {
    position: fixed;
    bottom: 180px;
    left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 40px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 50%;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    z-index: 50;
    transition: all 0.2s;
}

.scroll-to-bottom:hover {
    background: #f7fafc;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.scroll-to-bottom.visible {
    display: flex;
}

.scroll-to-bottom svg {
    width: 20px;
    height: 20px;
    color: #4a5568;
}

@media (max-width: 767px) {
    .scroll-to-bottom {
        bottom: 180px;
    }
}

/* Audio Player */
#audioPlayback {
    width: 100%;
    margin: 15px 0;
}

#examplePlayer {
    display: none;
}
</style>
</head>
<body>

<!-- Desktop Header -->
<div class="desktop-header">
    <h1>Strainless</h1>
</div>

<!-- Mobile Header -->
<div class="mobile-header">
    <h1>Strainless</h1>
</div>

<!-- Main Container -->
<div class="main-container" id="mainContainer">
    <!-- Centered Content (shown when no messages) -->
    <div class="centered-content" id="centeredContent">
        <h1 class="main-title">Experiencing vocal troubles?</h1>
    </div>
    
    <!-- Chat Messages -->
    <div class="chat-messages hidden" id="chatMessages"></div>
    
    <!-- Input Container -->
    <div class="input-container">
        <!-- Normal Input Mode -->
        <div class="input-wrapper" id="normalInput">
            <div 
                class="input-box" 
                id="chatInput" 
                contenteditable="true"
                data-placeholder="Ask anything singing-related..."
            ></div>
            <div class="button-group">
                <div style="display: flex; gap: 10px;">
                    <button class="record-voice-btn" id="recordVoiceBtn">Analyze Voice</button>
                    <button class="record-voice-btn" id="pitchTrainerBtn" onclick="window.location.href='/trainer'">Pitch Trainer</button>
                </div>
                <button class="send-btn" id="sendBtn">↗</button>
            </div>
        </div>
        
        <!-- Recording Mode -->
        <!-- Recording Mode -->
        <div class="recording-interface" id="recordingInterface">
            <div class="recording-status" id="recordingStatus">Click the microphone to begin...</div>
            <div class="recording-buttons">
                <button class="back-to-chat-btn" id="backToChatBtn">Back to chat</button>
                <div class="recording-buttons-right">
                    <button class="icon-btn info-btn" id="infoBtn" title="Show instructions">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" stroke-width="2"></circle>
                            <line x1="12" y1="16" x2="12" y2="12" stroke-width="2" stroke-linecap="round"></line>
                            <circle cx="12" cy="8" r="1" fill="currentColor"></circle>
                        </svg>
                    </button>
                    <button class="icon-btn" id="micBtn" title="Record now">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                        </svg>
                    </button>
                    <button class="icon-btn hidden" id="trashBtn" title="Discard recording">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                        </svg>
                    </button>
                    <button class="icon-btn hidden" id="checkBtn" title="Analyze recording">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Modal Overlay -->
<div class="modal-overlay" id="modalOverlay"></div>

<!-- Scroll to Bottom Button -->
<button class="scroll-to-bottom" id="scrollToBottomBtn">
    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
    </svg>
</button>

<!-- Instructions Modal -->
<div class="instructions-modal" id="instructionsModal">
    <h3>Voice Recording Instructions</h3>
    <p>Sing an arpeggio on an "ah" vowel. Listen to the examples:</p>
    <div class="example-buttons">
        <button class="example-btn" onclick="playExample('male_example.wav')">Male Example</button>
        <button class="example-btn" onclick="playExample('female_example.wav')">Female Example</button>
    </div>
    <button class="record-now-btn" onclick="closeInstructionsModal()">I'm Ready to Record</button>
</div>

<!-- Hidden Audio Players -->
<audio id="audioPlayback"></audio>
<audio id="examplePlayer"></audio>

<script>
// Vocabulary definitions
const VOCAB_DEFINITIONS = {{ vocab_json|safe }};

// State
let isRecording = false;
let hasRecordedOnce = false;
let audioChunks = [];
let audioContext, processor, input;
let recordedBlob = null;
let hasMessages = false;
let conversationHistory = [];
let isTyping = false;

function setTypingState(typing) {
    isTyping = typing;
    sendBtn.disabled = typing;
    sendBtn.style.opacity = typing ? '0.5' : '1';
    recordVoiceBtn.disabled = typing;
    recordVoiceBtn.style.opacity = typing ? '0.5' : '1';
}

// DOM Elements
const mainContainer = document.getElementById('mainContainer');
const centeredContent = document.getElementById('centeredContent');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const recordVoiceBtn = document.getElementById('recordVoiceBtn');
const normalInput = document.getElementById('normalInput');
const recordingInterface = document.getElementById('recordingInterface');
const recordingStatus = document.getElementById('recordingStatus');
const micBtn = document.getElementById('micBtn');
const trashBtn = document.getElementById('trashBtn');
const checkBtn = document.getElementById('checkBtn');
const backToChatBtn = document.getElementById('backToChatBtn');
const infoBtn = document.getElementById('infoBtn');
const modalOverlay = document.getElementById('modalOverlay');
const instructionsModal = document.getElementById('instructionsModal');
const scrollToBottomBtn = document.getElementById('scrollToBottomBtn');

// Check if scroll button should be visible
function checkScrollButtonVisibility() {
    if (!hasMessages) {
        scrollToBottomBtn.classList.remove('visible');
        return;
    }
    
    requestAnimationFrame(() => {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollBottom = scrollTop + windowHeight;
        
        if (scrollBottom < documentHeight - 100) {
            scrollToBottomBtn.classList.add('visible');
        } else {
            scrollToBottomBtn.classList.remove('visible');
        }
    });
}

// Scroll detection
window.addEventListener('scroll', checkScrollButtonVisibility);
window.addEventListener('resize', checkScrollButtonVisibility);

let mutationTimeout;
const observer = new MutationObserver(function() {
    clearTimeout(mutationTimeout);
    mutationTimeout = setTimeout(checkScrollButtonVisibility, 50);
});

if (chatMessages) {
    observer.observe(chatMessages, { childList: true, subtree: true, characterData: true });
}

scrollToBottomBtn.addEventListener('click', function() {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
});

// Handle Enter key
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('blur', function() {
    if (this.textContent.trim() === '') {
        this.textContent = '';
    }
});

// Send button
sendBtn.addEventListener('click', sendMessage);

// Record voice button
recordVoiceBtn.addEventListener('click', function() {
    if (!hasRecordedOnce) {
        showInstructionsModal();
        hasRecordedOnce = true;
    }
    
    normalInput.classList.add('hidden');
    recordingInterface.classList.add('active');
});

// Back to chat button
backToChatBtn.addEventListener('click', function() {
    recordedBlob = null;
    normalInput.classList.remove('hidden');
    recordingInterface.classList.remove('active');
    recordingStatus.textContent = 'Click the microphone to begin...';
    micBtn.classList.remove('hidden');
    trashBtn.classList.add('hidden');
    checkBtn.classList.add('hidden');
    isRecording = false;
});

// Info button
infoBtn.addEventListener('click', showInstructionsModal);

// Microphone button
micBtn.addEventListener('click', startRecording);

// Trash button
trashBtn.addEventListener('click', discardRecording);

// Check button
checkBtn.addEventListener('click', analyzeRecording);

// Modal overlay
modalOverlay.addEventListener('click', closeInstructionsModal);

function showInstructionsModal() {
    modalOverlay.classList.add('active');
    instructionsModal.classList.add('active');
}

function closeInstructionsModal() {
    modalOverlay.classList.remove('active');
    instructionsModal.classList.remove('active');
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 22050,
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false
            }
        });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
        input = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        audioChunks = [];
        
        processor.onaudioprocess = function(e) {
            if (isRecording) {
                const data = e.inputBuffer.getChannelData(0);
                audioChunks.push(new Float32Array(data));
            }
        };
        
        input.connect(processor);
        processor.connect(audioContext.destination);
        
        isRecording = true;
        recordingStatus.textContent = 'Listening...';
        micBtn.classList.add('hidden');
        trashBtn.classList.remove('hidden');
        checkBtn.classList.remove('hidden');
        
        setTimeout(() => {
            if (isRecording) stopRecording();
        }, 9000);
        
    } catch (e) {
        console.error('Microphone error:', e);
        recordingStatus.textContent = 'Error: ' + e.message;
    }
}

// Stop recording
function stopRecording() {
    if (isRecording && audioContext) {
        isRecording = false;
        processor.disconnect();
        input.disconnect();
        audioContext.close();
        
        let totalLength = 0;
        for (let chunk of audioChunks) totalLength += chunk.length;
        
        const combinedArray = new Float32Array(totalLength);
        let offset = 0;
        for (let chunk of audioChunks) {
            combinedArray.set(chunk, offset);
            offset += chunk.length;
        }
        
        const wavBuffer = encodeWAV(combinedArray, 22050);
        recordedBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        
        if (recordingStatus.textContent !== 'Analyzing...') {
            recordingStatus.textContent = 'Recording complete. Click ✓ to analyze.';
        }
        trashBtn.classList.remove('hidden');
        checkBtn.classList.remove('hidden');
    }
}

// Discard recording
function discardRecording() {
    recordedBlob = null;
    isRecording = false;
    recordingStatus.textContent = 'Click the microphone to begin...';
    micBtn.classList.remove('hidden');
    trashBtn.classList.add('hidden');
    checkBtn.classList.add('hidden');
}

// Analyze recording
async function analyzeRecording() {
    if (isTyping) return;
    if (!recordedBlob) {
        if (isRecording) {
            stopRecording();
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (!recordedBlob) {
            recordingStatus.textContent = 'No recording to analyze';
            return;
        }
    }
    
    recordingStatus.textContent = 'Analyzing...';
    checkBtn.disabled = true;
    checkBtn.style.opacity = '0.5';
    
    const formData = new FormData();
    formData.append('audio', recordedBlob, 'recording.wav');
    
    try {
        const response = await fetch('/analyze_voice', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayVoiceResult(result);
        
        recordedBlob = null;
        recordingStatus.textContent = 'Analysis complete! Click the microphone to record again.';
        micBtn.classList.remove('hidden');
        trashBtn.classList.add('hidden');
        checkBtn.classList.add('hidden');
        
    } catch (e) {
        console.error('Analysis error:', e);
        recordingStatus.textContent = 'Error analyzing voice';
    } finally {
        if (!isTyping) {
            sendBtn.disabled = false;
            sendBtn.style.opacity = '1';
            recordVoiceBtn.disabled = false;
            recordVoiceBtn.style.opacity = '1';
        }
    }
}

// Display voice result
function displayVoiceResult(result) {
    if (result.error) {
        addMessage('Error: ' + result.error, 'bot');
        return;
    }
    
    const prediction = result.prediction;
    
    // Create message with HTML for classification
    let resultText = `\n\n<span class="classification-result">${prediction}</span>\n`;
    resultText += `<strong>Breakdown:</strong>\n`;
    
    result.all_predictions.forEach(pred => {
        resultText += `${pred.class}: ${(pred.confidence * 100).toFixed(1)}%\n`;
    });
    
    if (result.coaching_response) {
        resultText += `\n${result.coaching_response}`;
    }
    
    addMessage(resultText, 'bot', true, true);
}

// Typewriter effect
function typeWriter(element, html, speed = 0.1) {
    setTypingState(true);
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    // Extract classification if it exists
    const classificationSpan = temp.querySelector('.classification-result');
    let classificationHTML = '';
    let remainingText = html;
    
    if (classificationSpan) {
        classificationHTML = classificationSpan.outerHTML;
        const classificationIndex = html.indexOf(classificationHTML);
        remainingText = html.substring(classificationIndex + classificationHTML.length);
    }
    
    temp.innerHTML = remainingText;
    const text = temp.textContent;
    
    element.innerHTML = '';
    
    // Add classification immediately
    if (classificationHTML) {
        element.innerHTML = classificationHTML;
    }
    
    // Create a single text node that we'll update
    const textContainer = document.createElement('span');
    element.appendChild(textContainer);
    
    let index = 0;
    
    function type() {
        if (index < text.length) {
            index = Math.min(index + 2, text.length);
            textContainer.textContent = text.substring(0, index);

            if (index % 10 === 0) {
                checkScrollButtonVisibility();
            }

            setTimeout(type, speed);
        } else {
            textContainer.replaceWith(document.createTextNode(text));
            highlightVocabInElement(element);
            checkScrollButtonVisibility();
            setTypingState(false);
        }
    }
    
    type();
}

// Send text message
async function sendMessage() {
    const message = chatInput.textContent.trim();
    if (!message || isTyping) return;
    
    addMessage(message, 'user');
    chatInput.textContent = '';
    chatInput.blur();
    
    const scrollBefore = window.pageYOffset;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    
    window.scrollTo(0, scrollBefore);
    
    chatInput.contentEditable = 'false';
    sendBtn.disabled = true;
    sendBtn.style.opacity = '0.5';
    recordVoiceBtn.disabled = true;
    recordVoiceBtn.style.opacity = '0.5';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, history: conversationHistory })
        });

        const result = await response.json();

        chatMessages.removeChild(typingDiv);

        conversationHistory.push({ role: 'user', content: message });
        conversationHistory.push({ role: 'assistant', content: result.response });
        if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);

        addMessage(result.response, 'bot', false, true);
        
    } catch (e) {
        console.error('Chat error:', e);
        chatMessages.removeChild(typingDiv);
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    } finally {
        chatInput.contentEditable = 'true';
        if (!isTyping) {
            sendBtn.disabled = false;
            sendBtn.style.opacity = '1';
            recordVoiceBtn.disabled = false;
            recordVoiceBtn.style.opacity = '1';
        }
        chatInput.focus();
    }
}

// Add message to chat
function addMessage(text, sender, isHTML = false, useTypewriter = false) {
    if (!hasMessages) {
        hasMessages = true;
        mainContainer.classList.add('has-messages');
        centeredContent.classList.add('hidden');
        chatMessages.classList.remove('hidden');
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (useTypewriter && sender === 'bot') {
        contentDiv.innerHTML = '';
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
        
        typeWriter(contentDiv, text, 1);
    } else {
        if (isHTML) { contentDiv.innerHTML = text; } else { contentDiv.textContent = text; }
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        if (sender === 'bot') {
            setTimeout(() => { highlightVocabInElement(contentDiv); }, 10);
        }

        setTimeout(() => {
            messageDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
            checkScrollButtonVisibility();
        }, 100);
    }
}

// FIXED: Highlight vocab words - preserves classification HTML
function highlightVocabInElement(element) {
    
    const words = Object.keys(VOCAB_DEFINITIONS).sort((a, b) => b.length - a.length);
    
    const walker = document.createTreeWalker(
        element,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: function(node) {
                // Skip vocab-word and classification-result spans
                let parent = node.parentElement;
                while (parent && parent !== element) {
                    if (parent.classList && (parent.classList.contains('vocab-word') || 
                        parent.classList.contains('classification-result'))) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    parent = parent.parentElement;
                }
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );
    
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
        textNodes.push(node);
    }
    
    textNodes.forEach(textNode => {
        const originalText = textNode.textContent;
        
        const matches = [];
        
        words.forEach(word => {
            const lowerText = originalText.toLowerCase();
            const lowerWord = word.toLowerCase();
            let index = lowerText.indexOf(lowerWord);
            
            while (index !== -1) {
                const beforeChar = index > 0 ? originalText[index - 1] : ' ';
                const afterChar = index + word.length < originalText.length ? originalText[index + word.length] : ' ';
                const isWordBoundary = /\\W/.test(beforeChar) && /\\W/.test(afterChar);
                
                if (isWordBoundary) {
                    matches.push({
                        word: word,
                        start: index,
                        end: index + word.length,
                        matchedText: originalText.substring(index, index + word.length)
                    });
                }
                
                index = lowerText.indexOf(lowerWord, index + 1);
            }
        });
        
        if (matches.length === 0) {
            return;
        }
        
        matches.sort((a, b) => a.start - b.start);
        const filteredMatches = [];
        let lastEnd = -1;
        matches.forEach(match => {
            if (match.start >= lastEnd) {
                filteredMatches.push(match);
                lastEnd = match.end;
            }
        });
        
        const fragment = document.createDocumentFragment();
        let lastIndex = 0;
        
        filteredMatches.forEach(match => {
            if (match.start > lastIndex) {
                fragment.appendChild(document.createTextNode(originalText.substring(lastIndex, match.start)));
            }
            
            const span = document.createElement('span');
            span.className = 'vocab-word';
            span.textContent = match.matchedText;
            span.dataset.vocabWord = match.word;
            
            span.onclick = function(e) {
                e.stopPropagation();
                showVocabTooltipDirect(this);
            };
            
            fragment.appendChild(span);
            lastIndex = match.end;
        });
        
        if (lastIndex < originalText.length) {
            fragment.appendChild(document.createTextNode(originalText.substring(lastIndex)));
        }
        
        textNode.parentNode.replaceChild(fragment, textNode);
    });
}

// Show tooltip
function showVocabTooltipDirect(element) {
    document.querySelectorAll('.vocab-tooltip').forEach(t => t.remove());
    
    const word = element.dataset.vocabWord;
    const definition = VOCAB_DEFINITIONS[word];
    
    if (!definition) {
        return;
    }
    
    const tooltip = document.createElement('div');
    tooltip.className = 'vocab-tooltip';
    tooltip.textContent = definition;
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = (window.scrollX + rect.left) + 'px';
    tooltip.style.top = (window.scrollY + rect.bottom + 8) + 'px';
    tooltip.style.position = 'absolute';
    
    setTimeout(() => {
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = (window.innerWidth - tooltipRect.width - 10) + 'px';
        }
        if (tooltipRect.bottom > window.innerHeight) {
            tooltip.style.top = (window.scrollY + rect.top - tooltipRect.height - 8) + 'px';
        }
    }, 0);
}

// Close tooltip
document.addEventListener('click', function(e) {
    if (!e.target.classList.contains('vocab-word')) {
        document.querySelectorAll('.vocab-tooltip').forEach(t => t.remove());
    }
});

// WAV encoding
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    const floatTo16 = (output, offset, input) => {
        for (let i = 0; i < input.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16(view, 44, samples);
    
    return buffer;
}

// Play example audio
function playExample(filename) {
    const player = document.getElementById('examplePlayer');
    player.src = `/static/audio/${filename}`;
    player.play();
}
</script>

</body>
</html>
'''

# ============================================================================
# STARTUP
# ============================================================================

logger.info("=" * 60)
logger.info("🎤 STRAINLESS AI - STARTING UP")
logger.info("=" * 60)

with open('trainer.html', 'r', encoding='utf-8') as f:
    TRAINER_HTML = f.read()

ml_loaded = initialize_ml_model()
rag_loaded = initialize_rag_system()

if ml_loaded and rag_loaded:
    logger.info("✅ ALL SYSTEMS OPERATIONAL")
elif ml_loaded:
    logger.warning("⚠️  ML ONLY (RAG unavailable)")
elif rag_loaded:
    logger.warning("⚠️  RAG ONLY (ML unavailable)")
else:
    logger.error("❌ NO SYSTEMS LOADED")

logger.info("=" * 60)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        debug=False,
        host='0.0.0.0',
        port=port,
        threaded=True
    )