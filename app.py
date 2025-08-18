import os
import sys
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import numpy as np
import librosa
import tensorflow as tf
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# RAG Chatbot imports
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONPATH'] = '/app'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Constants
CLASSES = ['Healthy', 'Strained', 'Weak']
SR, DURATION, N_MELS, HOP_LENGTH = 22050, 9, 128, 512

# Vocabulary definitions dictionary
VOCAB_DEFINITIONS = {
    "Falsetto": "A vocal quality where the true vocal folds (TVF) are held stiff and only partially come together. This produces a light, airy, or breathy tone.",
    "Twang": "A bright, brassy sound achieved by narrowing the aryepiglottic sphincter (AES). Comes in Oral Twang (bright but not nasal) and Nasalized Twang (bright and nasal).",
    "FVF": "False Vocal Fold (FVF) Retraction: The false vocal folds are drawn apart, opening the throat and reducing strain.",
    "Glottal Onset": "TVFs close before air is released, creating a percussive start.",
    "Aspirate Onset": "Air flows before folds touch, creating breathy start.",
    "Smooth Onset": "Air and fold vibration begin simultaneously.",
    "TVF": "True Vocal Fold (TVF): The main vibrating part of the vocal folds.",
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
}

model = None
rag_chain = None

def initialize_ml_model():
    global model
    model_path = "singing_health_cnn.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("ML Model loaded")
        return True
    print(f"ML Model not found at {model_path}")
    return False

def initialize_rag_system():
    global rag_chain
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not all([qdrant_url, qdrant_api_key, groq_api_key]):
            print("Missing environment variables for RAG")
            return False

        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=qdrant_url,
            collection_name="singing_chatbot",
            api_key=qdrant_api_key
        )

        llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        # Create prompt template
        prompt_template = """
        You are an intelligent voice teacher tasked with answering user queries based on provided context.
        You first explain what they are doing wrong, then how to fix it, and finally give them some exercises they can use to implement the techniques.
        Use the given context to respond to the user's question. Don't address the source of the context in your response. Only use the information
        provided in the context if it makes sense within the context of the question. For example, if the user were to greet you, ask for the
        definition of a term, or ask questions not related to singing, completely ignore the provided context - don't use or address the context in any way whatsoever
        (Don't even say that you are ignoring the context) - you should still answer the question normally, just
        rely on your own knowledge to answer the question. If someone asks you the function of a term, or how to alter it, use the information below (not the
        context) and your own knowledge to answer the  question. For example, if someone were to ask "How do I raise my larynx?" You would use the information below
        ("swallow to raise it"), as well as your own knowledge to answer the question. Make sure that, if a user is trying to initiate small talk, you respond normally;
        if they ask you how you are doing you should say you are doing good and ask them if they have any singing-related questions.

        Use this list of terms and definitions along with the provided context to thoroughly deliver your message in a way that a singer of any level could understand:
       
        Falsetto: A vocal quality where the true vocal folds (TVF) are held stiff and only partially come together. This produces a light, airy, or breathy tone.
       
        Twang: A bright, brassy sound achieved by narrowing the aryepiglottic sphincter (AES). Comes in Oral Twang (bright but not nasal) and Nasalized Twang (bright and nasal).
       
        False Vocal Fold (FVF) Retraction: The false vocal folds are drawn apart, opening the throat and reducing strain.
       
        Onsets - How vocal sound begins:
        - Glottal Onset: TVFs close before air is released, creating a percussive start
        - Aspirate Onset: Air flows before folds touch, creating breathy start
        - Smooth Onset: Air and fold vibration begin simultaneously
       
        TVF Covers (True Vocal Fold Body-Cover Control):
        - Thick: Entire vocal fold vibrates (chest voice, belt)
        - Thin: Only top layers vibrate (head voice, falsetto)
        - Stiff: Folds held tightly (pressed/tense sound)
        - Slack: Folds too loose (creaky/fry sound)
       
        Anchored Torso and Head: Using large muscles to stabilize the body and reduce vocal tract strain.
       
        Sob Quality: Low larynx, thin TVF cover, FVF retraction - creates sad, mellow tone.
        Opera Quality: Blend of Sob, Speech, and Twang - rich, resonant, soaring tone.
       
        Larynx Positions:
        - High: Brighter, more youthful sound, swallow to raise it
        - Mid: Neutral, conversational
        - Low: Darker, rounder tone, yawn to lower it
       
        Speech Quality: Natural, conversational sound with mid larynx and thick TVF.
        Belt Quality: Loud, high-energy sound using thick TVF and anchoring.

        Context: {context}
        Question: {query}
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

        print("RAG system initialized")
        return True
    except Exception as e:
        print(f"RAG error: {str(e)}")
        return False

def preprocess_audio(audio_data, sr=SR):
    target_length = sr * DURATION
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
    else:
        audio_data = audio_data[:target_length]
    
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return ((log_mel + 80) / 80).astype(np.float32)

def create_spectrogram_image(mel_spectrogram):
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_spectrogram, sr=SR, hop_length=HOP_LENGTH, 
                           x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return img_str

def get_rag_response(query):
    if rag_chain is None:
        return "RAG system not available"
    try:
        return rag_chain.invoke(query)
    except Exception as e:
        return f"Error: {str(e)}"

# Condensed HTML template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html><head><title>Strainless AI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f8f9fa;color:#333;line-height:1.6}
.header{text-align:center;padding:40px 20px 20px;background:white}
.header h1{font-size:2.5rem;font-weight:700;color:#2d3748;margin-bottom:8px}
.header p{color:#718096;font-size:1.1rem;margin-bottom:30px}
.nav-buttons{display:flex;gap:15px;justify-content:center;margin-bottom:20px}
.nav-btn{padding:12px 24px;background:#3182ce;color:white;border:none;border-radius:8px;font-size:1rem;font-weight:500;cursor:pointer;transition:all 0.2s}
.nav-btn:hover{background:#2c5aa0;transform:translateY(-1px)}
.nav-btn.active{background:#2c5aa0}
.container{max-width:1200px;margin:0 auto;padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:30px}
.card{background:white;border-radius:12px;padding:30px;box-shadow:0 2px 10px rgba(0,0,0,0.1);border:1px solid #e2e8f0;transition:border-color 0.3s}
.card.selected{border:2px solid #3182ce}
.card h2{font-size:1.5rem;font-weight:600;color:#2d3748;margin-bottom:8px}
.card p{color:#718096;margin-bottom:25px;line-height:1.5;display:flex;align-items:center;gap:15px}
.record-section{text-align:center;margin:20px 0}
.record-btn{width:80px;height:80px;border-radius:50%;background:#3182ce;color:white;font-size:24px;border:none;cursor:pointer;transition:all 0.3s;box-shadow:0 4px 15px rgba(49,130,206,0.3);display:flex;align-items:center;justify-content:center;margin:0 auto 15px}
.record-btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(49,130,206,0.4)}
.record-text{color:#3182ce;font-weight:500;font-size:0.9rem}
.status-text{color:#718096;font-size:0.9rem;margin-top:5px}
.chat-area{margin-top:20px}
.chat-messages{height:370px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:8px;padding:15px;margin-bottom:15px;background:#f8f9fa}
.chat-input-container{display:flex;gap:10px}
.chat-input{flex:1;padding:12px;border:1px solid #e2e8f0;border-radius:8px;font-size:1rem;outline:none;transition:border-color 0.2s}
.chat-input:focus{border-color:#3182ce}
.send-btn{padding:12px 20px;background:#3182ce;color:white;border:none;border-radius:8px;cursor:pointer;font-weight:500;transition:background 0.2s}
.send-btn:hover{background:#2c5aa0}
.chat-message{margin:10px 0;padding:10px 12px;border-radius:8px;font-size:0.95rem}
.chat-message.user{background:#e6fffa;border-left:3px solid #38b2ac}
.chat-message.bot{background:#f0fff4;border-left:3px solid #48bb78}
.result{margin-top:20px;padding:20px;border-radius:8px;border:1px solid #e2e8f0}
.result.healthy{background:#f0fff4;border-color:#48bb78}
.result.strained{background:#fef5e7;border-color:#ed8936}
.result.weak{background:#fed7d7;border-color:#f56565}
.vocab-word{color:#3182ce;text-decoration:underline;cursor:pointer;transition:color 0.2s}
.vocab-word:hover{color:#2c5aa0}
.vocab-tooltip{position:absolute;background:#fffbea;color:#2d3748;border:1px solid #ecc94b;border-radius:8px;padding:12px 16px;box-shadow:0 4px 16px rgba(0,0,0,0.15);z-index:9999;max-width:320px;font-size:0.9rem;line-height:1.4}
#audioPlayback{width:100%;margin:15px 0}
.example-buttons{display:flex;gap:8px}.example-btn{padding:6px 12px;background:#e2e8f0;color:#4a5568;border:1px solid #cbd5e0;border-radius:6px;font-size:0.8rem;font-weight:500;cursor:pointer;transition:all 0.2s}.example-btn:hover{background:#cbd5e0;color:#2d3748}.placeholder-text{color:#a0aec0;font-style:italic;text-align:center;padding:40px 20px}
@media(max-width:768px){.container{grid-template-columns:1fr;padding:15px}.nav-buttons{flex-direction:column;align-items:center}.header h1{font-size:2rem}}
</style></head>
<body>
<div class="header">
<h1>Strainless AI</h1>
<p>Record your voice for health analysis via ML and get expert coaching through a RAG chatbot.</p>
<div class="nav-buttons">
<button class="nav-btn active" onclick="showSection('recording')">Start Recording</button>
<button class="nav-btn" onclick="showSection('chat')">Ask the Coach</button>
</div></div>
<div class="container">
<div class="card" id="voiceCard">
<h2>Voice Health Analysis</h2>
<p>Sing an arpeggio on an "ah" vowel. <span class="example-buttons"><button class="example-btn" onclick="playExample('male_example.wav')">Male Example</button><button class="example-btn" onclick="playExample('female_example.wav')">Female Example</button></span></p>
<div class="record-section">
<div style="border:1px solid #e2e8f0;border-radius:8px;padding:30px;background:#f8f9fa;height:410px;display:flex;flex-direction:column;align-items:center">
<div style="flex:1;display:flex;align-items:center;padding-top:40px">
<div style="text-align:center">
<button id="recordBtn" class="record-btn" onclick="toggleRecording()">🎤</button>
<div style="margin-top:15px">
<div id="recordText" class="record-text">Click to Record</div>
<div id="statusText" class="status-text">Idle</div>
</div></div></div></div></div>
<audio id="audioPlayback" controls style="display:none"></audio>
<audio id="examplePlayer" controls style="display:none"></audio>
<div id="voiceResult"></div>
</div>
<div class="card" id="chatCard">
<h2>Ask the Voice Coach</h2>
<p>Chat about singing technique, exercises, and vocal health.</p>
<div class="chat-area">
<div class="chat-messages" id="chatMessages">
<div class="placeholder-text">No messages yet. Ask anything about singing.</div>
</div>
<div class="chat-input-container">
<input type="text" id="chatInput" class="chat-input" placeholder="Ask about technique, exercises, or voice problems..." onkeypress="if(event.key==='Enter')sendMessage()">
<button class="send-btn" onclick="sendMessage()">↗</button>
</div></div></div></div>
<script>
let mediaRecorder,audioChunks=[],isRecording=false,audioContext,processor,input;
const VOCAB_DEFINITIONS={{ vocab_json|safe }};
function showSection(s){document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));event.target.classList.add('active');document.querySelectorAll('.card').forEach(c=>c.classList.remove('selected'));document.getElementById(s==='recording'?'voiceCard':'chatCard').classList.add('selected')}
function highlightVocabWords(t){const w=Object.keys(VOCAB_DEFINITIONS).sort((a,b)=>b.length-a.length);w.forEach(word=>{const r=new RegExp("\\\\b"+word.replace(/[-\\/\\\\^$*+?.()|[\\]{}]/g,'\\\\$&')+"\\\\b","gi");t=t.replace(r,m=>`<span class="vocab-word" data-word="${m}">${m}</span>`)});return t}
document.addEventListener('click',function(e){if(e.target.classList.contains('vocab-word')){const w=e.target.getAttribute('data-word');const d=VOCAB_DEFINITIONS[w]||VOCAB_DEFINITIONS[w.charAt(0).toUpperCase()+w.slice(1)]||"Definition not found.";showVocabTooltip(e.target,d)}else{hideVocabTooltip()}});
function showVocabTooltip(t,d){hideVocabTooltip();const tt=document.createElement('div');tt.className='vocab-tooltip';tt.innerHTML=`<strong>${t.textContent}</strong><br>${d}`;document.body.appendChild(tt);const r=t.getBoundingClientRect();tt.style.left=(window.scrollX+r.left)+'px';tt.style.top=(window.scrollY+r.bottom+5)+'px'}
function hideVocabTooltip(){document.querySelectorAll('.vocab-tooltip').forEach(e=>e.remove())}
function encodeWAV(s,sr){const b=new ArrayBuffer(44+s.length*2);const v=new DataView(b);const ws=(o,str)=>{for(let i=0;i<str.length;i++)v.setUint8(o+i,str.charCodeAt(i))};const f16=(out,off,inp)=>{for(let i=0;i<inp.length;i++,off+=2){const s=Math.max(-1,Math.min(1,inp[i]));out.setInt16(off,s<0?s*0x8000:s*0x7FFF,true)}};ws(0,'RIFF');v.setUint32(4,36+s.length*2,true);ws(8,'WAVE');ws(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,s.length*2,true);f16(v,44,s);return b}
async function toggleRecording(){isRecording?stopRecording():await startRecording()}
async function startRecording(){try{const s=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:22050,channelCount:1,echoCancellation:false,noiseSuppression:false}});audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:22050});input=audioContext.createMediaStreamSource(s);processor=audioContext.createScriptProcessor(4096,1,1);audioChunks=[];processor.onaudioprocess=function(e){if(isRecording){const d=e.inputBuffer.getChannelData(0);audioChunks.push(new Float32Array(d))}};input.connect(processor);processor.connect(audioContext.destination);isRecording=true;document.getElementById('recordBtn').textContent='⏹️';document.getElementById('recordText').textContent='Click to Stop';document.getElementById('statusText').textContent='Recording... Sing your arpeggio!';setTimeout(()=>{if(isRecording)stopRecording()},9000)}catch(e){console.error('Mic error:',e);document.getElementById('statusText').textContent=`Error: ${e.message}`}}
function stopRecording(){if(isRecording&&audioContext){isRecording=false;processor.disconnect();input.disconnect();audioContext.close();document.getElementById('recordBtn').textContent='🎤';document.getElementById('recordText').textContent='Click to Record';document.getElementById('statusText').textContent='Processing audio...';let tl=0;for(let c of audioChunks)tl+=c.length;const ca=new Float32Array(tl);let off=0;for(let c of audioChunks){ca.set(c,off);off+=c.length}const wb=encodeWAV(ca,22050);const ab=new Blob([wb],{type:'audio/wav'});const au=URL.createObjectURL(ab);document.getElementById('audioPlayback').src=au;document.getElementById('audioPlayback').style.display='block';analyzeVoice(ab)}}
async function analyzeVoice(ab){const fd=new FormData();fd.append('audio',ab,'recording.wav');try{const r=await fetch('/analyze_voice',{method:'POST',body:fd});const res=await r.json();displayVoiceResult(res)}catch(e){console.error('Analysis error:',e);document.getElementById('statusText').textContent='Error analyzing voice'}}
function displayVoiceResult(r){if(r.error){document.getElementById('voiceResult').innerHTML=`<div class="result" style="background:#fed7d7;border-color:#f56565"><h3>Error</h3><p>${r.error}</p></div>`;return}const c=(r.confidence*100).toFixed(1);const p=r.prediction.toLowerCase();const h=r.all_predictions.find(pred=>pred.class==='Healthy').confidence*100;const g=h<=20?'F':h<=30?'D':h<=50?'C':h<=70?'B':'A';const gc=g==='D'||g==='F'?'red':g==='C'?'orange':g==='B'||g==='A'?'green':'black';let ch='';if(r.coaching_response){const hc=highlightVocabWords(r.coaching_response);ch=`<div style="margin-top:15px;padding:15px;background:#f8f9fa;border-radius:6px"><h4>Personalized Coaching:</h4><div style="white-space:pre-line;margin-top:8px">${hc}</div></div>`}document.getElementById('voiceResult').innerHTML=`<div class="result ${p}" style="position:relative;padding-bottom:10px"><span style="position:absolute;top:20px;right:20px;width:50px;height:50px;border:3px solid ${gc};border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.8em;color:${gc};font-weight:bold;text-shadow:1px 1px 2px rgba(0,0,0,0.3);line-height:1">${g}</span><h3>${r.prediction}</h3><p><strong>Classification:</strong> ${c}%</p>${ch}<div style="margin-top:15px"><h4 style="margin-bottom:10px">Breakdown:</h4>${r.all_predictions.map((pred,idx)=>`<p style="margin:10px 0">${idx===r.all_predictions.length-1?'<strong>'+pred.class+':</strong> '+(pred.confidence*100).toFixed(1)+'%':'<strong>'+pred.class+':</strong> '+(pred.confidence*100).toFixed(1)+'%'}</p>`).join('')}</div></div>`;document.getElementById('statusText').textContent='Analysis complete'}
async function sendMessage(){const i=document.getElementById('chatInput');const m=i.value.trim();if(!m)return;addMessage(m,'user');i.value='';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const res=await r.json();addMessage(res.response,'bot')}catch(e){console.error('Chat error:',e);addMessage('Sorry, I encountered an error. Please try again.','bot')}}
function addMessage(m,s){const cm=document.getElementById('chatMessages');const ph=cm.querySelector('.placeholder-text');if(ph)ph.remove();const md=document.createElement('div');md.className=`chat-message ${s}`;let mh=m.replace(/\\n/g,'<br>');if(s==='bot')mh=highlightVocabWords(mh);const l=s==='user'?'You:':'Voice Coach:';md.innerHTML=`<strong>${l}</strong><br>${mh}`;cm.appendChild(md);cm.scrollTop=cm.scrollHeight}
function playExample(filename){const player=document.getElementById('examplePlayer');player.src=`/static/audio/${filename}`;player.style.display='block';player.play();player.onended=function(){player.style.display='none'}}
</script></body></html>'''

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, vocab_json=json.dumps(VOCAB_DEFINITIONS))

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        audio_content = audio_file.read()
        if len(audio_content) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        
        audio_io = io.BytesIO(audio_content)
        audio_data, actual_sr = librosa.load(audio_io, sr=SR, mono=True)
        
        if actual_sr != SR:
            audio_data = librosa.resample(audio_data, orig_sr=actual_sr, target_sr=SR)
        
        target_length = SR * DURATION
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        if len(audio_data) == 0:
            return jsonify({'error': 'Audio empty after processing'}), 400
        
        mel_spectrogram = preprocess_audio(audio_data, SR)
        input_data = mel_spectrogram[np.newaxis, ..., np.newaxis]
        predictions = model.predict(input_data, verbose=0)[0]
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        spectrogram_img = create_spectrogram_image(mel_spectrogram)
        
        all_predictions = [
            {'class': CLASSES[i], 'confidence': float(predictions[i])}
            for i in range(len(CLASSES))
        ]
        
        response = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'all_predictions': all_predictions,
            'spectrogram': spectrogram_img
        }
        
        # Get coaching advice for problematic voices
        if predicted_class in ['Strained', 'Weak'] and rag_chain is not None:
            query = ("My tone feels thin or weak when I sing. What am I doing wrong and how can I fix it." 
                    if predicted_class == 'Weak' else 
                    "My high notes feel like shouting. What am I doing wrong and how can I fix it.")
            response['coaching_response'] = get_rag_response(query)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if rag_chain is None:
        return jsonify({'error': 'Chat system not available'}), 500
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = get_rag_response(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if (model and rag_chain) else 'partial',
        'ml_model': model is not None,
        'rag_system': rag_chain is not None
    })

if __name__ == '__main__':
    print("Initializing Strainless AI...")
    
    ml_loaded = initialize_ml_model()
    rag_loaded = initialize_rag_system()
    
    status = "All systems ready!" if (ml_loaded and rag_loaded) else \
             "ML only" if ml_loaded else "RAG only" if rag_loaded else "No systems loaded"
    print(status)
    
    # Production configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)