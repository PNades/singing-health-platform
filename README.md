# StrainlessAI

A web app that checks your singing voice for strain with a machine learning model, then coaches you through a voice-teacher chatbot.

## The Problem

Singers often don't notice strain until it's already hurting their voice, and private voice lessons are expensive. StrainlessAI gives singers instant feedback on vocal health and technique advice they can act on right away, from any browser.

## Features

- **Voice health analysis:** Record a 9-second arpeggio on an "ah" vowel. A CNN classifies it as **Healthy**, **Strained** or **Weak**, shows a confidence breakdown, and gives a letter grade (A–F) based on the Healthy score.
- **Automatic coaching:** If your voice is classified as Strained or Weak, the app asks the chatbot what's going wrong and shows personalized fixes and exercises.
- **Ask the Voice Coach:** A RAG chatbot answers questions about technique, exercises and vocal health.
- **Clickable vocabulary:** Technical terms in coaching answers (Twang, FVF, Sob Quality, etc.) are highlighted. Click one to see its definition.
- **Example recordings:** Male and female sample arpeggios show you what to sing.

## How It Works

**Classifier:** Audio is resampled to 22,050 Hz, trimmed or padded to 9 seconds, and converted to a normalized 128-band mel spectrogram. The spectrogram goes into a Keras CNN (`singing_health_cnn.keras`), which outputs probabilities for the three classes.

**Chatbot:** Questions are embedded with `BAAI/bge-large-en-v1.5`. The best-matching passage is retrieved from a Qdrant Cloud collection of vocal pedagogy material. That passage, plus a built-in glossary of vocal terms, goes to Llama 3.1 8B (via Groq), which answers in a set format: what you're doing wrong, how to fix it, and exercises to practice.

## Tech Stack

- **Backend:** Python, Flask
- **Audio processing:** librosa, NumPy
- **ML model:** TensorFlow / Keras (CNN on mel spectrograms)
- **RAG:** LangChain, Qdrant Cloud, Hugging Face embeddings, Groq (Llama 3.1 8B)
- **Frontend:** HTML, CSS and JavaScript served by Flask; Web Audio API for recording

## Setup

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/PNades/StrainlessAI
   cd StrainlessAI
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with your keys:
   ```
   QDRANT_CLOUD_URL=your_qdrant_url
   QDRANT_CLOUD_API_KEY=your_qdrant_key
   GROQ_API_KEY=your_groq_key
   ```
3. Make sure these files are in place:
   - `singing_health_cnn.keras` (the trained model) in the project root
   - `static/audio/male_example.wav` and `static/audio/female_example.wav`
4. Run the app and open http://localhost:5000:
   ```bash
   python app.py
   ```

The app still starts if one part is missing. Without the model you get chat only; without the API keys you get analysis only. Visit `/health` to see which parts loaded.

## API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web app |
| `/analyze_voice` | POST | Upload a WAV recording (`audio` field) and get the classification plus coaching |
| `/chat` | POST | Send `{"message": "..."}` and get the coach's reply |
| `/health` | GET | Shows whether the ML model and RAG system loaded |

## Author

Prem Nadesan, [@PNades](https://github.com/PNades)
