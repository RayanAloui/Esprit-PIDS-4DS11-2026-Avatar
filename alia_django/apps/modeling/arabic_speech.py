

# %%
# Import des bibliothèques
import whisper
import sounddevice as sd
#sd.default.device = 21  # Microphone Realtek
from scipy.io.wavfile import write
from IPython.display import Audio, display
import imageio_ffmpeg
import os
import soundfile as sf
import numpy as np
os.environ["PATH"] += os.pathsep + imageio_ffmpeg.get_ffmpeg_exe()
from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)
import edge_tts
import asyncio

# %% [markdown]
# ## 1. Traduction

# %%
LANGUES = {
    "fr": "Français",
    "en": "Anglais",
    "ar": "Arabe",
    "es": "Espagnol",
    "tn": "Tunisien (Darija)"
}

# %%
def traduire_tunisien(texte: str, src: str, tgt: str, max_retries: int = 3) -> str:
    
    src_nom = LANGUES.get(src, src)
    tgt_nom = LANGUES.get(tgt, tgt)

    if tgt == "tn":
        prompt = f"""
            Tu es un expert en traduction vers le dialecte tunisien écrit en arabe (دارجة تونسية).
            
            Tâche :
            Traduis du {src_nom} vers le dialecte tunisien en arabe vocalisé.
            
            Règles STRICTES :
            - Tunisien naturel uniquement
            - Aucune explication
            - Une seule phrase de sortie
            
            Texte :
            \"\"\"{texte}\"\"\"
            """
    else:
        prompt = f"Traduis du dialecte tunisien (Darija tunisienne) en {tgt_nom}. Réponds UNIQUEMENT avec la traduction.\n\nTexte : {texte}"

    last_response = ""

    for attempt in range(max_retries):

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3  # important pour stabilité
        )

        result = response.choices[0].message.content.strip()

        # ✔ validation
        if result and len(result) > 1:
            return result

        print(f" Réponse vide, retry {attempt + 1}/{max_retries}")
        last_response = result

    # fallback final
    raise ValueError(" Traduction échouée après plusieurs tentatives")

# %% [markdown]
# ## 2. Speech-to-Text (Reconnaissance Vocale) avec Whisper
# 
# Whisper est un modèle de reconnaissance vocale développé par OpenAI qui supporte très bien l'arabe.

# %%
import whisper
import torch

# 1. charger l’architecture Whisper
model = whisper.load_model("medium")

# 2. charger les poids sauvegardés
#state_dict = torch.load("model.pth", weights_only=True)
#model.load_state_dict(state_dict)

# 3. eval
model.eval()

# %%
def record_audio(filename="mic_audio.wav", duration=5, fs=16000):
    print(" Enregistrement (mode stable auto)...")

    # IMPORTANT : laisser sounddevice choisir le bon device
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )

    sd.wait()
    write(filename, fs, audio)

    print(" Terminé")
    return filename

# %%
def transcribe_arabic_audio(audio_path):
    print(f"Transcription de: {audio_path}")
    
    result = model.transcribe(
        audio_path,
        task='transcribe',
        language='ar',
        temperature=0,
        beam_size=5,

        verbose=True
    )
    
    return result

# %%
audio_file = record_audio(duration=6)
print(" Fichier généré :", audio_file)
print(" Existe ?", os.path.exists(audio_file))
# Chemin correct (utiliser la variable audio_file)
if os.path.exists(audio_file):
    result = transcribe_arabic_audio(audio_file)
    
    print("\n" + "="*50)
    print(" Transcription:")
    print("="*50)
    print(result["text"])
    
    print("\n Segments détaillés:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
else:
    print(" Erreur: fichier audio non trouvé")

# %%
# Exemple d'utilisation - Texte en arabe
texte = result["text"]
fr_text = traduire_tunisien(texte, src="tn", tgt="fr")

fr_text

# %% [markdown]
# ## 3. Text-to-Speech (Synthèse Vocale) avec gTTS
# 
# gTTS utilise l'API Google Text-to-Speech qui supporte l'arabe avec une bonne qualité.

# %%
async def text_to_speech_arabic(text, output_file="output_arabic.mp3", voice="ar-SA-HamedNeural"):

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

    return output_file

# %%
# Texte source
texte = "Le médicament est cher, on ne s’imagine pas que les gens l’achètent."

# Traduction tunisienne
arabic_text = traduire_tunisien(texte, src="fr", tgt="tn")
print("Texte traduit :", arabic_text)

# Génération audio
audio_path = await text_to_speech_arabic(arabic_text, "exemple_arabe.mp3")

# Lecture dans notebook
Audio(audio_path)


