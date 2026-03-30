import whisper
import sounddevice as sd
import numpy as np
import time

class AudioRecorder:
    def __init__(self):
        self.frequence = 16000
        self.silence_duree = 4.0
        self.seuil_silence = 0.01
        
    def mesurer_volume(self, audio):
        return np.abs(audio).mean()
    
    def enregistrer_avec_silence(self):
        print("\n" + "=" * 60)
        print("PARLEZ MAINTENANT...")
        print("(Arret apres 4 secondes de silence)")
        print("=" * 60)
        
        buffer = []
        silence_start = None
        enregistrement = True
        
        with sd.InputStream(samplerate=self.frequence, channels=1, dtype='float32') as stream:
            while enregistrement:
                chunk, _ = stream.read(int(self.frequence * 0.5))
                buffer.append(chunk)
                
                volume = self.mesurer_volume(chunk)
                barre = "#" * int(volume * 50)
                print(f"\rVolume: {volume:.4f} {barre}", end="")
                
                if volume < self.seuil_silence:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duree:
                        print("\n\nSilence detecte - Fin")
                        enregistrement = False
                else:
                    silence_start = None
        
        audio_complet = np.concatenate(buffer)
        return audio_complet.flatten()

class SpeechToText:
    def __init__(self):
        print("Chargement de Whisper 'small'...")
        self.model = whisper.load_model("small", device="cpu")
        self.recorder = AudioRecorder()
        print("Modele STT charge!")
    
    def ecouter_et_transcrire(self):
        audio = self.recorder.enregistrer_avec_silence()
        print("\nTranscription...")
        resultat = self.model.transcribe(
            audio,
            language="fr",
            fp16=False,
            verbose=False
        )
        texte = resultat["text"].strip()
        print(f"\nTexte detecte: {texte}")
        return texte

# Pour test indépendant
if __name__ == "__main__":
    stt = SpeechToText()
    texte = stt.ecouter_et_transcrire()
    print(f"Transcrit: {texte}")