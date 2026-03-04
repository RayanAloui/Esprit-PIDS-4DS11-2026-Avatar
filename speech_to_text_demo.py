import whisper
import sounddevice as sd
import numpy as np
import time

# ============================================================
# MICROPHONE AVEC DETECTION DE SILENCE
# ============================================================

class AudioRecorder:
    def __init__(self):
        self.frequence = 16000
        self.silence_duree = 4.0  # Secondes de silence pour arreter
        self.seuil_silence = 0.01  # Seuil de volume pour considerer comme silence
        
    def mesurer_volume(self, audio):
        """Mesure le volume moyen"""
        return np.abs(audio).mean()
    
    def enregistrer_avec_silence(self):
        """
        Enregistre jusqu'a 4 secondes de silence
        """
        print("\n" + "=" * 60)
        print("PARLEZ MAINTENANT...")
        print("(Le script s'arretera apres 4 secondes de silence)")
        print("=" * 60)
        
        buffer = []
        silence_start = None
        enregistrement = True
        
        # Tampon pour accumuler l'audio
        with sd.InputStream(samplerate=self.frequence, channels=1, dtype='float32') as stream:
            while enregistrement:
                # Lire un petit chunk (0.5 seconde)
                chunk, _ = stream.read(int(self.frequence * 0.5))
                buffer.append(chunk)
                
                # Calculer le volume du chunk
                volume = self.mesurer_volume(chunk)
                
                # Afficher un indicateur visuel
                barre = "#" * int(volume * 50)
                print(f"\rVolume: {volume:.4f} {barre}", end="")
                
                # Detection de silence
                if volume < self.seuil_silence:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duree:
                        print("\n\nSilence detecte - Fin de l'enregistrement")
                        enregistrement = False
                else:
                    silence_start = None
        
        # Concatener tous les chunks
        audio_complet = np.concatenate(buffer)
        print(f"Audio enregistre: {len(audio_complet)/self.frequence:.1f} secondes")
        
        return audio_complet.flatten()

class Transcriber:
    def __init__(self):
        print("Chargement de Whisper 'small'...")
        self.model = whisper.load_model("small", device="cpu")
        print("Modele charge!")
    
    def transcrire(self, audio):
        resultat = self.model.transcribe(
            audio,
            language="fr",
            fp16=False,
            verbose=False
        )
        return resultat["text"].strip()

def main():
    print("=" * 60)
    print("MICROPHONE - DETECTION AUTOMATIQUE")
    print("=" * 60)
    print("Instructions:")
    print("- Parlez normalement")
    print("- Le script detecte quand vous avez fini")
    print("- Apres 4 secondes de silence, il transcrit")
    print("=" * 60)
    
    # Initialiser
    recorder = AudioRecorder()
    transcriber = Transcriber()
    
    while True:
        # 1. Enregistrer avec detection de silence
        audio = recorder.enregistrer_avec_silence()
        
        # 2. Transcrire
        print("\nTranscription en cours...")
        texte = transcriber.transcrire(audio)
        
        # 3. Afficher le resultat
        print("\n" + "=" * 60)
        print("TRANSCRIPTION")
        print("=" * 60)
        print(texte if texte else "[RIEN DETECTE]")
        print("=" * 60)
        
        # 4. Demander si on continue
        print("\nAppuyez sur Entree pour continuer")
        print("Ou tapez 'q' pour quitter")
        choix = input().strip().lower()
        
        if choix == 'q':
            print("Au revoir!")
            break

if __name__ == "__main__":
    main()