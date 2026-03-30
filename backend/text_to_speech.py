import pyttsx3
import time

class TextToSpeech:
    def __init__(self):
        print("Initialisation de la voix...")
        self.vitesse = 150
        print("Voix prete!")
    
    def parler(self, texte):
        if not texte or not texte.strip():
            return
        
        print(f"\nALIA: {texte}")
        
        engine = pyttsx3.init()
        engine.setProperty('rate', self.vitesse)
        
        # Chercher voix francaise
        voices = engine.getProperty('voices')
        for v in voices:
            if 'french' in v.name.lower() or 'francais' in v.name.lower():
                engine.setProperty('voice', v.id)
                break
        
        engine.say(texte)
        engine.runAndWait()
        engine.stop()

# Pour test indépendant
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.parler("Bonjour, je suis ALIA")