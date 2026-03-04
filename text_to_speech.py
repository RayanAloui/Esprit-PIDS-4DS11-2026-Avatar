import pyttsx3
import time

def parler(texte):
    """Fonction simple qui parle"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(texte)
    engine.runAndWait()
    engine.stop() 


while True:
    texte = input("\n Vous: ").strip()
    
    if texte.lower() == 'quit':
        break
    
    if texte:
        print(f"ALIA: {texte}")
        parler(texte)
        time.sleep(0.5)