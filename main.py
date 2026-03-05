import time
from speech_to_text_demo import SpeechToText
from ragtest import RAGAssistant
from text_to_speech import TextToSpeech

# ============================================================
# ASSISTANT COMPLET (lie les 3 modules)
# ============================================================

class ALIAFullAssistant:
    def __init__(self, csv_path="vital_products.csv", persist_dir="./chroma_vital_db"):
        print("\n" + "=" * 60)
        print("ALIA - ASSISTANT VOCAL COMPLET")
        print("=" * 60)
        
        # Initialiser les 3 modules
        print("\n1. Initialisation du Speech-to-Text...")
        self.stt = SpeechToText()
        
        print("\n2. Initialisation du RAG...")
        self.rag = RAGAssistant(csv_path, persist_dir)
        
        print("\n3. Initialisation du Text-to-Speech...")
        self.tts = TextToSpeech()
        
        print("\n" + "=" * 60)
        print("ALIA EST PRETE - Parlez-lui!")
        print("=" * 60)
    
    def demarrer(self):
        """Boucle principale de l'assistant"""
        while True:
            print("\n" + "-" * 60)
            print("Appuyez sur Entree pour parler")
            print("Ou tapez 'q' pour quitter")
            choix = input().strip().lower()
            
            if choix == 'q':
                print("Au revoir!")
                break
            
            # 1. Ecouter et transcrire
            print("\n[ECOUTE]")
            question = self.stt.ecouter_et_transcrire()
            
            if not question:
                print("Aucun texte detecte, reessayez")
                continue
            
            # 2. RAG pour obtenir la reponse
            print("\n[RAG] Recherche dans la base...")
            
            # 3. Reponse vocale
            print("\n[VOIX] ALIA repond...")
            reponse = self.rag.ask(question)
            self.tts.parler(reponse)
            
            print("\n" + "-" * 60)
            print("Appuyez sur Entree pour une nouvelle question")
    
# ============================================================
# POINT D'ENTREE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Creer et demarrer l'assistant
    alia = ALIAFullAssistant()
    
    print("\n" + "-" * 60)
    print("MENU:")
    print("1. Mode vocal complet (microphone)")
    print("2. Quitter")
    print("-" * 60)
    
    choix = input("Votre choix (1-2): ").strip()
    
    if choix == "1":
        alia.demarrer()
    else:
        print("Au revoir!")