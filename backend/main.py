import sys
import requests
from speech_to_text_demo import SpeechToText
from text_to_speech import TextToSpeech

# ============================================================
# CONFIG
# ============================================================
SERVER_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{SERVER_URL}/ask_alia"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"

# ============================================================
# ASSISTANT VOCAL (client du serveur ALIA)
# ============================================================

class ALIAFullAssistant:
    def __init__(self):
        print("\n" + "=" * 60)
        print("ALIA - ASSISTANT VOCAL")
        print("=" * 60)

        # Vérifier que le serveur est actif
        print("\n1. Vérification du serveur ALIA...")
        self._check_server()

        # Initialiser STT
        print("\n2. Initialisation du Speech-to-Text...")
        self.stt = SpeechToText()

        # Initialiser TTS local (pyttsx3)
        print("\n3. Initialisation du Text-to-Speech...")
        self.tts = TextToSpeech()

        print("\n" + "=" * 60)
        print("ALIA EST PRÊTE - Je vous écoute...")
        print("=" * 60)
        print("[ECOUTE]")
        sys.stdout.flush()

    def _check_server(self):
        """Vérifie que alia_server est bien démarré"""
        try:
            r = requests.get(HEALTH_ENDPOINT, timeout=5)
            data = r.json()
            print(f"  ✓ Serveur actif — {data.get('num_products', '?')} produits chargés")
        except requests.exceptions.ConnectionError:
            print("\n  ✗ ERREUR : Le serveur ALIA n'est pas démarré !")
            print("  Lancez d'abord dans un autre terminal :")
            print("  → python alia_server.py")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Erreur inattendue : {e}")
            sys.exit(1)

    def _ask_server(self, question: str) -> dict:
        """Envoie la question au serveur et récupère la réponse texte"""
        try:
            r = requests.post(
                ASK_ENDPOINT,
                json={"text": question},
                timeout=120
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            return {"text": "Désolée, le serveur a mis trop de temps à répondre."}
        except requests.exceptions.RequestException as e:
            return {"text": f"Erreur de connexion au serveur : {e}"}

    def parler(self, texte: str):
        """Synthèse vocale locale via pyttsx3"""
        print(f"\nALIA: {texte}")
        print("[PARLE]")
        sys.stdout.flush()
        self.tts.parler(texte)
        print("[ECOUTE]")
        sys.stdout.flush()

    def demarrer(self):
        """Boucle principale"""
        while True:
            print("\n" + "-" * 60)
            print("Appuyez sur Entrée pour parler | 'q' pour quitter")
            choix = input().strip().lower()

            if choix == 'q':
                print("Au revoir !")
                break

            # 1. Écouter et transcrire
            print("\n[ECOUTE]")
            question = self.stt.ecouter_et_transcrire()

            if not question:
                print("Aucun texte détecté, réessayez.")
                continue

            print(f"\nVous : {question}")

            # 2. Envoyer au serveur RAG
            print("\n[RAG] Envoi au serveur ALIA...")
            resultat = self._ask_server(question)

            # 3. Répondre vocalement
            self.parler(resultat.get("text", "Pas de réponse."))


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("\n⚠  Assurez-vous que le serveur tourne : python alia_server.py")

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
        print("Au revoir !")