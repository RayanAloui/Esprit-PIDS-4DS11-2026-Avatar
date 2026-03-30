import sys
import os

print("=" * 60)
print("🔍 DIAGNOSTIC VENV")
print("=" * 60)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"VENV activé: {sys.prefix != sys.base_prefix}")
print(f"Site packages: {os.path.dirname(sys.executable)}\\Lib\\site-packages")
print("=" * 60)

# Tester whisper
try:
    import whisper
    print("✅ Whisper trouvé!")
    print(f"   Chemin: {whisper.__file__}")
except ImportError as e:
    print(f"❌ Whisper non trouvé: {e}")

print("=" * 60)