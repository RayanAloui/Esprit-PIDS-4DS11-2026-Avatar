import sys
sys.path.insert(0, r'C:\Users\MSI\Downloads\Projet DS\alia_django\models_ai')

import joblib, types, lstm_model_v2
from body_language_wrapper import SimpleBodyLanguageModel

fake = types.ModuleType('__main__')
fake.BodyLanguageLSTM = lstm_model_v2.BodyLanguageModel
for n in dir(lstm_model_v2): setattr(fake, n, getattr(lstm_model_v2, n))
sys.modules['__main__'] = fake

bundle = joblib.load(r'C:\Users\MSI\Downloads\Projet DS\alia_django\models_ai\lstm_body_language_v2.pkl')

lstm_model_v2._PRELOADED_MODEL = SimpleBodyLanguageModel(bundle['scaler'])

exec(open(r'C:\Users\MSI\Downloads\Projet DS\Body-language-detection.py').read())