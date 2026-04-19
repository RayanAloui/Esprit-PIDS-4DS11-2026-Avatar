import urllib.request
import json
prompt = 'Tu es un assistant. Donne moi 1 conseil en json. Structure: [{"axe": "nom", "description": "desc", "priorite": "haute", "seuil_cible": "1"}]'
payload = json.dumps({
    'model': 'llama3.2:latest',
    'prompt': prompt,
    'format': 'json',
    'stream': False
}).encode('utf-8')
req = urllib.request.Request('http://localhost:11434/api/generate', data=payload, headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req) as resp:
        print(json.loads(resp.read()).get('response'))
except Exception as e:
    print(e)
