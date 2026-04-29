import re
import os

mod_path = r'c:\Users\MSI\Downloads\Projet DS\alia_django\templates\modeling\index.html'
sim_path = r'c:\Users\MSI\Downloads\Projet DS\alia_django\templates\simulator\index.html'

with open(mod_path, 'r', encoding='utf-8') as f:
    mod_html = f.read()

# Extract the module script from modeling/index.html
match = re.search(r'(<script type="module">.*?</script>)', mod_html, re.DOTALL)
if not match:
    print("Could not find script block in modeling")
    exit(1)

mod_script = match.group(1)

# In simulator/index.html, we need to extract the existing module block
with open(sim_path, 'r', encoding='utf-8') as f:
    sim_html = f.read()

sim_new = re.sub(r'<script type="module">.*?</script>', mod_script, sim_html, count=1, flags=re.DOTALL)

with open(sim_path, 'w', encoding='utf-8') as f:
    f.write(sim_new)

print("Replaced script in simulator/index.html successfully.")
