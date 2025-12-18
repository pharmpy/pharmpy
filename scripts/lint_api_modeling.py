from pathlib import Path
import re

parent_dir = Path(__file__).parent.parent
docs_path = parent_dir / 'docs' / 'api_modeling.rst'
modeling_path = parent_dir / 'src' / 'pharmpy' / 'modeling' / '__init__.py'

with open(docs_path, 'r') as f:
    docs = f.read()

with open(modeling_path, 'r') as f:
    modeling = f.read()

matches = re.findall(re.compile(r'^[^\n]\s+[A-Za-z].*$', re.MULTILINE), docs)
docs_funcs = {func.strip() for func in matches}

match = re.search(r"__all__\s*=\s*\[\s*(.*?)\s*]", modeling, re.DOTALL).group(1)
match = match.replace(',', '').replace('\'', '')
modeling_funcs = {func.strip() for func in match.split('\n')}

if diff := sorted(docs_funcs - modeling_funcs):
    raise ValueError(f"Functions not found in modeling: {diff}")

if diff := sorted(modeling_funcs - docs_funcs):
    raise ValueError(f"Functions not found in documentation: {diff}")
