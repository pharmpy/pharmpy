# Use readme_renderer which is used by PyPI to render the package info
# to generate and display html

from pathlib import Path
from readme_renderer import rst
import webbrowser


readme_path = Path(__file__).parent.parent / 'README.rst'
html_path = Path("/tmp/readme.html")


with open(readme_path, 'r') as f:
    text = f.read()

html = rst.render(text)

with open(html_path, 'w') as f:
    f.write(html)

webbrowser.open_new("file:///tmp/readme.html")
