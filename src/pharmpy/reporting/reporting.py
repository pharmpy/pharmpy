import os
import re
import shutil
import warnings
from pathlib import Path
from urllib.request import urlopen

from bs4 import BeautifulSoup
from csscompressor import compress
from sphinx.application import Sphinx

from pharmpy.utils import TemporaryDirectory, TemporaryDirectoryChanger


def generate_report(rst_path, results_path):
    """Generate report from rst and results json"""
    results_path = Path(results_path)
    with TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        source_path = tmp_path / 'source'
        source_path.mkdir()
        shutil.copy(rst_path, source_path / 'results.rst')
        shutil.copy(results_path / 'results.json', source_path)

        conf_path = Path(__file__).parent

        # Change directory for results.json to be found
        with TemporaryDirectoryChanger(source_path):
            with open(os.devnull, 'w') as devnull:
                with warnings.catch_warnings():
                    # Don't display deprecation warnings.
                    # See https://github.com/pharmpy/pharmpy/issues/20
                    warnings.filterwarnings("ignore", message="The app.add_stylesheet")
                    warnings.filterwarnings("ignore", message="The app.add_javascript")
                    app = Sphinx(
                        str(source_path),
                        str(conf_path),
                        str(tmp_path),
                        str(tmp_path),
                        "singlehtml",
                        status=devnull,
                        warning=devnull,
                    )
                    app.build()

        # Write missing altair css
        with open(tmp_path / '_static' / 'altair-plot.css', 'w') as dh:
            dh.write(
                """.vega-actions a {
    margin-right: 12px;
    color: #757575;
    font-weight: normal;
    font-size: 13px;
}

.vega-embed {
    margin-bottom: 20px;
    margin-top: 20px;
    width: 100%;
}
"""
            )
        report_path = tmp_path / 'results.html'
        embed_css_and_js(tmp_path / 'results.html', report_path)
        shutil.copy(report_path, results_path)


def embed_css_and_js(html, target):
    """Embed all external css and javascript into an html"""
    with open(html, 'r', encoding='utf-8') as sh:
        soup = BeautifulSoup(sh, features='lxml')

    scripts = soup.findAll("script", attrs={"src": True})

    for script in scripts:
        source = script.attrs['src']
        if source.startswith('http'):
            infile = urlopen(source)
            content = infile.read().decode('utf-8')
            infile.close()
        else:
            path = html.parent / source
            with open(path, 'r') as sh:
                content = sh.read()

        # Minification with jsmin didn't work
        tag = soup.new_tag('script')
        tag['type'] = 'text/javascript'
        tag.append(content)
        script.replace_with(tag)

    stylesheets = soup.findAll("link", attrs={"rel": "stylesheet"})

    for stylesheet in stylesheets:
        stylesheet_src = stylesheet.attrs['href']
        tag = soup.new_tag("style")
        tag['type'] = 'text/css'
        path = html.parent / stylesheet_src
        with open(path, 'r') as sh:
            content = sh.read()
        if '@import' in content:
            import_files = re.findall(r'@import\s+url\("([A-Za-z0-9.]+)"\)', content)
            for name in import_files:
                with open(path.parent / name, 'r') as import_file:
                    import_content = import_file.read()
                content = re.sub(r'@import\s+url\("' + name + r'"\);', import_content, content)
        minified_content = compress(content)
        tag.append(minified_content)
        stylesheet.replace_with(tag)

    with open(target, 'w', encoding='utf-8') as dh:
        dh.write(str(soup))
