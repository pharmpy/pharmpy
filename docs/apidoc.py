# Find private modules to exclude and run sphinx-apidoc

import os
import re
from pathlib import Path

import importlib
import pkgutil
import inspect

import pharmpy

pharmpy_path = Path(pharmpy.__file__).parent

# Remove old rst files to make sure no old lingers
os.system('rm docs/reference/*.rst')

exclude = []
noclass = []
for _, modname, is_pkg in pkgutil.walk_packages([Path('src/pharmpy')], 'pharmpy.'):
    module = importlib.import_module(modname)
    if hasattr(module, '__doc__') and module.__doc__ is not None and re.search(':meta private:', module.__doc__):
        path = Path(module.__file__).relative_to(pharmpy_path)
        exclude.append('src/pharmpy/' + str(path))
    else:
        no_classes = True
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                qualname = obj.__module__ + '.' + obj.__name__
                if qualname.startswith(module.__name__):
                    no_classes = False
                    break
        if no_classes or is_pkg:        # No classes in subpackages to get rid of warnings about inheritance diagrams
            noclass.append(module.__name__)

noclass.append('pharmpy')

cmd = f'sphinx-apidoc -e -f -d 2 -o docs/reference src/pharmpy {" ".join(exclude)}'

os.system(cmd)

for path in Path('docs/reference').glob('pharmpy*.rst'):
    documented_module = str(path.name)[0:-4]
    if documented_module not in noclass:
        with open(path, 'a') as fh:
            print(f'''
Inheritance Diagram
-------------------

.. inheritance-diagram:: {documented_module}
   :parts: 4
''', file=fh)
