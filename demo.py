from os import listdir
from os.path import join, dirname, realpath
import sys
import importlib

def iprint(text, width=4):
    lines = text
    if isinstance(text, str):
        lines = text.splitlines()
    elif not all(isinstance(x, str) for x in lines):
        lines = ['%d : %s' % (i, str(x)) for i, x in enumerate(lines)]
    lines = [' '*width + line for line in lines]
    print('\n'.join(lines))

# load ./src/pysn as library 'pysn'
root = dirname(realpath(__file__))
sys.path.append(join(root, 'src'))
pysn = importlib.import_module('pysn')
sys.path.pop()

path = join(root, 'tests', 'testdata', 'nonmem')
for path_model in [join(path, file) for file in listdir(path)]:
    model = pysn.Model(path_model)
    print("pysn.Model('%s')" % path_model)
    print('='*80)

    print('str(model) =')
    iprint(str(model))

    names = model.input.column_names()
    print('\nmodel.input.columns_names() =')
    iprint(names)

    dataset = model.input.dataset_filename()
    print('\nmodel.input.dataset_filename() =')
    iprint(dataset)

    thetas = model.model.get_records('THETA')
    print("\nmodel.model.get_records('THETA') =")
    iprint(thetas)

    print("\nmodel.model.get_records('THETA')[0].tokens =")
    for tok in thetas[0].tokens:
        s = '%-30s %-30s' % (tok.type, repr(str(tok.content)))
        iprint(s)
