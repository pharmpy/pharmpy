import sys, os, importlib

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src')
sys.path.append(path)
pysn = importlib.import_module('pysn')
sys.path.pop()

x = pysn.Model(os.path.join('tests', 'test_data', 'pheno_real.mod'))
names = x.input.column_names()
print(names)
dataset = x.input.dataset_filename()
print(dataset)


a = x.model.get_records('THETA')
for tok in a[0].tokens:
    print(tok.type, repr(str(tok.content)))
