from psn.model import Model as Model

x = Model("pheno_real.mod")
names = x.input.column_names()
print(names)
dataset = x.input.dataset_filename()
print(dataset)


a = x.model.get_records('THETA')
for tok in a[0].tokens:
    print(tok.type, tok.content)
