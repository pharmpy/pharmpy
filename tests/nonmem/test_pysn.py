from pysn import Model


def test_model(pheno_real):
    with open(str(pheno_real), 'r') as f:
        buf = f.read()
    pheno = Model(pheno_real)
    assert str(pheno) == buf
