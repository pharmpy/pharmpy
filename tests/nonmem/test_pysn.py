from pysn import Model


def test_pheno(pheno_real):
    with open(pheno_real, 'r') as f:
        buf = f.read()
    pheno = Model(pheno_real)
    assert str(pheno) == buf
