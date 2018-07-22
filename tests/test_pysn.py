
import os.path
from pysn import Model


def test_main():
    path = os.path.join(os.path.realpath(os.path.dirname(__file__)))
    path = os.path.join(path, 'test_data')
    pheno = Model(os.path.join(path, 'pheno_real.mod'))
