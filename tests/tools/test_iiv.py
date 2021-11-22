from io import StringIO

from pharmpy import Model
from pharmpy.tools.iiv.tool import _get_iiv_combinations


def test_get_iiv_combinations(testdata, pheno_path):
    model = Model(
        StringIO(
            '''
$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V*EXP(ETA(3))*EXP(ETA(4))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA DIAGONAL(4)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
 0.031128
 0.031128
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
        )
    )

    model.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset

    assert _get_iiv_combinations(model) == [
        ['ETA(1)', 'ETA(2)'],
        ['ETA(1)', 'ETA(3)'],
        ['ETA(1)', 'ETA(4)'],
        ['ETA(2)', 'ETA(3)'],
        ['ETA(2)', 'ETA(4)'],
        ['ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)'],
        ['ETA(1)', 'ETA(2)', 'ETA(4)'],
        ['ETA(1)', 'ETA(3)', 'ETA(4)'],
        ['ETA(2)', 'ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)', 'ETA(4)'],
    ]
