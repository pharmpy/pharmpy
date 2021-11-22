import shutil
from io import StringIO

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_iiv(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)

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
        S1=V*EXP(ETA(3))

        $ERROR
        Y=F+F*EPS(1)

        $THETA (0,0.00469307) ; TVCL
        $THETA (0,1.00916) ; TVV
        $OMEGA DIAGONAL(3)
         0.0309626  ;       IVCL
         0.031128  ;        IVV
         0.031128
        $SIGMA 0.013241

        $ESTIMATION METHOD=1 INTERACTION
        '''
            )
        )
        import pandas as pd

        pd.set_option('display.max_columns', None)

        model.dataset_path = tmp_path / 'pheno.dta'
        res = run_tool('iiv', model)
        assert res
