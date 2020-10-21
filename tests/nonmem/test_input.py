from io import StringIO

import pharmpy
from pharmpy.data import ColumnType


def test_data_read(pheno):
    df = pheno.dataset
    # FIXME: should have numeric TIME
    assert list(df.iloc[1]) == [1.0, '2.0', 0.0, 1.4, 7.0, 17.3, 0.0, 0.0]
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']


def test_read_raw_dataset(pheno):
    df = pheno.read_raw_dataset()
    assert list(df.iloc[0]) == ['1', '0.', '25.0', '1.4', '7', '0', '1', '1']
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    assert df.pharmpy.column_type['ID'] == ColumnType.ID


def test_ignore_with_synonym(pheno_data):
    model = pharmpy.Model(
        StringIO(
            f"$PROBLEM dfs\n$INPUT ID TIME AMT WT APGR DV=CONC FA1 FA2\n"
            f"$DATA {pheno_data} IGNORE=@ IGNORE=(CONC.EQN.0)"
        )
    )
    df = model.dataset
    assert len(df) == 155
    model = pharmpy.Model(
        StringIO(
            f"$PROBLEM dfs\n$INPUT ID TIME AMT WT APGR DV=CONC FA1 FA2\n"
            f"$DATA {pheno_data} IGNORE=@ IGNORE=(DV.EQN.0)"
        )
    )
    df = model.dataset
    assert len(df) == 155
