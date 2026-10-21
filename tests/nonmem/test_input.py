import shutil


def test_data_read(pheno):
    df = pheno.dataset
    # FIXME: should have numeric TIME
    assert list(df.iloc[1]) == [1.0, 2.0, 0.0, 1.4, 7.0, 17.3, 0.0, 0.0]
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']


def test_read_raw_dataset(pheno):
    df = pheno.read_raw_dataset()
    assert list(df.iloc[0]) == ['1', '0.', '25.0', '1.4', '7', '0', '1', '1']
    assert list(df.columns) == ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']


def test_remove_individuals_without_observations(create_model_for_test, datadir):
    # First individual of data has no observations
    data = datadir / 'pheno_no_obs_1stID.dta'
    model = create_model_for_test(
        f"$PROBLEM\n$INPUT ID TIME AMT WT APGR DV FA1 FA2\n$DATA {data} IGNORE=@\n$PRED\nY=2\n"
    )
    df = model.dataset
    assert 1 in df['ID']
    assert len(df['ID'].unique()) == 59


def test_ignore_with_synonym(create_model_for_test, pheno_data):
    model = create_model_for_test(
        f"$PROBLEM dfs\n$INPUT ID TIME AMT WT APGR DV=CONC FA1 FA2\n"
        f"$DATA {pheno_data} IGNORE=@ IGNORE=(CONC.EQN.0)\n"
        f"$PRED\nY=1\n"
    )
    assert len(model.dataset) == 155
    model = create_model_for_test(
        f"$PROBLEM dfs\n$INPUT ID TIME AMT WT APGR DV=CONC FA1 FA2\n"
        f"$DATA {pheno_data} IGNORE=@ IGNORE=(DV.EQN.0)\n"
        f"$PRED\nY=1\n"
    )
    assert len(model.dataset) == 155


def test_idv_with_synonym(create_model_for_test, pheno_data, tmp_path):
    # Copy to avoid the .datainfo file
    shutil.copy(pheno_data, tmp_path)
    model = create_model_for_test(
        f"$PROBLEM dfs\n$INPUT ID TIME=TAD AMT WT APGR DV FA1 FA2\n"
        f"$DATA {tmp_path / "pheno.dta"} IGNORE=@\n"
        f"$PRED\nY=1\n"
    )
    assert 'TAD' in model.datainfo.names
