def test_split_rawrecord_name(records):
    split_raw_record_name = records.factory.split_raw_record_name

    assert split_raw_record_name('$INPUT ID TIME DV') == ('$INPUT', ' ID TIME DV')
    assert split_raw_record_name('$INPUT ID TIME DV\n') == ('$INPUT', ' ID TIME DV\n')
    assert split_raw_record_name('$PK') == ('$PK', '')
    assert split_raw_record_name('$etas FILE=run1.phi') == ('$etas', ' FILE=run1.phi')
    assert split_raw_record_name('  $DAT test.csv') == ('  $DAT', ' test.csv')
    assert split_raw_record_name('$PRED\nTVCL=THETA(1)') == ('$PRED', '\nTVCL=THETA(1)')
    assert split_raw_record_name('$EST\nMETH=1\nMAX=23') == ('$EST', '\nMETH=1\nMAX=23')


def test_get_canonical_record_name(records):
    get_canonical_record_name = records.factory.get_canonical_record_name

    assert get_canonical_record_name('$PROBLEM') == 'PROBLEM'
    assert get_canonical_record_name(' $PROB') == 'PROBLEM'
    assert get_canonical_record_name('$prob') == 'PROBLEM'
    assert get_canonical_record_name('  $PROBLE') == 'PROBLEM'


def test_create_record(records):
    create_record = records.factory.create_record

    obj = create_record("$PROBLEM ID TIME")
    assert obj.__class__.__name__ == "ProblemRecord"
    assert obj.raw_name == '$PROBLEM'

    obj = create_record(" $PROBL ID TIME")
    assert obj.__class__.__name__ == "ProblemRecord"
    assert obj.raw_name == ' $PROBL'

    obj = create_record("$PROB  MYPROB")
    assert obj.__class__.__name__, "ProblemRecord"
    assert obj.raw_name == '$PROB'
