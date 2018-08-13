
import pytest


def test_create_record(nm_api):
    create_record = nm_api.records.create_record

    obj = create_record("PROBLEM ID TIME")
    assert obj.__class__.__name__ == "ProblemRecord"
    assert obj.name == "PROBLEM"

    obj = create_record("PROB ID TIME")
    assert obj.__class__.__name__ == "ProblemRecord"
    assert obj.name == "PROBLEM"

    obj = create_record("PROB  MYPROB")
    assert obj.__class__.__name__, "ProblemRecord"
    assert obj.name, "PROBLEM"
    assert obj.raw_name, "PROB"
    assert obj.string, "MYPROB"


def test_lexical_tokens(nm_api):
    create_record = nm_api.records.create_record

    obj = create_record("SIZES LTH=28")
    assert list(obj._lexical_tokens()) == ["LTH=28"]


def test_get_raw_record_name(nm_api):
    get_raw_record_name = nm_api.records.get_raw_record_name

    assert get_raw_record_name("INPUT ID TIME DV") == "INPUT"
    assert get_raw_record_name("INPUT") == "INPUT"
    assert get_raw_record_name("etas FILE=run1.phi") == "etas"
    assert get_raw_record_name("DA&\nTA test.csv") == "DA&\nTA"
    assert get_raw_record_name("DA&\nT&\nA test.csv") == "DA&\nT&\nA"
    assert get_raw_record_name("DATA\ntest.csv") == "DATA"


def test_get_canonical_record_name(nm_api):
    get_canonical_record_name = nm_api.records.get_canonical_record_name

    assert get_canonical_record_name("INPUT") == "INPUT"
    assert get_canonical_record_name("INP") == "INPUT"
    assert get_canonical_record_name("inpu") == "INPUT"
    assert get_canonical_record_name("DA&\nTA") == "DATA"
    assert get_canonical_record_name("DA&\nT&\nA") == "DATA"


def test_get_record_content(nm_api):
    get_record_content = nm_api.records.get_record_content

    assert get_record_content("INP ID TIME") == " ID TIME"
    assert get_record_content("INP ID TIME\n") == " ID TIME\n"
    assert get_record_content("E&\nST  MAXEVAL=9999") == "  MAXEVAL=9999"
    assert get_record_content("E&\nST  METH=1\nMAX=23") == "  METH=1\nMAX=23"
    assert get_record_content("EST\nMETH=1\nMAX=23") == "\nMETH=1\nMAX=23"


# def test_OptionDescription(nm_api):
#     OptionDescription = nm_api.records.option_description.OptionDescription
#
#     od = OptionDescription(
#         {'name': 'MAXEVALS', 'type': OptionType.VALUE, 'abbreviate': True}
#     )
#     assert od.match("MAX")
