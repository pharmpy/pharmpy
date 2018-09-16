
import pandas as pd
from pandas.util.testing import assert_frame_equal

from pharmpy.input import InputFilterOperator, InputFilter, InputFilters


def test_input_filter_operator():
    op = InputFilterOperator.EQUAL
    assert str(op) == "=="


def test_input_filter():
    f = InputFilter("a", InputFilterOperator.EQUAL, 28)
    assert str(f) == "a == 28"


def test_input_filters():
    f1 = InputFilter("ID", InputFilterOperator.EQUAL, 2)
    f2 = InputFilter("AMT", InputFilterOperator.EQUAL, 10)
    filters = InputFilters([f1, f2])
    assert str(filters) == "not(ID == 2 or AMT == 10)"

    filters.accept = True
    assert str(filters) == "ID == 2 or AMT == 10"

    f = InputFilter("a", InputFilterOperator.EQUAL, 28)
    filters = InputFilters([f])
    d = {'a': [1, 2, 28], 'b': [2, 4, 8]}
    df = pd.DataFrame(d)
    filters.apply(df)
    answer = pd.DataFrame({'a': [1, 2], 'b': [2, 4]})
    assert_frame_equal(df, answer)

    filters.accept = True
    d = {'a': [1, 2, 28], 'b': [2, 4, 8]}
    df = pd.DataFrame(d)
    result = filters.apply(df, inplace=False)
    answer = pd.DataFrame({'a': [28], 'b': [8]})
    assert_frame_equal(result, answer)

    f2 = InputFilter("b", InputFilterOperator.EQUAL, 4)
    filters.append(f2)
    result = filters.apply(df, inplace=False)
    assert_frame_equal(result, pd.DataFrame({'a': [2, 28], 'b': [4, 8]}))
