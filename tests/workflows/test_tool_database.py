from pharmpy.workflows import NullToolDatabase


def test_null_tool_database():
    db = NullToolDatabase("any", sl1=23, model=45, opr=12, dummy="some dummy kwargs")
    db.store_local_file("path")
