from pathlib import Path


def test_title_limits(parser):
    rec = parser.parse("$ETAS FILE=run1.phi").records[0]
    assert rec.path == Path("run1.phi")
