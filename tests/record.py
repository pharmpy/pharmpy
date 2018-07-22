import sys
sys.path.append("..")
import unittest


class TestFunctions(unittest.TestCase):
    def test_get_raw_record_name(self):
        from records.record import get_raw_record_name
        self.assertEqual(get_raw_record_name("$INPUT ID TIME DV"), "INPUT")
        self.assertEqual(get_raw_record_name("$INPUT"), "INPUT")
        self.assertEqual(get_raw_record_name("$etas FILE=run1.phi"), "etas")
        self.assertEqual(get_raw_record_name("$DA&\nTA test.csv"), "DA&\nTA")
        self.assertEqual(get_raw_record_name("$DA&\nT&\nA test.csv"), "DA&\nT&\nA")
        self.assertEqual(get_raw_record_name("$DATA\ntest.csv"), "DATA")

    def test_get_canonical_record_name(self):
        from records.record import get_canonical_record_name
        self.assertEqual(get_canonical_record_name("INPUT"), "INPUT")
        self.assertEqual(get_canonical_record_name("INP"), "INPUT")
        self.assertEqual(get_canonical_record_name("inpu"), "INPUT")
        self.assertEqual(get_canonical_record_name("DA&\nTA"), "DATA")
        self.assertEqual(get_canonical_record_name("DA&\nT&\nA"), "DATA")

    def test_get_record_content(self):
        from records.record import get_record_content
        self.assertEqual(get_record_content("$INP ID TIME"), " ID TIME")
        self.assertEqual(get_record_content("$INP ID TIME\n"), " ID TIME\n")
        self.assertEqual(get_record_content("$E&\nST  MAXEVAL=9999"), "  MAXEVAL=9999")
        self.assertEqual(get_record_content("$E&\nST  METH=1\nMAX=23"), "  METH=1\nMAX=23")
        self.assertEqual(get_record_content("$EST\nMETH=1\nMAX=23"), "\nMETH=1\nMAX=23")

    def test_create_record(self):
        from records.record import create_record
        obj = create_record("$PROBLEM ID TIME")
        self.assertEqual(obj.__class__.__name__, "RawRecord")
        self.assertEqual(obj.name, "PROBLEM")
        obj = create_record("$PROB ID TIME")
        self.assertEqual(obj.__class__.__name__, "RawRecord")
        self.assertEqual(obj.name, "PROBLEM")

if __name__ == '__main__':
    unittest.main()
