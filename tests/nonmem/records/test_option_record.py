def test_create_record(api):
    record = api.records.create_record("ESTIMATION MAXEVALS=9999 INTER")
    pairs = record.option_pairs() 
    #create_record = api.records.create_record

    #obj = create_record("PROBLEM ID TIME")
    #assert obj.__class__.__name__ == "ProblemRecord"
    #assert obj.name == "PROBLEM"

    #obj = create_record("PROB ID TIME")
    #assert obj.__class__.__name__ == "ProblemRecord"
    #assert obj.name == "PROBLEM"

    #obj = create_record("PROB  MYPROB")
    #assert obj.__class__.__name__, "ProblemRecord"
    #assert obj.name, "PROBLEM"
    #assert obj.raw_name, "PROB"
    #assert obj.string, "MYPROB"
