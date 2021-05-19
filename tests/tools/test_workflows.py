from pharmpy import Model
from pharmpy.tools.workflows import Task, Workflow


def test_as_dict():
    wf = Workflow([Task('t1', 'func', 'x'), Task('t2', 'func', ('t1', 'y'))])

    assert wf.as_dict() == {'t1': ('func', 'x'), 't2': ('func', 't1', 'y')}


def test_merge(pheno_path):
    model_1 = Model(pheno_path)
    model_2 = Model(pheno_path)

    wf_1 = Workflow(Task('t1', 'func', 'x'))
    wf_1.add_infiles(('/test/infile/dir1/data.csv', 'dest1'))
    wf_1.add_outfiles({model_1: 'dest1/test.txt'})

    wf_2 = Workflow(Task('t2', 'func', ('t1', 'y')))
    wf_2.add_infiles(('/test/infile/dir2/data.csv', 'dest2'))
    wf_2.add_outfiles({model_2: 'dest2/test.txt'})

    wf_1.merge_workflows(wf_2)

    assert wf_1.as_dict() == {'t1': ('func', 'x'), 't2': ('func', 't1', 'y')}
