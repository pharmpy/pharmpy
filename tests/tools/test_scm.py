from io import StringIO

import pandas as pd
from numpy import isnan, nan
from pytest import approx

import pharmpy.tools.scm.results as scm


def test_psn_scm_options(testdata):
    options = scm.psn_scm_options(testdata / 'nonmem' / 'scm')
    assert options['included_relations'] is None
    assert options['test_relations'] is None
    assert options['logfile'] == 'scmlog.txt'

    options = scm.psn_scm_options(testdata / 'nonmem' / 'scm' / 'onlyforward_dir1')
    assert options['included_relations'] == {'CL': {'APGR': '2'}, 'V': {'CVD1': '2'}}
    assert options['test_relations'] == {
        'CL': ['WGT', 'APGR', 'CV1', 'CV2', 'CV3'],
        'V': ['CVD1', 'WGT'],
    }
    assert options['logfile'] == 'scmlog.txt'

    options = scm.psn_scm_options(testdata / 'nonmem' / 'scm' / 'backward_dir1')
    assert options['included_relations'] == {
        'CL': {'WGT': '2', 'CV1': '2'},
        'V': {'CV2': '2', 'WGT': '2', 'CVD1': '2'},
    }
    assert options['test_relations'] == {
        'CL': ['WGT', 'APGR', 'CV1', 'CV2', 'CV3'],
        'V': ['CVD1', 'CV2', 'WGT'],
    }
    assert options['logfile'] == 'scmlog.txt'


def test_main(testdata):
    basedir = testdata / 'nonmem' / 'scm'
    dirs = {
        'mer1': basedir / 'mergeofv_dir1',
        'mer2': basedir / 'mergeofv_dir2',
        'onlyf': basedir / 'onlyforward_dir1',
    }

    res1 = scm.psn_scm_results(dirs['mer1'])
    res2 = scm.psn_scm_results(dirs['mer2'])

    assert [float(res1.steps['extended_ofv'].iloc[5])] == approx([219208.47277])
    assert res1.steps['directory'].iloc[0] == 'm1'
    assert isnan(res1.steps['extended_ofv'].iloc[6])
    assert res1.ofv_summary(final_included=False) is not None
    assert res1.candidate_summary() is not None

    assert [float(res2.steps['ofv_drop'].iloc[2])] == approx([18.5591])
    assert res2.ofv_summary(iterations=False) is not None
    assert res2.candidate_summary() is not None

    parcovdict = {
        'CLCV2': ('CL', 'CV2'),
        'CLWGT': ('CL', 'WGT'),
        'CLCV3': ('CL', 'CV3'),
        'CLCV1': ('CL', 'CV1'),
        'CLAPGR': ('CL', 'APGR'),
        'VWGT': ('V', 'WGT'),
        'VCVD1': ('V', 'CVD1'),
        'VCV2': ('V', 'CV2'),
    }

    res3 = scm.psn_scm_results(dirs['onlyf'])
    assert res3.steps is not None
    sum1 = res3.ofv_summary()
    assert [float(x) for x in sum1['pvalue'].values] == approx([nan, nan, 2.22e-13], nan_ok=True)
    assert res3.ofv_summary(iterations=False) is not None

    cand1 = res3.candidate_summary()
    all_columns = [
        'parameter',
        'covariate',
        'extended_state',
        'N_test',
        'N_ok',
        'N_localmin',
        'N_failed',
        'StepIncluded',
    ]
    correct = """CL,APGR,3, 2,2,   0,0,
CL,CV1,2, 2,2,   0,0,
CL,CV2,2, 2,2,   0,0,
CL,CV3,2, 2,2,   0,0,
CL,WGT,2, 1,1,   0,0,1
CL,WGT,3,  1,1,   0, 0,
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1, 2], names=all_columns, header=None)
    pd.testing.assert_frame_equal(cand1, correct)

    df1 = scm.psn_scm_parse_logfile(
        basedir / 'scmplus_dir1' / 'scmlog.txt',
        {'directory': '/home/kajsa/sandbox/asrscm/asrscm_dir56', 'included_relations': None},
        parcovdict,
    )
    assert df1 is not None
    direct = df1['directory'].iloc[0]
    assert direct == 'rundir/m1' or direct == 'rundir\\m1'
    sum1 = scm.ofv_summary_dataframe(df1)
    assert [float(x) for x in sum1['pvalue'].values] == approx(
        [2.26e-10, 0.000002, 0.002178, 0.014192, 0.490010, 9999, 0.027672, 8.15e-24]
    )

    cand1 = scm.candidate_summary_dataframe(df1)
    all_columns = [
        'parameter',
        'covariate',
        'extended_state',
        'N_test',
        'N_ok',
        'N_localmin',
        'N_failed',
        'StepIncluded',
        'StepStashed',
        'StepReadded',
        'BackstepRemoved',
    ]
    correct = """CL,APGR,2,4,2,   2, 0,, 1,3,
CL,CV1,2, 2,2,   0, 0,3,1,3,7
CL,WGT,2, 2,2,   0, 0,2,,, 8
V,CV2,2,  3,3,   0, 0,4,1,3,6
V,CVD1,2, 4,3,   1, 0,,1,3,
V,WGT,2,  1,1,   0, 0,1,,,
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1, 2], names=all_columns, header=None)
    pd.testing.assert_frame_equal(cand1, correct)

    df2 = scm.psn_scm_parse_logfile(
        basedir / 'scm_dir1' / 'scmlog.txt',
        {'directory': '/home/kajsa/sandbox/asrscm/scm_dir7', 'included_relations': None},
        parcovdict,
    )
    assert df2 is not None
    sum2 = scm.ofv_summary_dataframe(df2)
    assert [float(x) for x in sum2['ofv_drop'].iloc[2:7].values] == approx(
        [2.90120, 3.00434, 1.18187, 1.18187, 3.00434]
    )
    cand2 = scm.candidate_summary_dataframe(df2)
    assert [int(t) for t in cand2['N_test']] == [6, 3, 2, 4, 5, 1]

    df3 = scm.psn_scm_parse_logfile(
        basedir / 'backward_dir1' / 'scmlog.txt',
        {
            'directory': '/home/kajsa/sandbox/asrscm/scm_dir10',
            'included_relations': {
                'CL': {'WGT': '2', 'CV1': '2'},
                'V': {'CV2': '2', 'WGT': '2', 'CVD1': '2'},
            },
        },
        parcovdict,
    )
    assert df3 is not None
    assert scm.ofv_summary_dataframe(df3) is not None
    cand1 = scm.candidate_summary_dataframe(df3)

    correct = """V,CVD1,2,1
V,CV2,2,2
CL,CV1,2,3
"""
    correct = pd.read_csv(
        StringIO(correct),
        index_col=[0, 1, 2],
        names=['parameter', 'covariate', 'extended_state', 'BackstepRemoved'],
        header=None,
    )
    pd.testing.assert_frame_equal(cand1, correct)

    df4 = scm.psn_scm_parse_logfile(
        basedir / 'localmin.logf',
        {'directory': '/home/kajsa/sandbox/asrscm/asrscm_dir5', 'included_relations': None},
        parcovdict,
    )
    assert df4 is not None
    assert scm.ofv_summary_dataframe(df4) is not None
    assert scm.candidate_summary_dataframe(df4) is not None

    gof1 = scm.psn_scm_parse_logfile(
        basedir / 'gofofv_dir1' / 'scmlog.txt',
        {'directory': '/home/kajsa/sandbox/asrscm/scm_dir8', 'included_relations': None},
        parcovdict,
    )
    assert gof1 is not None
    assert scm.ofv_summary_dataframe(gof1) is not None
    assert scm.candidate_summary_dataframe(gof1) is not None


def test_parse_runtable_block_gof_pval(testdata):
    basedir = testdata / 'nonmem' / 'scm'
    step = {
        'b_pval': basedir / 'log_steps' / 'backward_pval_1.txt',
        'f_pval': basedir / 'log_steps' / 'forward_pval_1.txt',
        'm_pos': basedir / 'log_steps' / 'forward_pval_2.txt',
        'm_neg': basedir / 'log_steps' / 'forward_pval_3.txt',
        'f_nosignif': basedir / 'log_steps' / 'forward_pval_4.txt',
        'b_nosignif': basedir / 'log_steps' / 'backward_pval_2.txt',
    }

    parcov = {
        'CLCV2': ('CL', 'CV2'),
        'CLWGT': ('CL', 'WGT'),
        'CLCV3': ('CL', 'CV3'),
        'CLCV1': ('CL', 'CV1'),
        'CLAPGR': ('CL', 'APGR'),
        'VWGT': ('V', 'WGT'),
        'VCVD1': ('V', 'CVD1'),
        'VCV2': ('V', 'CV2'),
    }

    columns = [
        'model',
        'parameter',
        'covariate',
        'extended_state',
        'reduced_ofv',
        'extended_ofv',
        'ofv_drop',
        'delta_df',
        'pvalue',
        'is_backward',
        'extended_significant',
    ]

    with open(step['b_pval']) as tab:
        block = [row for row in tab]

    correct = """0,CLWGT-1,CL,WGT,,657.31367,653.74502,3.56865,1,0.058880,True,False
1,VCV2-1,V,CV2,,654.03358,653.74502,0.28856,1,0.59115,True,False
2,VWGT-1,V,WGT,,729.06993,653.74502,75.32491,1,3.99e-18,True,True
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0], names=columns, header=None)
    table = scm.parse_runtable_block(block, parcov_dictionary=parcov)
    pd.testing.assert_frame_equal(table, correct)

    with open(step['f_pval']) as tab:
        block = [row for row in tab]

    correct = """0,CLAPGR-2,CL,APGR,2,724.0,725.6143599999999,-1.61436,9,9999.0,False,False
1,CLCV1-2,CL,CV1,2,724.0,723.11595,0.88405,1,0.34709,False,False
2,CLWGT-2,CL,WGT,2,724.0,699.47082,24.52918,1,1e-06,False,True
3,VCV2-2,V,CV2,2,724.0,724.9772,-0.9772,1,9999.0,False,False
4,VCVD1-2,V,CVD1,2,724.0,723.0435,0.9565,1,0.32807,False,False
5,VWGT-2,V,WGT,2,724.0,685.3777,38.6223,1,5.14e-10,False,True
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0], names=columns, header=None)
    table = scm.parse_runtable_block(block, parcov_dictionary=parcov)
    table = table.astype({'extended_state': 'int64'})
    pd.testing.assert_frame_equal(table, correct)

    # log files with merged base and new ofv. Small test to ensure correct
    # unmerging of column values
    with open(step['m_pos']) as tab:
        block = [row for row in tab]
    table = scm.parse_runtable_block(block)
    assert [float(x) for x in table['extended_ofv'].iloc[5:8].values] == approx(
        [219208.47277, nan, 217228.28027], nan_ok=True
    )

    with open(step['m_neg']) as tab:
        block = [row for row in tab]
    table = scm.parse_runtable_block(block)
    assert [float(x) for x in table['extended_ofv'].iloc[0:3].values] == approx(
        [-10620.89249, -10638.97128, -10629.37133]
    )

    with open(step['f_nosignif']) as tab:
        block = [row for row in tab]
    table = scm.parse_runtable_block(block)
    assert [float(x) for x in table['pvalue'].iloc[0:2].values] == approx([9999, 9999])
    assert [x for x in table['extended_significant'].iloc[0:2].values] == [False, False]

    with open(step['b_nosignif']) as tab:
        block = [row for row in tab]
    table = scm.parse_runtable_block(block)
    assert [float(table['pvalue'].iloc[0])] == approx([8.15e-24])
    assert table['extended_significant'].iloc[0]  # assert is True


def test_parse_runtable_block_gof_ofv(testdata):
    basedir = testdata / 'nonmem' / 'scm'
    step = {
        'b_ofv': basedir / 'log_steps' / 'backward_ofv_1.txt',
        'f_ofv': basedir / 'log_steps' / 'forward_ofv_1.txt',
    }

    parcov = {
        'CLCV2': ('CL', 'CV2'),
        'CLWGT': ('CL', 'WGT'),
        'CLCV3': ('CL', 'CV3'),
        'CLCV1': ('CL', 'CV1'),
        'CLAPGR': ('CL', 'APGR'),
        'VWGT': ('V', 'WGT'),
        'VCVD1': ('V', 'CVD1'),
        'VCV2': ('V', 'CV2'),
    }

    columns = [
        'model',
        'parameter',
        'covariate',
        'extended_state',
        'reduced_ofv',
        'extended_ofv',
        'ofv_drop',
        'goal_ofv_drop',
        'is_backward',
        'extended_significant',
    ]

    with open(step['b_ofv']) as tab:
        block = [row for row in tab]

    correct = """0,CLCV1-1,CL,CV1,,587.20249,584.15462,3.04787,7.63,True,False
1,CLWGT-1,CL,WGT,,622.67113,584.15462,38.51651,7.63,True,True
2,VCV2-1,V,CV2,,587.15896,584.15462,3.00434,7.63,True,False
3,VWGT-1,V,WGT,,669.56839,584.15462,85.41377,7.63,True,True
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0], names=columns, header=None)
    table = scm.parse_runtable_block(block, parcov_dictionary=parcov)
    pd.testing.assert_frame_equal(table, correct)

    with open(step['f_ofv']) as tab:
        block = [row for row in tab]

    correct = """0,CLAPGR-2,CL,APGR,2,725.602,705.46995,20.13205,16.92,False,True
1,CLCV1-2,CL,CV1,2,725.602,722.68945,2.91255,2.84,False,True
2,CLWGT-2,CL,WGT,2,725.602,672.98298,52.61901999999999,2.84,False,True
3,VCV2-2,V,CV2,2,725.602,724.65062,0.9513799999999999,2.84,False,False
4,VCVD1-2,V,CVD1,2,725.602,717.75649,7.845510000000001,2.84,False,True
5,VWGT-2,V,WGT,2,725.602,629.27205,96.32995,2.84,False,True
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0], names=columns, header=None)
    table = scm.parse_runtable_block(block, parcov_dictionary=parcov)
    table = table.astype({'extended_state': 'int64'})
    pd.testing.assert_frame_equal(table, correct)


def test_parse_mixed_block(testdata):
    block = [
        'Using user-defined ofv change criteria',
        'Degree of freedom  |  Required ofv change',
        '         1         -          7.63',
        '         2         -          10.21',
        '         3         -          11.34',
        '         4         -          13.28',
        '         5         -          15.09',
        '         6         -          16.81',
        '         7         -          18.48',
        '         8         -          20.09',
        '         9         -          21.67',
        '         10         -          23.21',
        r'Model directory' + ' /sandbox/asrscm/scm_dir8/backward_scm_dir1/scm_dir1/scm_dir1/m1',
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 == '/sandbox/asrscm/scm_dir8/backward_scm_dir1/scm_dir1/scm_dir1/m1'
    assert readded == list()
    assert stashed == list()
    assert included_relations == dict()

    block = [
        'Forward search done. Starting backward search inside forward top level directory',
        r'Model directory /home/run1080scm_3/backward_scm_dir1/m1',
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 == '/home/run1080scm_3/backward_scm_dir1/m1'
    assert readded == list()
    assert stashed == list()
    assert included_relations == dict()

    block = [
        'Taking a step forward in adaptive scope reduction scm after reducing scope '
        + r'with 15 relations : '
        + r'CL-SEX,CL-QUAT,KM-AGE,KM-RACE,KM-SEX,KM-QUAT,V1-AGE,V1-DOSE,'
        + r'V1-QUAT,V2-AGE,V2-RACE,VM-AGE,VM-RACE,VM-SEX,VM-QUAT',
        'Model directory /home/Drug/run15asr_2.dir/rundir/reduced_forward_scm_dir2/m1',
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 == '/home/Drug/run15asr_2.dir/rundir/reduced_forward_scm_dir2/m1'
    assert readded == list()
    assert stashed == [
        ('CL', 'SEX'),
        ('CL', 'QUAT'),
        ('KM', 'AGE'),
        ('KM', 'RACE'),
        ('KM', 'SEX'),
        ('KM', 'QUAT'),
        ('V1', 'AGE'),
        ('V1', 'DOSE'),
        ('V1', 'QUAT'),
        ('V2', 'AGE'),
        ('V2', 'RACE'),
        ('VM', 'AGE'),
        ('VM', 'RACE'),
        ('VM', 'SEX'),
        ('VM', 'QUAT'),
    ]
    assert included_relations == dict()

    block = [
        'Taking a step forward in adaptive scope reduction '
        + r'scm after reducing scope with 1 relations : V2-SEX',
        'Model directory /home/shared/rundir/reduced_forward_scm_dir3/m1',
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 == '/home/shared/rundir/reduced_forward_scm_dir3/m1'
    assert readded == list()
    assert stashed == [('V2', 'SEX')]
    assert included_relations == dict()

    block = [
        'Taking a step forward in adaptive scope reduction '
        + r'scm after reducing scope with 1 relations : V1-RACE'
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 is None
    assert readded == list()
    assert stashed == [('V1', 'RACE')]
    assert included_relations == dict()

    block = ['No models to test, there are no relations to add.']
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 is None
    assert readded == list()
    assert stashed == list()
    assert included_relations == dict()

    block = [
        'Scope reduction requested in adaptive scope reduction scm after forward step 6 but',
        'no relation was chosen for inclusion by scm in this step:',
        'adaptive scope reduction scm forward search with reduced scope is done.',
        'Included relations so far:  CL-CRCL-5,CL-DOSE-2,CL-RACE-2,V1-SEX-2,V2-QUAT-2',
        'Re-testing 19 relations after adaptive scope reduction '
        + r'scm reduced forward search : '
        + r'CL-SEX,CL-QUAT,KM-AGE,KM-RACE,KM-SEX,KM-QUAT,V1-AGE,V1-DOSE,'
        + r'V1-QUAT,V2-AGE,V2-RACE,VM-AGE,VM-RACE,VM-SEX,VM-QUAT,V2-SEX,CL-AGE,V2-DOSE,V1-RACE',
        'Model directory /home/run15asr_2.dir/rundir/readded_forward_scm_dir7/m1',
    ]
    m1, readded, stashed, included_relations = scm.parse_mixed_block(block)
    assert m1 == '/home/run15asr_2.dir/rundir/readded_forward_scm_dir7/m1'
    assert readded == [
        ('CL', 'SEX'),
        ('CL', 'QUAT'),
        ('KM', 'AGE'),
        ('KM', 'RACE'),
        ('KM', 'SEX'),
        ('KM', 'QUAT'),
        ('V1', 'AGE'),
        ('V1', 'DOSE'),
        ('V1', 'QUAT'),
        ('V2', 'AGE'),
        ('V2', 'RACE'),
        ('VM', 'AGE'),
        ('VM', 'RACE'),
        ('VM', 'SEX'),
        ('VM', 'QUAT'),
        ('V2', 'SEX'),
        ('CL', 'AGE'),
        ('V2', 'DOSE'),
        ('V1', 'RACE'),
    ]
    assert stashed == list()
    assert included_relations == {
        'CL': {'CRCL': '5', 'DOSE': '2', 'RACE': '2'},
        'V1': {'SEX': '2'},
        'V2': {'QUAT': '2'},
    }


def test_parse_chosen_relation_block(testdata):
    block = [
        'Parameter-covariate relation chosen in this forward step: PHA-DISDUR-5 \n',
        'CRITERION              PVAL < 0.01 \n',
        'BASE_MODEL_OFV      215748.07194\n',
        'CHOSEN_MODEL_OFV    215570.57352\n',
        'Relations included after this step:\n',
        'PHA     DISDUR-5MENS-2 \n',
        'PHIA\n',
        'PHIB    MENS-2\n',
        'SLP\n',
    ]
    chosen, criterion, included_relations = scm.parse_chosen_relation_block(block)
    assert chosen == {'parameter': 'PHA', 'covariate': 'DISDUR', 'state': '5'}
    assert criterion == {'gof': 'PVAL', 'pvalue': '0.01', 'is_backward': False}
    assert included_relations == {
        'PHA': {'DISDUR': '5', 'MENS': '2'},
        'PHIA': dict(),
        'PHIB': {'MENS': '2'},
        'SLP': dict(),
    }
    block = [
        'Parameter-covariate relation chosen in this forward step: PHA-WT-5',
        'CRITERION              PVAL < 0.01',
        'BASE_MODEL_OFV      215321.70954',
        'CHOSEN_MODEL_OFV    215306.67698',
        'Relations included after this step:',
        'PHA     AGE-5   BMI-5   DISDUR-5MENS-2  WT-5  ',
        'PHIA    MENS-2  ',
        'PHIB    GEOREG-2MENS-2  ',
        'SLP     GEOREG-2',
    ]

    chosen, criterion, included_relations = scm.parse_chosen_relation_block(block)
    assert chosen == {'parameter': 'PHA', 'covariate': 'WT', 'state': '5'}
    assert criterion == {'gof': 'PVAL', 'pvalue': '0.01', 'is_backward': False}

    assert included_relations == {
        'PHA': {'AGE': '5', 'BMI': '5', 'DISDUR': '5', 'MENS': '2', 'WT': '5'},
        'PHIA': {'MENS': '2'},
        'PHIB': {'GEOREG': '2', 'MENS': '2'},
        'SLP': {'GEOREG': '2'},
    }

    block = [
        'Parameter-covariate relation chosen in this forward step: --',
        'CRITERION              PVAL < 0.01',
    ]

    chosen, criterion, included_relations = scm.parse_chosen_relation_block(block)
    assert chosen == dict()
    assert criterion == {'gof': 'PVAL', 'pvalue': '0.01', 'is_backward': False}

    assert included_relations == dict()

    block = [
        'Parameter-covariate relation chosen in this backward step: --',
        'CRITERION              PVAL > 0.001',
    ]
    chosen, criterion, included_relations = scm.parse_chosen_relation_block(block)
    assert chosen == dict()
    assert criterion == {'gof': 'PVAL', 'pvalue': '0.001', 'is_backward': True}

    assert included_relations == dict()

    block = [
        'Parameter-covariate relation chosen in this forward step: V-CV2-2',
        'CRITERION              OFV',
        'BASE_MODEL_OFV         587.15896',
        'CHOSEN_MODEL_OFV       584.15462',
        'Relations included after this step:',
        'CL      CV1-2   WGT-2   ',
        'V       CV2-2   WGT-2',
    ]
    chosen, criterion, included_relations = scm.parse_chosen_relation_block(block)
    assert chosen == {'parameter': 'V', 'covariate': 'CV2', 'state': '2'}
    assert criterion == {'gof': 'OFV', 'pvalue': None, 'is_backward': False}

    assert included_relations == {'CL': {'CV1': '2', 'WGT': '2'}, 'V': {'CV2': '2', 'WGT': '2'}}
