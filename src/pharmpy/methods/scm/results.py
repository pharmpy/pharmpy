import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from pharmpy.results import Results


class SCMResults(Results):
    """SCM Results class



    """
    rst_path = Path(__file__).parent / 'report.rst'

    def __init__(self, steps=None):
        self.steps = steps

    def to_dict(self):
        return {'steps': self.steps}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __str__(self):
        return str(self.steps)


def psn_scm_parse_logfile(logfile, options, parcov_dictionary):
    """Read SCM results

    """

    logfile = Path(logfile)
    df = pd.concat([step for step in log_steps(logfile, options, parcov_dictionary)])
    if 'stashed' in df.columns:
        df.fillna(value={'stashed': False, 'readded': False}, inplace=True)
    return df


def file_blocks(path):
    block = []
    separator = re.compile(r'^[-\s]*$')
    with open(path) as file:
        for row in file:
            if separator.match(row):
                if block:
                    yield block
                block = []
            else:
                block.append(row)
    if block:
        yield block


def parse_runtable_block(block, parcov_dictionary=None, included_relations=None):
    header = {'ofv': ['model', 'test', 'base', 'new', 'dofv', 'cmp',
                      'goal', 'signif'],
              'pvalue': ['model', 'test', 'base', 'new', 'dofv', 'cmp',
                         'goal', 'ddf', 'signif', 'pval']}
    if not len(block) > 1:
        raise NotImplementedError('function parse_runtable_block called with empty table')

    gof = None
    is_backward = re.search(r'IN.?SIGNIFICANT', block[0])

    if re.match(r'^MODEL\s+TEST\s+BASE\s+OFV', block[0]):
        gof = 'pvalue'
    elif re.match(r'^MODEL\s+TEST\s+NAME\s+BASE\s+', block[0]):
        gof = 'ofv'
    else:
        raise NotImplementedError('Unsupported runtable header string')

    rawtable = pd.read_fwf(StringIO(str.join('', block)), skiprows=1, header=None,
                           names=header[gof], true_values=['YES!'],
                           na_values=['FAILED'])

    if len(rawtable.base.unique()) > 1:
        rawtable = split_merged_base_and_new_ofv(rawtable)

    if gof == 'pvalue':
        if np.all([np.isnan(v) or v is None for v in rawtable['pval'].values]):
            # No model signficant, signif column is empty and missed by read_fwf
            rawtable['pval'] = rawtable['signif']
            rawtable['signif'] = False
    else:
        if np.all([np.isnan(v) or v is None for v in rawtable['signif'].values]):
            rawtable['signif'] = False

    table = model_name_series_to_dataframe(rawtable.model, parcov_dictionary,
                                           is_backward, included_relations)
    if is_backward:
        table['reduced_ofv'] = rawtable.new
        table['extended_ofv'] = rawtable.base
        table['ofv_drop'] = rawtable.dofv.multiply(-1.0)
    else:
        table['reduced_ofv'] = rawtable.base
        table['extended_ofv'] = rawtable.new
        table['ofv_drop'] = rawtable.dofv

    if gof == 'pvalue':
        table['delta_df'] = rawtable.ddf.multiply(-1) if is_backward else rawtable.ddf
        table['pvalue'] = rawtable.pval
    else:
        table['goal_ofv_drop'] = rawtable.goal.multiply(-1.0) if is_backward else rawtable.goal

    significant = np.array([False if np.isnan(s) else s for s in rawtable.signif])
    table['is_backward'] = True if is_backward else False
    table['extended_significant'] = ~significant if is_backward else significant
    return table


def model_name_series_to_dataframe(modelname, parcov_dictionary,
                                   is_backward, included_relations):
    subdf = modelname.str.extract(r'(?P<parcov>.+)-(?P<state>\d+)$', expand=True)
    state = np.nan
    if is_backward:
        if included_relations:
            state = extended_states(modelname, included_relations)
    else:
        state = [int(s) for s in subdf.state.values]
    result = pd.DataFrame({'Model': modelname,
                           'Parameter': None,
                           'Covariate': None,
                           'Extended_State': state})
    if parcov_dictionary:
        temp = pd.DataFrame([parcov_dictionary[m] for m in subdf.parcov],
                            columns=('parameter', 'covariate'))
        result.Parameter = temp.parameter
        result.Covariate = temp.covariate
    return result


def parse_mixed_block(block):
    m1 = None
    readded = list()
    stashed = list()
    included_relations = dict()

    pattern = {'m1': re.compile(r'Model\s+directory\s+(?P<m1folder>\S+)'),
               'stashed':
                   re.compile(r'Taking a step.* reducing scope with .*:\s*(?P<relations>\S+)'),
               'readded': re.compile(r'Re-testing .* relations after .* :\s*(?P<relations>\S+)'),
               'included': re.compile(r'Included relations so far:\s*(?P<relations>\S+)')}

    for row in block:
        if pattern['stashed'].match(row):
            if stashed:
                raise NotImplementedError('Two scope reduction lines in the same logfile block')
            stashed = [tuple(relation.split('-'))
                       for relation in pattern['stashed'].match(row).group('relations').split(',')]
        elif pattern['readded'].match(row):
            if readded:
                raise NotImplementedError('Two Re-testing lines in the same logfile block')
            readded = [tuple(relation.split('-'))
                       for relation in pattern['readded'].match(row).group('relations').split(',')]
        elif pattern['m1'].match(row):
            if m1:
                raise NotImplementedError('Two Model directory lines in the same logfile block')
            m1 = pattern['m1'].match(row).group('m1folder')
        elif pattern['included'].match(row):
            for relation in pattern['included'].match(row).group('relations').split(','):
                par, cov, state = relation.split('-')
                if par not in included_relations.keys():
                    included_relations[par] = dict()
                included_relations[par][cov] = state

    return m1, readded, stashed, included_relations


def parse_chosen_relation_block(block):
    chosen = dict()
    criterion = dict()
    pattern = {'chosen': re.compile(r'Parameter-covariate relation chosen in this ' +
                                    r'(forward|backward) step: ' +
                                    r'(?P<parameter>\S+)-(?P<covariate>\S+)-(?P<state>\S+)'),
               'backward': re.compile(r'Parameter-covariate relation chosen in this backward st'),
               'gof_ofv': re.compile(r'CRITERION\s+OFV\b'),
               'gof_pvalue': re.compile(r'CRITERION\s+' +
                                        r'(?P<gof>PVAL)\s+(>|<)\s+(?P<pvalue>\S+)\s*$'),
               'included': re.compile(r'Relations included after this step'),
               'parameter': re.compile(r'(?P<parameter>\S+)'),
               'covstate': re.compile(r'\s*(?P<covariate>\D[^-\s]*)-(?P<state>\d+)')}

    chosen_match = pattern['chosen'].match(block[0])
    is_backward = False
    if pattern['backward'].match(block[0]):
        is_backward = True

    if chosen_match:
        chosen = chosen_match.groupdict()
    if pattern['gof_ofv'].match(block[1]):
        criterion = {'gof': 'OFV', 'pvalue': None}
    else:
        criterion = pattern['gof_pvalue'].match(block[1]).groupdict()
    criterion['is_backward'] = is_backward

    included_relations = dict()

    if len(block) > 4 and pattern['included'].match(block[4]):
        for row in block[5:]:
            row = row.strip()
            par = pattern['parameter'].match(row).group('parameter')
            if re.search(r'-\d+$', par):
                raise NotImplementedError('Missing whitespace between param and included covs')
            included_relations[par] = {p: c for p, c in pattern['covstate'].findall(row)}

    return chosen, criterion, included_relations


def names_without_state(model_names):
    return [parcov for parcov in model_names.str.extract(r'(.+)-\d+$').values.flatten()]


def extended_states(model_names, included_relations):
    model_parcov = names_without_state(model_names)
    statedict = {parcov: int(-99) for parcov in model_parcov}
    for par, d in included_relations.items():
        for cov, state in d.items():
            statedict[f'{par}{cov}'] = int(state)
    return [statedict[parcov] for parcov in model_parcov]


def step_data_frame(step, included_relations):
    df = step['runtable']
    is_backward = df['is_backward'].iloc[0]
    if is_backward and included_relations:
        if np.all(np.isnan(df['Extended_State'].values.flatten())):
            # This must be a backward step without preceeding steps of any kind
            df['Extended_State'] = extended_states(df['Model'],
                                                   included_relations)
    df['step'] = step['number']
    if 'pvalue' in df.columns:
        if step['criterion']:
            df.insert(9, 'goal_pvalue', step['criterion']['pvalue'])
        elif step['previous_criterion']:
            if step['previous_criterion']['is_backward'] == is_backward:
                # same direction as previous step with criterion
                df.insert(9, 'goal_pvalue', step['previous_criterion']['pvalue'])

    chosenmodel = 'no model'
    if step['chosen']:
        chosenmodel = step['chosen']['parameter'] + step['chosen']['covariate'] + \
            r'-' + step['chosen']['state']
    df['selected'] = [name == chosenmodel for name in df['Model'].values]

    df['folder'] = step['m1']
    model = df['Model']  # move to end
    df.drop(columns=['Model'], inplace=True)
    df['Model'] = model

    if step['stashed'] or step['readded']:
        model_parcov = names_without_state(df['Model'])
        if step['stashed']:
            stashed = {f'{par}{cov}' for par, cov in step['stashed']}
            df['stashed'] = [parcov in stashed for parcov in model_parcov]
        else:
            df['stashed'] = False
        if step['readded']:
            readded = {f'{par}{cov}' for par, cov in step['readded']}
            df['readded'] = [parcov in readded for parcov in model_parcov]
        else:
            df['readded'] = False

    return df.set_index(['step', 'Parameter', 'Covariate', 'Extended_State'])


def empty_step(previous_number, previous_criterion=None):
    return {'runtable': None, 'm1': None, 'chosen': None,
            'stashed': None, 'readded': None,
            'criterion': None, 'number': previous_number + 1,
            'previous_criterion': previous_criterion}


def have(something):
    return something is not None


def log_steps(path, options, parcov_dictionary=None):
    included_relations = None
    basepath = Path(options['directory'])

    pattern = {'runtable': re.compile(r'^MODEL\s+TEST\s+'),
               'm1': re.compile(r'Model\s+directory\s+(?P<m1folder>\S+)'),
               'chosen': re.compile(r'Parameter-covariate relation chosen in this')}

    step = empty_step(0)

    for block in file_blocks(path):
        if pattern['runtable'].match(block[0]):
            # can be empty table with header, only set table if have content
            if len(block) > 1:
                step['runtable'] = parse_runtable_block(block, parcov_dictionary,
                                                        included_relations)
        elif pattern['chosen'].match(block[0]):
            step['chosen'], step['criterion'], included = parse_chosen_relation_block(block)
            if included:
                included_relations = included
        elif pattern['m1'].match(block[0]):
            if have(step['runtable']):
                yield step_data_frame(step, included_relations)
                step = empty_step(step['number'], step['criterion'])
            step['m1'] = Path(pattern['m1'].match(block[0]).group('m1folder')).relative_to(basepath)
        else:
            # complex block, either gof ofv or scmplus. May contain m1
            m1, readded, stashed, included = parse_mixed_block(block)
            if stashed:
                # stashing belongs to previously read runtable: do not yield
                step['stashed'] = stashed
            if included:
                included_relations = included
            if readded:
                # readding is done first in new step: yield if have a runtable
                if have(step['runtable']):
                    yield step_data_frame(step, included_relations)
                    step = empty_step(step['number'], step['criterion'])
                step['readded'] = readded
            if m1:
                # Writing m1 is done first in step, unless readding was done first
                # in which case runtable will be empty
                if have(step['runtable']):
                    yield step_data_frame(step, included_relations)
                    step = empty_step(step['number'], step['criterion'])
                step['m1'] = Path(m1).relative_to(basepath)
    if have(step['runtable']):
        yield step_data_frame(step, included_relations)


def split_merged_base_and_new_ofv(rawtable):
    testnames = rawtable.test.unique()

    if len(testnames) > 1:
        raise ValueError('Non-unique TEST column value in scm-step, model name too wide?')
    elif not re.match(r'(PVAL|OFV)', testnames[0]):
        raise ValueError(f'Unrecognized TEST column value {testnames[0]}')

    if len(rawtable.base.unique()) > 1:
        # We have missing space between columns base and new.
        # These are formatted with the same widths, so split
        # at equal lengths, i.e. at half of max length. Handle
        # case where a run has ofv 'FAILED' by skipping whitespace
        # in second part after string split.
        if not all(pd.isna(rawtable.iloc[:, -1])):
            # expect all values in last column to be NaN if ofv columns are merged
            raise Exception
        # Save names and then drop last column. Names are unique so this drops 1 col
        column_names = list(rawtable.columns)
        rawtable.drop(columns=column_names[-1], inplace=True)

        baseofvlength = int(max({len(x) for x in rawtable.base.values})/2)
        pattern = f'(?P<base>.{{{baseofvlength}}})' + r'\s*(?P<newfixed>\S*)'

        subdf = rawtable.base.str.extract(pattern, expand=True)
        subdf.replace('FAILED', np.nan, inplace=True)
        if len(subdf.base.unique()) > 1:
            # all base values should be identical now
            raise Exception
        # replace base column and insert newfixed
        rawtable.base = subdf.base
        rawtable.insert(3, 'newfixed', subdf.newfixed)
        # rename columns to get original labels at correct positions
        old = list(rawtable.columns)
        rawtable.rename(columns={o: correct for o, correct in zip(old, column_names)},
                        inplace=True)

    return rawtable


def psn_scm_options(path):
    path = Path(path)
    options = {'directory': str(path),
               'logfile': 'scmlog.txt'}
    try:
        with open(path / 'meta.yaml') as meta:
            for row in meta:
                row = row.strip()
                if row.startswith('logfile: '):
                    options['logfile'] = Path(re.sub(r'\s*logfile:\s*', '', row)).name
                if row.startswith('directory: '):
                    options['directory'] = \
                        str(Path(re.sub(r'\s*directory:\s*', '', row)).absolute())
    except IOError:
        # if meta.yaml not found we return default name
        pass
    return options


def parse_scm_relations(path):
    path = Path(path)
    parcov = dict()

    pattern = re.compile(r'\$VAR1->{\'(?P<par1>\S+)\'}{\'(?P<cov1>\S+)\'}{\'ofv_changes\'} ' +
                         r'= \$VAR1->{\'(?P<par2>\S+)\'}{\'(?P<cov2>\S+)\'}{\'ofv_changes\'}')
    with open(path) as relations:
        for row in relations:
            if row.startswith(r'$VAR1->{'):
                m = pattern.match(row)
                if m:
                    parcov1 = m.groupdict()['par1'] + m.groupdict()['cov1']
                    parcov2 = m.groupdict()['par2'] + m.groupdict()['cov2']
                    parcov[parcov1] = (m.groupdict()['par1'], m.groupdict()['cov1'])
                    parcov[parcov2] = (m.groupdict()['par2'], m.groupdict()['cov2'])
    return parcov


def psn_scm_results(path):
    """ Create scm results from a PsN SCM run

        :param path: Path to PsN scm run directory
        :return: A :class:`SCMResults` object

    """
    path = Path(path)
    if not path.is_dir():
        raise IOError(f'Could not find scm folder: {str(path)}')

    options = psn_scm_options(path)
    logfile = path / options['logfile']

    if not logfile.is_file():
        raise IOError(f'Could not find scm logfile: {str(logfile)}')

    parcov_dictionary = parse_scm_relations(path / 'relations.txt')

    return SCMResults(steps=psn_scm_parse_logfile(logfile, options, parcov_dictionary))
