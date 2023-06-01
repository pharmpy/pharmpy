from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.results import ModelfitResults
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.psn_helpers import (
    arguments_from_command,
    options_from_command,
    tool_from_command,
)


@dataclass(frozen=True)
class SCMResults(Results):
    """SCM Results class"""

    rst_path = Path(__file__).resolve().parent / 'report.rst'

    steps: Optional[Any] = None
    ofv_summary: Optional[Any] = None
    candidate_summary: Optional[Any] = None


def candidate_summary_dataframe(steps):
    if steps is None:
        return None
    elif steps['is_backward'].all():
        selected = steps[steps['selected']].copy()
        df = pd.DataFrame(
            [{'BackstepRemoved': row.Index[0]} for row in selected.itertuples()],
            index=selected.index,
        )
        return df.droplevel('step')
    else:
        scmplus = True if 'stashed' in steps.columns else False
        backstep_removed = {
            f'{row.Index[1]}{row.Index[2]}-{row.Index[3]}': row.Index[0]
            for row in steps.itertuples()
            if row.is_backward and row.selected
        }
        forward_steps = steps.query('step > 0 & ~is_backward')
        df = pd.DataFrame(
            [
                {
                    'N_test': True,
                    'N_ok': (not np.isnan(row.ofv_drop) and row.ofv_drop >= 0),
                    'N_localmin': (not np.isnan(row.ofv_drop) and row.ofv_drop < 0),
                    'N_failed': np.isnan(row.ofv_drop),
                    'StepIncluded': row.Index[0] if row.selected else None,
                    'StepStashed': row.Index[0] if (scmplus and row.stashed) else None,
                    'StepReadded': row.Index[0] if (scmplus and row.readded) else None,
                    'BackstepRemoved': backstep_removed.pop(row.model, None),
                }
                for row in forward_steps.itertuples()
            ],
            index=forward_steps.index,
        )
        return df.groupby(level=['parameter', 'covariate', 'extended_state']).sum(
            numeric_only=True, min_count=1
        )


def ofv_summary_dataframe(steps, final_included=True, iterations=True):
    if steps is None or not (final_included or iterations):
        return None
    else:
        if final_included and iterations and not steps['is_backward'].iloc[-1]:
            # Will not be able to show final_included with additional info
            final_included = False
        # Use .copy() to ensure we do not work on original df
        df = steps[steps['selected']].copy() if iterations else pd.DataFrame()
        if iterations:
            df['is_backward'] = [
                'Backward' if backward else 'Forward' for backward in df['is_backward']
            ]
        if final_included:
            if steps['is_backward'].iloc[-1]:
                # all rows from last step where selected is False
                last_stepnum = steps.index[-1][steps.index.names.index('step')]
                try:
                    final = steps[~steps['selected']].loc[last_stepnum, :, :, :].copy()
                except KeyError:
                    # No final
                    final = None
            else:
                # all selected rows without ofv info
                final = pd.DataFrame(columns=steps.columns, index=steps[steps['selected']].index)
            if final is not None:
                final['is_backward'] = 'Final included'
                df = pd.concat([df, final])
        df.rename(columns={'is_backward': 'direction'}, inplace=True)
        columns = ['direction', 'reduced_ofv', 'extended_ofv', 'ofv_drop']
        if 'pvalue' in steps.columns:
            columns.extend(['delta_df', 'pvalue', 'goal_pvalue'])
        if df.empty:
            return df
        return df[columns]


def psn_scm_parse_logfile(logfile, options, parcov_dictionary):
    """Read SCM results"""

    logfile = Path(logfile)
    df = pd.concat(list(log_steps(logfile, options, parcov_dictionary)))
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
    header = {
        'ofv': ['model', 'test', 'base', 'new', 'dofv', 'cmp', 'goal', 'signif'],
        'pvalue': ['model', 'test', 'base', 'new', 'dofv', 'cmp', 'goal', 'ddf', 'signif', 'pval'],
    }
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

    # First column might be wider than 16 characters. Must be padded in that case
    lens = []
    for line in block:
        a = line.split(maxsplit=1)
        lens.append(len(a[0]))
    maxlen = max(lens)
    if maxlen > 16:
        for i, length in enumerate(lens):
            collen = 16 if length < 16 else length
            block[i] = block[i][:length] + (maxlen - collen) * ' ' + block[i][length:]

    rawtable = pd.read_fwf(
        StringIO(str.join('', block)),
        skiprows=1,
        header=None,
        names=header[gof],
        true_values=['YES!'],
        na_values=['FAILED'],
    )
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

    table = model_name_series_to_dataframe(
        rawtable.model, parcov_dictionary, is_backward, included_relations
    )
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


def model_name_series_to_dataframe(modelname, parcov_dictionary, is_backward, included_relations):
    subdf = modelname.str.extract(r'(?P<parcov>.+)-(?P<state>\d+)$', expand=True)
    state = np.nan
    if is_backward:
        if included_relations:
            state = extended_states(modelname, included_relations)
    else:
        state = [int(s) for s in subdf.state.values]
    result = pd.DataFrame(
        {'model': modelname, 'parameter': None, 'covariate': None, 'extended_state': state}
    )
    if parcov_dictionary:
        # Handling of binarized covariates
        parameter = []
        covariate = []
        for name in subdf.parcov:
            a = name.rsplit('_', maxsplit=1)
            if len(a) == 1:
                n = None
            else:
                try:
                    n = int(a[1])
                except ValueError:
                    n = None

            if n is None:
                parameter.append(parcov_dictionary[name][0])
                covariate.append(parcov_dictionary[name][1])
            else:
                parameter.append(parcov_dictionary[a[0]][0])
                covariate.append(parcov_dictionary[a[0]][1] + '_' + str(n))

        result.parameter = parameter
        result.covariate = covariate
    return result


def parse_mixed_block(block):
    m1 = None
    readded = []
    stashed = []
    included_relations = {}

    pattern = {
        'm1': re.compile(r'Model\s+directory\s+(?P<m1folder>\S+)'),
        'stashed': re.compile(r'Taking a step.* reducing scope with .*:\s*(?P<relations>\S+)'),
        'readded': re.compile(r'Re-testing .* relations after .* :\s*(?P<relations>\S+)'),
        'included': re.compile(r'Included relations so far:\s*(?P<relations>\S+)'),
    }

    for row in block:
        if match := pattern['stashed'].match(row):
            if stashed:
                raise NotImplementedError('Two scope reduction lines in the same logfile block')
            stashed = [
                tuple(relation.split('-')) for relation in match.group('relations').split(',')
            ]
        elif match := pattern['readded'].match(row):
            if readded:
                raise NotImplementedError('Two re-testing lines in the same logfile block')
            readded = [
                tuple(relation.split('-')) for relation in match.group('relations').split(',')
            ]
        elif match := pattern['m1'].match(row):
            if m1:
                raise NotImplementedError('Two model directory lines in the same logfile block')
            m1 = match.group('m1folder')
        elif match := pattern['included'].match(row):
            for relation in match.group('relations').split(','):
                par, cov, state = relation.split('-')
                if par not in included_relations.keys():
                    included_relations[par] = {}
                included_relations[par][cov] = state

    return m1, readded, stashed, included_relations


def parse_chosen_relation_block(block):
    chosen = {}
    criterion = {}
    pattern = {
        'chosen': re.compile(
            r'Parameter-covariate relation chosen in this '
            + r'(forward|backward) step: '
            + r'(?P<parameter>\S+)-(?P<covariate>\S+)-(?P<state>\S+)'
        ),
        'backward': re.compile(r'Parameter-covariate relation chosen in this backward st'),
        'gof_ofv': re.compile(r'CRITERION\s+OFV\b'),
        'gof_pvalue': re.compile(r'CRITERION\s+' + r'(?P<gof>PVAL)\s+(>|<)\s+(?P<pvalue>\S+)\s*$'),
        'included': re.compile(r'Relations included after this step'),
        'parameter': re.compile(r'(?P<parameter>\S+)'),
        'covstate': re.compile(r'\s*(?P<covariate>\D[^-\s]*)-(?P<state>\d+)'),
    }

    chosen_match = pattern['chosen'].match(block[0])
    is_backward = False
    if pattern['backward'].match(block[0]):
        is_backward = True

    if chosen_match:
        chosen = chosen_match.groupdict()
    if pattern['gof_ofv'].match(block[1]):
        criterion = {'gof': 'OFV', 'pvalue': None}
    else:
        match = pattern['gof_pvalue'].match(block[1])
        assert match is not None
        criterion = match.groupdict()
    criterion['is_backward'] = is_backward

    included_relations = {}

    if len(block) > 4 and pattern['included'].match(block[4]):
        for row in block[5:]:
            row = row.strip()
            match = pattern['parameter'].match(row)
            assert match is not None
            par = match.group('parameter')
            if re.search(r'-\d+$', par):
                raise NotImplementedError('Missing whitespace between param and included covs')
            included_relations[par] = dict(pattern['covstate'].findall(row))

    return chosen, criterion, included_relations


def names_without_state(model_names):
    return list(model_names.str.extract(r'(.+)-\d+$').values.flatten())


def extended_states(model_names, included_relations):
    model_parcov = names_without_state(model_names)
    placeholder = -99
    statedict = {
        f'{par}{cov}': int(state)
        for par, d in included_relations.items()
        for cov, state in d.items()
    }
    return [statedict.get(parcov, placeholder) for parcov in model_parcov]


def step_data_frame(step, included_relations):
    df = step['runtable']
    is_backward = df['is_backward'].iloc[0]
    if is_backward and included_relations:
        if np.all(np.isnan(df['extended_state'].values.flatten())):
            # This must be a backward step without preceeding steps of any kind
            # and where included_relations was not found from conf file
            df['extended_state'] = extended_states(df['model'], included_relations)
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
        chosenmodel = (
            step['chosen']['parameter']
            + step['chosen']['covariate']
            + r'-'
            + step['chosen']['state']
        )
    df['selected'] = [name == chosenmodel for name in df['model'].values]

    df['directory'] = str(step['m1'])
    model = df['model']  # move to end
    df.drop(columns=['model'], inplace=True)
    df['model'] = model

    if step['stashed'] or step['readded']:
        model_parcov = names_without_state(df['model'])
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

    return df.set_index(['step', 'parameter', 'covariate', 'extended_state'])


def prior_included_step(included_relations, gof_is_pvalue):
    states = []
    extra = None
    if gof_is_pvalue:
        extra = {'delta_df': 0, 'pvalue': np.nan, 'goal_pvalue': np.nan}
    else:
        extra = {'goal_ofv_drop': np.nan}
    for par, d in included_relations.items():
        for cov, state in d.items():
            states.append(
                {
                    'step': 0,
                    'parameter': par,
                    'covariate': cov,
                    'extended_state': int(state),
                    'reduced_ofv': np.nan,
                    'extended_ofv': np.nan,
                    'ofv_drop': np.nan,
                    **extra,
                    'is_backward': False,
                    'extended_significant': True,
                    'selected': True,
                    'directory': '',
                    'model': '',
                }
            )
    df = pd.DataFrame(states)
    return df.set_index(['step', 'parameter', 'covariate', 'extended_state'])


def empty_step(previous_number, previous_criterion=None):
    return {
        'runtable': None,
        'm1': None,
        'chosen': None,
        'stashed': None,
        'readded': None,
        'criterion': None,
        'number': previous_number + 1,
        'previous_criterion': previous_criterion,
    }


def have(something):
    return something is not None


def log_steps(path, options, parcov_dictionary=None):
    included_relations = options['included_relations']
    basepath = Path(options['directory'])

    pattern = {
        'runtable': re.compile(r'^MODEL\s+TEST\s+'),
        'm1': re.compile(r'Model\s+directory\s+(?P<m1folder>.*)'),
        'chosen': re.compile(r'Parameter-covariate relation chosen in this'),
    }

    step = empty_step(0)

    for block in file_blocks(path):
        if pattern['runtable'].match(block[0]):
            # can be empty table with header, only set table if have content
            if len(block) > 1:
                step['runtable'] = parse_runtable_block(
                    block, parcov_dictionary, included_relations
                )
                is_forward = not step['runtable']['is_backward'].iloc[-1]
                if step['number'] == 1 and included_relations is not None and is_forward:
                    yield prior_included_step(
                        included_relations, gof_is_pvalue=('pvalue' in step['runtable'].columns)
                    )
        elif pattern['chosen'].match(block[0]):
            step['chosen'], step['criterion'], included = parse_chosen_relation_block(block)
            if included:
                included_relations = included
        elif match := pattern['m1'].match(block[0]):
            if have(step['runtable']):
                yield step_data_frame(step, included_relations)
                step = empty_step(step['number'], step['criterion'])
            step['m1'] = Path(match.group('m1folder')).relative_to(basepath)
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

        baseofvlength = int(max({len(x) for x in rawtable.base.values}) / 2)
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
        rawtable.rename(columns=dict(zip(old, column_names)), inplace=True)

    return rawtable


def psn_config_file_argument_from_command(command, path):
    # arguments: everything on command-line which does not start with -
    # and where 'name' (i.e. last portion when treated as a path object) is found
    # as file in input folder path
    # this may include a model file, so we need to parse file(s) to see
    # if we actually find included_relations
    path = Path(path)
    return [
        Path(arg).name
        for arg in arguments_from_command(command)
        if (path / Path(arg).name).is_file()
    ]


def relations_from_config_file(path, files):
    # Allow multiple candidate files in case ambiguous command-line (unusual)
    path = Path(path)
    start_config_section = re.compile(r'\s*\[(?P<section>[a-z_]+)\]\s*$')
    empty_line = re.compile(r'^[-\s]*$')
    comment_line = re.compile(r'\s*;')
    included_lines = []
    test_lines = []
    scanning_included = False
    scanning_test = False
    for file in files:
        with open(path / file) as fn:
            for row in fn:
                match = start_config_section.match(row)
                if match:
                    section = match.group('section')
                    if section == 'included_relations':
                        scanning_included = True
                        scanning_test = False
                        included_lines = []  # NOTE This potentially resets?
                    elif section == 'test_relations':
                        scanning_included = False
                        scanning_test = True
                        test_lines = []  # NOTE This potentially resets?
                    else:
                        scanning_included = False
                        scanning_test = False
                elif scanning_included:
                    if not empty_line.match(row) and not comment_line.match(row):
                        included_lines.append(row.strip())
                elif scanning_test:
                    if not empty_line.match(row) and not comment_line.match(row):
                        test_lines.append(row.strip())
        if test_lines is not None and len(test_lines) > 0:
            break  # do not check any other file, if more than one

    if included_lines:
        included_relations = {}
        p = re.compile(r'\s*([^-]+)-(\d+)\s*')
        for row in included_lines:
            par, covstates = row.split(r'=')
            par = str.strip(par)
            included_relations[par] = {
                match.group(1): match.group(2)
                for val in covstates.split(r',')
                if (match := p.match(val)) is not None
            }
    else:
        included_relations = None

    if test_lines:
        test_relations = {}
        for row in test_lines:
            par, covs = row.split(r'=')
            par = str.strip(par)
            test_relations[par] = [str.strip(val) for val in covs.split(r',')]
    else:
        test_relations = None

    return included_relations, test_relations


def psn_scm_options(path):
    path = Path(path)
    options = {
        'directory': str(path),
        'logfile': 'scmlog.txt',
        'included_relations': None,
        'test_relations': None,
    }
    scmplus = False
    config_files = None
    with open(path / 'meta.yaml') as meta:
        cmd = None
        for row in meta:
            if cmd is not None:
                if re.match(r'\s', row):  # continuation is indented
                    cmd += row  # must not strip
                    continue
                else:  # no continuation: parse and remove
                    if tool_from_command(cmd) == 'scmplus':
                        scmplus = True
                    for k, v in options_from_command(cmd).items():
                        if 'config_file'.startswith(k):
                            config_files = [v]
                            break
                    if config_files is None:
                        # not option -config_file, must have been given as argument
                        config_files = psn_config_file_argument_from_command(cmd, path)
                    cmd = None
            row = row.strip()
            if row.startswith('logfile: '):
                options['logfile'] = Path(re.sub(r'\s*logfile:\s*', '', row)).name
            elif row.startswith('directory: '):
                options['directory'] = str(Path(re.sub(r'\s*directory:\s*', '', row)))
            elif row.startswith('command_line: '):
                cmd = row
    if scmplus and Path(options['directory']).parts[-1] == 'rundir':
        options['directory'] = str(Path(options['directory']).parents[0])
    if config_files is not None:
        options['included_relations'], options['test_relations'] = relations_from_config_file(
            path, config_files
        )

    return options


def parcov_dict_from_test_relations(test_relations):
    return {f'{par}{cov}': (par, cov) for par, covs in test_relations.items() for cov in covs}


def _add_covariate_effects_to_steps(steps, path):
    if 'delta_df' not in steps:
        return

    def fn(row):
        if row.is_backward:
            return np.nan
        degrees = row.delta_df
        model_name = row.model.replace(
            '-', ''
        )  # FIXME: The dash should not have been in the table!
        model_path = path / row.directory / f'{model_name}.mod'
        if not model_path.is_file():
            return np.nan
        model = Model.parse_model(model_path)
        results = read_modelfit_results(model_path)
        varpars = model.random_variables.free_symbols
        all_thetas = [param for param in model.parameters if param.symbol not in varpars]
        new_thetas = all_thetas[-degrees:]
        covariate_effects = {
            param.name: _parameter_estimates(results, param.name) for param in new_thetas
        }
        return covariate_effects

    steps['covariate_effects'] = steps.apply(fn, axis=1)


def _parameter_estimates(results: ModelfitResults, parameter: str):
    pe = results.parameter_estimates
    assert pe is not None
    return pe[parameter]


def psn_scm_results(path):
    """Create scm results from a PsN SCM run

    :param path: Path to PsN scm run directory
    :return: A :class:`SCMResults` object

    """
    path = Path(path)
    if not path.is_dir():
        raise IOError(f'Could not find scm directory: {str(path)}')

    options = psn_scm_options(path)
    logfile = path / options['logfile']

    if not logfile.is_file():
        raise IOError(f'Could not find scm logfile: {str(logfile)}')

    if options['test_relations'] is not None:
        parcov_dictionary = parcov_dict_from_test_relations(options['test_relations'])
    else:
        raise IOError(r'Could not find test_relations in scm config file')

    steps = psn_scm_parse_logfile(logfile, options, parcov_dictionary)
    _add_covariate_effects_to_steps(steps, path)

    return SCMResults(
        steps=steps,
        candidate_summary=candidate_summary_dataframe(steps),
        ofv_summary=ofv_summary_dataframe(steps, final_included=True, iterations=True),
    )
