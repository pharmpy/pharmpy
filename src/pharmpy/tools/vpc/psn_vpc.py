import re
from pathlib import Path

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem.table import NONMEMTableFile
from pharmpy.modeling import evaluate_expression, read_model
from pharmpy.tools.psn_helpers import options_from_command, psn_command

# Calculation of refcorr for PsN vpc


def calculate_reference_correction(dv, pred, predref, simdata, refdata, logdv=False, lb=0.0):
    if logdv:
        factor = predref - pred
    else:
        num = predref - lb
        den = pred - lb
        factor = np.divide(num, den, out=np.ones_like(num), where=den != 0.0)

    def correct(dv, factor, logdv, lb):
        if logdv:
            rpcdv = dv + factor
        else:
            rpcdv = lb + (dv - lb) * factor
        return rpcdv

    rpcdv = correct(dv, factor, logdv, lb)
    rpcdv_sim = [correct(sim[dv.name], factor, logdv, lb) for sim in simdata]

    refdvs = pd.concat((ref[dv.name] for ref in refdata), axis=1)
    refdvs.columns = range(len(refdvs.columns))  # To get unique column names
    rpcdvs = pd.concat(rpcdv_sim, axis=1)

    if logdv:
        refdv_sd = refdvs.std(axis=1)
        rpcdvs_sd = rpcdvs.std(axis=1)
    else:
        refdv_sd = np.log(refdvs - lb).std(axis=1)
        rpcdvs_sd = np.log(rpcdvs - lb).std(axis=1)
    var_factor = np.divide(refdv_sd, rpcdvs_sd, out=np.ones_like(refdv_sd), where=rpcdvs_sd != 0.0)

    def varcorrect(dv, factor, predref, logdv, lb):
        if logdv:
            rpvcdv = predref + (dv - predref) * factor
        else:
            log_predref = np.log(predref - lb)
            rpvcdv = lb + np.exp(log_predref + (np.log(dv - lb) - log_predref) * factor)
        return rpvcdv

    rpvcdv = varcorrect(rpcdv, var_factor, predref, logdv, lb)
    rpvcdv_sim = [varcorrect(x, var_factor, predref, logdv, lb) for x in rpcdv_sim]
    return rpvcdv, rpvcdv_sim


def reference_correction_from_psn_vpc(path):
    np.seterr(invalid='raise')
    path = Path(path)
    opts = options_from_command(psn_command(path))
    dv = opts.get('dv', 'DV')
    idv = opts.get('idv', 'TIME')
    refcorr_idv = opts.get('refcorr_idv', False)
    logdv = bool(int(opts.get('lnDV', '0')))  # NOTE: Not entirely correct
    lower_bound = opts.get('lower_bound', 0.0)

    m1 = path / 'm1'

    model = read_model(m1 / "vpc_original.mod")
    if refcorr_idv:
        have_eta_on_idv = not model.statements.before_odes.full_expression(
            idv
        ).free_symbols.isdisjoint(set(model.random_variables.symbols))
    else:
        have_eta_on_idv = False

    origfile_path = m1 / "vpc_original.npctab.dta"
    origfile = NONMEMTableFile(origfile_path)
    origdata = origfile.tables[0].data_frame
    origdata = origdata[origdata['MDV'] == 0.0]

    orig_pred_path = m1 / "vpc_pred.1.npctab.dta"
    if orig_pred_path.is_file():
        origpredfile = NONMEMTableFile(orig_pred_path)
        origpreddata = origpredfile.tables[0].data_frame
        origpreddata = origpreddata[origpreddata['MDV'] == 0.0]
        pred = origpreddata[dv]
        if have_eta_on_idv:
            idvpred = origpreddata[idv]
    else:
        pred = origdata['PRED']

    simfile_path = m1 / "vpc_simulation.1.npctab.dta"
    simfile = NONMEMTableFile(m1 / simfile_path)
    simdata = [tab.data_frame for tab in simfile.tables]
    simdata = [df[df['MDV'] == 0.0] for df in simdata]

    reffile = NONMEMTableFile(m1 / "vpc_simulation_refcorr.1.npctab.dta")
    refdata = [tab.data_frame for tab in reffile.tables]
    refdata = [df[df['MDV'] == 0.0] for df in refdata]
    ref = refdata[0]['REF'].astype(int) - 1
    refdata = [ser.reindex(ref).sort_index() for ser in refdata]

    ref_pred_path = m1 / "vpc_pred_refcorr.1.npctab.dta"
    if ref_pred_path.is_file():
        refpredfile = NONMEMTableFile(ref_pred_path)
        refpreddata = refpredfile.tables[0].data_frame
        refpreddata = refpreddata[refpreddata['MDV'] == 0.0]
        predref = refpreddata[dv]
        idvpredref = refpreddata[idv]
    else:
        predref = refdata[0]['PRED']

    lb = evaluate_expression(model, lower_bound)[origdata.index]
    rcdv, rcdv_sim = calculate_reference_correction(
        origdata[dv], pred, predref, simdata, refdata, logdv, lb
    )

    if have_eta_on_idv:
        rcidv, rcidv_sim = calculate_reference_correction(
            origdata[idv], idvpred, idvpredref, simdata, refdata, logdv, lb
        )

    simreffile_path = m1 / 'vpc_simulation.1.npctab.dta.refcorr'
    with open(simfile_path, "r") as sh, open(simreffile_path, "w") as dh:
        tab = -1
        for line in sh:
            if line.startswith("TABLE NO."):
                row = 0
                tab += 1
                dh.write(line)
            elif re.match(r'\s*[A-Za-z]', line):
                header = line.split()
                mdv_index = header.index("MDV")
                idv_index = header.index(idv)
                dv_index = header.index(dv)
                dh.write(line)
            else:
                values = line.split()
                if float(values[mdv_index]) == 1.0:
                    dh.write(line)
                else:
                    values[dv_index] = str(rcdv_sim[tab][row])
                    if have_eta_on_idv:
                        values[idv_index] = str(rcidv_sim[tab][row])
                    new_line = ' ' + ' '.join(values) + '\n'
                    dh.write(new_line)
                row += 1

    origreffile_path = m1 / 'vpc_original.npctab.dta.refcorr'
    with open(origfile_path, "r") as sh, open(origreffile_path, "w") as dh:
        row = 0
        for line in sh:
            if line.startswith("TABLE NO."):
                dh.write(line)
            elif re.match(r'\s*[A-Za-z]', line):
                header = line.split()
                mdv_index = header.index("MDV")
                idv_index = header.index(idv)
                dv_index = header.index(dv)
                dh.write(line)
            else:
                values = line.split()
                if float(values[mdv_index]) == 1.0:
                    dh.write(line)
                else:
                    values[dv_index] = str(rcdv[row])
                    if have_eta_on_idv:
                        values[idv_index] = str(rcidv[row])
                    new_line = ' ' + ' '.join(values) + '\n'
                    dh.write(new_line)
                row += 1
