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
        factor = (predref - lb) / (pred - lb)

    def correct(dv, factor, logdv, lb):
        if logdv:
            rpcdv = dv + factor
        else:
            rpcdv = lb + (dv - lb) * factor
        return rpcdv

    rpcdv = correct(dv, factor, logdv, lb)
    rpcdv_sim = [correct(sim['DV'], factor, logdv, lb) for sim in simdata]

    refdvs = pd.concat((ref['DV'] for ref in refdata), axis=1)
    rpcdvs = pd.concat(rpcdv_sim, axis=1)

    if logdv:
        refdv_sd = refdvs.std(axis=1)
        rpcdvs_sd = rpcdvs.std(axis=1)
    else:
        refdv_sd = np.log(refdvs).std(axis=1)
        rpcdvs_sd = np.log(rpcdvs).std(axis=1)
    var_factor = refdv_sd / rpcdvs_sd

    def varcorrect(dv, factor, predref, logdv):
        if logdv:
            rpvcdv = predref + (dv - predref) * factor
        else:
            log_predref = np.log(predref)
            rpvcdv = np.exp(log_predref + (np.log(dv) - log_predref) * factor)
        return rpvcdv

    rpvcdv = varcorrect(rpcdv, var_factor, predref, logdv)
    rpvcdv_sim = [varcorrect(x, var_factor, predref, logdv) for x in rpcdv_sim]
    return rpvcdv, rpvcdv_sim


def reference_correction_from_psn_vpc(path):
    path = Path(path)
    opts = options_from_command(psn_command(path))
    logdv = bool(int(opts.get('lnDV', '0')))  # NOTE: Not entirely correct
    lower_bound = opts.get('lower_bound', 0.0)

    m1 = path / 'm1'

    model = read_model(m1 / "vpc_original.mod")

    origfile_path = m1 / "vpc_original.npctab.dta"
    origfile = NONMEMTableFile(origfile_path)
    origdata = origfile.tables[0].data_frame
    origdata = origdata[origdata['MDV'] == 0.0]

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

    predref = refdata[0]['PRED']

    lb = evaluate_expression(model, lower_bound)[origdata.index]
    rcdv, rcdv_sim = calculate_reference_correction(
        origdata['DV'], pred, predref, simdata, refdata, logdv, lb
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
                dv_index = header.index("DV")
                dh.write(line)
            else:
                values = line.split()
                if float(values[mdv_index]) == 1.0:
                    dh.write(line)
                else:
                    values[dv_index] = str(rcdv_sim[tab][row])
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
                dv_index = header.index("DV")
                dh.write(line)
            else:
                values = line.split()
                if float(values[mdv_index]) == 1.0:
                    dh.write(line)
                else:
                    values[dv_index] = str(rcdv[row])
                    new_line = ' ' + ' '.join(values) + '\n'
                    dh.write(new_line)
                row += 1
