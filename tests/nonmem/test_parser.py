import pytest

from pharmpy.basic import Expr
from pharmpy.model import Assignment
from pharmpy.model.external.nonmem.nmtran_parser import NMTranParser
from pharmpy.model.external.nonmem.parsing import parse_table_columns
from pharmpy.modeling import read_model_from_string


def symbol(x):
    return Expr.symbol(x)


def test_parse_des():
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA testpharmpy/lib/python3.10/site-packages/pharmpy/internals/example_models/pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CLMM = TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$THETA (-.99,.1)
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ;        IVV

$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME AMT WGT APGR IPRED PRED TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=pheno.tab"""
    model = read_model_from_string(code)
    assert len(model.statements) == 16


def test_simple_parse():
    parser = NMTranParser()

    model = parser.parse('$PROBLEM MYPROB\n')

    assert len(model.records) == 1
    assert type(model.records[0]).__name__ == 'ProblemRecord'
    assert str(model) == '$PROBLEM MYPROB\n'

    model2_str = ';Comment\n   $PROBLEM     TW2\n'
    model2 = parser.parse(model2_str)

    assert len(model2.records) == 2
    assert type(model2.records[0]).__name__ == 'RawRecord'
    assert type(model2.records[1]).__name__ == 'ProblemRecord'

    assert str(model2) == model2_str


def test_round_trip(pheno_path):
    parser = NMTranParser()

    with open(pheno_path, 'r') as fh:
        content = fh.read()
    model = parser.parse(content)
    assert str(model) == content


@pytest.mark.parametrize(
    'buf,columns',
    [
        ('$TABLE ID TIME', [['ID', 'TIME', 'DV', 'PRED', 'RES', 'WRES']]),
        ('$TABLE ID TIME NOAPPEND', [['ID', 'TIME']]),
        (
            '$TABLE ID TIME CIPREDI=CONC NOAPPEND',
            [['ID', 'TIME', 'CIPREDI']],
        ),
        (
            '$TABLE ID TIME CONC=CIPREDI NOAPPEND',
            [['ID', 'TIME', 'CIPREDI']],
        ),
        (
            '$TABLE ID TIME CONC=CIPREDI NOAPPEND\n$TABLE ID TIME CONC',
            [['ID', 'TIME', 'CIPREDI'], ['ID', 'TIME', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES']],
        ),
    ],
)
def test_table_columns(buf, columns):
    parser = NMTranParser()

    cs = parser.parse(f'$PROBLEM\n{buf}')
    parsed_columns = parse_table_columns(cs, netas=2)
    assert parsed_columns == columns


def test_parse_multiple_dvs():
    code1 = """$PROBLEM PHENOBARB PD
$INPUT ID TIME DV AMT DVID
$DATA pheno_pd.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$ABBR REPLACE ETA_E0=ETA(3)
$ABBR REPLACE ETA_S=ETA(4)
$PK
S = THETA(4)*EXP(ETA_S)
E0 = THETA(3)*EXP(ETA_E0)
VC = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V=VC
$ERROR
C = A(1)/V
Y = C*(1+EPS(1))
E = A(1)*S/V + E0
Y_2 = E + E*EPS(2)
IF (DVID.EQ.1) THEN
    Y = Y
ELSE
    Y = Y_2
END IF
$ESTIMATION MAXEVAL=9999 SIGDIGITS=4 POSTHOC
$THETA (0, 95.0648) FIX ; POP_VC
$THETA (0, 10.3416) FIX ; POP_CL
$THETA  (0,0.1) ; POP_E0
$THETA  (0,0.1) ; POP_S
$OMEGA BLOCK(2) FIX
0.0605492	; IIV_VC
0.000294284	; IIV_VC_IIV_CL
1.13303e-05	; IIV_CL
$OMEGA  0.09 ; IIV_E0
$OMEGA  0.09 ; IIV_S
$SIGMA 1.16066 FIX; PK RUV_PROP
$SIGMA  0.09 ; sigma"""

    model = read_model_from_string(code1)
    assert Assignment.create(symbol('Y_1'), symbol('Y')) not in model.statements.after_odes
    assert Assignment.create(symbol('Y_2'), symbol('Y_2')) not in model.statements.after_odes
    assert (
        Assignment.create(symbol('Y'), symbol('C') * (1 + symbol('EPS_1')))
        in model.statements.after_odes
    )
    assert (
        Assignment.create(symbol('Y_2'), symbol('E') + symbol('E') * symbol('EPS_2'))
        in model.statements.after_odes
    )
    assert model.dependent_variables == {symbol('Y'): 1, symbol('Y_2'): 2}
    assert model.observation_transformation == {
        symbol('Y'): symbol('Y'),
        symbol('Y_2'): symbol('Y_2'),
    }

    code2 = """$PROBLEM PHENOBARB PD
$INPUT ID TIME DV AMT DVID
$DATA pheno_pd.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$ABBR REPLACE ETA_E0=ETA(3)
$ABBR REPLACE ETA_S=ETA(4)
$PK
S = THETA(4)*EXP(ETA_S)
E0 = THETA(3)*EXP(ETA_E0)
VC = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V=VC
$ERROR
C = A(1)/V
E = A(1)*S/V + E0
IF (DVID.EQ.1) THEN
    Y = C*(1+EPS(1))
ELSE
    Y = E + E*EPS(2)
END IF
$ESTIMATION MAXEVAL=9999 SIGDIGITS=4 POSTHOC
$THETA (0, 95.0648) FIX ; POP_VC
$THETA (0, 10.3416) FIX ; POP_CL
$THETA  (0,0.1) ; POP_E0
$THETA  (0,0.1) ; POP_S
$OMEGA BLOCK(2) FIX
0.0605492	; IIV_VC
0.000294284	; IIV_VC_IIV_CL
1.13303e-05	; IIV_CL
$OMEGA  0.09 ; IIV_E0
$OMEGA  0.09 ; IIV_S
$SIGMA 1.16066 FIX; PK RUV_PROP
$SIGMA  0.09 ; sigma"""

    model2 = read_model_from_string(code2)
    assert (
        Assignment.create(symbol('Y_1'), symbol('C') * (1 + symbol('EPS_1')))
        in model2.statements.after_odes
    )
    assert (
        Assignment.create(symbol('Y_2'), symbol('E') + symbol('E') * symbol('EPS_2'))
        in model2.statements.after_odes
    )
    assert model2.dependent_variables == {symbol('Y_1'): 1, symbol('Y_2'): 2}
    assert model2.observation_transformation == {
        symbol('Y_1'): symbol('Y_1'),
        symbol('Y_2'): symbol('Y_2'),
    }
