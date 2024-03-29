$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_dvid.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV DVID
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
IF (DVID.EQ.1) THEN
    Y = F + F*EPS(1)
ELSE
    Y = F + F*EPS(1) + EPS(2)
ENDIF

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.01     ; sigma_prop
$SIGMA 0.01     ; signa_add

$ESTIMATION METHOD=1 MAXEVAL=99999 INTERACTION

$TABLE ID TIME DV PRED NOAPPEND FILE=sdtab_pheno
