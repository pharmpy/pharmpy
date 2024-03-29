$PROBLEM SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$ABBR REPLACE THETA(CL)=THETA(1)
$ABBR REPLACE ETA(CL)=ETA(1)

$PK
CL=THETA(CL)*EXP(ETA(CL))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128
$SIGMA 0.1
