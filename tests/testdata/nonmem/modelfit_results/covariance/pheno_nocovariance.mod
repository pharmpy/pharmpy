$PROBLEM PHENOBARB SIMPLE MODEL+time_varying3
$DATA testdata/pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 TAD
$SUBROUTINE ADVAN1 TRANS2

$PK
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V

$ERROR
W=F
IF (TAD.LT.11.5) THEN
    Y = EPS(1)*THETA(4)*W*EXP(ETA(3)) + F
ELSE
    Y = EPS(1)*W*EXP(ETA(3)) + F
END IF
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469539) ; PTVCL
$THETA (0,0.98352) ; PTVV
$THETA (-.99,0.156691)
$THETA  0.801621 ; time_varying
$OMEGA DIAGONAL(2)
 0.0254791  ;       IVCL
 0.0276841  ;        IVV

$OMEGA  0.039145 ; IIV_RUV1
$SIGMA 0.0176838
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab2