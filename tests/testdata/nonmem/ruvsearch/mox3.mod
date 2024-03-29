$PROBLEM
$INPUT ID VISI AGE SEX WT COMP IACE DIG DIU TAD TIME CRCL AMT SS II VID1 DV EVID SHR SHR2
$DATA moxo_simulated_resmod.csv IGNORE=@
$SUBROUTINES ADVAN2  TRANS2
$PK
VC = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
MAT = THETA(3) * EXP(ETA(3))
KA = 1/MAT
V = VC
$ERROR
Y = A(2)/VC + A(2)/VC * EPS(1)
$THETA (0, 100, Inf) ; POP_VC
$THETA (0, 10.0, Inf) ; POP_CL
$THETA (0, 3.0, Inf) ; POP_MAT
$OMEGA BLOCK(2)
0.1	; IIV_VC
0.01	; IIV_VC_IIV_CL
0.1	; IIV_CL
$OMEGA 0.1 ; IIV_MAT
$SIGMA 0.1; RUV_PROP
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 SADDLE_RESET=1
$TABLE ID TIME CWRES CIPREDI VC CL MAT DV PRED RES WRES IPREDI IWRESI FILE=mytab NOAPPEND NOPRINT
