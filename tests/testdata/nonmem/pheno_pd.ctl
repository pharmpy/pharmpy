$PROBLEM direct_effect_linear
$INPUT ID TIME AMT WGT APGR DV DVID
$DATA ../.datasets/structsearch_run1.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$ABBR REPLACE ETA_CL=ETA(1)
$ABBR REPLACE ETA_VC=ETA(2)
$ABBR REPLACE ETA_E0=ETA(3)
$PK
SLOPE = THETA(4)
E0 = THETA(3)*EXP(ETA_E0)
CL = THETA(1)*EXP(ETA_CL)
VC = THETA(2)*EXP(ETA_VC)
V = VC
$ERROR
IPRED = A(1)/VC
IF (IPRED.EQ.0) THEN
    IPREDADJ = 2.22500000000000E-16
ELSE
    IPREDADJ = IPRED
END IF
Y = IPRED + EPS(1)*IPREDADJ
E = A(1)*SLOPE/V + E0
Y_2 = E + E*EPS(2)
IF (DVID.EQ.1) THEN
    Y = Y
ELSE
    Y = Y_2
END IF
$THETA  (0,0.00274199) FIX ; POP_CL
$THETA  (0,1.44692) FIX ; POP_VC
$THETA  (0,5.75005) FIX ; POP_E0
$THETA  (0,0.1) ; POP_Slope
$OMEGA BLOCK(2) FIX
1e-05	; IIV_CL
-3.08041e-07	; IIV_CL_IIV_VC
9.90949e-06	; IIV_VC
$OMEGA  9e-06 FIX ; IIV_E0
$SIGMA  0.00630501 FIX ; sigma
$SIGMA  0.338363 FIX ; sigma1
$ESTIMATION METHOD=COND INTER MAXEVAL=99999
$TABLE ID TIME DV CIPREDI CWRES FILE=mytab NOAPPEND NOPRINT