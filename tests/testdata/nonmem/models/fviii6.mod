;Basal FVIII model Clinical data ignored
;PREP (grouped) as covariate on V1
;PREP (grouped) as covariate on CL SAME THETA
;age as covariate on CL only

$PROBLEM 2-comp, allometric scaling, IOV on CL and V
$INPUT ID DAT2=DROP OTIM=DROP TIME DV AMT RATE AGE WT
BLGR PREP EVID OCC YBIN OCC2 STUD
$DATA fviii6.prn IGNORE=(STUD.EQ.1) IGN=@ 
$SUBROUTINES ADVAN3 TRANS4

$PK
Q1 = 0
Q2 = 0
Q3 = 0
Q4 = 0
Q5 = 0
Q6 = 0
Q7 = 0
Q8 = 0
IF(OCC.EQ.1)Q1 = 1
IF(OCC.EQ.2)Q2 = 1
IF(OCC.EQ.3)Q3 = 1
IF(OCC.EQ.4)Q4 = 1
IF(OCC.EQ.5)Q5 = 1
IF(OCC.EQ.6)Q6 = 1
IF(OCC.EQ.7)Q7 = 1
IF(OCC.EQ.8)Q8 = 1

IF (PREP.EQ.1) PREP2 = 1
IF (PREP.EQ.2) PREP2 = 2
IF (PREP.EQ.3) PREP2 = 1
IF (PREP.EQ.4) PREP2 = 1
IF (PREP.EQ.5) PREP2 = 1
IF (PREP.EQ.6) PREP2 = 2
IF (PREP.EQ.7) PREP2 = 2
;IF (PREP.EQ.8) PREP2 = 3
IF (PREP.EQ.9) PREP2 = 2

IF (PREP2.EQ.1) THEN
TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(13)*(AGE-40)) ;80 = Median of body weight
TVV1 = THETA(2)*((WT/80)**THETA(8)) 
ENDIF
IF (PREP2.EQ.2) THEN
TVCL = THETA(1)*((WT/80)**THETA(7))*(1+THETA(12))*(1+THETA(13)*(AGE-40))
TVV1 = THETA(2)*((WT/80)**THETA(8))*(1+THETA(12))
ENDIF
TVQ  = THETA(5)*(WT/80)**THETA(9)
TVV2 = THETA(6)*(WT/80)**THETA(10)
IOVCL= Q1*ETA(4)+Q2*ETA(5)+Q3*ETA(6)+Q4*ETA(7)+Q5*ETA(8)+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)
;IOVCL2 = IOVCL+Q6*ETA(9)+Q7*ETA(10)+Q8*ETA(11)
IOVV = Q1*ETA(12)+Q2*ETA(13)+Q3*ETA(14)+Q4*ETA(15)+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)
;IOVV2= IOVV+Q5*ETA(16)+Q6*ETA(17)+Q7*ETA(18)+Q8*ETA(19)
CL   = TVCL*EXP(ETA(2)+IOVCL)
V1   = TVV1*EXP(ETA(3)+IOVV)
Q    = TVQ
V2   = TVV2

TVBA = THETA(11)
BASE = TVBA*EXP(ETA(1))

S1 = V1

K = CL/V1
K12 = Q/V1
K21 = Q/V2

$ERROR
IPRED = F + BASE
IRES  = DV-IPRED
W     = SQRT(THETA(3)**2 + (THETA(4)*IPRED)**2)
IWRES = IRES/W
IF(W.EQ.0) W = 1

Y = IPRED + W*EPS(1)

$THETA             
(0,250)            ;1 CL (ml/h)
(0,3380)           ;2 V1 (ml)
(0,0.01)           ;3 Add res error (SD, units = U/ml)
(0,0.08)           ;4 Prop res error (Rel SD = CV)
(0,444)            ;5 Q (ml/h)
(0,268)            ;6 V2 (ml)
0.75 FIX           ;7 Scaling for CL
1 FIX              ;8 Scaling for V1
0.75 FIX           ;9 Scaling for Q
1 FIX              ;10 Scaling for V2
(0,0.01)           ;11 Baseline
(-0.2)             ;12 Difference in TVCL and TVV1 for full-lenght recombinant FVIII
(-0.0185,-0.0076,0.055) ;13 Fractional change in CL per year different from 40 years

$OMEGA BLOCK(3)
0.141                 ;varBase
0.0611 0.107          ;covarBase-CL varETACL
0 0.0509 0.0397       ;covar Base-V1 covar CL-V1 varV1


$OMEGA BLOCK(1) 0.0381 ;IOV_CL
$OMEGA BLOCK(1) SAME   
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME   
$OMEGA BLOCK(1) SAME   
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) 0.0410 ;IOV_V
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME   
$OMEGA BLOCK(1) SAME  
$OMEGA BLOCK(1) SAME 

$SIGMA 1 FIX    
$ESTIMATION METHOD=1 INTER PRINT=1 MAXEVAL=9999
$COVARIANCE PRECOND=1 PFCOND=1

$TABLE ID TIME IPRED CWRES
      NOPRINT ONEHEADER FILE=sdtabmod
