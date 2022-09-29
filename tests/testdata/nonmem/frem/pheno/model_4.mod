;; 1. Based on: 5
$PROBLEM    PHENOBARB SIMPLE MODEL
$DATA      frem_dataset.dta IGNORE=@
$INPUT      ID TIME AMT WGT APGR DV FA1 FA2 MDV FREMTYPE
$SUBROUTINE ADVAN1 TRANS2
$PK

IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
      TVCL=THETA(1)*WGT
      TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
      CL=TVCL*EXP(ETA(1))
      V=TVV*EXP(ETA(2))
      S1=V


    SDC3 = 2.23763568135
    SDC4 = 0.704564727537
$ERROR

      W=F
      Y=F+W*EPS(1)

      IPRED=F         ;  individual-specific prediction
      IRES=DV-IPRED   ;  individual-specific residual
      IWRES=IRES/W    ;  individual-specific weighted residual

;;;FREM CODE BEGIN COMPACT
;;;DO NOT MODIFY
    IF (FREMTYPE.EQ.100) THEN
;      APGR  2.23763568135
       Y = THETA(4) + ETA(3)*SDC3 + EPS(2)
       IPRED = THETA(4) + ETA(3)*SDC3
    END IF
    IF (FREMTYPE.EQ.200) THEN
;      WGT  0.704564727537
       Y = THETA(5) + ETA(4)*SDC4 + EPS(2)
       IPRED = THETA(5) + ETA(4)*SDC4
    END IF
;;;FREM CODE END COMPACT
$THETA  (0,0.00469555) ; CL
$THETA  (0,0.984258) ; V
$THETA  (-.99,0.15892)
$THETA  6.42372881356 FIX ; TV_APGR
 1.52542372881 FIX ; TV_WGT
$OMEGA  BLOCK(4)
 0.0293508  ;       IVCL
 0.000286193 0.027906  ;        IVV
 -0.0676481 0.0235094 1  ;   BSV_APGR
 0.0256033 -0.00161838 0.24458 1  ;    BSV_WGT
$SIGMA  0.013241
$SIGMA  0.0000001  FIX  ;     EPSCOV
$ESTIMATION METHOD=1 INTERACTION NONINFETA=1 MCETA=1
$COVARIANCE UNCONDITIONAL
$TABLE      ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE
            NOAPPEND NOPRINT ONEHEADER FILE=sdtab1
