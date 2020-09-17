$PROBLEM    PHENOBARB SIMPLE MODEL
$INPUT      ID TIME AMT WGT APGR DV CV1 CV2 CV3 CVD1 CVD2 CVD3
$DATA      pheno_ch.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK

      TVCL  = THETA(1)
      CL    = TVCL*EXP(ETA(1))

      TVV   = THETA(2)
      V     = TVV*EXP(ETA(2))

      S1    = V
$ERROR
      
      W     = THETA(3)
      Y     = F+W*EPS(1)

      IPRED = F          ;  individual-specific prediction
      IRES  = DV-IPRED   ;  individual-specific residual
      IWRES = IRES/W     ;  individual-specific weighted residual

$THETA  (0,0.0105)
$THETA  (0,1.0500)
$THETA  (0,0.4)
$OMEGA  .4
 .25
$SIGMA  1  FIX
$ESTIMATION MAXEVAL=3 SIGDIGITS=4 METHOD=CONDITIONAL
$COV PRINT=E UNCOND
