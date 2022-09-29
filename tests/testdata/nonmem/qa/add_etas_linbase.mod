$PROBLEM    PHENOBARB SIMPLE MODEL
$INPUT      ID DV MDV OPRED D_EPS1 TIME AMT WGT APGR D_ETA1 D_ETA2
            D_ETA3 D_ETA4 OETA1 OETA2 OETA3 OETA4 D_EPSETA1_1
            D_EPSETA1_2 D_EPSETA1_3 D_EPSETA1_4
$DATA      add_etas_linbase.dta IGNORE=@ IGNORE(MDV.NEN.0)
$PRED
BASE1=D_ETA1*(ETA(1)-OETA1)
BASE2=D_ETA2*(ETA(2)-OETA2)
BASE3=D_ETA3*(ETA(3)-OETA3)
BASE4=D_ETA4*(ETA(4)-OETA4)
BSUM1=BASE1+BASE2+BASE3+BASE4
BASE_TERMS=BSUM1
IPRED=OPRED+BASE_TERMS
ERR1=EPS(1)*(D_EPS1+D_EPSETA1_1*(ETA(1)-OETA1))
ERR2=EPS(1)*(D_EPSETA1_2*(ETA(2)-OETA2))
ERR3=EPS(1)*(D_EPSETA1_3*(ETA(3)-OETA3))
ERR4=EPS(1)*(D_EPSETA1_4*(ETA(4)-OETA4))
ESUM1=ERR1+ERR2+ERR3+ERR4
ERROR_TERMS=ESUM1
Y=IPRED+ERROR_TERMS
$OMEGA  0.111053  ;       IVCL
$OMEGA  0.201526  ;        IVV
$OMEGA  0.0001
$OMEGA  0.0001
$SIGMA  0.0164177
$ESTIMATION MCETA=1 METHOD=COND INTERACTION MAXEVALS=9999999 PRINT=1
