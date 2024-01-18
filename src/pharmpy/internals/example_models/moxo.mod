$PROBLEM    MOXONIDINE PK,FINAL ESTIMATES,simulated data
$INPUT      ID VISI AGE SEX WT COMP TIME CRCL AMT SS II CMT=DROP DV
            EVID
$DATA      moxo_simulated_final.csv IGNORE=@
$SUBROUTINE ADVAN2 TRANS1
$PK
;-----------OCCASIONS----------
   VIS3               = 0
   IF(VISI.EQ.3) VIS3 = 1
   VIS8               = 0
   IF(VISI.EQ.8) VIS8 = 1

;----------IOV--------------------
   
   KPCL  = VIS3*ETA(4)+VIS8*ETA(5)
   KPKA  = VIS3*ETA(6)+VIS8*ETA(7)

;---------- PK model ------------------

   CL    = THETA(1)*EXP(ETA(1)+KPCL)
   V     = THETA(2)*EXP(ETA(2))
   KA    = THETA(3)*EXP(ETA(3)+KPKA)
   ALAG1 = THETA(4)
   K     = CL/V
   S2    = V

$ERROR
    IPRED = A(2)/V
    IF (IPRED.EQ.0) THEN
        IPREDADJ = 2.22500000000000E-16
    ELSE
        IPREDADJ = IPRED
    END IF
    Y = IPRED + EPS(1)*IPREDADJ

$THETA  (0,23.1329) ; POP_CL
$THETA  (0,123.604) ; POP_V
$THETA  (0,8.02595) ; POP_KA
$THETA  (0,0.237773) ; LAG
$OMEGA  BLOCK(2)
 0.0751115
 0.0259926 0.144646  ;   IIV_CL_V
$OMEGA  BLOCK(1)
 2.78268  ;     IIV_KA
$OMEGA  BLOCK(1)
 0.0234871  ;     IOV_CL
$OMEGA  BLOCK(1) SAME
$OMEGA  BLOCK(1)
 0.51776  ;     IOV_KA
$OMEGA  BLOCK(1) SAME
$SIGMA  0.0764969
$ESTIMATION METHOD=1 MAXEVAL=9999
$COVARIANCE
$TABLE      ID TIME PRED IPRED CWRES CIPREDI DV NOAPPEND NOPRINT ONEHEADER
            FILE=moxtab1

