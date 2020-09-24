model=pheno_with_cov.mod
search_direction=forward
logfile=scmlog1.txt
p_forward=0.05

continuous_covariates=WGT,APGR,CV1,CV2,CV3
categorical_covariates=CVD1,CVD2,CVD3

[test_relations]
CL=WGT,APGR,CV1,CV2,CV3
V=CVD1,WGT

[valid_states]
continuous = 1,2,3,4,5
categorical = 1,2

[included_relations]
CL=APGR-2
V=CVD1-2
