model=pheno_with_cov.mod
search_direction=backward
logfile=scmlog1.txt
pvalue=0.01

continuous_covariates=WGT,APGR,CV1,CV2,CV3
categorical_covariates=CVD1,CVD2,CVD3

[test_relations]
CL=WGT,APGR,CV1,CV2,CV3
V=CVD1,CV2,WGT

[valid_states]
continuous = 1,2,3,4,5
categorical = 1,2

[included_relations]
CL=WGT-2,CV1-2
V=CV2-2,WGT-2,CVD1-2
