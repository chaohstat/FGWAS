# FGWAS
Functional Genome Wide Association analysiS (FGWAS) is a Python coding based toolbox for imaging genetic analysis. 

Functional phenotypes (e.g., subcortical surface representation), which commonly arise in imaging genetic
studies, have been used to detect putative genes for complexly inherited neuropsychiatric and neurodegenerative
disorders. However, existing statistical methods largely ignore the functional features (e.g., functional smoothness
and correlation). The aim of this paper is to develop a functional genome-wide association analysis (FGWAS)
framework to efficiently carry out whole-genome analyses of functional phenotypes. FGWAS consists of three
components: a multivariate varying coefficient model, a global sure independence screening procedure, and a test
procedure. Compared with the standard multivariate regression model, the multivariate varying coefficient model
explicitly models the functional features of functional phenotypes through the integration of smooth coefficient
functions and functional principal component analysis. Statistically, compared with existing methods for genomewide
association studies (GWAS), FGWAS can substantially boost the detection power for discovering important
genetic variants influencing brain structure and function. Simulation studies show that FGWAS outperforms
existing GWAS methods for searching sparse signals in an extremely large search space, while controlling for the
family-wise error rate.
