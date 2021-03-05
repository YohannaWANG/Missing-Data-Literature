
# Robust Inference under Missing Data Condition
## TODO:
```diff
- Full law identifiability (Missingness indicator as labels);
- Robust Inference (Without labels); 
```
## Missing Data and Causality
- [2013-Graphical Models for Inference with Missing Data](https://ftp.cs.ucla.edu/pub/stat_ser/r410.pdf)
- [2015-Missing Data as a Causal and Probabilistic Problem](https://auai.org/uai2015/proceedings/papers/204.pdf)
- [2016-Consistent estimation of functions of data missing non-monotonically and not at random](https://papers.nips.cc/paper/2016/hash/7bd28f15a49d5e5848d6ec70e584e625-Abstract.html)
- [2020-Full Law Identification in Graphical Models of Missing Data: Completeness Results](http://proceedings.mlr.press/v119/nabi20a/nabi20a.pdf)

## Odds Ratio Parameterization
- [2000-Regression Analysis under Non-Standard Situations: A Pairwise Pseudolikelihood Approach](https://sci-hub.se/https://www.jstor.org/stable/2680620?seq=1#metadata_info_tab_contents)
- [2003-A note on the prospective analysis of outcome‐dependent samples](https://sci-hub.se/https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00403)
- [2004-Nonparametric and Semiparametric Models for Missing Covariates in Parametric Regression](https://sci-hub.se/https://www.jstor.org/stable/27590495?seq=1#metadata_info_tab_contents)

## Robust Inference

# Causal_imputation
Missing data imputation using causal knowledge

## TODO:
```diff
+ Collect related works (NeurIPS, ICLR, ICML, AAAI, IJCAI, etc.) for missing data imputation with deep learning algorithms;
+ And the dataset used in their works;
- Graph neural network for i.i.d and time series data (with DAGs as the prior knowledge).
```

## Publications in NeurIPS 
[[2019]](https://nips.cc/Conferences/2019/Schedule)
  - [Missing Not at Random in Matrix Completion: The Effectiveness of Estimating Missingness Probabilities Under a Low Nuclear Norm Assumption](https://papers.nips.cc/paper/9628-missing-not-at-random-in-matrix-completion-the-effectiveness-of-estimating-missingness-probabilities-under-a-low-nuclear-norm-assumption)
    - [[Code]](https://github.com/georgehc/mnar_mc) [[PPT]](http://www.andrew.cmu.edu/user/georgech/presentations/mnar_mc_neurips2019.pdf)
    - [Dataset]: Synthetic data, [MovieLens-100k](https://grouplens.org/datasets/movielens/100k/)
  - [Scalable Structure Learning of Continuous-Time Bayesian Networks from Incomplete Data](https://arxiv.org/abs/1909.04570)
    - [[Code]](https://git.rwth-aachen.de/bcs/ssl-ctbn) [[Poster]](https://git.rwth-aachen.de/bcs/ssl-ctbn-poster)
    - [Dataset] [British household dataset](https://www.iser.essex.ac.uk/bhps/acquiring-the-data), IRMA gene-regulatory network data
  - [Neuropathic Pain Diagnosis Simulator for Causal Discovery Algorithm Evaluation](https://arxiv.org/abs/1906.01732) 
    - [[Code]](https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator) [[Poster]](https://drive.google.com/drive/folders/1Pup1r-R8FCseGXfHyltEevo3nQB6Edsg) [[Video]](https://www.youtube.com/watch?v=1UvVnIbjSX8&feature=youtu.be)
 
[[2018]](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
  - [Processing of missing data by neural networks](https://papers.nips.cc/paper/7537-processing-of-missing-data-by-neural-networks)
    - [[Code]](https://github.com/lstruski/Processing-of-missing-data-by-neural-networks) [[Video]](https://www.youtube.com/watch?v=dIiGW2vvCF0)
    - [Dataset] UCI repository, MNIST
  - [Modeling Dynamic Missingness of Implicit Feedback for Recommendation](https://papers.nips.cc/paper/7901-modeling-dynamic-missingness-of-implicit-feedback-for-recommendation)
    - [Dataset]: [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/), [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [LastFM dataset](http://millionsongdataset.com/lastfm/)
  - [Cluster Variational Approximations for Structure Learning of Continuous-Time Bayesian Networks from Incomplete Data](https://papers.nips.cc/paper/8013-cluster-variational-approximations-for-structure-learning-of-continuous-time-bayesian-networks-from-incomplete-data)
    - [[Code]](https://github.com/dlinzner-bcs/)

[[2016]](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016)
  - [Consistent Estimation of Functions of Data Missing Non-Monotonically and Not at Random](https://papers.nips.cc/paper/6297-consistent-estimation-of-functions-of-data-missing-non-monotonically-and-not-at-random)
  - [The Limits of Learning with Missing Data](https://papers.nips.cc/paper/6171-the-limits-of-learning-with-missing-data)
  - [Learning Influence Functions from Incomplete Observations](https://papers.nips.cc/paper/6181-learning-influence-functions-from-incomplete-observations)
  - [Dynamic matrix recovery from incomplete observations under an exact low-rank constraint](https://arxiv.org/pdf/1610.09420.pdf)
  - [High resolution neural connectivity from incomplete tracing data using nonnegative spline regression](https://papers.nips.cc/paper/6244-high-resolution-neural-connectivity-from-incomplete-tracing-data-using-nonnegative-spline-regression.pdf)
    - [[Code]](https://github.com/kamdh/high-res-connectivity-nips-2016)
  
## Publications in ICLR
[[2020]](https://iclr.cc/virtual_2020/index.html)
  - [Intensity-Free Learning of Temporal Point Processes](https://openreview.net/forum?id=HygOjhEYDH)
    - [[Code]](https://github.com/shchur/ifl-tpp) [[PPT]](https://iclr.cc/virtual_2020/poster_HygOjhEYDH.html)
    - [Dataset]: Synthetic data, [LastFM](http://millionsongdataset.com/lastfm/), [Reddit](https://github.com/srijankr/jodie/), [MOOC](https://github.com/srijankr/jodie/), [Wikipedia](https://github.com/srijankr/jodie/), [Stack Overflow](https://archive.org/details/stackexchange), [Yelp](https://www.kaggle.com/yelp-dataset/yelp-dataset)
  - [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://openreview.net/pdf?id=Hye1RJHKwB)
    - [[Code]](https://github.com/f90/FactorGAN) [[PPT]](https://iclr.cc/virtual_2020/poster_Hye1RJHKwB.html)
    - [Dataset]: image generation, image segmentation, and audio source separation
  - [Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks](https://openreview.net/pdf?id=BylsKkHYvH)
    - [[Code]](https://github.com/JoonyoungYi/sparsity-normalization) [[PPT]](https://iclr.cc/virtual_2020/poster_BylsKkHYvH.html)
    - [Dataset]: Collaborative Filtering (RECOMMENDATION) Datasets (5), Electronic Medical Records (EMR) Datasets (2), Single-Cell RNA Sequence Datasets, UCI Dataset 

[[2019]](https://iclr.cc/Conferences/2019/Schedule)
  - [MisGAN: Learning from Incomplete Data with Generative Adversarial Networks](https://openreview.net/forum?id=S1lDV3RcKm)
    - [[Code]](https://github.com/steveli/misgan)

[[2017]](https://openreview.net/group?id=ICLR.cc/2017/conference)
  - [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://openreview.net/pdf?id=BJC8LF9ex) 
  
## Publications in ICML 
[[2020]](https://icml.cc/Conferences/2020/Schedule)
  - [Full Law Identification in Graphical Models of Missing Data: Completeness Results](https://proceedings.icml.cc/static/paper_files/icml/2020/1396-Paper.pdf)
  - [Missing Data Imputation using Optimal Transport](https://arxiv.org/abs/2002.03860)
    - [[Code]](https://github.com/BorisMuzellec/MissingDataOT) [[PPT]](https://icml.cc/media/Slides/icml/2020/virtual(no-parent)-16-19-00UTC-6455-missing_data_im.pdf)
    - Dataset: Synthetic data, [23 datasets from the UCI machine learning repository](https://archive.ics.uci.edu/ml/index.php)
  - [Learning From Irregularly-Sampled Time Series: A Missing Data Perspective](https://proceedings.icml.cc/static/paper_files/icml/2020/3129-Paper.pdf)
    - [Dataset]: MNIST, [CelebA database](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [healthcare multivariate time series dataset, MIMIC-III](https://mimic.physionet.org/)
    
[[2019]](http://proceedings.mlr.press/v97/)
  - [Phase transition in PCA with missing data: Reduced signal-to-noise ratio, not sample size!](http://proceedings.mlr.press/v97/ipsen19a/ipsen19a.pdf)
    - [[Code]](https://github.com/nbip/ppca_ICML2019)
  - [Imputing Missing Events in Continuous-Time Event Streams](https://arxiv.org/pdf/1905.05570.pdf)
    - [[Code]](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing)
  - [MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets](http://proceedings.mlr.press/v97/mattei19a/mattei19a.pdf)
    - [[Code]](https://github.com/pamattei/miwae)
  - [Fast and Stable Maximum Likelihood Estimation for Incomplete Multinomial Models](http://proceedings.mlr.press/v97/zhang19o/zhang19o.pdf)
    - [[Code]](https://github.com/CYZhangHKU/Stable-Weaver)
  - [Co-manifold learning with missing data](http://proceedings.mlr.press/v97/mishne19a.html)
  - [Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random](http://proceedings.mlr.press/v97/wang19n.html)

[[2018]](https://icml.cc/Conferences/2018/Schedule?type=Poster)
  - [GAIN: Missing Data Imputation using Generative Adversarial Nets](http://proceedings.mlr.press/v80/yoon18a.html)
    - [[Code]](https://github.com/jsyoon0823/GAIN)

## Publications in AAAI 
[[2020]](https://aaai.org/Conferences/AAAI-20/wp-content/uploads/2020/01/AAAI-20-Accepted-Paper-List.pdf)
  - [The Missing Data Encoder: Cross-Channel Image Completion with Hide-And-Seek Adversarial Network](https://arxiv.org/abs/1905.01861) 
    - [[Project Page]](https://gitlab.com/adapo/themissingdataencoder/-/wikis/home)  [[Code]](https://gitlab.com/adapo/themissingdataencoder/-/wikis/MDE-code-wiki)
    - [Dataset]: MNIST, [CelebA database](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Oxford-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

  - [Joint Modeling of Local and Global Temporal Dynamics for Multivariate Time Series Forecasting with Missing Values](https://arxiv.org/abs/1911.10273)
    - [Dataset]: [Beijing Air](https://www.kdd.org/kdd2018/kdd-cup), [PhysioNet](https://physionet.org/about/database/), [Porto
Taxi](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data), [London Weather](https://www.kdd.org/kdd2018/kdd-cup)

  - [Polynomial Matrix Completion for Missing Data Imputation and Transductive Learning](https://arxiv.org/abs/1912.06989) 
    - [[Code]](https://github.com/jicongfan/Codes-of-Polynomial-Matrix-Completion-in-AAAI-2020) [[Poster]](https://jicongfan.github.io/poster/PMC.pdf)
    - [Dataset]: Synthetic data, [CMU Mocap dataset](http://mocap.cs.cmu.edu/)

[[2018]](https://aaai.org/Conferences/AAAI-18/wp-content/uploads/2017/12/AAAI-18-Accepted-Paper-List.Web_.pdf)
  - [Hawkes Process Inference with Missing Data](https://www.cs.ucr.edu/~cshelton/papers/index.cgi%3FSheQinShe18)
  - [Tracking Occluded Objects and Recovering Incomplete Trajectories by Reasoning about
Containment Relations and Human Actions](https://yzhu.io/publication/container2018aaai/paper.pdf)


## Publications in UAI
[[2019]](http://auai.org/uai2019/accepted.php)
  - [Identification In Missing Data Models Represented By Directed Acyclic Graphs](http://auai.org/uai2019/proceedings/papers/428.pdf)
 
## Publications in AISTATS
[[2020]](https://www.aistats.org/accepted.html)
  - [Imputation estimators for unnormalized models with missing data](https://arxiv.org/abs/1903.03630)
  - [Linear predictor on linearly-generated data with missing values: non consistency and solutions](https://arxiv.org/abs/2002.00658)
  
  [[2019]](http://proceedings.mlr.press/v89/)
  - [Causal Discovery in the Presence of Missing Data](https://arxiv.org/abs/1807.04010) [[Code]](https://github.com/TURuibo/MVPC)
  - [Precision Matrix Estimation with Noisy and Missing Data](https://arxiv.org/abs/1904.03548)
  
## Publications in IJCAI
[[2020]](http://static.ijcai.org/2020-accepted_papers.html)
  - [A Spatial Missing Value Imputation Method for Multi-view Urban Statistical Data](https://www.ijcai.org/Proceedings/2020/0182.pdf)
    - [[Code]](https://github.com/SMV-NMF/SMV-NMF/blob/master/SMV-NMF.zip) 
    - [Dataset]: Urban statistical datasets (Sydney,Melbourne, Brisbane, Perth, SYD-large, and MEL-large)

[[2019]](https://www.ijcai19.org/accepted-papers.html)
  - [HMLasso: Lasso with High Missing Rate](https://www.ijcai.org/Proceedings/2019/0491.pdf)
    - [[Code]](https://cran.r-project.org/web/packages/hmlasso/index.html)
    - [Datasets]: Synthetic data, UCI residental Building Data(https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set)
  - [What to Expect of Classifiers? Reasoning about Logistic Regression with Missing Feature](https://www.ijcai.org/Proceedings/2019/0377.pdf)
    - [[Code]](https://github.com/UCLA-StarAI/NaCL)
    - [[Dataset]](https://github.com/UCLA-StarAI/NaCL/tree/master/data): MNIST, ADULT, FASHION, COVTYPE, SPLICE

[[2018]](https://www.ijcai-18.org/accepted-papers/index.html)
  - [Temporal Belief Memory: Imputing Missing Data during RNN Training](https://www.ijcai.org/Proceedings/2018/0322.pdf) [[Code]](https://github.com/ykim32/TBM-missing-data-handling)
  - [Estimation with Incomplete Data: The Linear Case](https://ftp.cs.ucla.edu/pub/stat_ser/r480.pdf)
  - [Robust Feature Selection on Incomplete Data](https://www.ijcai.org/Proceedings/2018/0443.pdf)
    - [Dataset]: UCI Repository (Advertisement, Arrhythmia, Cvpu, and Mice) 
  
## Publications in KDD
[[2020]](https://www.kdd.org/kdd2020/accepted-papers)
  - [Missing Value Imputation for Mixed Data via Gaussian Copula](https://arxiv.org/pdf/1910.12845.pdf)
    - [[Code]](https://rdrr.io/github/udellgroup/mixedgcImp/)
    - [Dataset]: [General Social Survey (GSS) Data](https://gss.norc.org/), [MovieLens 1M Data](https://grouplens.org/datasets/movielens/1m/), [CAL500exp Data](http://slam.iis.sinica.edu.tw/demo/CAL500exp/), [classification datasets](https://waikato.github.io/weka-wiki/datasets/), [Lecturers Evaluation (LEV) and Employee
Selection (ESL), German Breast Cancer Study Group (GBSG)](https://cran.r-project.org/web/packages/mfp/), [Restaurant Tips (TIPS)](Restaurant Tips (TIPS))
  - [LogPar: Logistic PARAFAC2 Factorization for Temporal Binary Data with Missing Values](https://kejing.me/publications/2020_KDD_LogPar.pdf)
    - [[Code]](https://github.com/jakeykj/LogPar)
    - [Dataset]: [Sutter heart failure](https://catalog.data.gov/dataset?tags=sutter), [CMS Medical data](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF), [MIMIC ICU dataset](https://mimic.physionet.org/)
