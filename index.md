<div>
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3a/Singapore_Institute_of_Technology_logo.svg/640px-Singapore_Institute_of_Technology_logo.svg.png" height="100"/>
<img src="https://upload.wikimedia.org/wikipedia/en/8/8b/A%2ASTAR_logo.png" height="100"/>
</div>

# **Simplified machine learning approach for screening high hardness, low-density alloys**

Agency for Science, Technology and Research (A*STAR)

By: Quek Xiu Kun (1902711@sit.singaporetech.edu.sg)

Under supervision of: Dr Ng Chee Koon


---
## Introduction
High-entropy alloys (HEA), a novel class of alloys first introduced in 2004 by Yeh et al. and Cantor et al, are typically defined as containing five or more principal elements in a concentration of between 5 and 35 at.% [^1][^2]. Due to the number of different alloy systems and the composition range, HEAs may exhibit exceptional mechanical properties such as hardness/strength, ductility, as well as corrosion and oxidation resistances [^3]

In fields such as the Aerospace industry, materials are routinely required to undergo extreme exposure to stresses, strains, and rapid temperature changes. In addition, the ever increasing issue of global warming brought about by greenhouse gas emissions poses a significant challenge to aviation manufacturers. A key solution to reducing emissions is weight reduction as less Lift (and indirectly, Thrust) is required to sustain flight. Therefore, it is imperative that manufacturers use high-strength, light-weight materials.

It is entirely plausible that in the vast compositional search space of HEA there are high strength, light weight alloys that exceed the strength-to-weight ratio of traditional alloys. The challenge then becomes a matter of searching for these performant alloys within the search space. Due to the immense number of permutations brought about by the increased number of elements and the various element combinations, it becomes increasingly cost prohibitive for researchers to experimentally validate HEAs by trial and error. Therefore, we propose utilising Machine Learning (ML) algorithms to model the properties of HEAs.

---
## Machine Learning Framework
<div>
<img src="https://i.imgur.com/7PiGDoR.png"  height="300"/>
</div>

---
## Database
A total of 1,942 compositions are sourced from academic literature [^4][^5][^6].

### Pre-processing
During pre-processing, the following criteria were applied to the database:
- As-cast (no post-processing methods performed as they alter the alloy mechanical properties)
- Alloys with Vickers Hardness data
- All testing procedures performed at room temperature
- If there are duplicate entries of the same composition (arising from different academic literature), the mean value of their mechanical properties is used (provided none of the datapoints exceed 10% of the mean value).

### Data Analysis
There are a total of 30 elements commonly used in HEA research. They are:

Ag / Al / B / C / Ca / Co / Cr / Cu / Fe / Ga / Hf / Li / Mg / Mn / Mo / Nb / Nd / Ni / Pd / Re / Sc / Si / Sn / Ta / Ti / V / W / Y / Zn / Zr

We plot the input data to observe the feature distribution. Features with no variance are dropped.

![picture](https://i.imgur.com/KnC6rDk.png)

### Data Scaling
Data scaling converts each input feature to have zero-mean and unit variance of 1. This has proven to allow faster model convergence when used in ML [^7].

---
## Machine Learning
There are three prominent types of ML:
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Because the alloy composition and mechanical properties are labelled, supervised learning is the most effective model for this dataset. In addition, there are numerous supervised learning techniques in ML. A few of them are:

Method | Remarks
--- | ---
[Random Forest (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | Random forest uses decision trees to model data and is best used for classification although they can also be used for regression problems.
[Gradient Boosting (GB)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | Gradient Boosting builds upon and improves Decision Tree performance by optimising a differentiable loss function.
[Gaussian Process Regression (GPR)](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) | Gaussian Process uses probability distribution for inference and works well for normally distributed data. Works best for regression modelling.
[Artificial Neural Network (ANN)](https://keras.io/api/models/) | ANNs are based on biological neural networks and are applicable for both regression and classification modelling.

### Hyperparameters
Hyperparameters are training parameters variables used in ML. Changing these values may positively or negatively affect the training end result. Different machine learning methods have different hyperparameters.

### Training / Testing Data
Typically, the entire dataset is randomly split into training and testing datasets. For this project, 10% of the total dataset is extracted and reserved as unseen holdout data (data that is not seen by the ML model during training) and used to evaluate model performance.

### Cross Validation
To ensure proper tuning of the hyperparameters, training is conducted 5 times (known as k-fold) to ensure the full dataset is accurately represented. The average performance of all the models is then used to determine the performance of the hyperparameters.

When the optimal hyperparameters are obtained, the unseen holdout data is used to evaluate the ML model's performance.

### Metrics
There are two metrics that are of interest in regression problems. They are Mean Squared Error and Coefficient of Determination.

The Mean Squared Error (MSE) represents the average of the squared difference between the original and predicted values in a dataset.

The Coefficient of Determination (or R2) represents the goodness-of-fit of a model and is a statistical measure of how well the model approximates the real data points.

### Baseline Models
The four different ML models were created by using the default hyperparameters. MSE was used as the loss function. 10-fold cross validation was used to evaluate the models' performances.

Metric   |RF     |GB     |XGB    |GPR    |ANN
---      |---:   |---:   |---:   |---:   |---:
Test R2  |0.84   |0.78   |0.78   |0.81   |0.56
Train R2 |0.98   |0.93   |0.92   |0.88   |0.70
Test MSE |7454   |9822   |10117  |8690   |19452
Train MSE|1056   |3285   |3852   |5471   |13654

### Hyperparameter Tuning
We tune the hyperparamaters of each individual model.

Metric   |RF     |GB     |XGB    |GPR    |ANN
---      |---:   |---:   |---:   |---:   |---:
Test R2  |0.84   |0.85   |0.82   |0.88   |0.81
Train R2 |0.98   |0.99   |0.98   |0.99   |0.98
Test MSE |7373   |6843   |8294   |5430   |8185
Train MSE|1054   |481    |1023   |209    |962

### Validation on Holdout Data

RF |GB |XGB |GPR |ANN
:---: | :---: | :---: | :---: | :---:
R2 : 0.85|R2 : 0.88|R2 : 0.85|R2 : 0.92|R2 : 0.93|
<img src="https://i.imgur.com/PywwM0n.png"  height="400"/> | <img src="https://i.imgur.com/q1IZiwZ.png"  height="400"/> | <img src="https://i.imgur.com/6qjd60z.png"  height="400"/> | <img src="https://i.imgur.com/01EQwWi.png"  height="400"/> | <img src="https://i.imgur.com/Jj7svoG.png"  height="400"/>

It is clear that the ANN model performs the best. ANN was chosen as the algorithm of choice. The model was retrained using the entire dataset with the optimal hyperparameters.

## Search Space
A synthetic composition generator generates all possible permutations of an alloy under a given criteria. For example:

Number of Elements  |Step size (at.%)   |Max concentration (at.%)   |Permutations
:---                |:---               |:---                       |---:
3                   |5                  |35                         |3
5                   |5                  |35                         |2,226
7                   |5                  |35                         |104,692
9                   |5                  |35                         |1,992,195

## Exhaustive Search
Exhaustive search (also known as brute-force search systematically enumerates over all possible candidates within the search space. In our case, we ran all 1,992,195 compositions through our ML model.


---
## References
[^1]: B. Cantor, I.T.H. Chang, P. Knight, A.J.B. Vincent, Microstructural development in equiatomic multicomponent alloys, Materials Science and Engineering: A. 375–377 (2004) 213–218. https://doi.org/10.1016/j.msea.2003.10.257.
[^2]: J.-W. Yeh, S.-K. Chen, S.-J. Lin, J.-Y. Gan, T.-S. Chin, T.-T. Shun, C.-H. Tsau, S.-Y. Chang, Nanostructured High-Entropy Alloys with Multiple Principal Elements: Novel Alloy Design Concepts and Outcomes, Advanced Engineering Materials. 6 (2004) 299–303. https://doi.org/10.1002/adem.200300567.
[^3]: M.-H. Tsai, J.-W. Yeh, High-Entropy Alloys: A Critical Review, Materials Research Letters. 2 (2014) 107–123. https://doi.org/10.1080/21663831.2014.912690.
[^4]: Borg, C.K.H., Frey, C., Moh, J. et al. Expanded dataset of mechanical properties and observed phases of multi-principal element alloys. Sci Data 7, 430 (2020). https://doi.org/10.1038/s41597-020-00768-9
[^5]: Stéphane Gorsse, Daniel B. Miracle, Oleg N. Senkov, Mapping the world of complex concentrated alloys, Acta Materialia, Volume 135, 2017, Pages 177-187, ISSN 1359-6454, https://doi.org/10.1016/j.actamat.2017.06.027.
[^6]: Chen Yang, Chang Ren, Yuefei Jia, Gang Wang, Minjie Li, Wencong Lu, A machine learning-based alloy design system to facilitate the rational design of high entropy alloys with enhanced hardness, Acta Materialia, Volume 222, 2022, 117431, ISSN 1359-6454, https://doi.org/10.1016/j.actamat.2021.117431.
[^7]: A.Y.-T. Wang, R.J. Murdock, S.K. Kauwe, A.O. Oliynyk, A. Gurlo, J. Brgoch, K.A. Persson, T.D. Sparks, Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices, Chem. Mater. 32 (2020) 4954–4965. https://doi.org/10.1021/acs.chemmater.0c01907.
