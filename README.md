# DeepMirror
Project for benchmarking ML models on TDC datasets

## Approach and times

To tackle the problem of benchmarking several models, the first necessary step was to learn about the SOTA and the field involving the simulation and predicton of ADME markers.
For that I researched into TDC common datasets, different existing approaches and repositories on how to tackle the problem. 
I came across several libraries such as DeepChem  and DeepPurpose which provided methods to develop solutions.
After that, it was necessary to design a structure for the problem that allowed for scalability and reutilization of the code.
To do this, an Abstract class was created to define the necessary functions each model has to have in order to be able to benchmark it following the TDC approach and standards, this way, any model developed in the system can be added to the scoreboard provided at TDC commons.
To develop the models further details are provided in Model Development section.
The time used for this project can be devided in research and development, 4-6 hours were used to research the different libraries and context of the project. After that around 6-8 hours were used for the development of the code, including solving dependency problems, and fixing data types. Finally extra time was used for compiling and running the models and the benchmarking process.

## Files
- models.py:  proposed models to solve this task can be found as well as an abstract class indicating what functions and parameters every new model must have in order to follow the same standards and be able to benchmark it.
- benchmark.py: the benchmark function is here with several parameters, evaluates the given models on the chosen datasets.
- main.py: models and datasets are selected and given to the benchmark function.

## Model development

Three different models were proposed by DeepMirror to establish a benchmark for the different datasets. A Decision Tree, a Graph Neural Network and a Transformer model. This would be the initial job, therefore, three different approaches were chosen in order to give more options for future models or developers. 
For the Tree model, sklearn was used to implement it, making it straightforward. GridSearch was used to find the best parameters given a grid of possibilities. Featurizing the input was done with DeepChem library and its provided featurizers, in this case Mol2VecFingerprint() was used.
Gradient Boosting was also implemented through sklearn's GradientBoostingRegressor out of curiosity in order to compare it to the performance of the Tree model.
The Graph model was implemented through DeepChem library and their provided models. Graph Convolutional Network was used in regression mode. The chosen featurizer for this model was MolGraphConvFeaturizer().
Finally the transformer model was the most complex one. Bert for sequence classification was implemented from the transformers Python library. To train this model Pytorch was used, therefore the train and predict methods were implemented step by step, validation methods are ready but not used due to training constraints. To featurize the input, several actions were needed. First, it was tokenized by by DeepChem BasicSmilesTokenizer(), after that, sequence vectors were formed by assigning a unique key to each token of the vocabulary and swapping each token by their corresponding key. 

## Datasets 
The selected datasets were the 18 datasets that formed the ADME category from TDC https://tdcommons.ai/benchmark/admet_group/overview/ :
- caco2_wang
- hia_hou
- pgp_broccatelli
- bioavailability_ma
- lipophilicity_astrazeneca
- solubility_aqsoldb
- bbb_martins
- ppbr_az
- vdss_lombardo
- cyp2d6_veith
- cyp3a4_veith
- cyp2c9_veith
- cyp2d6_substrate_carbonmangels
- cyp3a4_substrate_carbonmangels
- cyp2c9_substrate_carbonmangels
- half_life_obach
- clearance_microsome_az
- clearance_hepatocyte_az

## Use 
First it is necessary to install the requirements from requirements.txt, it can be done with pip.
To benchmark the wanted models, each model has to be imported and benchmark function from benchmark.py has to be called. The arguments are an array with all the models to benchmark and a list with all the datasets to test(selected from the list under datasets setcion), if no dataset is selected, all are tested.
It can be done through terminal or by running main.py with the desired modifications.

## New Models
To implement new models, a new file can be created for each new model, it is necessary for each model to import the abstract class Model from models.py and inherit from that class, implementing each of the functions stated plus any othe necessary function.

## Results 

### Tree:
- 'caco2_wang': [0.497, 0.017]
- 'hia_hou': [0.837, 0.01]
- 'pgp_broccatelli': [0.697, 0.011]
- 'bioavailability_ma': [0.62, 0.009]
- 'lipophilicity_astrazeneca': [0.941, 0.0]
- 'solubility_aqsoldb': [1.291, 0.0]
- 'bbb_martins': [0.774, 0.0]
- 'ppbr_az': [11.539, 0.037]
- 'vdss_lombardo': [0.253, 0.018]
- 'cyp2d6_veith': [0.348, 0.0]
- 'cyp3a4_veith': [0.68, 0.0]
- 'cyp2c9_veith': [0.567, 0.0]
- 'cyp2d6_substrate_carbonmangels': [0.455, 0.022]
- 'cyp3a4_substrate_carbonmangels': [0.597, 0.021]
- 'cyp2c9_substrate_carbonmangels': [0.313, 0.023]
- 'half_life_obach': [0.063, 0.001]
- 'clearance_microsome_az': [0.455, 0.001]
- 'clearance_hepatocyte_az': [0.256, 0.01]

### Transformer:
- 'caco2_wang': [0.627, 0.111]
- 'hia_hou': [0.569, 0.137]
- 'pgp_broccatelli': [0.764, 0.026]
- 'bioavailability_ma': [0.576, 0.19]
- 'lipophilicity_astrazeneca': [0.975, 0.013]
- 'solubility_aqsoldb': [1.478, 0.195]
- 'bbb_martins': [0.772, 0.025]
- 'ppbr_az': [13.115, 0.286]
- 'vdss_lombardo': [0.188, 0.126]
- 'cyp2d6_veith': [0.224, 0.065]
- 'cyp3a4_veith': [0.567, 0.08]
- 'cyp2c9_veith': [0.267, 0.046]
- 'cyp2d6_substrate_carbonmangels': [0.366, 0.049]
- 'cyp3a4_substrate_carbonmangels': [0.522, 0.019]
- 'cyp2c9_substrate_carbonmangels': [0.344, 0.009]
- 'half_life_obach': [-0.004, 0.062]
- 'clearance_microsome_az': [0.015, 0.037]
- 'clearance_hepatocyte_az': [-0.041, 0.107]


### CGN:
- 'caco2_wang': [0.493, 0.158]
- 'hia_hou': [0.923, 0.05]
- 'pgp_broccatelli': [0.897, 0.007]
- 'solubility_aqsoldb': error while making features
- 'bioavailability_ma': [0.534, 0.078]
- 'lipophilicity_astrazeneca': [0.694, 0.116]
- 'bbb_martins': [0.852, 0.018]
- 'ppbr_az': [13.159, 3.565]
- 'vdss_lombardo': [0.475, 0.06]
- 'cyp2d6_veith': [0.611, 0.04]
- 'cyp3a4_veith': [0.821, 0.008]
- 'cyp2c9_veith': [0.695, 0.013]
- 'cyp2d6_substrate_carbonmangels': [0.586, 0.055]
- 'cyp3a4_substrate_carbonmangels': [0.584, 0.034]
- 'cyp2c9_substrate_carbonmangels': [0.365, 0.028]
- 'half_life_obach': [0.288, 0.025]
- 'clearance_microsome_az': [0.542, 0.015]
- 'clearance_hepatocyte_az': [0.323, 0.018]


