# DeepMirror
Project for benchmarking ML models on TDC datasets

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

