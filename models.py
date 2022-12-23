#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:33:57 2022

@author: josemiguelacitores
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import deepchem as dc
import numpy as np
from abc import ABC, abstractmethod
import os
import json

path=r'/featurized/'


 
class Model(ABC):
 
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    def featurize(self, benchmark, name):
        
        train_val, test = benchmark['train_val'], benchmark['test']
        train_val_y = np.array(train_val.Y)
        test_y = np.array(test.Y)
        features_json = {}
        
        filename = path + name + '.json'
        if os.path.exists(filename):
                with open(filename) as json_file:
                    features_json = json.load(json_file)
                    
                    if self.feat_name + '_train' in features_json.keys():
                        f_train_val = features_json[self.feat_name + '_train']
                        f_test = features_json[self.feat_name + '_test']
                        
                        return f_train_val, train_val_y, f_test, test_y
            
        
        f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
        f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
        features_json[self.feat_name + '_train'] = f_train_val
        features_json[self.feat_name + '_test']  = f_test
        
        with open(path + name + '.json', 'w') as fp:
            json.dump(features_json, fp)
        
        
        
        return f_train_val, train_val_y, f_test, test_y
        
    
class Tree(Model):
    
    model = None
    featurizer = dc.feat.Mol2VecFingerprint()
    feat_name = 'mol2vec'
    
    def train(self, train_set, y):
        dt = DecisionTreeRegressor()

        # define hyperparameter grid
        param_grid = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf':[1,2,3]
        }
        
        # define grid search
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', return_train_score=True)
        
        # fit grid search to training data
        grid_search.fit(train_set, y)
        best_params = grid_search.best_params_
        
        self.model = grid_search
    
    def predict(self, test_set):
        res = self.model.predict(test_set)
        return res
        
        
    # def featurize(self, train_val, test):
        
    #     f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
    #     f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
    #     return f_train_val, f_test
       
class GBR(Model):
    
    model = None
    featurizer = dc.feat.Mol2VecFingerprint()
    
    def train(self, train_set, y):
        dt = GradientBoostingRegressor()

        # define hyperparameter grid
        param_grid = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf':[1,2,3]
        }
        
        # define grid search
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1', return_train_score=True)
        
        # fit grid search to training data
        grid_search.fit(train_set, y)
        best_params = grid_search.best_params_
        
        self.model = grid_search
    
    def predict(self, test_set):
        res = self.model.predict(test_set)
        return res
        
        
    # def featurize(self, train_val, test):
        
    #     f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
    #     f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
    #     return f_train_val, f_test
        
    

class GCN(Model):
    '''    
        import deepchem as dc
    >> from deepchem.models import GCNModel
    >> featurizer = dc.feat.MolGraphConvFeaturizer()
    >> tasks, datasets, transformers = dc.molnet.load_tox21(
    ..     reload=False, featurizer=featurizer, transformers=[])
    >> train, valid, test = datasets
    >> model = GCNModel(mode='classification', n_tasks=len(tasks),
    ..                  batch_size=32, learning_rate=0.001)
    >> model.fit(train, nb_epoch=50)'''
    
    model = None
    featurizer = dc.feat.MolGraphConvFeaturizer()
    
    def train(self, train_set, val_set):
        print(train_set)
    
    def predict(self, test_set):
        pass
    

class Transformer(Model):
    drug_encoding = 'Transformer'
    
    def train(self, train_set, val_set):
        print(train_set)
    
    def predict(self, test_set):
        pass