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

from deepchem.models import GCNModel

from tqdm import trange
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

path=r'./featurized/'


 
class Model(ABC):
    '''Abstract class to show the methos every model class should have'''
 
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def featurize(self, benchmark, name):
        pass
        
        # train_val, test = benchmark['train_val'], benchmark['test']
        # train_val_y = np.array(train_val.Y)
        # test_y = np.array(test.Y)
        # features_json = {}
        
        # filename = path + name + '.json'
        # if os.path.exists(filename):
        #         with open(filename) as json_file:
        #             features_json = json.load(json_file)
                    
        #             if self.feat_name + '_train' in features_json.keys():
        #                 f_train_val = features_json[self.feat_name + '_train']
        #                 f_test = features_json[self.feat_name + '_test']
                        
        #                 return f_train_val, train_val_y, f_test, test_y
            
        
        # f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
        # f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
        # features_json[self.feat_name + '_train'] = f_train_val
        # features_json[self.feat_name + '_test']  = f_test
        
        # json_object = json.dumps(features_json, indent=4)
        
        # with open(path + name + '.json', 'w') as fp:
        #     json.dump(json_object, fp)
        
        
        
        # return f_train_val, train_val_y, f_test, test_y
        
    
class Tree(Model):
    '''Class for the Tree model, it can store the best parameters'''
    
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
        
        
    def featurize(self, benchmark, name):
        
        train_val, test = benchmark['train_val'], benchmark['test']
        train_val_y = np.array(train_val.Y)
        test_y = np.array(test.Y)
        
        f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
        f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
        return f_train_val, train_val_y, f_test, test_y
    
       
class GBR(Model):
    '''Class for the GBR model, it can store the best parameters'''
    
    model = None
    featurizer = dc.feat.Mol2VecFingerprint()
    feat_name = 'mol2vec'
    best_params = {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
    
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
        
        
    def featurize(self, benchmark, name):
        
        train_val, test = benchmark['train_val'], benchmark['test']
        train_val_y = np.array(train_val.Y)
        test_y = np.array(test.Y)
        
        f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
        f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
        return f_train_val, train_val_y, f_test, test_y
        
    

class GCN(Model):
    '''Class for the GCN model'''
    
    model = None
    featurizer = dc.feat.MolGraphConvFeaturizer()
    feat_name = 'molGCN'
    
    def train(self, train_set, y):
        gcn = GCNModel(n_tasks=1, mode='regression',batch_size=32, learning_rate=0.001)
        train_ds = dc.data.NumpyDataset(X=train_set, y=np.array(y))
        gcn.fit(train_ds, 10)
        
        self.model = gcn
    
    def predict(self, test_set):
        
        test_ds = dc.data.NumpyDataset(X=test_set)
        return self.model.predict(test_ds)

    def featurize(self, benchmark, name):
        
        train_val, test = benchmark['train_val'], benchmark['test']
        train_val_y = np.array(train_val.Y)
        test_y = np.array(test.Y)
        
        f_train_val = self.featurizer.featurize(train_val.iloc[:, 1].to_list())
        f_test = self.featurizer.featurize(test.iloc[:, 1].to_list())
        
        return f_train_val, train_val_y, f_test, test_y
    

class Transformer(Model):
    '''Class for the Transformer model, it stores the vocab for SMILES.
    This model is trained with PyTorch and the input is tokenized and transformed
    into a sequence vector to feed into BERT model. A pretrained version of the
    model is used to accelerate the training.'''
    
    model = None
    featurizer = BasicSmilesTokenizer()
    feat_name = 'transformer'
    vocab = None
    max_len = None
    f_train_val_att_m = None
    f_test_att_m = None
    
    def _create_dataloaders(self, train_val,labels, attention=0):
        val_ratio = 0.2
        # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
        batch_size = 16
        
        #validation methods implemented, but for the trial of the models with low epochs it is not useful
        
        # Indices of the train and validation splits stratified by labels
        # train_x, val_x, train_y, val_y, train_mask, val_mask = train_test_split(train_val, labels, self.f_train_val_att_m,
        #     test_size = val_ratio,
        #     shuffle = True)
        train_x = train_val
        train_y = labels
        train_mask = self.f_train_val_att_m


        train_set = TensorDataset(torch.from_numpy(train_x.astype(int)),
                                  torch.from_numpy(train_mask.astype(int)),
                                  torch.from_numpy(train_y.astype('float32')))
        # val_set = TensorDataset(torch.from_numpy(val_x.astype(int)),
        #                         torch.from_numpy(val_mask.astype(int)),
        #                         torch.from_numpy(val_y.astype('float32')))
        
        # Prepare DataLoader
        train_dataloader = DataLoader(
                    train_set,
                    sampler = RandomSampler(train_set),
                    batch_size = batch_size
                )
        
        # validation_dataloader = DataLoader(
        #             val_set,
        #             sampler = SequentialSampler(val_set),
        #             batch_size = batch_size
        #         )
        return train_dataloader, None#validation_dataloader
    
    def train(self, train_val_set, y):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        
        train_dataloader, validation_dataloader = self._create_dataloaders(train_val_set, y)
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )
        
        epochs = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        for _ in trange(epochs, desc = 'Epoch'):
            
            # ========== Training ==========
            
            # Set model to training mode
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
        
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids, 
                                     token_type_ids = None, 
                                     attention_mask = b_input_mask, 
                                     labels = b_labels)
                # Backward pass
                train_output.loss.backward()
                optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
        
            # Validation
        
            # Set model to evaluation mode
        #     self.model.eval()
        
        #     for batch in validation_dataloader:
        #         batch = tuple(t.to(device) for t in batch)
        #         b_input_ids, b_input_mask, b_labels = batch
        #         with torch.no_grad():
        #           # Forward pass
        #           eval_output = self.model(b_input_ids, 
        #                               token_type_ids = None, 
        #                               attention_mask = b_input_mask)
        #         logits = eval_output.logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
    
    def predict(self, test_set):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        test_set = TensorDataset(torch.from_numpy(test_set.astype(int)),
                                 torch.from_numpy(self.f_test_att_m.astype(int)))
        
        # Prepare DataLoader
        test_dataloader = DataLoader(
                    test_set,
                    sampler = SequentialSampler(test_set),
                    batch_size = 1
                )
        output=np.array([])

        for element in test_dataloader:
            b_input_ids, b_input_mask =tuple( element)

            with torch.no_grad():
              # Forward pass
              test_output = self.model(b_input_ids.to(device), 
                                  token_type_ids = None, 
                                  attention_mask = b_input_mask.to(device))
              output = np.append(output,test_output.logits.detach().cpu().numpy())
              
        return output
    
    def featurize(self, benchmark, name):
        
        train_val, test = benchmark['train_val'], benchmark['test']
        train_val_y = np.array(train_val.Y)
        test_y = np.array(test.Y)

        vocab = np.array([])
        max_len = 0
        for i in range(len(train_val)):
          tokenized = self.featurizer.tokenize(train_val.iloc[:, 1].to_list()[i])
          vocab = np.append(vocab, np.array(tokenized))
          if len(tokenized) > max_len:
            max_len = len(tokenized)

        self.vocab = np.unique(vocab)
        self.max_len = max_len + 10
        
        self.f_train_val_att_m = np.zeros((len(train_val),self.max_len))
        self.f_test_att_m = np.zeros((len(test),self.max_len))
        
        f_train_val = np.zeros((len(train_val),self.max_len))
        for i in range(len(train_val)):
            tokenized = self.featurizer.tokenize(train_val.iloc[:, 1].to_list()[i])
            for j in range(len(tokenized)):
                position = np.where(self.vocab == tokenized[j])
                f_train_val[i,j] = position[0]
                self.f_train_val_att_m[i,j] = 1
        
        f_test = np.zeros((len(test),self.max_len))
        for i in range(len(test)):
            tokenized = self.featurizer.tokenize(test.iloc[:, 1].to_list()[i])
            for j in range(len(tokenized)):
                if j > self.max_len:
                  break
                position = np.where(self.vocab == tokenized[j])
                if len(position[0]) ==0:
                  f_test[i,j] = 0
                  self.f_test_att_m[i,j] = 0
                else:
                  f_test[i,j] = position[0]
                  self.f_test_att_m[i,j] = 1
                
        
        return f_train_val, train_val_y, f_test, test_y