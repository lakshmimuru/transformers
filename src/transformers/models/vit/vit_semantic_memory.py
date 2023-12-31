
#Author: Bhishma Dedhia

import numpy as np 
import torch as torch
import math
import torch.nn as nn


class ViTMemory():

    def __init__(self, config):

        print('Memory initialized')
        self.k = config.k
        self.hidden_dims = config.hidden_size
        self.means = torch.randn(self.k,self.hidden_dims)
        self.networks = [SemanticNetwork(config) for i in range(self.k)]
        self.ema_decay = config.ema_decay
        self.l = config.l
        self.top_m = config.top_m

    def add_to_memory(self, query, key,  value, centres, labels):

        query = query.reshape(-1,self.hidden_dims)
        key = key.reshape(-1,self.hidden_dims)
        value = value.reshape(-1,self.hidden_dims)
        centre = centres.reshape(-1,self.l)

        for i in range(self.k):
            inds_i = torch.sum((centre == i),dim=1).nonzero().reshape(-1)
            if inds_i.shape[0] > 0:
                self.networks[i].push_to_memory(query[inds_i],key[inds_i],value[inds_i],labels[inds_i])
                '''
                print(f'center {i} pushed successfully: {inds_i.shape[0]} tokens')
                print(f'center {i} memory status: {self.networks[i].memory_full}')
                '''
                self.networks[i].compute_network()
                self.update_means(i)

    def return_center(self, query):

        sim = torch.matmul(query,self.means.transpose(0,1))
        _,top_l = torch.topk(sim,self.l)

        return top_l

    def retrieve(self, query):

        sim = torch.matmul(query,self.means.transpose(0,1))
        _,top_l = torch.topk(sim,self.l)
        top_1 = top_l[:,0]
        key = torch.zeros(query.shape[0],self.top_m,self.hidden_dims)
        value = torch.zeros(query.shape[0],self.top_m,self.hidden_dims)

        for i in range(self.k):

            key_i, value_i = self.networks[i].return_key_value()
            key[(top_1==i).nonzero()] = key_i
            value[(top_1 ==i).nonzero()] = value_i            

        return key, value, top_l



    def update_means(self,i):

        self.means[i] = self.ema_decay*self.means[i]\
                        + (1-self.ema_decay)*self.networks[i].return_mean()

    def check_minimum_entries(self):

        has_min = True

        for i in range(self.k):
            has_min = has_min and (self.networks[i].ptr>self.top_m-1 or self.networks[i].memory_full)
            if has_min == False:
                break
            
        return has_min

    def save_memory(self,fname):

        checkpoint = dict()
        for i in range(self.k):
            checkpoint[str(i)] = self.networks[i].return_network_dict()
            checkpoint['mean'] = self.means[i]
        print(fname)
        torch.save(checkpoint,fname+'memory.pt') 


    def load_memory(self,fname):

        checkpoint = torch.load(fname+'memory.pt')
        for i in range(self.k):
            self.networks[i].load_network(checkpoint[str(i)])
            self.networks[i].compute_network()
            self.means[i] = checkpoint['mean']



class SemanticNetwork():

    def __init__(self, config):

        self.size = config.size
        self.top_m = config.top_m
        self.hidden_dims = config.hidden_size
        self.ptr = 0
        self.query = torch.zeros(self.size,self.hidden_dims)
        self.key = torch.zeros(self.size, self.hidden_dims)
        self.value = torch.zeros(self.size,self.hidden_dims)
        self.label = torch.zeros(self.size)
        self.memory_full = False
        self.token_rank = torch.zeros(self.size)

    def push_to_memory(self, queries, keys, values, labels):

        n = queries.shape[0]
        self.memory_full = self.memory_full or n+self.ptr>self.size - 1
        if n+self.ptr>self.size-1:
            self.query[self.ptr:] = queries[:(self.size-self.ptr)]
            self.query[:n-(self.size-self.ptr)] = queries[self.size-self.ptr:] 
            self.key[self.ptr:] = queries[:(self.size-self.ptr)]
            self.key[:n-(self.size-self.ptr)] = keys[self.size-self.ptr:] 
            self.value[self.ptr:] = values[:(self.size-self.ptr)]
            self.value[:n-(self.size-self.ptr)] = values[self.size-self.ptr:] 
            self.label[self.ptr:] = labels[:(self.size-self.ptr)]
            self.label[:n-(self.size-self.ptr)] = labels[self.size-self.ptr:]
        else:
            self.query[self.ptr:self.ptr+n] = queries
            self.key[self.ptr:self.ptr+n] = keys
            self.value[self.ptr:self.ptr+n] = values
            self.label[self.ptr:self.ptr+n] = labels
        self.ptr = (self.ptr+n) %self.size


    def compute_network(self):

        if self.memory_full == False:
            attention_scores = torch.matmul(self.query[:self.ptr],self.key[:self.ptr].transpose(0,1))/math.sqrt(self.hidden_dims)
        else:
            attention_scores = torch.matmul(self.query,self.key.transpose(0,1))/math.sqrt(self.hidden_dims)
        self.M = nn.Softmax(dim=-1)(attention_scores).transpose(0,1)
        v,e = torch.linalg.eig(self.M)
        e = torch.real(e)
        v = torch.real(v)
        largest = torch.argsort(v)[-1].item()
        self.token_rank = e[:,largest]/torch.sum(e[:,largest])
        self.inds_rank = torch.argsort(self.token_rank)

    def return_mean(self):
        
        if self.memory_full == False:
            return torch.mean(self.key[:self.ptr],dim=0)
        else:
            return torch.mean(self.key)

    def return_key_value(self):

        inds = self.inds_rank[-self.top_m:]
        return self.key[inds], self.value[inds]

    def return_network_dict(self):

        checkpoint = {'ptr':self.ptr,
                      'query':self.query,
                      'key':self.key,
                      'value':self.value,
                      'label':self.label,
                      'memory_full':1 if self.memory_full else 0
                     }

        return checkpoint


    def load_network(self,checkpoint):

        self.ptr = checkpoint['ptr']
        self.query = checkpoint['query']
        self.key = checkpoint['key']
        self.value = checkpoint['value']
        self.label = checkpoint['label']
        self.memory_full = True if checkpoint['memory_full'] == 1 else False








        





