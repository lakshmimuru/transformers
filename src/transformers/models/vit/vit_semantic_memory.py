import numpy as np 
import torch as torch


class ViTMemory():

	def __init__(self, config):

		self.k = config.k
		self.hidden_dims = config.hidden_size
		self.means = torch.randn(self.k,self.hidden_dims)
		self.networks = [SemanticNetwork(config) for i in range(self.k)]
		self.ema_decay = ema_decay

	def add_to_memory(self, key, query, value, centre):
		#Call push_to_memory followed by compute network


	def retrieve(self, query):
		#Do knn and return {key,value} pairs

	def update_means(self,i):

		self.means[i] = self.ema_decay*self.means[i]\
						+ (1-self.ema_decay)*self.networks[i].return_mean()


		

class SemanticNetwork():

	def __init__(self, config, mean):

		self.size = config.memory_size
		self.hidden_dims = config.hidden_size
		self.ptr = 0
		self.query = torch.zeros(self.size,self.hidden_size)
		self.key = torch.zeros(self.size, self.hidden_size)
		self.value = torch.zeros(self.size,self.hidden_size)
		self.memory_full = False
		self.token_rank = torch.zeros(self.size)

	def push_to_memory(self, queries, keys, values):

		n = queries.shape[0]
		self.memory_full = self.memory_full or n+self.ptr>self.size
		if n+self.ptr>self.size:
			#memory overflow so wraparound
			self.query[self.ptr:] = queries[:(self.size-self.ptr)]
			self.query[:n-(self.size-self.ptr)] = queries[self.size-self.ptr:] 
			self.key[self.ptr:] = queries[:(self.size-self.ptr)]
			self.key[:n-(self.size-self.ptr)] = keys[self.size-self.ptr:] 
			self.value[self.ptr:] = values[:(self.size-self.ptr)]
			self.value[:n-(self.size-self.ptr)] = values[self.size-self.ptr:] 
		else:
			self.query[self.ptr:self.ptr+n] = queries
			self.keys[self.ptr:self.ptr+n] = keys
			self.values[self.ptr:self.ptr+n] = values

		self.ptr = (self.ptr+n) %self.size


	def compute_network(self):

		if self.memory_full == False:
			attention_scores = torch.matmul(query[:self.ptr],key[:self.ptr].transpose(0,1))		
		else:
			attention_scores = torch.matmul(query,key)
	
		self.M = nn.Softmax(dim=-1)(attention_scores).transpose(0,1)
		v,e = torch.linalg.eig(self.M)
		largest = torch.argsort(torch.real(v))[-1].item()
		self.token_rank = e[:,ind]/torch.sum(e[:,largest])

	def return_mean(self):

		return torch.mean(self.key,dim=0)

	def return_key_value(self,top_m =5):

		inds = torch.argsort(self.token_rank)[-top_m:]
		return self.key[inds], self.value[inds]







		





