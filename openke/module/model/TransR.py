import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransR(Model):

	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, rand_init = False, margin = None):
		super(TransR, self).__init__(ent_tot, rel_tot)
		
		# 初始化 TransR 模型 参数
		self.dim_e = dim_e # 实体嵌入的维度
		self.dim_r = dim_r # 关系嵌入的维度
		self.norm_flag = norm_flag # 是否进行归一化
		self.p_norm = p_norm # 范数的幂次
		self.rand_init = rand_init # 是否使用初始随机化

		# 定义实体嵌入、关系嵌入和关系投影矩阵
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

		self.transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)
		if not self.rand_init: # 如果不使用随机初始化，则将关系投影矩阵初始化为单位矩阵
			identity = torch.zeros(self.dim_e, self.dim_r)
			for i in range(min(self.dim_e, self.dim_r)):
				identity[i][i] = 1
			identity = identity.view(self.dim_r * self.dim_e)
			for i in range(self.rel_tot):
				self.transfer_matrix.weight.data[i] = identity
		else:
			nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

		if margin != None: # 如果提供了边际参数，则启用边际损失
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _calc(self, h, t, r, mode): # 计算TransR模型的得分
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score
	
	def _transfer(self, e, r_transfer):
		# 对实体进行关系投影
		r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
			e = torch.matmul(e, r_transfer).permute(1, 0, 2)
		else:
			e = e.view(-1, 1, self.dim_e)
			e = torch.matmul(e, r_transfer)
		return e.view(-1, self.dim_r)

	def forward(self, data):
		# 模型的前向传播
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_transfer = self.transfer_matrix(batch_r)
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		# 正则化项
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_transfer = self.transfer_matrix(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_transfer ** 2)) / 4
		return regul * regul

	def predict(self, data):
		# 预测函数
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()
		

