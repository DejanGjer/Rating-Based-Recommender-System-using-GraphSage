import math
import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import time

class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			#just for params that are matrices
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists

class Projection(nn.Module):

	def __init__(self, emb_size, projection_size):
		super(Projection, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer_user = nn.Sequential(
			nn.Linear(emb_size, projection_size)
		)
		self.layer_movie = nn.Sequential(
			nn.Linear(emb_size, projection_size)
		)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			#just for params that are matrices
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, user_batch_embeds, movie_batch_embeds):
		users_projection = self.layer_user(user_batch_embeds)
		movies_projection = self.layer_movie(movie_batch_embeds)
		# print("Projection")
		# print("Movies projection")
		# print(type(movies_projection))
		# print(movies_projection)
		# print("Users projection")
		# print(type(users_projection))
		# print(users_projection)

		result = torch.sum((users_projection * movies_projection), 1)

		return result


# class Classification(nn.Module):

# 	def __init__(self, emb_size, num_classes):
# 		super(Classification, self).__init__()

# 		self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
# 		self.init_params()

# 	def init_params(self):
# 		for param in self.parameters():
# 			nn.init.xavier_uniform_(param)

# 	def forward(self, embeds):
# 		logists = torch.log_softmax(torch.mm(embeds,self.weight), 1)
# 		return logists

class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.MARGIN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print(neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print(pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0))
		
		return loss

	def get_loss_margin(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			# nodes_score.append((-pos_score - neg_score).view(1,-1))

		loss = torch.mean(torch.cat(nodes_score, 0),0)

		# loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
		
		return loss


	def extend_nodes(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes, num_neg)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		assert set(self.target_nodes) < set(self.unique_nodes_batch)
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if next_node != node and next_node in self.train_nodes:
						self.positive_pairs.append((node,next_node))
						cur_pairs.append((node,next_node))
					curr_node = next_node

			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs
		

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, gcn=False): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size


		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		combined = F.relu(self.weight.mm(combined.t())).t()
		return combined

class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
		super(GraphSage, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func

		self.raw_features = raw_features
		self.adj_lists = adj_lists

		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

	def forward(self, nodes_batch):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""

		lower_layer_nodes = list(nodes_batch)
		nodes_batch_layers = [(lower_layer_nodes,)]
		# self.dc.logger.info('get_unique_neighs.')
		for i in range(self.num_layers):
			lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
		#Testing
		# print("First")
		# print(f"Nodes - {len(nodes_batch_layers[2][0])}")
		# print(nodes_batch_layers[2][0])
		# print("Dict")
		# print(nodes_batch_layers[2][2])
		# print("Neighs")
		# print(nodes_batch_layers[2][1])
		# exit()

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features
		for index in range(1, self.num_layers+1):
			nb = nodes_batch_layers[index][0]
			pre_neighs = nodes_batch_layers[index-1]
			# self.dc.logger.info('aggregate_feats.')
			aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
			sage_layer = getattr(self, 'sage_layer'+str(index))
			if index > 1:
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			# self.dc.logger.info('sage_layer.')
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			pre_hidden_embs = cur_hidden_embs

		return pre_hidden_embs

	def _nodes_map(self, nodes, hidden_embs, neighs):
		layer_nodes, samp_neighs, layer_nodes_dict = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes_dict[x] for x in nodes]
		return index

	def _get_unique_neighs_list(self, nodes, num_sample=10):
		_set = set
		to_neighs = [self.adj_lists[int(node)] for node in nodes]
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		_unique_nodes_list = list(set.union(*samp_neighs))
		i = list(range(len(_unique_nodes_list)))
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		return samp_neighs, unique_nodes, _unique_nodes_list

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
		unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

		assert len(nodes) == len(samp_neighs)
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		assert (False not in indicator)
		if not self.gcn:
			samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
		# self.dc.logger.info('2')
		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
		# self.dc.logger.info('3')
		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		# self.dc.logger.info('4')

		if self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True)
			mask = mask.div(num_neigh).to(embed_matrix.device)
			aggregate_feats = mask.mm(embed_matrix)

		elif self.agg_func == 'MAX':
			# print(mask)
			indexs = [x.nonzero() for x in mask==1]
			aggregate_feats = []
			# self.dc.logger.info('5')
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)

		# self.dc.logger.info('6')
		
		return aggregate_feats


class GraphSage2(nn.Module):
	"""docstring for GraphSage"""

	def __init__(self, num_layers, input_size, hidden_size, out_size, raw_movie_features, movie_adj_lists, user_adj_lists, num_ratings, device, agg_func='SUM'):
		super(GraphSage2, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.num_ratings = num_ratings
		self.device = device
		self.agg_func = agg_func

		self.raw_movie_features = raw_movie_features
		self.movie_adj_lists = movie_adj_lists
		self.user_adj_lists = user_adj_lists

		for index in range(1, 2 * num_layers + 1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'rating_layer' + str(index), RatingLayer(layer_size, hidden_size, num_ratings, movie_adj_lists, user_adj_lists))
			setattr(self, 'sage_layer' + str(index), SageLayer2(hidden_size, out_size))


	def forward(self, edge_batch, to_gen):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		#start_time = time.time()

		lower_layer_movie_nodes = list()
		lower_layer_user_nodes = list()
		movie_edges = list()
		user_edges = list()
		nodes_batch_layers = None
		layers = 0

		if to_gen == "movie":
			to_gen = 0
			lower_layer_movie_nodes = [edge[1] for edge in edge_batch]
			movie_edges = [(edge[0], edge[2]) for edge in edge_batch]
			nodes_batch_layers = [(lower_layer_movie_nodes,)]
			layers = 2 * self.num_layers
		elif to_gen == "user":
			to_gen = 1
			lower_layer_user_nodes = [edge[0] for edge in edge_batch]
			user_edges = [(edge[1], edge[2]) for edge in edge_batch]
			nodes_batch_layers = [(lower_layer_user_nodes,)]
			layers = 2 * self.num_layers - 1

		# self.dc.logger.info('get_unique_neighs.')
		for i in range(layers):
			if (i + to_gen) % 2 == 0:
				# print("Adding users for movies")
				# print(lower_layer_movie_nodes)
				lower_samp_movie_neighs, lower_samp_movie_neighs_ratings, lower_layer_user_dict, lower_layer_user_nodes \
					= self._get_unique_neighs_list(lower_layer_movie_nodes, "movie",
												   opposite_edges=movie_edges if i == 0 else None)
				nodes_batch_layers.insert(0, (lower_layer_user_nodes, lower_layer_user_dict, lower_samp_movie_neighs,
											  lower_samp_movie_neighs_ratings))
			else:
				# print("Adding movies for users")
				# print(lower_layer_user_nodes)
				lower_samp_user_neighs, lower_samp_user_neighs_ratings, lower_layer_movie_dict, lower_layer_movie_nodes \
					= self._get_unique_neighs_list(lower_layer_user_nodes, "user",
												   opposite_edges=user_edges if i == 0 else None)
				nodes_batch_layers.insert(0, (lower_layer_movie_nodes, lower_layer_movie_dict, lower_samp_user_neighs,
											  lower_samp_user_neighs_ratings))
		# print(nodes_batch_layers[0])
		# print(nodes_batch_layers[1])
		# print(nodes_batch_layers[2])
		# print(nodes_batch_layers[3])
		# if to_gen == "movie":
		# 	print(nodes_batch_layers[4])

		#print(f"Expanding nodes {time.time() - start_time}")

		pre_hidden_embs = self.raw_movie_features
		#print(pre_hidden_embs)
		movie_embs = []
		user_embs = []
		tip = ""
		for index in range(1, layers + 1):
			nb = nodes_batch_layers[index][0]
			node_type = "movie" if index % 2 == 0 else "user"
			#print(f"Current nodes ({node_type}) \n {nb}")
			pre_neighs = nodes_batch_layers[index - 1]
			# print(f"Layer before: ")
			# print("Nodes")
			# print(pre_neighs[0])
			# print("Dict")
			# print(pre_neighs[1])
			# print("Neighborhood")
			# print(pre_neighs[2])
			# print("Rating neighborhood")
			# print(pre_neighs[3])
			# self.dc.logger.info('aggregate_feats.')
			rating_layer = getattr(self, 'rating_layer' + str(index))
			hidden_embs = rating_layer(nb, pre_neighs, pre_hidden_embs, node_type)
			# print(f"{tip} hidden feats")
			# print(hidden_embs)
			aggregate_feats = self.aggregate(nb, hidden_embs, pre_neighs)
			# print(f"{tip} agg feats")
			# print(aggregate_feats)
			sage_layer = getattr(self, 'sage_layer' + str(index))
			# self.dc.logger.info('sage_layer.')
			cur_hidden_embs = sage_layer(aggregate_feats=aggregate_feats)
			# print(f"{tip} gen embeds")
			# print(cur_hidden_embs)
			# if index == self.num_layers:
			# 	if index % 2 == 0:
			# 		movie_embs = cur_hidden_embs
			# 		user_embs = pre_hidden_embs[self._nodes_map(user_batch, pre_neighs)]
			# 	else:
			# 		user_embs = cur_hidden_embs
			# 		movie_embs = pre_hidden_embs[self._nodes_map(movie_batch, pre_neighs)]
			pre_hidden_embs = cur_hidden_embs
			#print("-------------------------------------")
			#print(f"Layer {index}: {time.time() - start_time}")

		return pre_hidden_embs

	def _nodes_map(self, nodes, neighs):
		layer_nodes, layer_nodes_dict, samp_neighs, samp_neighs_ratings = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes_dict[x] for x in nodes]
		return index

	def _get_unique_neighs_list(self, nodes, node_type, opposite_edges=None, num_sample=10):
		_set = set
		to_neighs = []
		if node_type == "movie":
			to_neighs = [self.movie_adj_lists[int(node)] for node in nodes]
		elif node_type == "user":
			to_neighs = [self.user_adj_lists[int(node)] for node in nodes]

		if not num_sample is None:
			_sample = random.sample
			samp_neighs_ratings = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
						   in to_neighs]
		else:
			samp_neighs_ratings = to_neighs

		#Sample and remove for label leakage issue
		if opposite_edges != None:
			samp_neighs_ratings = [samp_neigh_rating - set([opposite_edges[i]]) for i, samp_neigh_rating in
									   enumerate(samp_neighs_ratings)]

		samp_neighs = []
		for samp_neigh_rating in samp_neighs_ratings:
			samp_neigh = set()
			for pair in samp_neigh_rating:
				samp_neigh.add(pair[0])
			samp_neighs.append(samp_neigh)

		#DODAVANJE SAMOG SEBE U SUSEDE
		#samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		_unique_nodes_list = list(set.union(*samp_neighs))
		i = list(range(len(_unique_nodes_list)))
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		return samp_neighs, samp_neighs_ratings, unique_nodes, _unique_nodes_list

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
		unique_nodes_list, unique_nodes, samp_neighs, samp_neighs_ratings = pre_neighs

		assert len(nodes) == len(samp_neighs)
		#indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		#assert (False not in indicator)

		#IZBRISATI CVOR IZ SVOG NEIGHBORHOODA
		#if not self.gcn:
		#	samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]
		# self.dc.logger.info('2')
		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
		# self.dc.logger.info('3')
		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		# self.dc.logger.info('4')

		if self.agg_func == 'SUM':
			num_neigh = mask.sum(1, keepdim=True)
			mask = mask.to(embed_matrix.device)
			aggregate_feats = mask.mm(embed_matrix)

		elif self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True)
			mask = mask.div(num_neigh).to(embed_matrix.device)
			aggregate_feats = mask.mm(embed_matrix)

		elif self.agg_func == 'MAX':
			# print(mask)
			indexs = [x.nonzero() for x in mask == 1]
			aggregate_feats = []
			# self.dc.logger.info('5')
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)

		# self.dc.logger.info('6')

		return aggregate_feats

class SageLayer2(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size):
		super(SageLayer2, self).__init__()

		self.input_size = input_size
		self.out_size = out_size

		self.weight = nn.Parameter(torch.FloatTensor(out_size, input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		combined = F.leaky_relu(aggregate_feats, negative_slope=0.1)
		combined = F.leaky_relu(self.weight.mm(combined.t()), negative_slope=0.1).t()
		return combined

class RatingLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, hidden_size, num_ratings, movie_adj_lists, user_adj_lists):
		super(RatingLayer, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_ratings = num_ratings
		self.movie_adj_lists = movie_adj_lists
		self.user_adj_lists = user_adj_lists

		self.weight = nn.ParameterList()
		rating = 0
		for i in range(num_ratings):
			rating += 0.5
			self.weight.append(nn.Parameter(torch.FloatTensor(hidden_size, input_size)))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def num_of_rating_neigh(self, node, rating, node_type):
		result = 0
		if node_type == "movie":
			for neigh in self.movie_adj_lists[node]:
				if neigh[1] == rating:
					result += 1
		elif node_type == "user":
			for neigh in self.user_adj_lists[node]:
				if neigh[1] == rating:
					result += 1
		return result

	def forward(self, nodes, nodes_before, pre_hidden_embs, node_type):
		unique_nodes_list, unique_nodes, samp_neighs, samp_neighs_ratings = nodes_before

		assert len(nodes) == len(samp_neighs_ratings)

		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]

		result_embed_matrix = [None] * len(unique_nodes)
		for i,samp_neigh in enumerate(samp_neighs_ratings):
			for neigh_node in samp_neigh:
				index = unique_nodes[neigh_node[0]]
				rating = neigh_node[1]
				param_index = int(rating / 0.5) - 1
				c_ij = math.sqrt(
					self.num_of_rating_neigh(nodes[i], rating, node_type) * self.num_of_rating_neigh(neigh_node[0],
																									 rating,
																									 "movie" if node_type == "user" else "user"))
				hidden_emb = torch.mul(torch.matmul(self.weight[param_index], embed_matrix[index].t()), 1/c_ij)
				result_embed_matrix[index] = hidden_emb

		result_embed_matrix = torch.stack(result_embed_matrix)

		return result_embed_matrix
