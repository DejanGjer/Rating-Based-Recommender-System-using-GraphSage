import gc
import sys
import os
import pickle
import sys

from collections import defaultdict
import numpy as np
import pandas as pd

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config
		
	def load_dataSet(self, dataSet='cora'):
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(cora_cite_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)

		elif dataSet == 'movielens':
			a_file = open("dataset_preprocessing/prepared_data/data-small-better.pkl", "rb")
			data = pickle.load(a_file)
			a_file.close()
			movie_feats = data["movies_feat"]
			movie_adj_list = data["movie_adj_list_train"]
			user_adj_list = data["user_adj_list_train"]
			edge_list_train = data["edge_list_train"]
			edge_list_valid = data["edge_list_valid"]
			edge_list_test = data["edge_list_test"]
			rating_distrib_movie = data["rating_distrib_movie"]
			rating_distrib_user = data["rating_distrib_user"]

			del data
			gc.collect()

			setattr(self, dataSet + '_movie_feats', movie_feats)
			setattr(self, dataSet + '_movie_adj_list', movie_adj_list)
			setattr(self, dataSet + '_user_adj_list', user_adj_list)

			setattr(self, dataSet + '_edge_list_train', edge_list_train)
			setattr(self, dataSet + '_edge_list_valid', edge_list_valid)
			setattr(self, dataSet + '_edge_list_test', edge_list_test)

			setattr(self, dataSet + '_rating_distrib_movie', rating_distrib_movie)
			setattr(self, dataSet + '_rating_distrib_user', rating_distrib_user)

			print("Ovo radi!")

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(pubmed_cite_file) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)


	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs


