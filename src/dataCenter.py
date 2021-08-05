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
			# a_file = open("dataset_preprocessing/prepared_data/data.pkl", "rb")
			# data = pickle.load(a_file)
			# a_file.close()
			# movie_feats = data["movies_feat"]
			# user_feats = []
			# movie_map = data["movie_map"]
			# movie_adj_list = data["movie_adj_list"]
			# user_adj_list = data["user_adj_list"]
			# edge_list = data["edge_list"]

			# movie_feats = [[0,0,0,3],
			# 			   [0,0,0,5],
			# 			   [0,0,2,0],
			# 			   [0,7,0,0],
			# 			   [0,5,4,0],
			# 			   [2,0,0,6],
			# 			   [1,0,0,0]]

			movie_feats = [[1,0,0.2,0],
						   [0.4,1,0,0.1],
						   [1,0.1,0.3,0],
						   [0,0.8,0,0.5],
						   [0.6,0,1,0],
						   [0,0,0,1]]

			movie_adj_list = [{(0,4.5), (1,4.5), (3,4), (5,3.5), (6,4.5), (7,3.5)},
							  {(0,3.5), (1,3), (2,5), (4,4.5), (5,4), (6,4.5), (7,2.5)},
							  {(0,5), (2,4), (3,4.5), (4,4.5), (6,5), (7,4)},
							  {(1,2), (2,4.5), (4,4), (6,4.5)},
							  {(0,4), (1,4.5), (2,3.5), (4,3.5), (5,4), (6,3.5), (7,4)},
							  {(2,4),(5,2)}]

			user_adj_list = [{(0,4.5), (1,3.5), (2,5), (4,4)},
							 {(0,4.5), (1,3), (3,2), (4,4.5)},
							 {(1,5), (2,4), (3,4.5), (4,3.5), (5,4)},
							 {(0,4), (2,4.5)},
							 {(1,4.5), (2,4.5), (3,4), (4,3.5)},
							 {(0,3.5), (1,4), (4,4), (5,2)},
							 {(0,4.5), (1,4.5), (2,5), (3,4.5), (4,3.5)},
							 {(0,3.5), (1,2.5), (2,4), (4,4)}]

			edge_list = [(0,0,4.5),
						 (0,1,3.5),
						 (0,2,5),
						 (0,4,4),
						 (1,0,4.5),
						 (1,1,3),
						 (1,3,2),
						 (1,4,4.5),
						 (2,1,5),
						 (2,2,4),
						 (2,3,4.5),
						 (2,4,3.5),
						 (2,5,4),
						 (3,0,4),
						 (3,2,4.5),
						 (4,1,4.5),
						 (4,2,4.5),
						 (4,3,4),
						 (4,4,3.5),
						 (5,0,3.5),
						 (5,1,4),
						 (5,4,4),
						 (5,5,2),
						 (6,0,4.5),
						 (6,1,4.5),
						 (6,2,5),
						 (6,3,4.5),
						 (6,4,3.5),
						 (7,0,3.5),
						 (7,1,2.5),
						 (7,2,4),
						 (7,4,4)]

			# movie_adj_list = [{(7,3)},
			# 				  {(0,4.5),(7,4)},
			# 				  {(0,3.5)},
			# 				  {(3,3),(6,2.5)},
			# 				  {(2,1)},
			# 				  {(3,3.5),(6,2)},
			# 				  {(0,4),(4,5),(5,5),(7,4.5)}]
			#
			# user_adj_list = [{(1,4.5),(2,3.5),(6,4)},
			# 				 {},
			# 				 {(4,1)},
			# 				 {(3,3),(5,3.5)},
			# 				 {(6,5)},
			# 				 {(6,5)},
			# 				 {(3,2.5),(5,2),(6,5)},
			# 				 {(0,3),(1,4)}]
			#
			# edge_list = [(0,1,4.5),
			# 			 (0,2,3.5),
			# 			 (0,6,4),
			# 			 (3,3,3),
			# 			 (3,5,3.5),
			# 			 (4,6,5),
			# 			 (5,6,5),
			# 			 (6,3,2.5),
			# 			 (6,5,2),
			# 			 (6,6,4.5),
			# 			 (7,0,3),
			# 			 (7,1,4)]


			# print(f"{len(movie_feats)} x {len(movie_feats[0])} - {sys.getsizeof(movie_feats[1000])}")
			# print(f"{len(movie_map)} - {sys.getsizeof(movie_map)}")
			# print(f"{len(movie_adj_list)} x {len(movie_adj_list[500])}- {sys.getsizeof(movie_adj_list[100])}")
			# print(f"{len(user_adj_list)} - {sys.getsizeof(user_adj_list)}")
			# print(f"{len(edge_list)} - {sys.getsizeof(edge_list)}")

			setattr(self, dataSet + '_movie_feats', movie_feats)
			#setattr(self, dataSet + '_user_feats', user_feats)
			#setattr(self, dataSet + '_movie_map', movie_map)

			setattr(self, dataSet + '_movie_adj_list', movie_adj_list)
			setattr(self, dataSet + '_user_adj_list', user_adj_list)
			setattr(self, dataSet + '_edge_list', edge_list)

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


