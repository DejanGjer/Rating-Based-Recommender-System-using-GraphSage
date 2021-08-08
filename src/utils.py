import sys
import os
import datetime

import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

import time

from src.regularization import *

def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	embs = graphSage(val_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)

	vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
	print("Validation F1:", vali_f1)

	if vali_f1 > max_vali_f1:
		max_vali_f1 = vali_f1
		embs = graphSage(test_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_test = labels[test_nodes]
		assert len(labels_test) == len(predicts)
		comps = zip(labels_test, predicts.data)

		test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
		print("Test F1:", test_f1)

		for param in params:
			param.requires_grad = True

		torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

	for param in params:
		param.requires_grad = True

	return max_vali_f1

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	for epoch in range(epochs):
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = features[nodes_batch]

			logists = classification(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()

		max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
	return classification, max_vali_f1

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')

	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal':
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)

	train_nodes = shuffle(train_nodes)

	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.SGD(params, lr=0.7)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_nodes) / b_sz)

	visited_nodes = set()
	for index in range(batches):
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning

		#zakomentarisana linija ispod

		#nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
		visited_nodes |= set(nodes_batch)

		# get ground-truth for the nodes batch
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		embs_batch = graphSage(nodes_batch)

		if learn_method == 'sup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_sup + loss_net
		else:
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_net

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, classification

def get_optimizer(graphSage, projection):
	models = [graphSage, projection]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.Adam(params, lr=0.01)
	return optimizer

def apply_model2(dataCenter, ds, graphSage, projection, optimizer, b_sz, device, learn_method):
	train_edges = getattr(dataCenter, ds+'_edge_list_train')

	train_edges = shuffle(train_edges)

	models = [graphSage, projection]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	regularizer = RatingMatrixRegularizer(graphSage)

	# for model in models:
	# 	for name, param in model.named_parameters():
	# 		if param.requires_grad:
	# 			print(name)
	# 			print(param)

	#optimizer = torch.optim.SGD(params, lr=0.7)
	if optimizer == None:
		optimizer = torch.optim.Adam(params, lr=0.05)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_edges) / b_sz)

	visited_edges = set()
	train_losses = []
	for index in range(batches):
		# print("NEW BATCH")
		#print("========================================================================================================")
		edge_batch = train_edges[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning

		#zakomentarisana linija ispod

		#nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
		visited_edges |= set(edge_batch)

		# get ground-truth for the nodes batch
		ratings_batch = torch.tensor([edge[2] for edge in edge_batch]).to(device)

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		#print("MOVIE GENERATION")
		movie_embs = graphSage(edge_batch, "movie")
		#print("-------------------------------------------------------")
		#print("USER GENERATION")
		user_embs = graphSage(edge_batch, "user")

		#start_time = time.time()

		if learn_method == 'sup':
			# superivsed learning
			result_batch = projection(user_embs, movie_embs)
			# print("Results")
			# print(result_batch)
			# print("Ratings")
			# print(ratings_batch)
			loss_sup = torch.sum(torch.square(result_batch - ratings_batch))
			loss_sup /= len(edge_batch)
			reg_loss = regularizer.regularization()
			loss = loss_sup + reg_loss
			train_losses.append(loss.item())

		print('Step [{}/{}], Loss-Sup: {:.4f}, Loss-Reg: {:.4f}, Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1,
																											 batches,
																											 loss_sup.item(),
																											 reg_loss.item(),
																											 loss.item(),
																											 len(visited_edges),
																											 len(train_edges)))

		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, projection, optimizer, train_losses

def evaluate2(dataCenter, ds, graphSage, projection, optimizer, device, name, cur_epoch, save_dir):
	edge_list_valid = getattr(dataCenter, ds + "_edge_list_valid")

	models = [graphSage, projection]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	ratings = torch.tensor([edge[2] for edge in edge_list_valid]).to(device)

	movie_embs = graphSage(edge_list_valid, "movie")
	user_embs = graphSage(edge_list_valid, "user")
	results = projection(user_embs, movie_embs)

	val_loss = torch.sum(torch.square(results - ratings))
	val_loss /= len(edge_list_valid)

	val_rmse = torch.sqrt(val_loss)

	print(f"Validation after epoch {cur_epoch} - loss: {val_loss.item()} , rmse: {val_rmse.item()}")

	#torch.save(models, f"models/model_graphsage_{name}_ep{cur_epoch}.torch")

	torch.save({
		'epoch': cur_epoch,
		'graphsage_state_dict': models[0].state_dict(),
		'projection_state_dict': models[1].state_dict(),
		'optimizer_state_dict': optimizer.state_dict()
	}, f"{save_dir}/model_graphsage_train_ep{cur_epoch}.tar")

	for param in params:
		param.requires_grad = True

	return val_loss.item(), val_rmse.item()

def evaluate_on_steps(dataCenter, ds, graphSage, projection, optimizer, device, steps, call, num_edges, save_dir):
	edge_list_valid = getattr(dataCenter, ds + "_edge_list_valid")

	models = [graphSage, projection]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	boundary = math.ceil(len(edge_list_valid) / num_edges)
	call = call % boundary
	edge_list = edge_list_valid[call * num_edges: (call + 1) * num_edges]

	ratings = torch.tensor([edge[2] for edge in edge_list]).to(device)

	movie_embs = graphSage(edge_list, "movie")
	user_embs = graphSage(edge_list, "user")
	results = projection(user_embs, movie_embs)

	val_loss = torch.sum(torch.square(results - ratings))
	val_loss /= len(edge_list)

	val_rmse = torch.sqrt(val_loss)

	print(f"Validation after step {steps} - loss: {val_loss.item()} , rmse: {val_rmse.item()}")

	torch.save({
		'steps': steps,
		'graphsage_state_dict': models[0].state_dict(),
		'projection_state_dict': models[1].state_dict(),
		'optimizer_state_dict': optimizer.state_dict()
	}, f"{save_dir}/model_graphsage_train_step{steps}.tar")

	for param in params:
		param.requires_grad = True

	return val_loss.item(), val_rmse.item()

def apply_model_on_steps(dataCenter, ds, graphSage, projection, optimizer, b_sz, num_of_steps, val_on_step, device, learn_method, save_dir):
	train_edges = getattr(dataCenter, ds+'_edge_list_train')
	valid_edges = getattr(dataCenter, ds + '_edge_list_valid')
	train_edges = shuffle(train_edges)
	valid_edges = shuffle(valid_edges)
	setattr(dataCenter, ds + '_edge_list_valid', valid_edges)

	val_losses = []
	val_rmse_losses = []

	models = [graphSage, projection]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	if optimizer is None:
		optimizer = torch.optim.Adam(params, lr=0.05)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_edges) / b_sz)
	batches = min(batches, num_of_steps)

	visited_edges = set()
	train_losses = []
	for index in range(batches):
		if index % val_on_step == 0:
			val_loss, val_rmse = evaluate_on_steps(dataCenter, ds, graphSage, projection, optimizer, device, index, index // val_on_step, 10000, save_dir)
			val_losses.append(val_loss)
			val_rmse_losses.append(val_rmse)

		edge_batch = train_edges[index*b_sz:(index+1)*b_sz]
		visited_edges |= set(edge_batch)

		# get ground-truth for the nodes batch
		ratings_batch = torch.tensor([edge[2] for edge in edge_batch]).to(device)

		# feed edges batch to the graphSAGE
		# returning the nodes embeddings

		movie_embs = graphSage(edge_batch, "movie")
		user_embs = graphSage(edge_batch, "user")

		if learn_method == 'sup':
			# superivsed learning
			result_batch = projection(user_embs, movie_embs)
			loss_sup = torch.sum(torch.square(result_batch - ratings_batch))
			loss_sup /= len(edge_batch)
			loss = loss_sup
			train_losses.append(loss.item())

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_edges), len(train_edges)))

		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, projection, optimizer, train_losses, val_losses, val_rmse_losses
