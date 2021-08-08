import sys
import os
import torch
import argparse
import pyhocon
import random
import pickle
import datetime

from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=2000)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--train_on_steps', action='store_true')
parser.add_argument('--steps', type=int, default=301)
parser.add_argument('--val_on_step', type=int, default=50)
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	dataCenter = DataCenter(config)
	dataCenter.load_dataSet(ds)
	features = torch.FloatTensor(getattr(dataCenter, ds+'_movie_feats')).to(device)

	graphsage = None
	projection = None
	classification = None
	train_losses = []
	val_losses = []
	val_rmse_losses = []

	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	save_dir = f"models/{current_time}"
	os.mkdir(save_dir)

	epochs_before = 0

	if ds == "movielens":
		graphSage = GraphSage2(config['setting.num_layers'], features.size(1), config['setting.rating_emb_size'],
							   config['setting.hidden_emb_size'],
							   features, getattr(dataCenter, ds + '_movie_adj_list'),
							   getattr(dataCenter, ds + '_user_adj_list'),getattr(dataCenter, ds + '_rating_distrib_movie'),
							   getattr(dataCenter, ds + '_rating_distrib_user'),
							   config['setting.num_ratings'], device,
							   agg_func=args.agg_func)
		graphSage.to(device)

		# edge_batch = [(0,2,3.5),
		# 			  (6,3,2.5)]
		#
		# print("GENERATING USERS")
		# print("===============================================")
		# user_embs = graphSage(edge_batch, "user")
		# print("GENERATING MOVIES")
		# print("===============================================")
		# movie_embs = graphSage(edge_batch, "movie")
		#
		# print("Movie embeddings")
		# print(movie_embs)
		# print("User embeddings")
		# print(user_embs)
		#
		# print("===============================================")
		projection = Projection(config['setting.hidden_emb_size'], config['setting.projection_size'])
		projection.to(device)
		optimizer = get_optimizer(graphSage, projection)

		if args.continue_training:
			#LOADING PRETRAINED MODEL
			path = config['setting.model']
			checkpoint = torch.load(path)
			graphSage.load_state_dict(checkpoint["graphsage_state_dict"])
			projection.load_state_dict(checkpoint["projection_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			epochs_before = checkpoint["epoch"]

		# result = projection(user_embs, movie_embs)
		#
		# print("Result")
		# print(type(result))
		# print(result)
		#
		# exit()
	else:
		graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
		graphSage.to(device)

		num_labels = len(set(getattr(dataCenter, ds+'_labels')))
		classification = Classification(config['setting.hidden_emb_size'], num_labels)
		classification.to(device)

		unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')

	for epoch in range(epochs_before + 1, epochs_before + args.epochs + 1):
		print('----------------------EPOCH %d-----------------------' % epoch)
		if args.train_on_steps:
			graphSage, projection, optimizer, train_losses, val_losses, val_rmse_losses = apply_model_on_steps(
				dataCenter, ds, graphSage, projection, optimizer, args.b_sz, args.steps, args.val_on_step, device,
				args.learn_method, save_dir)
		elif ds == "movielens":
			graphSage, projection, optimizer, losses = apply_model2(dataCenter, ds, graphSage, projection, optimizer, args.b_sz, device, args.learn_method)
			train_losses.extend(losses)
			val_loss, val_rmse = evaluate2(dataCenter, ds, graphSage, projection, optimizer, device, args.name, epoch, save_dir)
			val_losses.append(val_loss)
			val_rmse_losses.append(val_rmse)
		else:
			graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)

		# if (epoch+1) % 2 == 0 and args.learn_method == 'unsup':
		# 	classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name)
		# if args.learn_method != 'unsup':
		# 	args.max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, args.max_vali_f1, args.name, epoch)

	data = dict()
	data["train_losses"] = train_losses
	data["val_losses"] = val_losses
	data["val_rmse_losses"] = val_rmse_losses
	file_data = open(f"{save_dir}/result_losses.pkl", "wb")
	pickle.dump(data, file_data)
	file_data.close()