import torch

nodes = [3, 5, 2, 4, 1, 8, 9]
embs = [[0,0,1,1],
        [0,0,2,1],
        [0,3,0,1],
        [0,4,0,1],
        [0,5,0,1],
        [6,0,0,1],
        [7,0,0,1]]
embs = torch.FloatTensor(embs)
unique_nodes = {3:0, 5:1, 2:2, 4:3, 1:4, 8:5, 9:6}
samp_neighs = [{2,4}, {1,8,9}, {3}]

column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
print(column_indices)
row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
print(row_indices)
mask = torch.zeros(3,7)
mask[row_indices, column_indices] = 1
print(mask)
aggregate_feats = mask.mm(embs)
print(aggregate_feats)

