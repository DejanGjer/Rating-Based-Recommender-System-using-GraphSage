import torch
import torch.nn as nn
import torch.nn.functional as F

class RatingMatrixRegularizer():

    def __init__(self, model, lambda_reg=4000):
        self.model = model
        self.lambda_reg = lambda_reg

    def regularization(self):
        prev_param = None
        layer = 1
        result_loss = 0
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.startswith("rating_layer"):
                # print(model_param_name)
                # print(model_param_value)
                # print(type(model_param_value))
                if prev_param is None:
                    prev_param = model_param_value
                else:
                    result_loss += RatingMatrixRegularizer.__add_l2(prev_param, model_param_value)
            else:
                prev_param = None
        return self.lambda_reg * result_loss

    @staticmethod
    def __add_l2(var1, var2):
        return torch.mean(torch.square(var2 - var1))
