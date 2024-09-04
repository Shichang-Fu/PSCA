import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, feature_dim=256, need_weights=(True, True, True, True, True)):
        super(FeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        if need_weights[0]:
            self.linear_w_p3 = nn.Linear(int(feature_dim), 1)
        if need_weights[1]:
            self.linear_w_p4 = nn.Linear(int(feature_dim), 1)
        if need_weights[2]:
            self.linear_w_p5 = nn.Linear(int(feature_dim), 1)
        if need_weights[3]:
            self.linear_w_p6 = nn.Linear(int(feature_dim), 1)
        if need_weights[4]:
            self.linear_w_p7 = nn.Linear(int(feature_dim), 1)
        
    def forward(self, p3, p4, p5, p6, p7):

        if hasattr(self, 'linear_w_p3'):
            w_p3 = self.linear_w_p3(p3)
            w_p3 = w_p3.view(-1)
            w_p3 = F.softmax(w_p3, dim=0)


        if hasattr(self, 'linear_w_p4'):
            w_p4 = self.linear_w_p4(p4)
            w_p4 = w_p4.view(-1)
            w_p4 = F.softmax(w_p4, dim=0)


        if hasattr(self, 'linear_w_p5'):
            w_p5 = self.linear_w_p5(p5)
            w_p5 = w_p5.view(-1)
            w_p5 = F.softmax(w_p5, dim=0)


        if hasattr(self, 'linear_w_p6'):
            w_p6 = self.linear_w_p6(p6)
            w_p6 = w_p6.view(-1)
            w_p6 = F.softmax(w_p6, dim=0)


        if hasattr(self, 'linear_w_p7'):
            w_p7 = self.linear_w_p7(p7)
            w_p7 = w_p7.view(-1)
            w_p7 = F.softmax(w_p7, dim=0)

        

        fusion_features = []
        for i in range(p3.shape[0]):
            denominator = 0
            if hasattr(self, 'linear_w_p3'):
                denominator += w_p3[i]
            if hasattr(self, 'linear_w_p4'):
                denominator += w_p4[i]
            if hasattr(self, 'linear_w_p5'):
                denominator += w_p5[i]
            if hasattr(self, 'linear_w_p6'):
                denominator += w_p6[i]
            if hasattr(self, 'linear_w_p7'):
                denominator += w_p7[i]
            
            a = torch.zeros_like(p3[i])
            if hasattr(self, 'linear_w_p3'):
                a += w_p3[i] / denominator * p3[i]
            if hasattr(self, 'linear_w_p4'):
                a += w_p4[i] / denominator * p4[i]
            if hasattr(self, 'linear_w_p5'):
                a += w_p5[i] / denominator * p5[i]
            if hasattr(self, 'linear_w_p6'):
                a += w_p6[i] / denominator * p6[i]
            if hasattr(self, 'linear_w_p7'):
                a += w_p7[i] / denominator * p7[i]

            fusion_features.append(a)


        fusion_features = torch.stack(fusion_features)
        
        return fusion_features
