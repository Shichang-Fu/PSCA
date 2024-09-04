import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal_(self.conv.weight, std=0.01)
        torch.nn.init.constant_(self.conv.bias, 0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DJSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation
        return -mutual_info
    
    
class FCs(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FCs, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2
    

class Memory(nn.Module):
    def __init__(self, memory_size, 
                 feature_dim, 
                 key_dim, 
                 temp_update,
                 temp_gather, 
                 use_MI = False,
                 # feature_shapes = [(88,168),(44,84),(22,42),(11,21),(6,11)],
                 feature_shapes = [(96,144),(48,72),(24,36),(12,18),(6,9)], 
                 classes_number=7):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        self.use_MI = use_MI
        
        if use_MI:
            self.djs_loss = DJSLoss()
            self.MI_nets = nn.ModuleDict()
            for level in range(3, 8):
                for i in range(classes_number):
                    MI_net = nn.Sequential(Conv(c1=feature_dim, c2=8),
                                  Conv(c1=feature_dim, c2=8), 
                                  FCs(int(feature_shapes[level-3][0]) * int(feature_shapes[level-3][1]), 64))
                    
                    self.MI_nets[f'{level}_{i}'] = MI_net
                    
        self.feature_shapes = feature_shapes

        self.memory = nn.ParameterDict()
        for level in range(3, 8):
            for i in range(classes_number):
                memory_item = nn.Parameter(F.normalize(torch.rand((memory_size, key_dim), dtype=torch.float), dim=1))
                self.memory[f'{level}_{i}'] = memory_item

            
    def get_update_query(self, mem, max_indices, update_indices, score, query):
        
        m, d = mem.size()
        query_update = torch.zeros((m,d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1)==i)
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) * query[idx].squeeze(1)), dim=0)
            else:
                query_update[i] = 0
        return query_update 


    def get_score(self, mem, query):
        bs, h,w,d = query.size()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)   
        score_memory = F.softmax(score,dim=1)  
        
        return score_query, score_memory
    
    def forward(self, query_source, keys, only_update = True, query_target = None, fusion_keys = None):

        batch_size_source, dims,h_1,w_1 = query_source.size() # b X c X h_1 X w_1 
        if self.use_MI and (query_target is not None):   
            #if query_source.size()[2:] != query_target.size()[2:]:
            query_source = F.interpolate(query_source, size=self.feature_shapes[keys[0]-3], mode='bilinear', align_corners=False)# ------> #插值到固定尺寸
        query_source = F.normalize(query_source, dim=1)
        query_source = query_source.permute(0,2,3,1) # b X h_1 X w_1 X c

        if not only_update:
            if query_target is not None:
                batch_size_target, dims,h_2,w_2 = query_target.size() # b X c X h_2 X w_2             
                query_target = F.interpolate(query_target, size=self.feature_shapes[keys[0]-3], mode='bilinear', align_corners=False)# ------> #插值到固定尺寸
                query_target = F.normalize(query_target, dim=1)
                query_target = query_target.permute(0,2,3,1) # b X h_2 X w_2 X c
                if batch_size_target != batch_size_source:
                    mi_loss = torch.tensor(0.0, dtype=query_target.dtype, device=query_target.device)
                    return mi_loss
                if fusion_keys is not None:
                    updated_query_source, softmax_score_query_source,softmax_score_memory_source = self.read(query_source,fusion_keys)
                    updated_query_target, softmax_score_query_target,softmax_score_memory_target = self.read(query_target,fusion_keys)
                else:
                    updated_query_source, softmax_score_query_source,softmax_score_memory_source = self.read(query_source,self.memory[f'{keys[0]}_{keys[1]}'])
                    updated_query_target, softmax_score_query_target,softmax_score_memory_target = self.read(query_target,self.memory[f'{keys[0]}_{keys[1]}'])
                updated_query_source = self.MI_nets[f"{keys[0]}_{keys[1]}"][0](updated_query_source)
                updated_query_target = self.MI_nets[f"{keys[0]}_{keys[1]}"][1](updated_query_target)

                updated_query_source = updated_query_source.reshape(8 * batch_size_source, -1)     #  b x h_1 x w_1 x 8  ----> b*8, h1 x w1
                updated_query_target = updated_query_target.reshape(8 * batch_size_target, -1)     #  b x h_2 x w_2 x 8  ----> b*8, h2 x w2
                
                updated_query_target_shuffle = torch.randperm(updated_query_target.size(0))
                updated_query_target_shuffle = updated_query_target[updated_query_target_shuffle]

                pred_xy = self.MI_nets[f"{keys[0]}_{keys[1]}"][2](updated_query_source, updated_query_target)
                pred_x_y = self.MI_nets[f"{keys[0]}_{keys[1]}"][2](updated_query_source, updated_query_target_shuffle)

                mi_loss = self.djs_loss(pred_xy, pred_x_y)
                
                return mi_loss
            else:
                if fusion_keys is not None:
                    updated_query, softmax_score_query,softmax_score_memory = self.read(query_source, fusion_keys)
                else:
                    updated_query, softmax_score_query,softmax_score_memory = self.read(query_source, self.memory[f'{keys[0]}_{keys[1]}'])
                return updated_query, softmax_score_query, softmax_score_memory
        else:
            updated_memory = self.update(query_source, self.memory[f'{keys[0]}_{keys[1]}'])
            
            return updated_memory
    
    
    def update(self, query, keys):
        
        batch_size, h,w,dims = query.size() # b X h X w X d 
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)   
        
        query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape)
        updated_memory = F.normalize(query_update + keys, dim=1)
        
        return updated_memory.detach()
        

    def read(self, query, updated_memory):
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        updated_query = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d
        updated_query = updated_query.view(batch_size, h, w, dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query, softmax_score_query, softmax_score_memory