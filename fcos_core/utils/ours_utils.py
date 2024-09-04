import torch
import torch.nn.functional as F

def gen_class_feature(targets, feature, nc):
    batch_size, C, H, W = feature.size()
    class_feature_maps = []
    class_targets = {}
    for batch_idx, boxlist in enumerate(targets):
        boxes = boxlist.bbox
        labels = boxlist.get_field("labels")
        image_width = boxlist.size[0]
        image_height = boxlist.size[1]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            class_idx = label.item() - 1 
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            if class_idx not in class_targets:
                class_targets[class_idx] = []
            
            class_targets[class_idx].append((batch_idx, x_center, y_center, width, height))

    for class_idx in range(nc):
        if class_idx in class_targets:
            class_feature_map = torch.zeros(batch_size, C, H, W).to(feature.device)
            for target in class_targets[class_idx]:
                batch_idx, x_center, y_center, width, height = target
                batch_idx = int(batch_idx)
                x = int(x_center * W)
                y = int(y_center * H)
                w = int(width * W)
                h = int(height * H)
                class_feature_map[batch_idx, :, y - h // 2:y + h // 2, x - w // 2:x + w // 2] = feature[batch_idx, :, y - h // 2:y + h // 2, x - w // 2:x + w // 2]
            class_feature_maps.append(class_feature_map)
        else:
            class_feature_maps.append(torch.zeros_like(feature))
    class_feature_maps = torch.stack(class_feature_maps, dim=0)

    return class_feature_maps



def vis_gen_class_feature(targets, feature, nc):
    batch_size, C, H, W = feature.size()
    class_features = {i: [] for i in range(nc)}  # 初始化字典以存储每个类别的特征
    
    for batch_idx, boxlist in enumerate(targets):
        boxes = boxlist.bbox
        labels = boxlist.get_field("labels")
        image_width = boxlist.size[0]
        image_height = boxlist.size[1]
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            class_idx = label.item() - 1 
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            # 计算区域
            x = int(x_center * W)
            y = int(y_center * H)
            w = int(width * W)
            h = int(height * H)
            
            # 确保区域在特征图的范围内
            x_start = max(x - w // 2, 0)
            y_start = max(y - h // 2, 0)
            x_end = min(x + w // 2, W)
            y_end = min(y + h // 2, H)
            
            # 提取特征区域
            feature_patch = feature[batch_idx, :, y_start:y_end, x_start:x_end]
            
            class_features[class_idx].append(feature_patch.cpu().numpy())  # 将特征转换为numpy数组并存储
    
    return class_features


def gen_class_masks(targets, feature, nc):
    batch_size, C, H, W = feature.size()
    class_masks = []
    class_targets = {}
    for batch_idx, boxlist in enumerate(targets):
        boxes = boxlist.bbox
        labels = boxlist.get_field("labels")
        image_width = boxlist.size[0]
        image_height = boxlist.size[1]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            class_idx = label.item() - 1 
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            if class_idx not in class_targets:
                class_targets[class_idx] = []
            
            class_targets[class_idx].append((batch_idx, x_center, y_center, width, height))
            
    for class_idx in range(nc):
        batch_masks = []
        for batch_idx in range(batch_size):
            if class_idx in class_targets:
                class_mask = torch.zeros(1, H, W, dtype=torch.float32, device=feature.device)
                for target in class_targets[class_idx]:
                    tgt_batch_idx, x_center, y_center, width, height = target
                    if batch_idx == tgt_batch_idx:
                        x = int(x_center * W)
                        y = int(y_center * H)
                        w = int(width * W)
                        h = int(height * H)
                        # Ensure coordinates are within bounds
                        x1 = max(x - w // 2, 0)
                        y1 = max(y - h // 2, 0)
                        x2 = min(x + w // 2, W)
                        y2 = min(y + h // 2, H)
                        # Set the corresponding region in the mask to 1
                        class_mask[0, y1:y2, x1:x2] = 1.0
                batch_masks.append(class_mask)
            else:
                # If no targets for this class, append a zero mask
                batch_masks.append(torch.zeros(1, H, W, dtype=torch.float32, device=feature.device))

        # Append the masks for the current class to class_masks
        class_masks.append(torch.stack(batch_masks, dim=0))  # shape: (batch_size, 1, H, W)

    return class_masks

def process_memory_features(feature, mask):

    batch_size, C, H, W = feature.size()
    
    # Expand mask to match feature shape
    mask = mask.expand(batch_size, 1, H, W)  # shape: (batch_size, 1, H, W)

    # Apply mask to features
    masked_features = feature * mask  # shape: (batch_size, C, H, W)
    complement_features = feature * (1 - mask)  # shape: (batch_size, C, H, W)
    return masked_features, complement_features


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    计算三元组对比损失的函数
    
    参数:
    - anchor: 锚点样本特征, 形状为 (batch_size, feature_dim)
    - positive: 正样本特征, 形状为 (batch_size, feature_dim)
    - negative: 负样本特征, 形状为 (batch_size, feature_dim)
    - margin: 边界值
    
    返回:
    - loss: 三元组对比损失标量
    """
    # 检查输入的尺寸
    if anchor.size()[0] != positive.size()[0] or anchor.size()[0] != negative.size()[0]:
        return torch.tensor(0.0, dtype=anchor.dtype, device=anchor.device)

    # 调整特征图尺寸
    if anchor.size()[2:] != positive.size()[2:]:
        anchor = F.interpolate(anchor, size=positive.size()[2:], mode='bilinear', align_corners=False)

    if anchor.size()[2:] != negative.size()[2:]:
        anchor = F.interpolate(anchor, size=negative.size()[2:], mode='bilinear', align_corners=False)

    # 计算正样本和负样本之间的距离
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    
    # 计算三元组对比损失
    loss = torch.relu(pos_dist - neg_dist + margin)
    
    # 返回平均损失
    return loss.mean()