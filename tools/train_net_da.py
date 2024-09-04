# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip
import random
import argparse
import os
import numpy as np
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.modeling.discriminator import FCOSDiscriminator, FCOSDiscriminator_CA
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from fcos_core.modeling.ours.memory_bank import Memory
from fcos_core.modeling.ours.fusion import FeatureFusion
from fcos_core.modeling.ours.contra import Contrast
import torch.nn.functional as F
from fcos_core.utils.metric_logger import (
    MetricLogger, TensorboardLogger)

def train(cfg, local_rank, distributed, test_only=False, use_tensorboard=False):


    with_DA = cfg.MODEL.DA_ON

    ##########################################################################
    ############################# Initial MODEL ##############################
    ##########################################################################
    MODEL = {}
    device = torch.device(cfg.MODEL.DEVICE)
    backbone = build_backbone(cfg).to(device)
    fcos = build_rpn(cfg, backbone.out_channels).to(device)


    ##########################################################################
    #################### Initial Optimizer and Scheduler #####################
    ##########################################################################
    optimizer = {}
    scheduler = {}


    optimizer["backbone"] = make_optimizer(cfg, backbone, name='backbone')
    optimizer["fcos"] = make_optimizer(cfg, fcos, name='fcos')

    scheduler["backbone"] = make_lr_scheduler(cfg, optimizer["backbone"], name='backbone')
    scheduler["fcos"] = make_lr_scheduler(cfg, optimizer["fcos"], name='fcos')

    if with_DA:
        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P7_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P7,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P6_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P6,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P5_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P5,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P4_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P4,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = FCOSDiscriminator(
                    patch_stride=cfg.MODEL.ADV.PATCH_STRIDE,
                    num_convs=cfg.MODEL.ADV.DIS_P3_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P3,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P7_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P7,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P6_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P6,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P5_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P5,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P4_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P4,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = FCOSDiscriminator_CA(
                    num_convs=cfg.MODEL.ADV.CA_DIS_P3_NUM_CONVS,
                    grad_reverse_lambda=cfg.MODEL.ADV.CA_GRL_WEIGHT_P3,
                    center_aware_weight=cfg.MODEL.ADV.CENTER_AWARE_WEIGHT,
                    center_aware_type=cfg.MODEL.ADV.CENTER_AWARE_TYPE,
                    grl_applied_domain=cfg.MODEL.ADV.GRL_APPLIED_DOMAIN).to(device)
            ##############################  Ousr methods Initial #########################################
        if cfg.MODEL.ADV.USE_MEMORY:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
            use_mi = False
            if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY:
                if cfg.MODEL.ADV.USE_MEMORY_P7:
                    discriminators_cda_P7 = []
                    for _ in range(num_classes):
                        in_channels = 256
                        discriminatorcls = FCOSDiscriminator(num_convs=cfg.MODEL.ADV.DIS_P7_NUM_CONVS,
                                                                in_channels=in_channels, 
                                                                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P7).to(device)
                        discriminators_cda_P7.append(discriminatorcls)
                        
                if cfg.MODEL.ADV.USE_MEMORY_P6:
                    discriminators_cda_P6 = []
                    for _ in range(num_classes):
                        in_channels = 256
                        discriminatorcls = FCOSDiscriminator(num_convs=cfg.MODEL.ADV.DIS_P6_NUM_CONVS,
                                                                in_channels=in_channels, 
                                                                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P6).to(device)
                        discriminators_cda_P6.append(discriminatorcls)
                if cfg.MODEL.ADV.USE_MEMORY_P5:
                    discriminators_cda_P5 = []
                    for _ in range(num_classes):
                        in_channels = 256
                        discriminatorcls = FCOSDiscriminator(num_convs=cfg.MODEL.ADV.DIS_P5_NUM_CONVS,
                                                                in_channels=in_channels, 
                                                                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P5).to(device)
                        discriminators_cda_P5.append(discriminatorcls)
                    
                if cfg.MODEL.ADV.USE_MEMORY_P4:
                    discriminators_cda_P4 = []
                    for _ in range(num_classes):
                        in_channels = 256
                        discriminatorcls = FCOSDiscriminator(num_convs=cfg.MODEL.ADV.DIS_P4_NUM_CONVS,
                                                                in_channels=in_channels, 
                                                                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P4).to(device)
                        discriminators_cda_P4.append(discriminatorcls)
                if cfg.MODEL.ADV.USE_MEMORY_P3:
                    discriminators_cda_P3 = []
                    for _ in range(num_classes):
                        in_channels = 256
                        discriminatorcls = FCOSDiscriminator(num_convs=cfg.MODEL.ADV.DIS_P3_NUM_CONVS,
                                                                in_channels=in_channels, 
                                                                grad_reverse_lambda=cfg.MODEL.ADV.GRL_WEIGHT_P3).to(device)
                        discriminators_cda_P3.append(discriminatorcls)
                    
            #############使用MI进行类别对齐：需要初始化带网络的memory################
            if cfg.MODEL.ADV.USE_MI:
                use_mi = True
                if cfg.MODEL.ADV.USE_FUSION:
                    if cfg.MODEL.ADV.USE_FUSION_P3:
                        fusion_module_P3 = FeatureFusion(feature_dim=256).to(device)
                    if cfg.MODEL.ADV.USE_FUSION_P4:
                        fusion_module_P4 = FeatureFusion(feature_dim=256).to(device)
                    if cfg.MODEL.ADV.USE_FUSION_P5:
                        fusion_module_P5 = FeatureFusion(feature_dim=256).to(device)
                    if cfg.MODEL.ADV.USE_FUSION_P6:
                        fusion_module_P6 = FeatureFusion(feature_dim=256).to(device)
                    if cfg.MODEL.ADV.USE_FUSION_P7:
                        fusion_module_P7 = FeatureFusion(feature_dim=256).to(device)
            if cfg.MODEL.ADV.USE_CONTRAST:
                contra_P3 = Contrast(1,256).to(device)
                contra_P4 = Contrast(1,256).to(device)
                contra_P5 = Contrast(1,256).to(device)
                contra_P6 = Contrast(1,256).to(device)
                contra_P7 = Contrast(1,256).to(device)
                
            memory_size = 10
            feature_dim = key_dim = 256
            memory = Memory(memory_size=memory_size, 
                            feature_dim=feature_dim, 
                            key_dim=key_dim,
                            temp_update=0.1, 
                            temp_gather=0.1,
                            use_MI=use_mi,
                            classes_number=num_classes).to(device)
        ##############################  Ousr methods Initial ending #########################################
                

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                optimizer["dis_P7"] = make_optimizer(cfg, dis_P7, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                optimizer["dis_P6"] = make_optimizer(cfg, dis_P6, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                optimizer["dis_P5"] = make_optimizer(cfg, dis_P5, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                optimizer["dis_P4"] = make_optimizer(cfg, dis_P4, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                optimizer["dis_P3"] = make_optimizer(cfg, dis_P3, name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                optimizer["dis_P7_CA"] = make_optimizer(cfg, dis_P7_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                optimizer["dis_P6_CA"] = make_optimizer(cfg, dis_P6_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                optimizer["dis_P5_CA"] = make_optimizer(cfg, dis_P5_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                optimizer["dis_P4_CA"] = make_optimizer(cfg, dis_P4_CA, name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                optimizer["dis_P3_CA"] = make_optimizer(cfg, dis_P3_CA, name='discriminator')
        ######################################类别-尺度鉴别器#########################################################
        if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY:
            if cfg.MODEL.ADV.USE_MEMORY_P7:
                for idx, discriminatorcls in enumerate(discriminators_cda_P7):
                    optimizer[f"dis_P7_SCA_{idx}"] = make_optimizer(cfg, discriminatorcls, name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P6:
                for idx, discriminatorcls in enumerate(discriminators_cda_P6):
                    optimizer[f"dis_P6_SCA_{idx}"] = make_optimizer(cfg, discriminatorcls, name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P5:
                for idx, discriminatorcls in enumerate(discriminators_cda_P5):
                    optimizer[f"dis_P5_SCA_{idx}"] = make_optimizer(cfg, discriminatorcls, name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P4:
                for idx, discriminatorcls in enumerate(discriminators_cda_P4):
                    optimizer[f"dis_P4_SCA_{idx}"] = make_optimizer(cfg, discriminatorcls, name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P3:
                for idx, discriminatorcls in enumerate(discriminators_cda_P3):
                    optimizer[f"dis_P3_SCA_{idx}"] = make_optimizer(cfg, discriminatorcls, name='discriminator')
        if not cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY and cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_FUSION:
            optimizer["fusion_P3"] = make_optimizer(cfg, fusion_module_P3, name='fusion')
            optimizer["fusion_P4"] = make_optimizer(cfg, fusion_module_P3, name='fusion')
            optimizer["fusion_P5"] = make_optimizer(cfg, fusion_module_P3, name='fusion')
            optimizer["fusion_P6"] = make_optimizer(cfg, fusion_module_P3, name='fusion')
            optimizer["fusion_P7"] = make_optimizer(cfg, fusion_module_P3, name='fusion')
        ######################################最大化互信息#########################################################
        if cfg.MODEL.ADV.USE_MEMORY:
            optimizer["memory"] = make_optimizer(cfg, memory, name='memory')
        if cfg.MODEL.ADV.USE_CONTRAST:
            optimizer["contra_P3"] = make_optimizer(cfg, contra_P3, name='fusion')
            optimizer["contra_P4"] = make_optimizer(cfg, contra_P4, name='fusion')
            optimizer["contra_P5"] = make_optimizer(cfg, contra_P5, name='fusion')
            optimizer["contra_P6"] = make_optimizer(cfg, contra_P6, name='fusion')
            optimizer["contra_P7"] = make_optimizer(cfg, contra_P7, name='fusion')
        
        
        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                scheduler["dis_P7"] = make_lr_scheduler(cfg, optimizer["dis_P7"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                scheduler["dis_P6"] = make_lr_scheduler(cfg, optimizer["dis_P6"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                scheduler["dis_P5"] = make_lr_scheduler(cfg, optimizer["dis_P5"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                scheduler["dis_P4"] = make_lr_scheduler(cfg, optimizer["dis_P4"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                scheduler["dis_P3"] = make_lr_scheduler(cfg, optimizer["dis_P3"], name='discriminator')
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                scheduler["dis_P7_CA"] = make_lr_scheduler(cfg, optimizer["dis_P7_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P6:
                scheduler["dis_P6_CA"] = make_lr_scheduler(cfg, optimizer["dis_P6_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P5:
                scheduler["dis_P5_CA"] = make_lr_scheduler(cfg, optimizer["dis_P5_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P4:
                scheduler["dis_P4_CA"] = make_lr_scheduler(cfg, optimizer["dis_P4_CA"], name='discriminator')
            if cfg.MODEL.ADV.USE_DIS_P3:
                scheduler["dis_P3_CA"] = make_lr_scheduler(cfg, optimizer["dis_P3_CA"], name='discriminator')
        ######################################类别-尺度鉴别器#########################################################
        if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY:
            if cfg.MODEL.ADV.USE_MEMORY_P7:
                for idx, discriminatorcls in enumerate(discriminators_cda_P7):
                    scheduler[f"dis_P7_SCA_{idx}"] = make_lr_scheduler(cfg, optimizer[f"dis_P7_SCA_{idx}"], name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P6:
                for idx, discriminatorcls in enumerate(discriminators_cda_P6):
                    scheduler[f"dis_P6_SCA_{idx}"] = make_lr_scheduler(cfg, optimizer[f"dis_P6_SCA_{idx}"], name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P5:
                for idx, discriminatorcls in enumerate(discriminators_cda_P5):
                    scheduler[f"dis_P5_SCA_{idx}"] = make_lr_scheduler(cfg, optimizer[f"dis_P5_SCA_{idx}"], name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P4:
                for idx, discriminatorcls in enumerate(discriminators_cda_P4):
                    scheduler[f"dis_P4_SCA_{idx}"] = make_lr_scheduler(cfg, optimizer[f"dis_P4_SCA_{idx}"], name='discriminator')
            if cfg.MODEL.ADV.USE_MEMORY_P3:
                for idx, discriminatorcls in enumerate(discriminators_cda_P3):
                    scheduler[f"dis_P3_SCA_{idx}"] = make_lr_scheduler(cfg, optimizer[f"dis_P3_SCA_{idx}"], name='discriminator')
        if not cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY and cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_FUSION:
            scheduler["fusion_P3"] = make_lr_scheduler(cfg, optimizer["fusion_P3"], name='fusion')
            scheduler["fusion_P4"] = make_lr_scheduler(cfg, optimizer["fusion_P4"], name='fusion')
            scheduler["fusion_P5"] = make_lr_scheduler(cfg, optimizer["fusion_P5"], name='fusion')
            scheduler["fusion_P6"] = make_lr_scheduler(cfg, optimizer["fusion_P6"], name='fusion')
            scheduler["fusion_P7"] = make_lr_scheduler(cfg, optimizer["fusion_P7"], name='fusion')
        ######################################最大化互信息#########################################################
        if cfg.MODEL.ADV.USE_MEMORY :
            scheduler["memory"] = make_lr_scheduler(cfg, optimizer["memory"], name='memory')
        if cfg.MODEL.ADV.USE_CONTRAST:
            scheduler["contra_P3"] = make_lr_scheduler(cfg, optimizer["contra_P3"], name='fusion')
            scheduler["contra_P4"] = make_lr_scheduler(cfg, optimizer["contra_P4"], name='fusion')
            scheduler["contra_P5"] = make_lr_scheduler(cfg, optimizer["contra_P5"], name='fusion')
            scheduler["contra_P6"] = make_lr_scheduler(cfg, optimizer["contra_P6"], name='fusion')
            scheduler["contra_P7"] = make_lr_scheduler(cfg, optimizer["contra_P7"], name='fusion')

        
        
        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                MODEL["dis_P7"] = dis_P7
            if cfg.MODEL.ADV.USE_DIS_P6:
                MODEL["dis_P6"] = dis_P6
            if cfg.MODEL.ADV.USE_DIS_P5:
                MODEL["dis_P5"] = dis_P5
            if cfg.MODEL.ADV.USE_DIS_P4:
                MODEL["dis_P4"] = dis_P4
            if cfg.MODEL.ADV.USE_DIS_P3:
                MODEL["dis_P3"] = dis_P3
        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                MODEL["dis_P7_CA"] = dis_P7_CA
            if cfg.MODEL.ADV.USE_DIS_P6:
                MODEL["dis_P6_CA"] = dis_P6_CA
            if cfg.MODEL.ADV.USE_DIS_P5:
                MODEL["dis_P5_CA"] = dis_P5_CA
            if cfg.MODEL.ADV.USE_DIS_P4:
                MODEL["dis_P4_CA"] = dis_P4_CA
            if cfg.MODEL.ADV.USE_DIS_P3:
                MODEL["dis_P3_CA"] = dis_P3_CA
        ######################################类别-尺度鉴别器#########################################################
        if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY:
            if cfg.MODEL.ADV.USE_MEMORY_P7:
                for idx, discriminatorcls in enumerate(discriminators_cda_P7):
                    MODEL[f"dis_P7_SCA_{idx}"] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P6:
                for idx, discriminatorcls in enumerate(discriminators_cda_P6):
                    MODEL[f"dis_P6_SCA_{idx}"] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P5:
                for idx, discriminatorcls in enumerate(discriminators_cda_P5):
                    MODEL[f"dis_P5_SCA_{idx}"] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P4:
                for idx, discriminatorcls in enumerate(discriminators_cda_P4):
                    MODEL[f"dis_P4_SCA_{idx}"] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P3:
                for idx, discriminatorcls in enumerate(discriminators_cda_P3):
                    MODEL[f"dis_P3_SCA_{idx}"] = discriminatorcls
                    
        ######################################最大化互信息#########################################################
        if cfg.MODEL.ADV.USE_MEMORY:
                MODEL[f"memory"] = memory
        if not cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY and cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_FUSION:
            MODEL["fusion_P3"] = fusion_module_P3
            MODEL["fusion_P4"] = fusion_module_P4
            MODEL["fusion_P5"] = fusion_module_P5
            MODEL["fusion_P6"] = fusion_module_P6
            MODEL["fusion_P7"] = fusion_module_P7
            
        if cfg.MODEL.ADV.USE_CONTRAST:
            MODEL["contra_P3"] = contra_P3
            MODEL["contra_P4"] = contra_P4
            MODEL["contra_P5"] = contra_P5
            MODEL["contra_P6"] = contra_P6
            MODEL["contra_P7"] = contra_P7


    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        fcos = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fcos)

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3)

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P7_CA)
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P6_CA)
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P5_CA)
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P4_CA)
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis_P3_CA)
        ###########################  ours methods ################################
        ######################################类别-尺度鉴别器#########################################################
        if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY:
            if cfg.MODEL.ADV.USE_MEMORY_P7:
                for idx, discriminatorcls in enumerate(discriminators_cda_P7):
                    discriminatorcls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminatorcls)
                    discriminators_cda_P7[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P6:
                for idx, discriminatorcls in enumerate(discriminators_cda_P6):
                    discriminatorcls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminatorcls)
                    discriminators_cda_P6[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P5:
                for idx, discriminatorcls in enumerate(discriminators_cda_P5):
                    discriminatorcls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminatorcls)
                    discriminators_cda_P5[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P4:
                for idx, discriminatorcls in enumerate(discriminators_cda_P4):
                    discriminatorcls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminatorcls)
                    discriminators_cda_P4[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P3:
                for idx, discriminatorcls in enumerate(discriminators_cda_P3):
                    discriminatorcls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminatorcls)
                    discriminators_cda_P3[idx] = discriminatorcls
        
        ######################################最大化互信息#########################################################
        if cfg.MODEL.ADV.USE_MEMORY:
            memory = torch.nn.SyncBatchNorm.convert_sync_batchnorm(memory)
        if cfg.MODEL.ADV.USE_CONTRAST:
            contra_P3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(contra_P3)
            contra_P4 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(contra_P4)
            contra_P5 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(contra_P5)
            contra_P6 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(contra_P6)
            contra_P7 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(contra_P7)


    ##########################################################################
    ######################## DistributedDataParallel #########################
    ##########################################################################

    if distributed:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )
        
        fcos = torch.nn.parallel.DistributedDataParallel(
            fcos, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False
        )

        if cfg.MODEL.ADV.USE_DIS_GLOBAL:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7 = torch.nn.parallel.DistributedDataParallel(
                    dis_P7, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6 = torch.nn.parallel.DistributedDataParallel(
                    dis_P6, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5 = torch.nn.parallel.DistributedDataParallel(
                    dis_P5, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4 = torch.nn.parallel.DistributedDataParallel(
                    dis_P4, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3 = torch.nn.parallel.DistributedDataParallel(
                    dis_P3, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

        if cfg.MODEL.ADV.USE_DIS_CENTER_AWARE:
            if cfg.MODEL.ADV.USE_DIS_P7:
                dis_P7_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P7_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P6:
                dis_P6_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P6_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P5:
                dis_P5_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P5_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P4:
                dis_P4_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P4_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            if cfg.MODEL.ADV.USE_DIS_P3:
                dis_P3_CA = torch.nn.parallel.DistributedDataParallel(
                    dis_P3_CA, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )

        ######################################类别-尺度鉴别器#########################################################
        if cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY:
            if cfg.MODEL.ADV.USE_MEMORY_P7:
                for idx, discriminatorcls in enumerate(discriminators_cda_P7):
                    discriminatorcls = torch.nn.parallel.DistributedDataParallel(
                        discriminatorcls,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False
                    )
                    discriminators_cda_P7[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P6:
                for idx, discriminatorcls in enumerate(discriminators_cda_P6):
                    discriminatorcls = torch.nn.parallel.DistributedDataParallel(
                        discriminatorcls,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False
                    )
                    discriminators_cda_P6[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P5:
                for idx, discriminatorcls in enumerate(discriminators_cda_P5):
                    discriminatorcls = torch.nn.parallel.DistributedDataParallel(
                        discriminatorcls,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False
                    )
                    discriminators_cda_P5[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P4:
                for idx, discriminatorcls in enumerate(discriminators_cda_P4):
                    discriminatorcls = torch.nn.parallel.DistributedDataParallel(
                        discriminatorcls,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False
                    )
                    discriminators_cda_P4[idx] = discriminatorcls
            if cfg.MODEL.ADV.USE_MEMORY_P3:
                for idx, discriminatorcls in enumerate(discriminators_cda_P3):
                    discriminatorcls = torch.nn.parallel.DistributedDataParallel(
                        discriminatorcls,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False
                    )
                    discriminators_cda_P3[idx] = discriminatorcls
        
        ######################################最大化互信息#########################################################
        if cfg.MODEL.ADV.USE_MEMORY:
            memory = torch.nn.parallel.DistributedDataParallel(
                memory, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                broadcast_buffers=False
            )

        if not cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY and cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_FUSION:
            fusion_module_P3 = torch.nn.parallel.DistributedDataParallel(
                    fusion_module_P3, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            fusion_module_P4 = torch.nn.parallel.DistributedDataParallel(
                    fusion_module_P4, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            fusion_module_P5 = torch.nn.parallel.DistributedDataParallel(
                    fusion_module_P5, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            fusion_module_P6 = torch.nn.parallel.DistributedDataParallel(
                    fusion_module_P6, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            fusion_module_P7 = torch.nn.parallel.DistributedDataParallel(
                    fusion_module_P7, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            
        if cfg.MODEL.ADV.USE_CONTRAST:
            contra_P3 = torch.nn.parallel.DistributedDataParallel(
                    contra_P3, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            contra_P4 = torch.nn.parallel.DistributedDataParallel(
                    contra_P4, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            contra_P5 = torch.nn.parallel.DistributedDataParallel(
                    contra_P5, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            contra_P6 = torch.nn.parallel.DistributedDataParallel(
                    contra_P6, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )
            contra_P7 = torch.nn.parallel.DistributedDataParallel(
                    contra_P7, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False
                )



    ##########################################################################
    ########################### Save MODEL to Dict ###########################
    ##########################################################################
    MODEL["backbone"] = backbone
    MODEL["fcos"] = fcos

    ##########################################################################
    ################################ Training ################################
    ##########################################################################
    arguments = {}
    arguments["iteration"] = 0
    if with_DA:
        arguments["use_dis_global"] = cfg.MODEL.ADV.USE_DIS_GLOBAL
        arguments["use_dis_ca"] = cfg.MODEL.ADV.USE_DIS_CENTER_AWARE
        arguments["ga_dis_lambda"] = cfg.MODEL.ADV.GA_DIS_LAMBDA
        arguments["ca_dis_lambda"] = cfg.MODEL.ADV.CA_DIS_LAMBDA

        arguments["use_feature_layers"] = []
        arguments["use_feature_layers"].append("P7") if cfg.MODEL.ADV.USE_DIS_P7 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P6") if cfg.MODEL.ADV.USE_DIS_P6 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P5") if cfg.MODEL.ADV.USE_DIS_P5 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P4") if cfg.MODEL.ADV.USE_DIS_P4 else arguments["use_feature_layers"]
        arguments["use_feature_layers"].append("P3") if cfg.MODEL.ADV.USE_DIS_P3 else arguments["use_feature_layers"]


    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, MODEL, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, load_dis=True, load_opt_sch=False)
    # arguments.update(extra_checkpoint_data)

    # Initial dataloader (both target and source domain)
    data_loader = {}
    data_loader["source"] = make_data_loader_source(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    if cfg.SOLVER.ADAPT_VAL_ON:
        data_loader["val"] = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

    if with_DA:
        data_loader["target"] = make_data_loader_target(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        dir = os.path.join(cfg.OUTPUT_DIR,'tensorboard_logs/')

        if not os.path.exists(dir):
            mkdir(dir)
        meters = TensorboardLogger(
            log_dir=dir,
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")
    print(MODEL)

    if not test_only:
        do_train(
            MODEL,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
            distributed,
            meters,
        )
    return MODEL


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def run_test(cfg, MODEL, distributed):
    if distributed:
        MODEL["backbone"] = MODEL["backbone"].module
        MODEL["fcos"] = MODEL["fcos"].module

    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    # print(MODEL)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model=MODEL,
            data_loader=data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="da_ga_MI_VGG_16_FPN_memory_4x_contrast_fusion.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--test_only",
        dest="test_only",
        help="Test the input MODEL directly, without training",
        action="store_true",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final MODEL",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # Check if domain adaption
    # assert cfg.MODEL.DA_ON, "Domain Adaption"

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    setup_seed(1234)
    MODEL = train(cfg, args.local_rank, args.distributed, args.test_only,args.use_tensorboard)

    if not args.skip_test:
        run_test(cfg, MODEL, args.distributed)


if __name__ == "__main__":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    main()
