# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection (CVPR22-ORAL)
# Modified by Wuyang Li
# Based on https://github.com/chengchunhsu/EveryPixelMatters/blob/master/fcos_core/engine/trainer.py
# --------------------------------------------------------
import torch.nn as nn
import datetime
import logging
import time
import ipdb
import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list
import os
from fcos_core.data import make_data_loader, make_data_loader_source, make_data_loader_target
from fcos_core.utils.miscellaneous import mkdir
from .validation import _inference
from fcos_core.utils.comm import synchronize
from fcos_core.structures.image_list import ImageList
from fcos_core.utils.ours_utils import gen_class_feature, gen_class_masks, process_memory_features, triplet_loss

def forward_mi_contrast_fusion(cfg, model, targets=None, features_FPN=(None, None)):
    num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
    
    P3_neck_s = features_FPN[0][0]
    P4_neck_s = features_FPN[0][1]
    P5_neck_s = features_FPN[0][2]
    P6_neck_s = features_FPN[0][3]
    P7_neck_s = features_FPN[0][4]
    
    P3_neck_t = features_FPN[1][0]
    P4_neck_t = features_FPN[1][1]
    P5_neck_t = features_FPN[1][2]
    P6_neck_t = features_FPN[1][3]
    P7_neck_t = features_FPN[1][4]
    
    class_feature_maps_P3 = gen_class_feature(targets, P3_neck_s, num_classes)
    class_feature_maps_P4 = gen_class_feature(targets, P4_neck_s, num_classes) 
    class_feature_maps_P5 = gen_class_feature(targets, P5_neck_s, num_classes)
    class_feature_maps_P6 = gen_class_feature(targets, P6_neck_s, num_classes)
    class_feature_maps_P7 = gen_class_feature(targets, P7_neck_s, num_classes)
    
    memory = model["memory"]
    for i in range(num_classes):
        if torch.any(class_feature_maps_P3[i] != 0):
            updated_memory = memory(class_feature_maps_P3[i],(3,i))
            memory.memory[f'3_{i}'].data = updated_memory

        if torch.any(class_feature_maps_P4[i] != 0):
            updated_memory = memory(class_feature_maps_P4[i],(4,i))
            memory.memory[f'4_{i}'].data = updated_memory
            
        if torch.any(class_feature_maps_P5[i] != 0):
            updated_memory = memory(class_feature_maps_P5[i],(5,i))
            memory.memory[f'5_{i}'].data = updated_memory
            
        if torch.any(class_feature_maps_P6[i] != 0):
            updated_memory = memory(class_feature_maps_P6[i],(6,i))
            memory.memory[f'6_{i}'].data = updated_memory
            
        if torch.any(class_feature_maps_P7[i] != 0):
            updated_memory = memory(class_feature_maps_P7[i],(7,i))
            memory.memory[f'7_{i}'].data = updated_memory


    
    
    mitems_multi_after_fusion = {'P3': [], 'P4': [], 'P5': [],'P6': [], 'P7': []}
    for i in range(num_classes):
        fusion_P3_i = model["fusion_P3"](memory.memory[f'3_{i}'], memory.memory[f'4_{i}'], memory.memory[f'5_{i}'], memory.memory[f'6_{i}'], memory.memory[f'7_{i}'])
        fusion_P4_i = model["fusion_P4"](memory.memory[f'3_{i}'], memory.memory[f'4_{i}'], memory.memory[f'5_{i}'], memory.memory[f'6_{i}'], memory.memory[f'7_{i}'])
        fusion_P5_i = model["fusion_P5"](memory.memory[f'3_{i}'], memory.memory[f'4_{i}'], memory.memory[f'5_{i}'], memory.memory[f'6_{i}'], memory.memory[f'7_{i}'])
        fusion_P6_i = model["fusion_P6"](memory.memory[f'3_{i}'], memory.memory[f'4_{i}'], memory.memory[f'5_{i}'], memory.memory[f'6_{i}'], memory.memory[f'7_{i}'])
        fusion_P7_i = model["fusion_P7"](memory.memory[f'3_{i}'], memory.memory[f'4_{i}'], memory.memory[f'5_{i}'], memory.memory[f'6_{i}'], memory.memory[f'7_{i}'])
        mitems_multi_after_fusion['P3'].append(fusion_P3_i)
        mitems_multi_after_fusion['P4'].append(fusion_P4_i)
        mitems_multi_after_fusion['P5'].append(fusion_P5_i)
        mitems_multi_after_fusion['P6'].append(fusion_P6_i)
        mitems_multi_after_fusion['P7'].append(fusion_P7_i)
        
    
    targets_features_P3 = []
    targets_features_P4 = []
    targets_features_P5 = []
    targets_features_P6 = []
    targets_features_P7 = []
    
    sources_features_P3 = []
    sources_features_P4 = []
    sources_features_P5 = []
    sources_features_P6 = []
    sources_features_P7 = []
    
    fusion_mitems_P3 = mitems_multi_after_fusion['P3']
    fusion_mitems_P4 = mitems_multi_after_fusion['P4']
    fusion_mitems_P5 = mitems_multi_after_fusion['P5']
    fusion_mitems_P6 = mitems_multi_after_fusion['P6']
    fusion_mitems_P7 = mitems_multi_after_fusion['P7']
    
    mi_loss = {}
    for i in range(num_classes):
        mi_loss_p3= memory(P3_neck_s, (3,i), only_update = False, query_target = P3_neck_t, fusion_keys=fusion_mitems_P3[i])
        mi_loss_p4= memory(P4_neck_s, (4,i), only_update = False, query_target = P4_neck_t, fusion_keys=fusion_mitems_P4[i])
        mi_loss_p5= memory(P5_neck_s, (5,i), only_update = False, query_target = P5_neck_t, fusion_keys=fusion_mitems_P5[i])
        mi_loss_p6= memory(P6_neck_s, (6,i), only_update = False, query_target = P6_neck_t, fusion_keys=fusion_mitems_P6[i])
        mi_loss_p7= memory(P7_neck_s, (7,i), only_update = False, query_target = P7_neck_t, fusion_keys=fusion_mitems_P7[i]) 
        
        mi_loss["loss_mi_%s" % i] = (mi_loss_p3 + mi_loss_p4 + mi_loss_p5 + mi_loss_p6 + mi_loss_p7) * cfg.MODEL.ADV.MI_LAMBDA
        
    
    for i in range(num_classes):
        targets_features_P3_i, _, _ = memory(P3_neck_t, (3,i), only_update = False, fusion_keys=fusion_mitems_P3[i])
        sources_features_P3_i, _, _ = memory(P3_neck_s, (3,i), only_update = False, fusion_keys=fusion_mitems_P3[i])
        targets_features_P3.append(targets_features_P3_i)
        sources_features_P3.append(sources_features_P3_i)
        
        targets_features_P4_i, _, _ = memory(P4_neck_t, (4,i), only_update = False, fusion_keys=fusion_mitems_P4[i])
        sources_features_P4_i, _, _ = memory(P4_neck_s, (4,i), only_update = False, fusion_keys=fusion_mitems_P4[i])
        targets_features_P4.append(targets_features_P4_i)
        sources_features_P4.append(sources_features_P4_i)
        
        
        targets_features_P5_i, _, _ = memory(P5_neck_t, (5,i), only_update = False, fusion_keys=fusion_mitems_P5[i])
        sources_features_P5_i, _, _ = memory(P5_neck_s, (5,i), only_update = False, fusion_keys=fusion_mitems_P5[i])
        targets_features_P5.append(targets_features_P5_i)
        sources_features_P5.append(sources_features_P5_i)
        
        targets_features_P6_i, _, _ = memory(P6_neck_t, (6,i), only_update = False, fusion_keys=fusion_mitems_P6[i])
        sources_features_P6_i, _, _ = memory(P6_neck_s, (6,i), only_update = False, fusion_keys=fusion_mitems_P6[i])
        targets_features_P6.append(targets_features_P6_i)
        sources_features_P6.append(sources_features_P6_i)
        
        targets_features_P7_i, _, _ = memory(P7_neck_t, (7,i), only_update = False, fusion_keys=fusion_mitems_P7[i])
        sources_features_P7_i, _, _ = memory(P7_neck_s, (7,i), only_update = False, fusion_keys=fusion_mitems_P7[i])
        targets_features_P7.append(targets_features_P7_i)
        sources_features_P7.append(sources_features_P7_i)
        
        
    P3_masks = gen_class_masks(targets, P3_neck_s, num_classes)
    P4_masks = gen_class_masks(targets, P4_neck_s, num_classes)
    P5_masks = gen_class_masks(targets, P5_neck_s, num_classes)
    P6_masks = gen_class_masks(targets, P6_neck_s, num_classes)
    P7_masks = gen_class_masks(targets, P7_neck_s, num_classes)
    
    contra_P3 = model["contra_P3"]
    contra_P4 = model["contra_P4"]
    contra_P5 = model["contra_P5"]
    contra_P6 = model["contra_P6"]
    contra_P7 = model["contra_P7"]
    
    for i in range(num_classes):
        sources_features_P3[i], targets_features_P3[i] = contra_P3(sources_features_P3[i], targets_features_P3[i])
        sources_features_P4[i], targets_features_P4[i] = contra_P4(sources_features_P4[i], targets_features_P4[i])
        sources_features_P5[i], targets_features_P5[i] = contra_P5(sources_features_P5[i], targets_features_P5[i])
        sources_features_P6[i], targets_features_P6[i] = contra_P6(sources_features_P6[i], targets_features_P6[i])
        sources_features_P7[i], targets_features_P7[i] = contra_P7(sources_features_P7[i], targets_features_P7[i])
    
    triplet_losses = {}
    for i in range(num_classes):
        positive_P3_features, negative_P3_features = process_memory_features(sources_features_P3[i], P3_masks[i])
        triplet_loss_P3 = triplet_loss(targets_features_P3[i], positive_P3_features, negative_P3_features)

        positive_P4_features, negative_P4_features = process_memory_features(sources_features_P4[i], P4_masks[i])
        triplet_loss_P4 = triplet_loss(targets_features_P4[i], positive_P4_features, negative_P4_features)

        positive_P5_features, negative_P5_features = process_memory_features(sources_features_P5[i], P5_masks[i])
        triplet_loss_P5 = triplet_loss(targets_features_P5[i], positive_P5_features, negative_P5_features)
        
        positive_P6_features, negative_P6_features = process_memory_features(sources_features_P6[i], P6_masks[i])
        triplet_loss_P6 = triplet_loss(targets_features_P6[i], positive_P6_features, negative_P6_features)
        
        positive_P7_features, negative_P7_features = process_memory_features(sources_features_P7[i], P7_masks[i])
        triplet_loss_P7 = triplet_loss(targets_features_P7[i], positive_P7_features, negative_P7_features)
        
        triplet_losses["loss_contrast_%s" % i] = (triplet_loss_P3 + triplet_loss_P4 + triplet_loss_P5 + triplet_loss_P6 + triplet_loss_P7) * cfg.MODEL.ADV.CONTRAST_LAMBDA
            
    return mi_loss, triplet_losses


def foward_detector(cfg, model, images, targets=None, return_maps=True,  DA_ON=True):
    use_scale_category_dis = cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY
    use_mi = cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_MEMORY
    use_fusion = cfg.MODEL.ADV.USE_MI and cfg.MODEL.ADV.USE_MEMORY and cfg.MODEL.ADV.USE_FUSION
    use_contrast = cfg.MODEL.ADV.USE_MEMORY  and cfg.MODEL.ADV.USE_CONTRAST
    map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
    feature_layers = map_layer_to_index.keys()
    model_backbone = model["backbone"]
    model_fcos = model["fcos"]

    if model_fcos.training and DA_ON:
        losses = {}
        images_s, images_t = images
        features_s = model_backbone(images_s.tensors)
        features_t = model_backbone(images_t.tensors)
        proposals, proposal_losses, _ = model_fcos(
            images_s, features_s, targets=targets)

        # Convert feature representations
        f_s = {
            layer: features_s[map_layer_to_index[layer]]
            for layer in feature_layers
        }
        f_t = {
            layer: features_t[map_layer_to_index[layer]]
            for layer in feature_layers
        }

        losses.update(proposal_losses)
        
        if not use_scale_category_dis:
            if use_mi and use_contrast and use_fusion:
                mi_loss, contrast_loss = forward_mi_contrast_fusion(cfg, model, targets=targets, features_FPN=(features_s, features_t))
                losses.update(mi_loss)
                losses.update(contrast_loss)
            return losses, (f_s, f_t), ()


    elif model_fcos.training  and not DA_ON:
        losses = {}
        features = model_backbone(images.tensors)
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets)
        losses.update(proposal_losses)
        return losses, []
    else:
        images = to_image_list(images)
        features = model_backbone(images.tensors)
        proposals, proposal_losses, score_maps = model_fcos(
            images, features, targets=targets, return_maps=return_maps)

        return proposals


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        # for i, loss in enumerate(all_losses):
        #     print(f"Element {i}: Type={type(loss)}, Value={loss}")
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def validataion(cfg, model, data_loader, distributed=False):
    if distributed:
        model["backbone"] = model["backbone"].module
        model["fcos"] = model["fcos"].module
    iou_types = ("bbox",)
    dataset_name = cfg.DATASETS.TEST
    assert len(data_loader) == 1, "More than one validation sets!"
    data_loader = data_loader[0]
    # for  dataset_name, data_loader_val in zip( dataset_names, data_loader):
    results, _ = _inference(
        cfg,
        model,
        data_loader,
        dataset_name=dataset_name,
        iou_types=iou_types,
        box_only=False if cfg.MODEL.ATSS_ON or cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
    )
    synchronize()
    return results

def do_train(
        model,
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
):
    with_DA = cfg.MODEL.DA_ON
    data_loader_source = data_loader["source"]
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
    for k in model:
        model[k].train()
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    AP50 = cfg.SOLVER.INITIAL_AP50
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    print('DA_ON: {}'.format(str(with_DA)))
    if not with_DA: # without domain adaptation (we report the results from https://chengchunhsu.github.io/EveryPixelMatters/)
        max_iter = len(data_loader_source)
        for iteration, (images_s,targets_s, _) in enumerate(data_loader_source, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            if not pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()
            images_s = images_s.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]
            for k in optimizer:
                optimizer[k].zero_grad()

            loss_dict, features_s = foward_detector(cfg, model, images_s, targets=targets_s, DA_ON=False)
            loss_dict = {k + "_gs": loss_dict[k] for k in loss_dict}
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_gs=losses_reduced, **loss_dict_reduced)
            losses.backward()

            for k in optimizer:
                optimizer[k].step()
            if pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()
            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, ))
            if cfg.SOLVER.ADAPT_VAL_ON:
                if iteration % cfg.SOLVER.VAL_ITER == 0:
                    val_results = validataion(cfg, model, data_loader["val"], distributed)
                    # used for saving model
                    AP50_emp = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # used for logging
                    meter_AP50= val_results.results['bbox']['AP50'] * 100
                    meter_AP = val_results.results['bbox']['AP']* 100
                    meters.update(AP = meter_AP, AP50 = meter_AP50 )

                    if AP50_emp > AP50:
                        AP50 = AP50_emp
                        checkpointer.save("model_{}_{:07d}".format(AP50, iteration), **arguments)
                        print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50,
                                                                                           iteration))
                    if distributed:
                        model["backbone"] = model["backbone"].module
                        model["fcos"] = model["fcos"].module
                    for k in model:
                        model[k].train()
            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # save the last model
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
    else: # With domain adaptation
        data_loader_target = data_loader["target"]
        max_iter = max(len(data_loader_source), len(data_loader_target))
        ga_dis_lambda = arguments["ga_dis_lambda"]
        used_feature_layers = arguments["use_feature_layers"]

        assert len(data_loader_source) == len(data_loader_target)
        
        use_scale_category_dis = cfg.MODEL.ADV.USE_DIS_SCALE_CATEGORY and cfg.MODEL.ADV.USE_MEMORY
        
        for iteration, ((images_s, targets_s, _), (images_t, targets_t, _)) in enumerate(zip(data_loader_source, data_loader_target), start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration
            if not pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()

            images_s = images_s.to(device)
            images_t = images_t.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]

            for k in optimizer:
                optimizer[k].zero_grad()

            loss_dict, features_s_t , cls_feature_s_t = foward_detector(cfg,
                model, (images_s, images_t), targets=targets_s, return_maps=True)

            for layer in used_feature_layers:
                features_s, features_t = features_s_t
                loss_dict["loss_adv_%s" % layer] = \
                    ga_dis_lambda * model["dis_%s" % layer]((features_s[layer],features_t[layer]))
                
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_ds=losses_reduced, **loss_dict_reduced)

            losses.backward()
            del loss_dict, losses

            for k in optimizer:
                optimizer[k].step()

            if pytorch_1_1_0_or_later:
                for k in scheduler:
                    scheduler[k].step()

            # End of training
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            sample_layer = used_feature_layers[0]  # sample any one of used feature layer
            sample_optimizer = optimizer["dis_%s" % sample_layer]

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr_backbone: {lr_backbone:.6f}",
                        #"ls_dis_SCA: {ls_dis_SCA_:.6f}",
                        "lr_fcos: {lr_fcos:.6f}",
                        "lr_dis: {lr_dis:.6f}",
                        "max mem: {memory:.0f}",
                    ]).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_backbone=optimizer["backbone"].param_groups[0]["lr"],
                        #ls_dis_SCA=optimizer["middle_head"].param_groups[0]["lr"],
                        lr_fcos=optimizer["fcos"].param_groups[0]["lr"],
                        lr_dis=sample_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    ))
            if cfg.SOLVER.ADAPT_VAL_ON:

                # This is similar to https://github.com/ChrisAllenMing/GPA-detection/blob/master/iterative_test.py to get a benchmark result,
                # Without this iterative validation, we even cannot reproduce our baseline results (EPM).

                if iteration % cfg.SOLVER.VAL_ITER== 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
                    #val_results = validataion(cfg, model, data_loader["val"], distributed)
                    #val_results = validataion(cfg, model, data_loader["val"], distributed)
                    # # used for saving model
                    # AP50_emp = val_results.results['bbox'][cfg.SOLVER.VAL_TYPE] * 100
                    # # used for logging
                    # meter_AP50 = val_results.results['bbox']['AP50'] * 100
                    # meter_AP = val_results.results['bbox']['AP'] * 100
                    # meters.update(AP=meter_AP, AP50=meter_AP50)
                    # if (AP50_emp > AP50): # saving better models
                    # # if (AP50_emp > AP50) or (AP50_emp> 40.8): # saving more models
                    #     if  (AP50_emp > AP50):
                    #         AP50 = AP50_emp
                    #     checkpointer.save("model_{}_{:07d}".format(AP50_emp, iteration), **arguments)
                    #     print('***warning****,\n best model updated. {}: {}, iter: {}'.format(cfg.SOLVER.VAL_TYPE, AP50_emp, iteration))
                    # if distributed:
                    #     model["backbone"] = model["backbone"].module
                    #     model["fcos"] = model["fcos"].module
                    # for k in model:
                    #         model[k].train()
            else:
                if iteration % checkpoint_period == 0:
                    checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # if iteration % 1000 == 0:
            #     checkpointer.save("model_check_{}".format(iteration), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))
