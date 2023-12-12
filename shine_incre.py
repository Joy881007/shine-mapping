import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import copy
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.pose import *
from utils.incre_learning import cal_feature_importance
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def run_shine_mapping_incremental():

    config = SHINEConfig()
    # if len(sys.argv) > 1:
    #     config.load(sys.argv[1])
    # else:
    #     sys.exit(
    #         "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
    #     )
    # use for debugging
    config_path = "/media/yunan/hdd/research/SHINE_mapping/config/kitti/kitti_incre_reg.yaml"
    config.load(config_path)

    run_path = setup_experiment(config)
    # shutil.copy2(sys.argv[1], run_path) # copy the config file to the result folder
    
    dev = config.device

    # initialize the feature octree
    octree = FeatureOctree(config)
    # initialize the mlp decoder
    geo_mlp = Decoder(config, is_geo_encoder=True)
    sem_mlp = Decoder(config, is_geo_encoder=False)

    # Load the decoder model
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        print("Pretrained decoder loaded")
        freeze_model(geo_mlp) # fixed the decoder
        if config.semantic_on:
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
            freeze_model(sem_mlp) # fixed the decoder
        if 'feature_octree' in loaded_model.keys(): # also load the feature octree  
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    # dataset
    dataset = LiDARDataset(config, octree)

    # mesh reconstructor
    mesher = Mesher(config, octree, geo_mlp, sem_mlp)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Non-blocking visualizer
    if config.o3d_vis_on:
        vis = MapVisualizer()

    # learnable parameters
    geo_mlp_param = list(geo_mlp.parameters())
    sem_mlp_param = list(sem_mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    processed_frame = 0
    total_iter = 0
    if config.continual_learning_reg:
        config.loss_reduction = "sum" # other-wise "mean"

    if config.normal_loss_on or config.ekional_loss_on or config.proj_correction_on or config.consistency_loss_on:
        require_gradient = True
    else:
        require_gradient = False

    #[Joy] **** testing **** Trying to let the pose are read in here
    if config.calib_path != '':
        calib = read_calib_file(config.calib_path)
    else:
        calib['Tr'] = np.eye(4)
    if config.pose_path.endswith('txt'):
        poses_w = read_poses_file(config.pose_path, calib)
    elif config.pose_path.endswith('csv'):
        poses_w = csv_odom_to_transforms(config.pose_path)
    else:
        sys.exit(
        "Wrong pose file format. Please use either *.txt (KITTI format) or *.csv (xyz+quat format)"
        )

    pose_ref = poses_w

    # for each frame
    for frame_id in tqdm(range(25, dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0): 
            continue
        
        vis_mesh = False 

        if processed_frame == config.freeze_after_frame: # freeze the decoder after certain frame
            print("Freeze the decoder")
            freeze_model(geo_mlp) # fixed the decoder
            if config.semantic_on:
                freeze_model(sem_mlp) # fixed the decoder

        T0 = get_time()
        # preprocess, sample data and update the octree
        # if continual_learning_reg is on, we only keep the current frame's sample in the data pool,
        # otherwise we accumulate the data pool with the current frame's sample

        local_data_only = False # this one would lead to the forgetting issue

        # [Joy] **** testing **** 
        cur_pose_ref = pose_ref[frame_id]
        prev_pose_ref = pose_ref[frame_id]
        # if frame_id != 0:
        #     prev_pose_ref = pose_ref[frame_id - 1]

        dataset.process_frame(frame_id, cur_pose_ref, incremental_on=config.continual_learning_reg or local_data_only)

        # load two framwes of point cloud (support *pcd, *ply and kitti *bin format)
        # if frame_id == 0:
        #     if not dataset.config.semantic_on:
        #         frame_filename_prev = os.path.join(dataset.config.pc_path, dataset.pc_filenames[frame_id])
        #         frame_filename_cur = os.path.join(dataset.config.pc_path, dataset.pc_filenames[frame_id])
        #         pc_s = dataset.read_point_cloud(frame_filename_prev)
        #         pc_t = dataset.read_point_cloud(frame_filename_cur)
        #     else:
        #         label_filename_prev = os.path.join(dataset.config.label_path, dataset.pc_filenames[frame_id].replace('bin', 'label'))
        #         label_filename_cur = os.path.join(dataset.config.label_path, dataset.pc_filenames[frame_id].replace('bin', 'label'))
        #         pc_s = dataset.read_semantic_point_label(frame_filename_prev, label_filename_prev)
        #         pc_t = dataset.read_semantic_point_label(frame_filename_cur, label_filename_cur)
        if not dataset.config.semantic_on:
            frame_filename_prev = os.path.join(dataset.config.pc_path, dataset.pc_filenames[frame_id - 1])
            frame_filename_cur = os.path.join(dataset.config.pc_path, dataset.pc_filenames[frame_id])
            pc_s = dataset.read_point_cloud(frame_filename_prev)
            pc_t = dataset.read_point_cloud(frame_filename_cur)          
        else:
            label_filename_prev = os.path.join(dataset.config.label_path, dataset.pc_filenames[frame_id - 1].replace('bin','label'))
            label_filename_cur = os.path.join(dataset.config.label_path, dataset.pc_filenames[frame_id].replace('bin','label'))
            pc_s = dataset.read_semantic_point_label(frame_filename_prev, label_filename_prev)
            pc_t = dataset.read_semantic_point_label(frame_filename_cur, label_filename_cur)

        # Point cloud registration using open3d-icp
        threshold = 1.0
        # trans_init = prev_pose_ref
        trans_init = pose_ref[0]
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     pc_s, pc_t, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
            pc_s, pc_t, threshold, trans_init, 
            estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria =  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
        

        print("Frame id:", frame_id)
        print("Transformation is:")
        print(reg_gicp.transformation)
        # if (frame_id % 10 == 0 and frame_id != 0):
        #     draw_registration_result(pc_s, pc_t, reg_gicp.transformation)

        # [Joy] **** Testing ****
        # use function by Si-yu to calaulate the loss of point cloud registration transformation
        # prev_pose_ref = pose_ref[frame_id - 1]
        trans_cur = reg_gicp.transformation
        trans_ref = np.dot(inv(cur_pose_ref), prev_pose_ref)
        trans_error = calc_pose_error(trans_cur, trans_ref)

        print("Transformation Error is:", trans_error, "\n")
        _, trans_error = calc_pose_error(trans_ref, trans_cur)

        if (frame_id % 5 == 0):
            draw_registration_result(pc_s, pc_t, reg_gicp.transformation)
        
        octree_feat = list(octree.parameters())
        opt = setup_optimizer(config, octree_feat, geo_mlp_param, sem_mlp_param, sigma_size)
        octree.print_detail()

        T1 = get_time()

        for iter in tqdm(range(config.iters)):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, _, _, _, sem_label, weight = dataset.get_batch() 
            
            if require_gradient:
                coord.requires_grad_(True)

            # interpolate and concat the hierachical grid features
            feature = octree.query_feature(coord)
            
            # predict the scaled sdf with the feature
            sdf_pred = geo_mlp.sdf(feature)
            if config.semantic_on:
                sem_pred = sem_mlp.sem_label_prob(feature)

            # calculate the loss
            surface_mask = weight > 0

            if require_gradient:
                g = get_gradient(coord, sdf_pred)*sigma_sigmoid

            if config.consistency_loss_on:
                near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count,coord.shape[0]),), device=dev)
                shift_scale = config.consistency_range * config.scale # 10 cm
                random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
                coord_near = coord + random_shift 
                coord_near = coord_near[near_index, :] # only use a part of these coord to speed up
                coord_near.requires_grad_(True)
                feature_near = octree.query_feature(coord_near)
                pred_near = geo_mlp.sdf(feature_near)
                g_near = get_gradient(coord_near, pred_near)*sigma_sigmoid

            cur_loss = 0.
            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
            cur_loss += sdf_loss

            # incremental learning regularization loss 
            reg_loss = 0.
            if config.continual_learning_reg:
                reg_loss = octree.cal_regularization()
                cur_loss += config.lambda_forget * reg_loss

            # optional ekional loss
            eikonal_loss = 0.
            if config.ekional_loss_on: # MSE with regards to 1  
                # eikonal_loss = ((g.norm(2, dim=-1) - 1.0) ** 2).mean() # both the surface and the freespace
                # eikonal_loss = ((g[~surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # only the freespace
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # only close to the surface
                cur_loss += config.weight_e * eikonal_loss
            
            consistency_loss = 0.
            if config.consistency_loss_on:
                consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
                cur_loss += config.weight_c * consistency_loss
            
            # semantic classification loss
            sem_loss = 0.
            if config.semantic_on:
                loss_nll = nn.NLLLoss(reduction='mean')
                sem_loss = loss_nll(sem_pred[::config.sem_label_decimation,:], sem_label[::config.sem_label_decimation])
                cur_loss += config.weight_s * sem_loss

            opt.zero_grad(set_to_none=True)
            cur_loss.backward() # this is the slowest part (about 10x the forward time)
            opt.step()

            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {'iter': total_iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, \
                    'loss/reg_loss':reg_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/consistency_loss': consistency_loss, 'loss/sem_loss': sem_loss} 
                wandb.log(wandb_log_content)
        
        # calculate the importance of each octree feature
        if config.continual_learning_reg:
            opt.zero_grad(set_to_none=True)
            cal_feature_importance(dataset, octree, geo_mlp, sigma_sigmoid, config.bs, \
                config.cal_importance_weight_down_rate, config.loss_reduction)

        T2 = get_time()
        
        # reconstruction by marching cubes
        if processed_frame == 0 or (processed_frame+1) % config.mesh_freq_frame == 0:
            print("Begin mesh reconstruction from the implicit map")       
            vis_mesh = True 
            # print("Begin reconstruction from implicit mapn")               
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id+1) + ".ply"
            map_path = run_path + '/map/sdf_map_frame_' + str(frame_id+1) + ".ply"
            if config.mc_with_octree: # default
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
            else:
                if config.mc_local: # only build the local mesh to speed up
                    cur_mesh = mesher.recon_bbx_mesh(dataset.cur_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        T3 = get_time()

        if config.o3d_vis_on:
            if vis_mesh: 
                cur_mesh.transform(dataset.begin_pose_inv) # back to the globally shifted frame for vis
                vis.update(dataset.cur_frame_pc, dataset.cur_pose, cur_mesh)
            else: # only show frame and current point cloud
                vis.update(dataset.cur_frame_pc, dataset.cur_pose)

            # visualize the octree (it is a bit slow and memory intensive for the visualization)
            # vis_octree = True
            # if vis_octree: 
            #     cur_mesh.transform(dataset.begin_pose_inv)
            #     vis_list = [] # create a list of bbx for the octree nodes
            #     for l in range(config.tree_level_feat):
            #         nodes_coord = octree.get_octree_nodes(config.tree_level_world-l)/config.scale
            #         box_size = np.ones(3) * config.leaf_vox_size * (2**l)
            #         for node_coord in nodes_coord:
            #             node_box = o3d.geometry.AxisAlignedBoundingBox(node_coord-0.5*box_size, node_coord+0.5*box_size)
            #             node_box.color = random_color_table[l]
            #             vis_list.append(node_box)
            #     vis_list.append(cur_mesh)
            #     o3d.visualization.draw_geometries(vis_list)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': processed_frame, 'timing(s)/preprocess': T1-T0, 'timing(s)/mapping': T2-T1, 'timing(s)/reconstruct': T3-T2} 
            wandb.log(wandb_log_content)

        processed_frame += 1
    
    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_shine_mapping_incremental()