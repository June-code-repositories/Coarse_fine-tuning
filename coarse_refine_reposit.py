import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import scipy.io
import numpy as np
from models.Refine_models import *
from models import quaternion
from torch.optim.lr_scheduler import MultiStepLR


current_dir = os.getcwd()
data_folder = os.path.join(current_dir, "data")

filename = "Fiducial_data_CT.mat"
file_path = os.path.join(data_folder, filename)
data = scipy.io.loadmat(file_path)
tensor_data = torch.tensor(data['Rot_PC'])
sampled_tensor = tensor_data   

filename = "Target_data_CT.mat"
file_path = os.path.join(data_folder, filename)
data = scipy.io.loadmat(file_path)
target_db = torch.tensor(data['Rot_Candi'])
    
class FiducialPointCloudDataset(Dataset):
    def __init__(self, data, target_db, start_idx=0, end_idx=None):
        self.input_data = data[6:, :, start_idx:end_idx]  # Shape: (10, 4, end_idx-start_idx)
        self.target_data = data[0:6, :, start_idx:end_idx]  # Shape: (1, 4, end_idx-start_idx)
        self.target_db = target_db[:, :, start_idx:end_idx]  # Shape: (1, 4, end_idx-start_idx)

    def __len__(self):
        return self.input_data.shape[2]

    def __getitem__(self, idx):
        return self.input_data[:,:,idx], self.target_data[:,:,idx], self.target_db[:,:,idx] 
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
     
    @staticmethod
    def chamfer_distance(pc1, pc2):
        """
        Compute the Chamfer Distance between two point clouds.
    
        Args:
        - pc1: Tensor of shape (B, N, 3) representing the first point cloud
        - pc2: Tensor of shape (B, M, 3) representing the second point cloud
    
        Returns:
        - chamfer_dist: Tensor of shape (B,) containing the Chamfer Distance for each batch
        """       
        
        dist = torch.cdist(pc1, pc2, p=2)

        # Find minimum distance from each point in pc1 to pc2
        min_dist_pc1_to_pc2 = torch.min(dist, dim=-1)[0]  # (B, N)
        
        # Find minimum distance from each point in pc2 to pc1
        min_dist_pc2_to_pc1 = torch.min(dist, dim=-2)[0]  # (B, M)
        
        # Compute the Chamfer Distance
        chamfer_dist = torch.mean(min_dist_pc1_to_pc2, dim=-1) + torch.mean(min_dist_pc2_to_pc1, dim=-1)  # (B,)
        
        
        return torch.mean(chamfer_dist)

    @staticmethod
    def angular_distance_loss(pred_quat, target_quat, eps=1e-7):
        pred_quat = F.normalize(pred_quat, dim=-1)
        target_quat = F.normalize(target_quat, dim=-1)
        
        dot_product = torch.sum(pred_quat * target_quat, dim=-1)
        dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
        
        distance = 2 * torch.acos(torch.abs(dot_product))
        return distance.mean()
    
    @staticmethod
    def normalize_quatern(q):
        return q / q.norm(dim=-1, keepdim=True)
    
    @staticmethod            
    def quat2euler_zyx(q):
        """
        Convert quaternion to Euler angles (ZYX order).
        
        Args:
        q (torch.Tensor): Quaternion tensor of shape (..., 4) where last dimension is [qw, qx, qy, qz]
        
        Returns:
        torch.Tensor: Euler angles tensor of shape (..., 3) where last dimension is [z, y, x]
        """
        
        min_values = torch.tensor([-14.3309,-14.3309,-14.3309]) # Male 20240808
        max_values = torch.tensor([10.7794,10.7794,10.7794])       
        
        #min_values = torch.tensor([-10.7488,-10.7488,-10.7488]) # Male 20240810
        #max_values = torch.tensor([13.8192,13.8192,13.8192])        
                
        min_values = min_values.view(1, -1).cuda()
        max_values = max_values.view(1, -1).cuda()
        
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Calculate asin input
        asin_input = -2 * (qx * qz - qw * qy)
        
        # Create masks for clamping
        eps = torch.finfo(q.dtype).eps
        mask1 = asin_input >= 1 - 10 * eps
        mask2 = asin_input <= -1 + 10 * eps
        
        # Clamp asin input
        asin_input = torch.clamp(asin_input, -1 + 10 * eps, 1 - 10 * eps)
        
        # Calculate Euler angles
        eul = torch.stack([
            torch.atan2(2 * (qx * qy + qw * qz), qw**2 + qx**2 - qy**2 - qz**2),
            torch.asin(asin_input),
            torch.atan2(2 * (qy * qz + qw * qx), qw**2 - qx**2 - qy**2 + qz**2)
        ], dim=-1)
        
        # Handle singularity cases
        mask = mask1 | mask2
        eul[mask, 0] = -torch.sign(asin_input[mask]) * 2 * torch.atan2(qx[mask], qw[mask])
        eul[mask, 2] = 0
        
        euler_angles_deg = torch.rad2deg(eul)
         
        scaled_angles = ((euler_angles_deg - min_values) / (max_values - min_values) )
        
        return scaled_angles
    
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([Encoder() for _ in range(4)])
        self.fusion = nn.ModuleList([Fusion() for _ in range(4)])
        self.regression = nn.ModuleList([Regression() for _ in range(4)])

    def forward(self, coarse, batch_db):

        xyz_src = coarse[:, :, :3]
        xyz_ref = batch_db[:, :, :3]
        

        # init params
        B = xyz_src.size()[0]
        init_quatern = torch.tensor([1, 0, 0, 0]).expand(B, 4)
        init_translation = torch.from_numpy(np.array([[0., 0., 0.]])).expand(B, 3)
        
        pose_pred = torch.cat((init_quatern, init_translation), dim=1).float().cuda()  # B, 7
        
        translations = []
        transforms = []
        eulers = []
        rotations = []

        # rename
        xyz_src_iter = xyz_src

        min_trans = torch.tensor([-0.0461,-0.0461,-0.0461])
        max_trans = torch.tensor([0.0440,0.0440,0.0440]) 
           
        min_trans = min_trans.view(1, -1).cuda()
        max_trans = max_trans.view(1, -1).cuda()


        for i in range(4):
            # encoder
            
            encoder = self.encoder[i] # encoder = encoder[i]

            enc_input = torch.cat((xyz_src_iter.transpose(1, 2).detach(), xyz_ref.transpose(1, 2)), dim=0)  # 2B, C, N

            enc_feats = encoder(enc_input)
            
            src_enc_feats = [feat[:B, ...] for feat in enc_feats]
            ref_enc_feats = [feat[B:, ...] for feat in enc_feats]
            enc_src_R_feat = src_enc_feats[0]  # B, C
            enc_src_t_feat = src_enc_feats[1]  # B, C
            enc_ref_R_feat = ref_enc_feats[0]  # B, C
            enc_ref_t_feat = ref_enc_feats[1]  # B, C

            # fusion
            src_R_cat_feat = torch.cat((enc_src_R_feat, enc_ref_R_feat), dim=-1)  # B, 2C
            ref_R_cat_feat = torch.cat((enc_ref_R_feat, enc_src_R_feat), dim=-1)  # B, 2C
            src_t_cat_feat = torch.cat((enc_src_t_feat, enc_ref_t_feat), dim=-1)  # B, 2C
            ref_t_cat_feat = torch.cat((enc_ref_t_feat, enc_src_t_feat), dim=-1)  # B, 2C
            fusion_R_input = torch.cat((src_R_cat_feat, ref_R_cat_feat), dim=0)  # 2B, C
            fusion_t_input = torch.cat((src_t_cat_feat, ref_t_cat_feat), dim=0)  # 2B, C
            
            fusion_feats = self.fusion[i](fusion_R_input, fusion_t_input) # fusion_feats = fusion[i](fusion_R_input, fusion_t_input)
            
            src_fusion_feats = [feat[:B, ...] for feat in fusion_feats]
            ref_fusion_feats = [feat[B:, ...] for feat in fusion_feats]
            src_R_feat = src_fusion_feats[0]  # B, C
            src_t_feat = src_fusion_feats[1]  # B, C
            ref_R_feat = ref_fusion_feats[0]  # B, C
            ref_t_feat = ref_fusion_feats[1]  # B, C
            

            # R feats
            R_feats = torch.cat((src_t_feat, src_R_feat, ref_t_feat, ref_R_feat), dim=-1)  # B, 4C
            
            # t feats
            src_t_feats = torch.cat((src_t_feat, src_R_feat, ref_t_feat), dim=-1)  # B, 3C
            ref_t_feats = torch.cat((ref_t_feat, ref_R_feat, src_t_feat), dim=-1)  # B, 3C
            
            t_feats = torch.cat((src_t_feats, ref_t_feats), dim=0)  # 2B, 3C or 2B, 2C
            
            pred_quat, pred_translate = self.regression[i](R_feats, t_feats) # pred_quat, pred_translate = regression[i](R_feats, t_feats) 
            src_pred_center, ref_pred_center = torch.chunk(pred_translate, 2, dim=0)
            pred_translate = ref_pred_center - src_pred_center

            
            pose_pred_iter = torch.cat((pred_quat, pred_translate), dim=-1)  # B, 7


            xyz_src_iter = quaternion.torch_quat_transform(pose_pred_iter, xyz_src_iter.detach())

            pose_pred = quaternion.torch_transform_pose(pose_pred.detach(), pose_pred_iter)
            
           
            euler = (get_loss.quat2euler_zyx(pose_pred[:,:4]))           
            distance = ((pose_pred[:,4:] - min_trans) / (max_trans - min_trans))
            
            translations.append(distance)
            transforms.append(xyz_src_iter)
            eulers.append(euler)
            rotations.append(pose_pred[:,:4])


        return  translations, transforms, eulers, rotations 


# Split the dataset into training and validation
batch_size=40  
train_dataset = FiducialPointCloudDataset(sampled_tensor, target_db, end_idx=1)
val_dataset = FiducialPointCloudDataset(sampled_tensor, target_db, start_idx=12900, end_idx=13000)

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

# Model, Criterion, and Optimizer Definition
model = Net().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)

scheduler = MultiStepLR(optimizer, milestones=[40,80,120,160,260], gamma=0.2)

global_step = 0
global_epoch = 0

# Training and Validation
epochs = 400

loss_before = 1.0

identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)

for epoch in range(epochs):
    
    # Training Loop
    model.train()

    train_loss = 0.0

    for batch_data, batch_labels, batch_db in train_dataloader:
        optimizer.zero_grad()
        
        batch_data, batch_labels, batch_db = batch_data.cuda(), batch_labels.cuda(), batch_db.cuda()

        batch_db = batch_db[:, :, :].float()
        batch_data = batch_data[:, :, :].float()
        point_cloud_data = batch_data[:, :, :]
        
        coarse = point_cloud_data[:, 8:, :]
        target = point_cloud_data[:, 4:8, :]
        pick = point_cloud_data[:, 0:4, :]
     
        gt_euler = batch_labels[:, 4, :3].float()
        gt_quat = batch_labels[:, 5, :].float()
        gt_RT = batch_labels[:, 0:4, :].float()
        
        
        translation, transforms, euler, rotations  = model(coarse, batch_db)


        loss = {}
        for i in range(4):

            loss["euler_{}".format(i)] = F.mse_loss(euler[i], gt_euler)  * 1

            loss["translate_{}".format(i)] = get_loss.chamfer_distance(transforms[i][:, :4, :3], pick[:, :, :3]) * 1

            loss["translation_{}".format(i)] = F.mse_loss(translation[i] * 1, gt_RT[:,:3,3:].squeeze(-1) * 1)  * 1
            
            loss["quat_{}".format(i)] = get_loss.angular_distance_loss(rotations[i], gt_quat) * 1

        total_loss = []

        for k in loss:
            total_loss.append(loss[k].float())
  
        loss["total"] = torch.sum(torch.stack(total_loss), dim=0)
        losses = loss["total"]
        loss["total"].backward()
        
                
        train_loss += losses.item()
        

        optimizer.step()
        global_step += 1  
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_dataloader):.4f}")


    # Validation Loop
    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        
        euler_list = []
        trans_list = []

        for batch_data, batch_labels, batch_db in val_dataloader:
            
            batch_data, batch_labels, batch_db = batch_data.cuda(), batch_labels.cuda(), batch_db.cuda()

            batch_db = batch_db[:, :, :].float()
            batch_data = batch_data[:, :, :].float()
            point_cloud_data = batch_data[:, :, :]
             
            coarse = point_cloud_data[:, 8:, :]
            target = point_cloud_data[:, 4:8, :]
            pick = point_cloud_data[:, 0:4, :] 
            
            gt_euler = batch_labels[:, 4, :3].float()
            gt_quat = batch_labels[:, 5, :].float()           
            gt_RT = batch_labels[:, 0:4, :].float()
            
            
            translation, transforms, euler, rotations  = model(coarse, batch_db)
    

            loss = {}
            for i in range(4):

                loss["euler_{}".format(i)] = F.mse_loss(euler[i], gt_euler) * 1

                loss["translate_{}".format(i)] = get_loss.chamfer_distance(transforms[i][:, :4, :3], pick[:, :, :3]) * 1

                loss["translation_{}".format(i)] = F.mse_loss(translation[i] * 1, gt_RT[:,:3,3:].squeeze(-1) * 1)  * 1

                loss["quat_{}".format(i)] = get_loss.angular_distance_loss(rotations[i], gt_quat) * 1
        
            total_loss = []

            for k in loss: 
                total_loss.append(loss[k].float())

            loss["total"] = torch.sum(torch.stack(total_loss), dim=0)
            losses = loss["total"]
            
            val_loss += losses.item()
            
            euler_list.append(torch.cat((euler[3], gt_euler), dim=0))   
            trans_list.append(torch.cat((translation[3], gt_RT[:,:3,3:].squeeze(-1)), dim=0))
            
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_dataloader):.4f}, Eul: {loss['euler_{}'.format(3)]:.4f}, Trans: {loss['translation_{}'.format(3)]:.4f}")
        
        if (loss_before >= val_loss/len(val_dataloader)):

            best_euler = euler_list
            
            best_translation = trans_list
            
            best_loss = loss
            
            loss_before = val_loss/len(val_dataloader)
            
            
    scheduler.step()       
    
    global_epoch += 1
    