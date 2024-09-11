import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # R
        self.R_layer1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.R_layer2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.R_layer3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        self.R_layer4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), self.relu)
        self.R_layer5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)


        # t
        self.t_layer1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.t_layer2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.t_layer3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        self.t_layer4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), self.relu)
        self.t_layer5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        B, C, N = x.size()

        # R stage1
        R_feature1 = self.R_layer1(x)
        R_feature2 = self.R_layer2(R_feature1)
        R_feat_glob2 = torch.max(R_feature2, dim=-1, keepdim=True)[0]

        # t stage1
        t_feature1 = self.t_layer1(x)
        t_feature2 = self.t_layer2(t_feature1)
        t_feat_glob2 = torch.max(t_feature2, dim=-1, keepdim=True)[0]

        # exchange1
        src_R_feat_glob2, ref_R_feat_glob2 = torch.chunk(R_feat_glob2, 2, dim=0)
        src_t_feat_glob2, ref_t_feat_glob2 = torch.chunk(t_feat_glob2, 2, dim=0)
        interaction_R_feat = torch.cat((ref_R_feat_glob2.repeat(1, 1, N), src_R_feat_glob2.repeat(1, 1, N)), dim=0)
        interaction_t_feat = torch.cat((ref_t_feat_glob2.repeat(1, 1, N), src_t_feat_glob2.repeat(1, 1, N)), dim=0)
        interaction_R_feat = torch.cat((R_feature2, interaction_R_feat.detach()), dim=1)
        interaction_t_feat = torch.cat((t_feature2, interaction_t_feat.detach()), dim=1)

        # R stage2
        R_feature3 = self.R_layer3(interaction_R_feat)
        R_feature4 = self.R_layer4(R_feature3)
        R_feat_glob4 = torch.max(R_feature4, dim=-1, keepdim=True)[0]

        # t stage2
        t_feature3 = self.t_layer3(interaction_t_feat)
        t_feature4 = self.t_layer4(t_feature3)
        t_feat_glob4 = torch.max(t_feature4, dim=-1, keepdim=True)[0]

        # exchange2
        src_R_feat_glob4, ref_R_feat_glob4 = torch.chunk(R_feat_glob4, 2, dim=0)
        src_t_feat_glob4, ref_t_feat_glob4 = torch.chunk(t_feat_glob4, 2, dim=0)
        interaction_R_feat = torch.cat((ref_R_feat_glob4.repeat(1, 1, N), src_R_feat_glob4.repeat(1, 1, N)), dim=0)
        interaction_t_feat = torch.cat((ref_t_feat_glob4.repeat(1, 1, N), src_t_feat_glob4.repeat(1, 1, N)), dim=0)
        interaction_R_feat = torch.cat((R_feature4, interaction_R_feat.detach()), dim=1)
        interaction_t_feat = torch.cat((t_feature4, interaction_t_feat.detach()), dim=1)

        # R stage3
        R_feature5 = self.R_layer5(interaction_R_feat)

        # t stage3
        t_feature5 = self.t_layer5(interaction_t_feat)


        # final
        R_final_feature = torch.cat((R_feature1, R_feature2, R_feature3, R_feature4, R_feature5), dim=1)
        t_final_feature = torch.cat((t_feature1, t_feature2, t_feature3, t_feature4, t_feature5), dim=1)


        R_glob_feat, R_glob_idx = torch.max(R_final_feature, dim=-1, keepdim=False)
        t_glob_feat, t_glob_idx = torch.max(t_final_feature, dim=-1, keepdim=False)

        return [R_glob_feat, t_glob_feat]



class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # R
        self.R_layer1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), self.relu)
        self.R_layer2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), self.relu)
        self.R_layer3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), self.relu)

        # t
        self.t_layer1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), self.relu)
        self.t_layer2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), self.relu)
        self.t_layer3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), self.relu)

    def forward(self, R_feat, t_feat):
        # R
        fuse_R_feat = self.R_layer1(R_feat)
        fuse_R_feat = self.R_layer2(fuse_R_feat)
        fuse_R_feat = self.R_layer3(fuse_R_feat)
        
        # t
        fuse_t_feat = self.t_layer1(t_feat)
        fuse_t_feat = self.t_layer2(fuse_t_feat)
        fuse_t_feat = self.t_layer3(fuse_t_feat)

        return [fuse_R_feat, fuse_t_feat]


class Regression(nn.Module):
    def __init__(self, ):
        super().__init__()

        R_in_channel = 1024 *4
        t_in_channel = 1024 *3
        
        self.relu = nn.ReLU(inplace=True)

        self.R_net = nn.Sequential(
            # layer 1
            nn.Linear(R_in_channel, 2048),
            nn.BatchNorm1d(2048),
            self.relu,
            # layer 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            self.relu,
            # layer 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.relu,
            # layer 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.relu,
            # final fc
            nn.Linear(256, 4),
        )

        self.t_net = nn.Sequential(
            # layer 1
            nn.Linear(t_in_channel, 2048),
            nn.BatchNorm1d(2048),
            self.relu,
            # layer 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            self.relu,
            # layer 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.relu,
            # layer 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.relu,
            # final fc
            nn.Linear(256, 3),
        )

    def forward(self, R_feat, t_feat):

        pred_quat = self.R_net(R_feat)
        pred_quat = F.normalize(pred_quat, dim=1)

        pred_translate = self.t_net(t_feat)

        return [pred_quat, pred_translate]