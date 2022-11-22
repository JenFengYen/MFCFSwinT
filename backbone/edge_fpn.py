#This file has been modified by the author
import torch
from torch import nn
import torch.nn.functional as F


class EdgeFPN(nn.Module):

    def __init__(self,cfg):
        super(EdgeFPN, self).__init__()
        self.freeze_edge = False
        if 'FREEZE_EDGE' in cfg.MODEL:
            self.freeze_edge = cfg.MODEL.FREEZE_EDGE
        self.Conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        self.final_Conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.seg_Conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
        self._freeze_stages()


    def forward(self, origine_features, gt_Edge):

        edge_seg0 = self.Conv0(origine_features['p2'])
        b , f , w , h = edge_seg0.shape
        edge_seg1 = self.Conv1(origine_features['p3'])
        edge_seg1 = F.interpolate(edge_seg1, size=[w, h], mode="bilinear")

        edge_seg2 = self.Conv2(origine_features['p4'])
        edge_seg2 = F.interpolate(edge_seg2, size=[w, h], mode="bilinear")

        edge_seg3 = self.Conv3(origine_features['p5'])
        edge_seg3 = F.interpolate(edge_seg3, size=[w, h], mode="bilinear")

        edge_seg4 = self.Conv4(origine_features['p6'])
        
        edge_seg4 = F.interpolate(edge_seg4, size=[w, h], mode="bilinear")
        #edge_seg4 = F.interpolate(edge_seg4, scale_factor=16, mode="bilinear")
        edge_feature = edge_seg0 + edge_seg1 + edge_seg2 + edge_seg3 + edge_seg4

        edge_feature = F.relu(self.final_Conv0(edge_feature))
        edge_feature = F.relu(self.final_Conv1(edge_feature))
        edge_feature = F.relu(self.final_Conv2(edge_feature))
        edge_feature = F.relu(self.final_Conv3(edge_feature))
        
        if self.training:

            edge_seg_feature = self.seg_Conv(edge_feature)

            b, w, h = gt_Edge.shape
            #edge_seg_feature = F.interpolate(edge_seg_feature, scale_factor=[4], mode="bilinear")
            edge_seg_feature = F.interpolate(edge_seg_feature, size=[w,h], mode="bilinear")
            #edgev3 使用weight
            posiitive_num = torch.sum(gt_Edge > 0)
            negative_num = torch.sum(gt_Edge == 0)
            total_num = posiitive_num + negative_num
            total_num = total_num.float()
            weight = torch.tensor([posiitive_num / total_num, negative_num/ total_num]).float().cuda()
            #print(edge_seg_feature.shape)
            #print(gt_edge.shape)
            #print(gt_edge.shape)
            gt_Edge = gt_Edge.long()
            loss_edge = F.cross_entropy(edge_seg_feature, gt_Edge, weight=weight)


            loss = loss_edge

            # loss = loss_salient_segmentation0 + loss_salient_segmentation1 + loss_salient_segmentation2 + \
            #        loss_salient_segmentation3 + loss_salient_segmentation4
            predict_contour = F.softmax(edge_seg_feature, dim=1)[:, 1, :, :].unsqueeze(0)


            return edge_feature, dict(loss_edge=loss), predict_contour
        # salient_seg0 = self.Conv0(salient_features[0])
        # salient_seg0 = F.interpolate(salient_seg0, scale_factor=4, mode="bilinear")
        # salient_seg_feature = self.seg_Conv(salient_feature)
        # salient_seg_feature = F.interpolate(salient_seg_feature, scale_factor=16, mode="bilinear")
        # salient_seg_feature = F.softmax(salient_seg_feature, dim=1)
        #

        edge_seg_feature = self.seg_Conv(edge_feature)
        edge_seg_feature = F.interpolate(edge_seg_feature, scale_factor=4, mode="bilinear")
        predict_contour = F.softmax(edge_seg_feature, dim=1)[:, 1, :, :].unsqueeze(0)
        return edge_feature, {}, predict_contour
    def _freeze_stages(self):
        freeze_list = [self.Conv0,self.Conv1,self.Conv2,self.Conv3,self.Conv4,self.final_Conv0,self.final_Conv1,self.final_Conv2,self.final_Conv3,self.seg_Conv]
        if self.freeze_edge:
            for block in freeze_list:
                for param in block.parameters():
                    param.requires_grad = False