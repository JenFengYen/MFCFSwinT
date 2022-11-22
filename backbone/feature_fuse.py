#This file has been modified by the author
import torch
from torch import nn
import torch.nn.functional as F



class Attention_Feature(nn.Module):
    def __init__(self):
        super(Attention_Feature, self).__init__()


    def forward(self, salient_feature):
        """
        :param salient_feature: [1, 256, h, w]
        :return:[1, 256, 1, 1]
        """
        salient_pool_feature0 = F.adaptive_max_pool2d(salient_feature, (1, 1)) #[1, 256, 1, 1]
        salient_pool_feature0 = salient_pool_feature0.view(256, 1) #[256, 1]
        salient_pool_feature0 = salient_pool_feature0.transpose(1, 0) #[1, 256]

        salient_pool_feature1 = F.adaptive_max_pool2d(salient_feature, (4, 5)) #[1, 256, 4, 5]
        salient_pool_feature1 = salient_pool_feature1.view(256, 20) #[256, 20]
        salient_pool_feature1 = salient_pool_feature1.transpose(1, 0) #[20, 256]

        salient_pool_feature2 = F.adaptive_max_pool2d(salient_feature, (8, 10)) #[1, 256, 8, 10]
        salient_pool_feature2 = salient_pool_feature2.view(256, 80) #[256, 80]
        salient_pool_feature2 = salient_pool_feature2.transpose(1, 0) #[80, 256]

        salient_pool_feature3 = F.adaptive_max_pool2d(salient_feature, (15, 20)) #[1, 256, 15, 20]
        salient_pool_feature3 = salient_pool_feature3.view(256, 300) #[256, 300]
        salient_pool_feature3 = salient_pool_feature3.transpose(1, 0) #[300, 256]


        salient_pool_feature = torch.cat(
            (salient_pool_feature0, salient_pool_feature1, salient_pool_feature2, salient_pool_feature3),
            dim=0)  # [401, 256]

        return salient_pool_feature #[401, 256]


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.FC = nn.Linear(256, 1)


    def forward(self, salient_pool_feature):
        """
        :param salient_pool_feature: #[401, 256]
        :return:
        """


        salient_pool_feature_attention = F.softmax(self.FC(salient_pool_feature), dim=0) #[401, 1]
        salient_pool_feature_attention = salient_pool_feature_attention.transpose(1, 0) #[1, 401]

        salient_attention_feature = torch.mm(salient_pool_feature_attention, salient_pool_feature)  # [1, 256]

        return salient_attention_feature # [1, 256]

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, guidance_feature):
        #guidance_feature[1, 256, w, h]


        #print(guidance_feature.shape)
        guidance_pool_feature0 = F.adaptive_max_pool2d(guidance_feature, (1, 1)) #[1, 256, 1, 1]
        #print(guidance_pool_feature0.shape)
        
        guidance_pool_feature0 = guidance_pool_feature0.view(guidance_pool_feature0.shape[0], 256, 1) # [256, 1]
        #print(guidance_pool_feature0.shape)
        guidance_pool_feature0 = guidance_pool_feature0.permute(0, 2, 1) #[1, 256]
        #print(guidance_pool_feature0.shape)
        
        guidance_pool_feature1 = F.adaptive_max_pool2d(guidance_feature, (4, 5)) #[1, 256, 4, 5]
        guidance_pool_feature1 = guidance_pool_feature1.view(guidance_pool_feature1.shape[0],256, 20) # [256, 20]
        guidance_pool_feature1 = guidance_pool_feature1.permute(0, 2, 1) #[20, 256]

        guidance_pool_feature2 = F.adaptive_max_pool2d(guidance_feature, (8, 10)) #[1, 256, 8, 10]
        guidance_pool_feature2 = guidance_pool_feature2.view(guidance_pool_feature2.shape[0],256, 80) # [256, 80]
        guidance_pool_feature2 = guidance_pool_feature2.permute(0, 2, 1) #[80, 256]

        guidance_pool_feature3 = F.adaptive_max_pool2d(guidance_feature, (15, 20))  # [1, 256, 15, 20]
        guidance_pool_feature3 = guidance_pool_feature3.view(guidance_pool_feature3.shape[0],256, 300)  # [256, 300]
        guidance_pool_feature3 = guidance_pool_feature3.permute(0, 2, 1)  # [300, 256]


        guidance_pool_feature = torch.cat((guidance_pool_feature0, guidance_pool_feature1, guidance_pool_feature2, guidance_pool_feature3), dim=1)  # [401, 256]
        #print(guidance_pool_feature.shape)


        return guidance_pool_feature

class Mga(nn.Module):
    def __init__(self):
        super(Mga, self).__init__()
        self.FC = nn.Linear(256, 1)
    def forward(self, pool_feature):
        # pool_feature[401, 256]
        attention_feature = F.softmax(self.FC(pool_feature), dim=0) #[401, 1]
        attention_feature = attention_feature.permute(0, 2, 1)  #[1, 401]
        
        out_feature = torch.matmul(attention_feature, pool_feature)

        return out_feature #[1, 256]






class FeatureFuse(nn.Module):

    def __init__(self, cfg):
        super(FeatureFuse, self).__init__()
        self.freeze_edge = False
        if 'FREEZE_EDGE' in cfg.MODEL:
            self.freeze_edge = cfg.MODEL.FREEZE_EDGE
        self.use_MGA = False
        if 'MGA' in cfg.MODEL:
            if cfg.MODEL.MGA:
                self.use_MGA = cfg.MODEL.MGA
                self.edge_Attention0 = Mga()
                self.edge_Attention1 = Mga()
                self.edge_Attention2 = Mga()
                self.edge_Attention3 = Mga()
                self.edge_Attention4 = Mga()
                self.edge_FC0 = nn.Linear(256, 256)
                self.edge_FC1 = nn.Linear(256, 256)
                self.edge_FC2 = nn.Linear(256, 256)
                self.edge_FC3 = nn.Linear(256, 256)
                self.edge_FC4 = nn.Linear(256, 256)
                self.edge_pool = Pool()
        self.edge_Conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.edge_Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.edge_Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.edge_Conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.edge_Conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self._freeze_stages()
    def forward(self, origine_features, edge_features):
        # salient_features [1, 256, 120, 160]
        # edge_featrues [1, 256, 120, 160]
        if self.use_MGA:
            
            
            edge_feature0 = self.edge_Conv0(edge_features) #[1, 256, 120, 160]
            #print(origine_features.keys())

            _,_,w,h = origine_features['p3'].shape
            edge_feature1 = F.interpolate(self.edge_Conv1(edge_features), size=[w, h], mode="bilinear") #[1, 256, 60, 80]
            _,_,w,h = origine_features['p4'].shape
            edge_feature2 = F.interpolate(self.edge_Conv2(edge_features), size=[w, h], mode="bilinear") #[1, 256, 30, 40]
            _,_,w,h = origine_features['p5'].shape
            edge_feature3 = F.interpolate(self.edge_Conv3(edge_features), size=[w, h], mode="bilinear") #[1, 256, 15, 20]
            _,_,w,h = origine_features['p6'].shape
            edge_feature4 = F.interpolate(self.edge_Conv4(edge_features), size=[w, h], mode="bilinear") #[1, 256, 8, 10]

            edge_pool_features = self.edge_pool(edge_features) #[401, 256]
            edge_pool_feature0 = self.edge_FC0(edge_pool_features)
            edge_pool_feature1 = self.edge_FC1(edge_pool_features)
            edge_pool_feature2 = self.edge_FC2(edge_pool_features)
            edge_pool_feature3 = self.edge_FC3(edge_pool_features)
            edge_pool_feature4 = self.edge_FC4(edge_pool_features)

            b = edge_pool_feature0.shape[0]
            edge_attention_feature0 = self.edge_Attention0(edge_pool_feature0).view(b, 256, 1, 1)
            edge_attention_feature1 = self.edge_Attention1(edge_pool_feature1).view(b, 256, 1, 1)
            edge_attention_feature2 = self.edge_Attention2(edge_pool_feature2).view(b, 256, 1, 1)
            edge_attention_feature3 = self.edge_Attention3(edge_pool_feature3).view(b, 256, 1, 1)
            edge_attention_feature4 = self.edge_Attention4(edge_pool_feature4).view(b, 256, 1, 1)

            features = {}


            feature0 = torch.add(edge_feature0, origine_features['p2'])
            feature0 = torch.add(edge_attention_feature0, feature0)
            features['p2'] = feature0

            feature1 = torch.add(edge_feature1, origine_features['p3'])
            feature1 = torch.add(edge_attention_feature1, feature1)
            features['p3'] = feature1

            feature2 = torch.add(edge_feature2, origine_features['p4'])
            feature2 = torch.add(edge_attention_feature2, feature2)
            features['p4'] = feature2

            feature3 = torch.add(edge_feature3, origine_features['p5'])
            feature3 = torch.add(edge_attention_feature3, feature3)
            features['p5'] = feature3

            feature4 = torch.add(edge_feature4, origine_features['p6'])
            feature4 = torch.add(edge_attention_feature4, feature4)
            features['p6'] = feature4
            return features
        else:
            edge_feature0 = self.edge_Conv0(edge_features) #[1, 256, 120, 160]
            _,_,w,h = origine_features['p3'].shape
            edge_feature1 = F.interpolate(self.edge_Conv1(edge_features), size=[w, h], mode="bilinear") #[1, 256, 60, 80]
            _,_,w,h = origine_features['p4'].shape
            edge_feature2 = F.interpolate(self.edge_Conv2(edge_features), size=[w, h], mode="bilinear") #[1, 256, 30, 40]
            _,_,w,h = origine_features['p5'].shape
            edge_feature3 = F.interpolate(self.edge_Conv3(edge_features), size=[w, h], mode="bilinear") #[1, 256, 15, 20]
            _,_,w,h = origine_features['p6'].shape
            edge_feature4 = F.interpolate(self.edge_Conv4(edge_features), size=[w, h], mode="bilinear") #[1, 256, 8, 10]



            features = {}


            feature0 = torch.add(edge_feature0, origine_features['p2'])
            features['p2'] = feature0

            feature1 = torch.add(edge_feature1, origine_features['p3'])
            features['p3'] = feature1

            feature2 = torch.add(edge_feature2, origine_features['p4'])
            features['p4'] = feature2

            feature3 = torch.add(edge_feature3, origine_features['p5'])
            features['p5'] = feature3

            feature4 = torch.add(edge_feature4, origine_features['p6'])
            features['p6'] = feature4
            return features
    def _freeze_stages(self):
        freeze_list = [self.edge_Conv0,self.edge_Conv1,self.edge_Conv2,self.edge_Conv3,self.edge_Conv4]
        if self.freeze_edge:
            for block in freeze_list:
                for param in block.parameters():
                    param.requires_grad = False