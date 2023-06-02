import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from .encoder import Encoder
from .decoder import Decoder
import pytorch_lightning as pl
from dataset.dataset import LightFormerDataset
from pathlib import Path
from functools import partial


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x) # C: 256
    x = self.layer4(x) # C: 512

    return x

class LightFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = self.config["embed_dim"] # 256
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = None
        self.resnet.avgpool = None
        self.resnet.forward = partial(forward, self.resnet)
        self.down_conv = nn.Sequential(
            nn.Conv2d(512, 1024,3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256,3),
            nn.BatchNorm2d(256)
        )
        self.mlp = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
        )
        self.head1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )
        self.head2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )
        self.num_query = self.config["num_query"]
        self.query_embed = nn.Embedding(self.num_query, self.embed_dim)
        self.encoder = Encoder(self.config)


    def forward(self, images, features=None):
        """
        images: 10 buffered sequential images
        """
        image_num = self.config['image_num']
        B,_,c,h,w = images.shape
        images = images.reshape(B*image_num,c,h,w)
        vectors = self.resnet(images)
        vectors = self.down_conv(vectors) # 512 -> 256
        _,c,h,w = vectors.shape
        vectors = vectors.view(B, image_num, c, h, w) # [bs,num_img, 256, h, w]
    
        query = self.query_embed.weight
        agent_all_feature = self.encoder(query, vectors) # [bs, 1, 256]
        agent_all_feature = self.mlp(agent_all_feature)
        head1_out = self.head1(agent_all_feature)
        head2_out = self.head2(agent_all_feature)
        head1_out = head1_out.unsqueeze(3)
        head2_out = head2_out.unsqueeze(3)
       
        return head1_out, head2_out


class LightFormerPredictor(pl.LightningModule, nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LightFormer(self.config)
        self.index = 0
        self.class_decoder_st = Decoder(self.config)
        self.class_decoder_lf = Decoder(self.config)

    def forward(self, images):
        return self.model(images) 

    def cal_loss_step(self, batch):
        images = batch["images"]
        head1_out, head2_out = self.model(images) # (bs, 1, 1024, 1)
        st_lightstatus_class = self.class_decoder_st(head1_out, batch["label"][:,:2])
        lf_lightstatus_class = self.class_decoder_lf(head2_out, batch["label"][:,2:4])
        st_class_loss = self.prob_loss(st_lightstatus_class, batch["label"][:,:2])
        lf_class_loss = self.prob_loss(lf_lightstatus_class, batch["label"][:,2:4])
        class_loss = st_class_loss + lf_class_loss
        self.index = self.index+1
        return class_loss
    
    def cal_ebeding_step(self, batch):
        images = batch["images"]
        head1_out, head2_out = self.model(images)
        st_lightstatus = self.class_decoder_st(head1_out,None)
        lf_lightstatus = self.class_decoder_lf(head2_out,None)
        B, K, _ = st_lightstatus.shape
        st_prob = st_lightstatus.view(B, self.config["out_class_num"], 1)
        lf_prob = lf_lightstatus.view(B, self.config["out_class_num"], 1)
        st_predict=st_prob.argmax(dim=1)[0][0]
        lf_predict=lf_prob.argmax(dim=1)[0][0]
        st_target=batch["label"][:,:2].argmax(dim=1)[0]
        lf_target=batch["label"][:,2:4].argmax(dim=1)[0]
        with open("complete_model_Kaggle_daytime_n=1_res1.txt","a+") as f:
            flag='right'
            if(st_predict!=st_target) or (lf_predict!=lf_target):
                flag = 'error'
            ss = "{} {} {} {} {} {}\n".format(batch["name"], st_predict, st_target, lf_predict, lf_target, flag)
            f.write(ss)
            f.flush()
        return 0
    
    def prob_loss(self, lightstatus, gt_label):
        gt_label_idx = torch.argmax(gt_label,dim=-1)
        pred_cls_score = torch.log(lightstatus)
        loss = F.nll_loss(pred_cls_score.squeeze(-1), gt_label_idx, reduction='mean')

        return loss
    
    def training_step(self, batch, batch_idx):
        loss  = self.cal_loss_step(batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.cal_loss_step(batch)
        loss = self.cal_loss_step(batch)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        _ = self.cal_ebeding_step(batch)
        return
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])

        return [optimizer], [scheduler]

    def train_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        print("what is this", self.config['training']['sample_database_folder'])
        train_set = LightFormerDataset(self.config['training']['sample_database_folder'], image_norm)
        print(f"...............................Total Samples {len(train_set)} .......................................")
        train_loader = DataLoader(dataset=train_set,
                                                                batch_size=self.config['training']['batch_size'],
                                                                shuffle=True,
                                                                # collate_fn=AgentClosureBatch.from_data_list,
                                                                num_workers=self.config['training']['loader_worker_num'],
                                                                drop_last=True,
                                                                pin_memory=True)

        return train_loader

    def val_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        val_set = LightFormerDataset(self.config['validation']['sample_database_folder'], image_norm)
        val_loader = DataLoader(val_set,
                                batch_size=self.config['validation']['batch_size'],
                                shuffle=False,
                                # collate_fn=AgentClosureBatch.from_data_list,
                                num_workers=self.config['validation']['loader_worker_num'],
                                drop_last=True,
                                pin_memory=True)

        return val_loader

    def test_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        test_set = LightFormerDataset(self.config['test']['sample_database_folder'], image_norm)
        test_loader = DataLoader(test_set,
                                 batch_size=self.config['test']['batch_size'],
                                 shuffle=False,
                                #  collate_fn=AgentClosureBatch.from_data_list,
                                 num_workers=self.config['test']['loader_worker_num'],
                                 drop_last=False,
                                 pin_memory=True)

        return test_loader