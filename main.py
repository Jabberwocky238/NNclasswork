from Norms import BatchNorm, LayerNorm, InstanceNorm, GroupNorm, UnifiedNorm, PowerNorm, MABatchNorm
from t2t_vit import T2T_ViT
from trainer import Trainer
from vit import ViT
from swin_transformer_v2 import SwinTransformerV2
import torch

def train_task(task_name, norm_method, dataset):
    model_VIT = ViT(
        image_size = 32, patch_size = 4, num_classes = 100,
        dim = 512, depth = 6, heads = 8, 
        mlp_dim = 1024, norm_method=norm_method, dropout = 0.1, emb_dropout = 0.1
    )
    model_STV2 = SwinTransformerV2(
        img_size=32, patch_size=4, in_chans=3, num_classes=100,
        embed_dim=512, depths=[6], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=norm_method, ape=False, patch_norm=True,
        use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
    )
    model_T2T = T2T_ViT(
        img_size=32, tokens_type='transformer', in_chans=3, num_classes=100, 
        embed_dim=512, depth=6, num_heads=8, 
        mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., norm_layer=norm_method, token_dim=64
    )

    lr = 1e-3
    batch = 128
    root = './data'
    start_epoch = 0
    epochs = 10
    from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

    optimizer_VIT = torch.optim.Adam(model_VIT.parameters(), lr)
    optimizer_T2T = torch.optim.Adam(model_T2T.parameters(), lr)
    optimizer_STV2 = torch.optim.Adam(model_STV2.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    trainer_VIT = Trainer(task_name+'_VIT', model_VIT, optimizer_VIT, criterion, 
                          dataset, root, batch, start_epoch)
    trainer_T2T = Trainer(task_name+'_T2T', model_T2T, optimizer_T2T, criterion, 
                          dataset, root, batch, start_epoch)
    trainer_STV2 = Trainer(task_name+'_STV2', model_STV2, optimizer_STV2, criterion, 
                           dataset, root, batch, start_epoch)
    
    for epoch in range(start_epoch, epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        trainer_STV2.train_step(epoch)
        trainer_STV2.test_step(epoch)
        # trainer_STV2.save(epoch)

        trainer_T2T.train_step(epoch)
        trainer_T2T.test_step(epoch)
        # trainer_T2T.save(epoch)

        # trainer_VIT.train_step(epoch)
        # trainer_VIT.test_step(epoch)
        # trainer_VIT.save(epoch)

    trainer_STV2.save(epoch)
    trainer_T2T.save(epoch)
    # trainer_VIT.save(epoch)

# train_task('cifar100_BN', BatchNorm, 'cifar100')
# train_task('cifar100_LN', LayerNorm, 'cifar100')
# train_task('cifar100_GN', GroupNorm, 'cifar100')
# train_task('cifar100_IN', InstanceNorm, 'cifar100')
# train_task('cifar100_PN', PowerNorm, 'cifar100')
# train_task('cifar100_UN', UnifiedNorm, 'cifar100')
# train_task('cifar100_MABN', MABatchNorm, 'cifar100')

train_task('cifar10_BN', BatchNorm, 'cifar10')
train_task('cifar10_LN', LayerNorm, 'cifar10')
train_task('cifar10_GN', GroupNorm, 'cifar10')
train_task('cifar10_IN', InstanceNorm, 'cifar10')
train_task('cifar10_PN', PowerNorm, 'cifar10')
train_task('cifar10_UN', UnifiedNorm, 'cifar10')
train_task('cifar10_MABN', MABatchNorm, 'cifar10')

# tensorboard --logdir=logs  

# import os
# os.system("zip -r logs.zip /tmp/code/logs/*")
# os.system("zip -r models.zip /tmp/code/models/*")
# os.system("cp logs.zip /tmp/output/ ")
# os.system("cp models.zip /tmp/output/ ")

