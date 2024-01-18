import torch
from t2t_vit import T2T_ViT
from vit import ViT
from swin_transformer_v2 import SwinTransformerV2

def checkNorm(norm_method):
    try:
        img = torch.randn(16, 3, 32, 32).cuda()

        model_ViT = ViT(
            image_size = 32, patch_size = 4, num_classes = 100,
            dim = 512, depth = 6, heads = 8, 
            mlp_dim = 1024, norm_method=norm_method, dropout = 0.1, emb_dropout = 0.1
        ).cuda()
        model_STV2 = SwinTransformerV2(
            img_size=32, patch_size=4, in_chans=3, num_classes=100,
            embed_dim=512, depths=[6], num_heads=[8],
            window_size=4, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=norm_method, ape=False, patch_norm=True,
            use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
        ).cuda()
        model_T2T = T2T_ViT(
            img_size=32, tokens_type='transformer', in_chans=3, num_classes=100, 
            embed_dim=512, depth=6, num_heads=8, 
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=norm_method, token_dim=64
        ).cuda()
        
        preds = model_T2T(img) 
        preds = model_ViT(img) 
        preds = model_STV2(img) 
        print(f"{norm_method} CHECK")
    except Exception as err:
        print(f"{norm_method} ERROR")
        raise err

from Norms import BatchNorm, LayerNorm, InstanceNorm, GroupNorm, UnifiedNorm, PowerNorm, MABatchNorm
checkNorm(BatchNorm)
checkNorm(LayerNorm)
checkNorm(InstanceNorm)
checkNorm(GroupNorm)
checkNorm(UnifiedNorm)
checkNorm(PowerNorm)
checkNorm(MABatchNorm)