from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_CT_dsv_3D import *
from .unet_CT_single_att_dsv_3D import *
from .unet_CT_multi_att_dsv_3D import *
from .unet_CT_2D_add_pic import *
from .unet_CT_single_att_dsv_2D import *
from .sononet import *
from .sononet_grid_attention import *
from .attention_unet import AttU_Net,AttU_Net_ASFF
from .vit_seg_modeling import VisionTransformer as Vit_seg
from .vit_seg_configs import get_b16_config,get_r50_b16_config
from .deeplabv3_plus import DeepLab
from .se_p_resunet import Se_PPP_ResUNet

def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2),
                aggregation_mode='concat'):
    model = _get_model_instance(name, tensor_dim)
    if name in ['atten_unet','atten_unet_asff']:
        model = model(img_ch=in_channels, output_ch=n_classes)
    if name in ['se_resunet']:
        model = model(n_channels=in_channels, n_classes=n_classes)
    elif name in ['unet_trans']:
        model = model(config=get_r50_b16_config(),img_size=256,num_classes=n_classes)
    elif name in ['deeplab']:
        model = model(num_classes=n_classes,backbone="xception",downsample_factor=16,pretrained=False)
    elif name in ['unet_CT_att','unet_CT_two_seg']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)
    elif name in ['unet_CT_2D_ASFF','unet_CT_2D_ASPP']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)
    elif name in ['unet_CT_att_add']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)

    elif name in ['unet', 'unet_ct_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False)
    elif name in ['unet_nonlocal','unet_nonlocal_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)

    elif name in ['unet_nonlocal_ASFF']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)

    elif name in ['unet_grid_gating',
                  'unet_ct_single_att_dsv',
                  'unet_ct_multi_att_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False)
    elif name in ['sononet','sononet2']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)
    elif name in ['sononet_grid_attention']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      nonlocal_mode=nonlocal_mode,
                      aggregation_mode=aggregation_mode)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'2D': unet_2D, '3D': unet_3D},
        'unet_nonlocal':{'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
        'unet_nonlocal_dsv':{'2D': unet_nonlocal_dsv_2D},
        'unet_nonlocal_ASFF': {'2D': unet_nonlocal_2D_ASFF},
        'unet_grid_gating': {'3D': unet_grid_attention_3D},
        'unet_ct_dsv': {'3D': unet_CT_dsv_3D},
        'unet_ct_single_att_dsv': {'3D': unet_CT_single_att_dsv_3D},
        'unet_ct_multi_att_dsv': {'3D': unet_CT_multi_att_dsv_3D},
        'sononet': {'2D': sononet},
        'sononet2': {'2D': sononet2},
        'sononet_grid_attention': {'2D': sononet_grid_attention},
        'atten_unet':{'2D':AttU_Net},
        'atten_unet_asff': {'2D': AttU_Net_ASFF},
        'unet_CT_att':{'2D':unet_CT_att_dsv_2D},
        'unet_CT_att_add':{'2D':unet_CT_2D_add_pic},
        'unet_CT_2D_ASFF':{'2D':unet_CT_2D_ASFF},
        'unet_CT_2D_ASPP':{'2D':unet_CT_2D_ASPP},
        'unet_CT_two_seg':{'2D':unet_CT_two_seg},
        'unet_trans':{'2D':Vit_seg},
        'deeplab':{'2D':DeepLab},
        'se_resunet':{'2D':Se_PPP_ResUNet}
    }[name][tensor_dim]

