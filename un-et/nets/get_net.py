from .unet import vgg_Unet_2d
from .unet_2D import unet_2D

def get_model_instance(name):
    return {
        'vgg_unet': vgg_Unet_2d,
        'unet_2D':unet_2D,
        'unet_CT_att':unet_2D,
        'unet_CT_two_seg':unet_2D
    }[name]

def get_network(opts):
    name=opts.model.model_type
    Model = get_model_instance(name)
    return Model(in_channels=opts.model.input_nc,n_classes=opts.model.output_nc)

    # if name in ['vgg_unet','unet_2D','unet_CT_att']:
    #     model = Model(n_classes=n_classes,
    #                   is_batchnorm=True,
    #                   in_channels=in_channels,)
    # elif name in ['unet_CT_two_seg']:
    #     model = Model()
    # else:
    #     raise "Not model"


