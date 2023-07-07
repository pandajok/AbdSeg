import torch

from model.UNet import UNet
from model.MSENet import MSENet


def get_coarse_model(cfg, phase):
    model_cfg = {'NUM_CLASSES': cfg.COARSE_MODEL.NUM_CLASSES,
                 'NUM_CHANNELS': cfg.COARSE_MODEL.NUM_CHANNELS,
                 'NUM_DEPTH': cfg.COARSE_MODEL.NUM_DEPTH,
                 'NUM_BLOCKS': cfg.COARSE_MODEL.NUM_BLOCKS,
                 'DECODER_NUM_BLOCK': cfg.COARSE_MODEL.DECODER_NUM_BLOCK,
                 'AUXILIARY_TASK': cfg.COARSE_MODEL.AUXILIARY_TASK,
                 'AUXILIARY_CLASS': cfg.COARSE_MODEL.AUXILIARY_CLASS,
                 'ENCODER_CONV_BLOCK': cfg.COARSE_MODEL.ENCODER_CONV_BLOCK,
                 'DECODER_CONV_BLOCK': cfg.COARSE_MODEL.DECODER_CONV_BLOCK,
                 'CONTEXT_BLOCK': cfg.COARSE_MODEL.CONTEXT_BLOCK,
                 'INPUT_SIZE': cfg.COARSE_MODEL.INPUT_SIZE,
                 'WINDOW_LEVEL': cfg.DATA_LOADER.WINDOW_LEVEL,
                 'IS_PREPROCESS': False if phase != 'test' else cfg.COARSE_MODEL.IS_PREPROCESS,
                 'IS_POSTPROCESS': False if phase != 'test' else cfg.COARSE_MODEL.IS_POSTPROCESS}
    if cfg.COARSE_MODEL.META_ARCHITECTURE == 'UNet':
        return UNet(model_cfg)
    elif cfg.COARSE_MODEL.META_ARCHITECTURE == 'MSENet':
        return MSENet(model_cfg)
    else:
        raise ValueError("Don't exist network: {}}".format(cfg.COARSE_MODEL.META_ARCHITECTURE))


def get_fine_model(cfg, phase):
    model_cfg = {'NUM_CLASSES': cfg.FINE_MODEL.NUM_CLASSES,
                 'NUM_CHANNELS': cfg.FINE_MODEL.NUM_CHANNELS,
                 'NUM_DEPTH': cfg.FINE_MODEL.NUM_DEPTH,
                 'NUM_BLOCKS': cfg.FINE_MODEL.NUM_BLOCKS,
                 'DECODER_NUM_BLOCK': cfg.FINE_MODEL.DECODER_NUM_BLOCK,
                 'AUXILIARY_TASK': cfg.FINE_MODEL.AUXILIARY_TASK,
                 'AUXILIARY_CLASS': cfg.FINE_MODEL.AUXILIARY_CLASS,
                 'ENCODER_CONV_BLOCK': cfg.FINE_MODEL.ENCODER_CONV_BLOCK,
                 'DECODER_CONV_BLOCK': cfg.FINE_MODEL.DECODER_CONV_BLOCK,
                 'CONTEXT_BLOCK': cfg.FINE_MODEL.CONTEXT_BLOCK,
                 'INPUT_SIZE': cfg.FINE_MODEL.INPUT_SIZE,
                 'WINDOW_LEVEL': cfg.DATA_LOADER.WINDOW_LEVEL,
                 'IS_PREPROCESS': False if phase != 'test' else cfg.FINE_MODEL.IS_PREPROCESS,
                 'IS_POSTPROCESS': False if phase != 'test' else cfg.FINE_MODEL.IS_POSTPROCESS}
    if cfg.FINE_MODEL.META_ARCHITECTURE == 'UNet':
        return UNet(model_cfg)
    elif cfg.FINE_MODEL.META_ARCHITECTURE == 'MSENet':
        return MSENet(model_cfg)
    else:
        raise ValueError("Don't exist network: {}".format(cfg.FINE_MODEL.META_ARCHITECTURE))


if __name__ == '__main__':
    data = torch.randn([1, 1, 160, 160, 160]).float().cuda()

    model_cfg = {'NUM_CLASSES': 4,
                 'NUM_CHANNELS': [16, 32, 64, 128, 256],
                 'NUM_DEPTH': 4,
                 'NUM_BLOCKS': [2, 2, 2, 2],
                 'DECODER_NUM_BLOCK': 2,
                 'AUXILIARY_TASK': False,
                 'AUXILIARY_CLASS': 1,
                 'ENCODER_CONV_BLOCK': None,
                 'DECODER_CONV_BLOCK': None,
                 'CONTEXT_BLOCK': None,
                 'INPUT_SIZE': [160, 160, 160],
                 'WINDOW_LEVEL': [-325, 325],
                 'IS_PREPROCESS': False,
                 'IS_POSTPROCESS': False}

    model = MSENet(model_cfg).cuda()

    with torch.no_grad():
        outputs = model(data)

