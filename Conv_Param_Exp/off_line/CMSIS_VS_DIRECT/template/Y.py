from torch import nn
from torch.functional import F
from fxpmath import Fxp

def conv_format(CONFIG, bit=8, **kwargs):
    if CONFIG['ARGS']['CONV_TYPE'] == 'conv2d':
        print(CONFIG['CONV']['CONV_IN_CH'])
        conv2d = nn.Conv2d(in_channels= int(CONFIG['CONV']['CONV_IN_CH']), 
                        out_channels= CONFIG['CONV']['CONV_OUT_CH'], 
                        kernel_size=(CONFIG['CONV']['CONV_KER_DIM_H'], CONFIG['CONV']['CONV_KER_DIM_W']),
                        stride=(CONFIG['CONV']['CONV_STRIDE_H'], CONFIG['CONV']['CONV_STRIDE_W']),
                        padding = (kwargs['padding_h'], kwargs['padding_w']),
                        dilation=CONFIG['CONV']['DILATION'],
                        groups=CONFIG['CONV']['GROUPS'],
                        bias=CONFIG['CONV']['BIAS'])
        weight = conv2d.weight.detach().cpu().numpy()
        weight = Fxp(weight, signed=True, n_word=bit, overflow='saturate')
        weight.config.dtype_notation = 'Q'
        weight.config.array_output_type = 'array'
        w_frac = weight.n_frac
        ############################################
        b_bias = conv2d.bias.detach().cpu().numpy()
        b_bias = Fxp(b_bias, signed=True, n_word=bit, overflow='saturate')
        b_bias.config.dtype_notation = 'Q'
        b_bias.config.array_output_type = 'array'
        b_frac = b_bias.n_frac
        return (weight, weight.val, w_frac), (b_bias, b_bias.val, b_frac)
    elif CONFIG.conv == 'conv2d_dw':
        conv2d_dw = nn.Conv2d(in_channels= CONFIG['CONV']['CONV_IN_CH'], 
                        out_channels= CONFIG['CONV']['CONV_OUT_CH'], 
                        kernel_size=(CONFIG['CONV']['CONV_KER_DIM_H'], CONFIG['CONV']['CONV_KER_DIM_W']),
                        stride=(CONFIG['CONV']['CONV_STRIDE_H'], CONFIG['CONV']['CONV_STRIDE_W']),
                        padding_mode= 1 if CONFIG['CONV']['PADMODE'] == 'same' else 0,
                        dilation=CONFIG['CONV']['DILATION'],
                        groups=CONFIG['CONV']['CONV_IN_CH'], 
                        bias=CONFIG['CONV']['BIAS'])
        weight = conv2d_dw.weight.detach().cpu().numpy()
        weight = Fxp(weight, signed=True, n_word=bit, overflow='saturate')
        weight.config.dtype_notation = 'Q'
        weight.config.array_output_type = 'array'
        w_frac = weight.n_frac
        ############################################
        b_bias = conv2d_dw.bias.detach().cpu().numpy()
        b_bias = Fxp(b_bias, signed=True, n_word=bit, overflow='saturate')
        b_bias.config.dtype_notation = 'Q'
        b_bias.config.array_output_type = 'array'
        b_frac = b_bias.n_frac
        return (weight, weight.val, w_frac), (b_bias, b_bias.val, b_frac)
    elif CONFIG.conv == 'conv2d_pw':
        conv2d_pw = nn.Conv2d(in_channels= CONFIG['CONV']['CONV_IN_CH'], 
                        out_channels= CONFIG['CONV']['CONV_OUT_CH'], 
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding_mode= 1 if CONFIG['CONV']['PADMODE'] == 'same' else 0,
                        dilation=CONFIG['CONV']['DILATION'],
                        groups=1,
                        bias=CONFIG['CONV']['BIAS'])
        weight = conv2d_pw.weight.detach().cpu().numpy()
        weight = Fxp(weight, signed=True, n_word=bit, overflow='saturate')
        weight.config.dtype_notation = 'Q'
        weight.config.array_output_type = 'array'
        w_frac = weight.n_frac
        ############################################
        b_bias = conv2d_pw.bias.detach().cpu().numpy()
        b_bias = Fxp(b_bias, signed=True, n_word=bit, overflow='saturate')
        b_bias.config.dtype_notation = 'Q'
        b_bias.config.array_output_type = 'array'
        b_frac = b_bias.n_frac
        return (weight, weight.val, w_frac), (b_bias, b_bias.val, b_frac)
    else:
        raise NotImplementedError