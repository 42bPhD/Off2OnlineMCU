from torch import nn
from torch.functional import F
from fxpmath import Fxp

def conv_format(args, bit=8):
    if args.conv == 'conv2d':
        conv2d = nn.Conv2d(in_channels= args.ic, 
                        out_channels= args.oc, 
                        kernel_size=(args.kernel_size[0], args.kernel_size[1]),
                        stride=(args.stride[0], args.stride[1]),
                        padding_mode=args.padmode,
                        dilation=args.dilation,
                        groups=args.groups,
                        bias=args.bias)
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
    elif args.conv == 'conv2d_dw':
        conv2d_dw = nn.Conv2d(in_channels= args.ic, 
                        out_channels= args.oc, 
                        kernel_size=args.ks,
                        stride=args.stride,
                        padding_mode=args.padmode,
                        dilation=args.dilation,
                        groups=args.ic,
                        bias=args.bias)
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
    elif args.conv == 'conv2d_pw':
        conv2d_pw = nn.Conv2d(in_channels= args.ic, 
                        out_channels= args.oc, 
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding_mode=args.padmode,
                        dilation=args.dilation,
                        groups=1,
                        bias=args.bias)
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
    elif args.conv == 'conv2d_dw_pw':
        conv2d_dw_pw = nn.Conv2d(in_channels= args.ic, 
                        out_channels= args.oc, 
                        kernel_size=args.ks,
                        stride=(args.stride),
                        padding_mode=args.padmode,
                        dilation=args.dilation,
                        groups=args.ic,
                        bias=args.bias)
        weight = conv2d_dw_pw.weight.detach().cpu().numpy()
        weight = Fxp(weight, signed=True, n_word=bit, overflow='saturate')
        weight.config.dtype_notation = 'Q'
        weight.config.array_output_type = 'array'
        w_frac = weight.n_frac
        ############################################
        b_bias = conv2d_dw_pw.bias.detach().cpu().numpy()
        b_bias = Fxp(b_bias, signed=True, n_word=bit, overflow='saturate')
        b_bias.config.dtype_notation = 'Q'
        b_bias.config.array_output_type = 'array'
        b_frac = b_bias.n_frac
        return (weight, weight.val, w_frac), (b_bias, b_bias.val, b_frac)
    elif args.conv == 'conv2d_pw_dw':
        conv2d_pw_dw = nn.Conv2d(in_channels= args.ic, 
                        out_channels= args.oc, 
                        kernel_size=(args.ks),
                        stride=(args.stride),
                        padding_mode=args.padmode,
                        dilation=args.dilation,
                        groups=1,
                        bias=args.bias)
        weight = conv2d_pw_dw.weight.detach().cpu().numpy()
        weight = Fxp(weight, signed=True, n_word=bit, overflow='saturate')
        weight.config.dtype_notation = 'Q'
        weight.config.array_output_type = 'array'
        w_frac = weight.n_frac
        ############################################
        b_bias = conv2d_pw_dw.bias.detach().cpu().numpy()
        b_bias = Fxp(b_bias, signed=True, n_word=bit, overflow='saturate')
        b_bias.config.dtype_notation = 'Q'
        b_bias.config.array_output_type = 'array'
        b_frac = b_bias.n_frac
        return (weight, weight.val, w_frac), (b_bias, b_bias.val, b_frac)
    
    else:
        raise NotImplementedError