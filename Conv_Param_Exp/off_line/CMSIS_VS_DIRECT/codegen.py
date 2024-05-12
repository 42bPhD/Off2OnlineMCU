from template import include, CCNT, setup, gen_code
from utils import weight_quantization, reproducibility, ddict, list_bracket
from argparse import ArgumentParser
from template.X import input_format
from template.Y import conv_format
from torch.functional import F
from fxpmath import Fxp
import numpy as np
import torch
import os


class CodeGenerator():
    def __init__(self, config:dict, root='./'):
        # #include "parameter.h"
        self.header = config['header'] # specific variable
        # #define CONV_WT {wt};
        self.body = config['body'] # weight array param
        self.foot = config['foot'] # declare variable code
        self.root = root
    def header_gen(self, filename='parameter.h'):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_PARAMETER_H\n")
            F.write("#define _TEST_PARAMETER_H\n\n")
            for k, v in self.header.items():
                F.write(f"#define {k} {v}\n")
            F.write("\n#endif\n")

    def body_gen(self, filename=None):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_INPUTS_H\n")
            F.write("#define _TEST_INPUTS_H\n\n")
            F.write("#include <stdint.h>\n")
            F.write('#include "parameter.h"\n')
            F.write('#include "weights.h"\n\n')
            F.write('#define DTCM __attribute__((section(".DTCM")))\n')
            for k, v in self.body.items():
                F.write(f"{v}\n")
            F.write("\n#endif\n")
            
    def foot_gen(self, filename):
        with open(os.path.join(self.root, filename), 'w') as F:
            F.write("#ifndef _TEST_WEIGHTS_H\n")
            F.write("#define _TEST_WEIGHTS_H\n\n")
            for k, v in self.foot.items():
                F.write(f"#define {k} {v}\n")
            F.write("\n#endif\n")
    
    def codegen(self, 
                path:str= './',
                header= True, 
                body= True,
                foot= True):
        self.root = path
        self.header_gen(filename = 'parameter.h.')
        self.body_gen(filename = 'inputs.h')
        self.foot_gen(filename = 'weights.h')
        
if __name__ == '__main__':
    from config import TOMLConfig as Config
    CONFIG = Config(filepath='./config/conv.toml')
    
    reproducibility(CONFIG['ARGS']['REPRODUCE'])
    FILENAME = os.path.join(CONFIG['ARGS']['FILENAME'], 'exp2.cpp')
    with open(FILENAME, 'w') as f:
        f.write('\n'.join(include()))
        f.write('\n'.join(CCNT()))
        f.write('\n'.join(setup(CONFIG)))
        f.write('\n'.join(gen_code(CONFIG)))
    config = ddict()
    config['header']['IM_IN_CH'] = CONFIG['INPUT_PARAMS']['IM_IN_CH']
    config['header']['IM_DIM_H'] = CONFIG['INPUT_PARAMS']['IM_DIM_H']
    config['header']['IM_DIM_W'] = CONFIG['INPUT_PARAMS']['IM_DIM_W']
    
    config['header']['CONV_IN_CH'] = CONFIG['CONV']['CONV_IN_CH']
    config['header']['CONV_OUT_CH'] = CONFIG['CONV']['CONV_OUT_CH']
    config['header']['CONV_KER_DIM_H'] = CONFIG['CONV']['CONV_KER_DIM_H']
    config['header']['CONV_KER_DIM_W'] = CONFIG['CONV']['CONV_KER_DIM_W']
    config['header']['CONV_STRIDE_H'] = CONFIG['CONV']['CONV_STRIDE_H']
    # config['header']['CONV_STRIDE_W'] = CONFIG['CONV']['CONV_STRIDE_W']
    config['header']['CONV_PADDING_H'] = CONFIG['CONV']['CONV_PADDING_H']
    config['header']['CONV_PADDING_W'] = CONFIG['CONV']['CONV_PADDING_W']
    
    fxpX, X_q, X_bit  = input_format((1, CONFIG['INPUT_PARAMS']['IM_IN_CH'], 
                                      CONFIG['INPUT_PARAMS']['IM_DIM_W'], 
                                      CONFIG['INPUT_PARAMS']['IM_DIM_H']), bit=8)
    
    raw_X_q, W_bit = weight_quantization(fxpX, bits=CONFIG['INPUT_PARAMS']['BIT'])
    assert np.allclose(X_q, raw_X_q)

    W, B  = conv_format(CONFIG, bit=CONFIG['INPUT_PARAMS']['BIT'])
    fxpW, W_q, W_bit = W # W: Weight(fp16bit), WQ: Fxp16(int8), W_BIT: Weight Shift
    fxpB, B_q, B_bit = B # B: Bias(fp16bit), BQ: Fxp16(int8), B_BIT: Bias Shift
    raw_B_q, raw_B_bit = weight_quantization(fxpB, bits=CONFIG['INPUT_PARAMS']['BIT'])
    # print(B_q, B_bit)
    # shift right 
    B2 = np.right_shift(raw_B_q.astype(np.int16), raw_B_bit)
    # print(raw_B_q, raw_B_bit)
    assert np.allclose(B_q, B2)

    ########################################
    X = torch.tensor(fxpX.val)
    W = torch.tensor(fxpW.val)
    B = torch.tensor(fxpB.val)

    conv_out = F.conv2d(input=X, weight=W, bias=B, 
                        dilation= CONFIG['CONV']['DILATION'], 
                        stride = (CONFIG['CONV']['CONV_STRIDE_W'], CONFIG['CONV']['CONV_STRIDE_H']), 
                        groups= CONFIG['CONV']['GROUPS'],
                        padding= (CONFIG['CONV']['CONV_PADDING_W'], CONFIG['CONV']['CONV_PADDING_H']))
    conv_out = conv_out.detach().cpu().numpy()

    fxp_conv_Out = Fxp(conv_out, signed=True, n_word=8, overflow='saturate')
    OUT_RSHIFT = (X_bit + W_bit) - fxp_conv_Out.n_frac
    BIAS_LSHIFT = (X_bit + W_bit) - B_bit
    config['header']['CONV_WEIGHT_BIT'] = X_bit
    config['header']['CONV_BIAS_BIT'] = B_bit
    config['header']['INPUT_RSHIFT'] = fxp_conv_Out.n_frac

    config['header']['CONV_OUT_RSHIFT'] = OUT_RSHIFT
    config['header']['CONV_BIAS_LSHIFT'] = BIAS_LSHIFT
    B_q = B_q.ravel().tolist()
    B_q = list_bracket(B_q)
    config['foot']['CONV_BIAS'] = B_q
    config['body']['CONV_BIAS'] = 'static q7_t conv_bias[CONV_OUT_CH] = CONV_BIAS;'
    
    
    X_q = np.transpose(X_q, (0, 2, 3, 1)).astype(np.int8)
    X_q = X_q.ravel().tolist() #NCHW -> NHWC
    X_q = list_bracket(X_q)
    config['foot']['IM_IN'] = X_q
    config['body']['IM_IN'] = 'static q7_t input_data[IM_IN_CH * IM_DIM_H * IM_DIM_W] = IM_IN;'
    
    # WEIGHT * INPUT + BIAS = OUTPUT
    _, Im_out_c, Im_out_h, Im_out_w = fxp_conv_Out.shape
    NHWC_weight = np.transpose(W_q, (0, 2, 3, 1)).astype(np.int8) #NCHW -> NHWC
    print(NHWC_weight.shape)
    HWNC_weight = np.transpose(NHWC_weight, (1, 2, 0, 3)) #NHWC(0, 1, 2, 3) -> HWNC(1, 2, 0, 3)
    print(HWNC_weight.shape)
    # Settings for codegen
    config['foot']['NHWC_WEIGHT'] = list_bracket(NHWC_weight.ravel().tolist())
    if CONFIG['OPTIMIZING']['IM2COL']:
        config['body']['NHWC_WEIGHT'] = 'static q7_t conv_wt_nhwc[CONV_IN_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = NHWC_WEIGHT;'
        config['body']['COL_BUFFER'] = 'static q7_t col_buffer[2 * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH * 2];'
        config['body']['CONV_OUT1'] = 'static q7_t conv_out_nhwc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];'
    # DEPLOY_WT_FORMAT = WT_Format.USE_RGB
    # assert not DEPLOY_WT_FORMAT in [i.value for i in WT_Format], "Not supported weight format"
    # if DEPLOY_WT_FORMAT == WT_Format.USE_RGB: # RGB -> RRR GGG BBB (ODD)
    if CONFIG['OPTIMIZING']['WEIGHT_INTERLEAVING'] == 'RGB':
        h, w, n, c = HWNC_weight.shape # [0, 3, 2, 5, 6, 9, 8, 11, 1, 4, 7, 10] -> [0, 2, 3, 5, 6, 8, 9, 11, (1, 4, 7, 10)]
        indice = np.array([np.array([0, 3, 2, 5, 6, 9, 8, 11, 1, 4, 7, 10]) + 12*i for i in range(n*c//12)])
        indice = indice.ravel()
        HWNC_weight = HWNC_weight.reshape(h, w, n*c)
        HWNC_weight = HWNC_weight[:, :, indice]
    # elif DEPLOY_WT_FORMAT == WT_Format.USE_EVEN_CH_V1: # EVEN Channels
    #     HWNC_weight = np.concatenate([i for i in np.split(HWNC_weight, 
    #                                                       config['header']['CONV_IN_CH']//4, 
    #                                                       axis=3)], axis=2)
    # elif DEPLOY_WT_FORMAT== WT_Format.USE_EVEN_CH_V2: # EVEN Channels
    elif CONFIG['OPTIMIZING']['WEIGHT_INTERLEAVING'] == 'EVEN':
        HWNC_weight = HWNC_weight.reshape(config['header']['CONV_KER_DIM_H'], 
                                        config['header']['CONV_KER_DIM_W'], 
                                        config['header']['CONV_OUT_CH'] *(config['header']['CONV_IN_CH']//4), 4)
    else:
        raise NotImplementedError
    config['foot']['HWNC_WEIGHT'] = list_bracket(HWNC_weight.ravel().tolist())
    config['body']['HWNC_WEIGHT'] = 'static q7_t conv_wt_hwnc[CONV_IN_CH * CONV_KER_DIM_H * CONV_KER_DIM_W * CONV_OUT_CH] = HWNC_WEIGHT;'
    config['header']['CONV_OUT_DIM_H'] = Im_out_h
    config['header']['CONV_OUT_DIM_W'] = Im_out_w
    
    
    # for proposed direct conv only
    if CONFIG['OPTIMIZING']['TCM']:        
        config['body']['CONV_OUT2'] = 'static DTCM q7_t conv_out_hwnc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];'
        config['body']['CONV_BUF'] = 'static DTCM q31_t conv_buf[CONV_OUT_CH];'
    else:
        config['body']['CONV_OUT2'] = 'static q7_t conv_out_hwnc[CONV_OUT_CH*CONV_OUT_DIM_H*CONV_OUT_DIM_W];'
        config['body']['CONV_BUF'] = 'static q31_t conv_buf[CONV_OUT_CH];'
        
    ####
    codegen = CodeGenerator(config=config)
    codegen.codegen(path=CONFIG['ARGS']['FILENAME'])
