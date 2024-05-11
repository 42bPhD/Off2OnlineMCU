import torch
import argparse
from torchsummary import summary
import os
from utils.dataloaders import get_dataloader, get_subnet_dataloader
from utils.train_eval import evaluate
from utils.functions import reconstruction_model
from utils.io import load_weights
from etc.vww_model import mobilenet_v1

from bn_fold import bn_fold
from torch import nn

from fxpmath import Fxp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-ckpt",
        default="./weights/mcu_vggrepc1_vww.pth",
        type=str,
        help="Path to model checkpoint for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default value is 50 according to TF training procedure.",
    )
    parser.add_argument(
        "--data-dir",
        default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
        type=str,
        help="Path to dataset (will be downloaded).",
    )
    parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
    parser.add_argument(
        "--workers", default=8, type=int, help="Number of data loading processes."
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model_q import MCU_VGGRep, MCU_VGGRepC1
    # model = mobilenet_v1()
    # model = MCU_VGGRep()
    model = MCU_VGGRepC1()
    
    # Load pre-trained weight
    model = load_weights(model, args.model_ckpt)
    from utils.dataloaders import load_cifar
    # train_loader, val_loader, _ = load_cifar(args.data_dir, args.batch_size,args.workers,
    #                                          val_num=100)

    # train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir,
    #                                             batch_size=args.batch_size,
    #                                             image_size=args.image_size,
    #                                             shuffle=True,
    #                                             num_workers=args.workers)
    
    val_loader = get_subnet_dataloader(args.data_dir, 100, args.batch_size, args.image_size, args.workers)
    
    
    ########## Evaluation (float 32)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    from utils.train_eval import get_accuracy
    from utils.utils import count_net_flops
    print(f"Before accuracy: {get_accuracy(model.to(device), val_loader):.2f}%",
          f"MAC+BN={count_net_flops(model, (1, 3, args.image_size, args.image_size), False):,}")
    # flatten_model = reconstruction_model(model, device)
    
    import copy
    # flatten_model = copy.deepcopy(model)
    # flatten_model = flatten_model.to(device)
    
    ########  the accuracy will remain the same after fusion
    # fused_acc = get_accuracy( model=flatten_model, dataloader=val_loader)
    # fold_model = bn_fold(flatten_model, val_loader)
    # torch.save(fold_model, 'cfiles/fold_model.pth')
    # print(f'Accuracy of the fused model={fused_acc:.2f}%, MAC={count_net_flops(model, (1, 3, args.image_size, args.image_size))}')
    # summary(flatten_model, (3, args.image_size, args.image_size))

    model = model.cpu()
    with torch.no_grad():
        get_shape = model.get_shape(args.batch_size, (3, args.image_size, args.image_size))
    model = model.to(device)
    
    
    # print(f'Pytorch Accuracy: {get_accuracy(model, val_loader)}')
    
    vs_proj_path = os.path.join(os.getcwd(), "CMSIS_NN_PC_simulator/Deploy_Simulator")
    compilation_bat = os.path.join(os.getcwd(), 'compile.bat')
    model.eval()
    print(model)

    from torch2cmsis.converter2 import CMSISConverter
    # from torch2cmsis.converter import CMSISConverter
    cm_converter = CMSISConverter(root = vs_proj_path,
                                model = model,
                                weight_file_name="weights.h",
                                parameter_file_name="parameters.h",
                                weight_bits=8,
                                linear_features = get_shape,
                                compilation_config=compilation_bat)
    cm_converter.convert_model(val_loader)
    # weight값 percentile해봤는데 안좋음. <- 적용안함.
    # weight값 quantize시 -128 min, 127 max해봄.
    # 모델 search파라미터 1로 줄임.
    # 모델 refine하지 않고 그냥 deploy해보자. <- 이게 제일 좋음.

    execute_path = os.path.join(vs_proj_path, "Debug/Deploy_Simulator.exe")
    cm_converter.evaluate_cmsis(execute_path, val_loader)
    input, label = next(iter(val_loader))
    cm_converter.sample_inference_checker(execute_path, input)
    # torch.save(cm_converter.model_qq, 'cfiles/quantized_model.pth')
    
    
    #! Step1 TODO: c_code_gen.py will import the model and generate the c code
    #! Step2 TODO: Add make_test_inp.py to generate the input data for the c code