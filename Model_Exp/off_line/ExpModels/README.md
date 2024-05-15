# MLPerfTiny Pytorch Versions

This was made in conjunction with the WSN-DL framework.

## Usage

When you need one of the models, just do one of the following:

```Python
    import mlperftinytorch_models

    ad_model = mlperftinytorch_models.get_ad_model()
    ic_model = mlperftinytorch_models.get_ic_model()
    mlperftinytorch_models.get_vww_model()
    mlperftinytorch_models.get_ks_model()
```

Inside `mlperftinytorch_models` the models are defined. See inside for info:

```Python
# Anomaly Detection Model (ToyADMOS, DCASE2020-ToyCar Subset)
fc_ae.FC_AE()

# Keyword Spotting Model (SpeechCommands, Hello Edge Subset)
ds_cnn.DS_CNN()

# Image Classification Model (CIFAR-10)
resnet.MLPerfTiny_ResNet_Baseline(num_classes=10) # Long name, apologies.

# Visual Wakewords Model (Visual Wakewords)
mbv2.KL_MBV2_forVWW()
```

## Model Architectures

Same as the ones from WSN-DL. See the repo [Lawrence-lugs/WSN-DL](https://github.com/Lawrence-lugs/cidr-ufl).
