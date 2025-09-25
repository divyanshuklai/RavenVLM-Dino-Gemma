# RavenVLM-Dino-Gemma

Please excuse my bad code.

This repo uses Gemma-3-270M and DinoV3-ViT-S+ (29M) to create a VLM.

Currently i am in Stage-1 : Image Captioning. if i am succesful, then i will move on to VQA and then full VLM.

The ViT always remains freezed and the outputs are projected to LM dimension using adapter inspired by Intern-VL3 (LN -> Linear -> GELU -> Linear)

Several Configurations have been tried, all being overrides on base confguration that can be found under configs/

training using amp is resulting in NaN outputs (NaN train loss, NaN validation loss).

As of 25 Sept 25, here are the best training runs:
https://wandb.ai/divyanshukla/DinoGemmaCaptioner/reports/Best-Runs-25-9-25--VmlldzoxNDUyMjEyNQ?accessToken=03x0cw9enescf2o3b2n4fc1ce2yrh37fyfpiuvx4kznue74f3hr0rsg9dn74emwp

### Overview of the project:
#### Model:
LM used : Gemma-3-270M [huggingface](https://huggingface.co/google/gemma-3-270m)
ViT Encoder used: DinoV3-ViT-S+ (29M) [huggingface](http://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
Adapter Arch : LN -> Linear(Vit_dim, Gemma_dim) -> GELU -> Linear(Gemma_dim, Gemma_Dim)
Model definition : src/models/caption_modelling.py

#### Dataset:
coco-captions-train/validation/test huggingface: [train](https://huggingface.co/datasets/Multimodal-Fatima/COCO_captions_train), [validation](https://huggingface.co/datasets/Multimodal-Fatima/COCO_captions_validation)
each image has 5 captions, and because LLaVA used a 535K subset of CC3M for 1 epoch, and coco_captions have 113K samples, i am doing 5 epochs with random caption selection for an equivalent effective train set size.
* src/data/dataloader.py contains dataloader that returns a tensor of image and String of caption
* src/data/dataloader.py contains coco_collate function that returns a tensor of batched images and a list of String captions.

#### Training:
I am Using Pytorch Lightning + Hydra for training configurations. 
* src/engine/train.py is the local entrypoint to commence training.
* src/utils/training.py contains the lightining module and helper functions to construct model, datasets and trainer.

I am using modal.com for my compute platform, i just need to wrap a deployer function with @app_image and call this image functionw within sweep with .map that deploys multiple images at the same time.
* scripts/modal_train.py is the entry point i use from my pc to deploy training on modal servers, i change the sweep grid of hparams and other common hparam overrides in the sweep function inside this file.

