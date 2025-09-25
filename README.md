# RavenVLM-Dino-Gemma

This repo uses Gemma-3-270M and DinoV3-ViT-S+ (29M) to create a VLM.

Currently i am in Stage-1 : Image Captioning. if i am succesful, then i will move on to VQA and then full VLM.

The ViT always remains freezed and the outputs are projected to LM dimension using adapter inspired by Intern-VL3 (LN -> Linear -> GELU -> Linear)

Several Configurations have been tried, all being overrides on base confguration that can be found under configs/

training using amp is resulting in NaN outputs (NaN train loss, NaN validation loss).

As of 25 Sept 25, here are the best training runs:
https://wandb.ai/divyanshukla/DinoGemmaCaptioner/reports/Best-Runs-25-9-25--VmlldzoxNDUyMjEyNQ?accessToken=03x0cw9enescf2o3b2n4fc1ce2yrh37fyfpiuvx4kznue74f3hr0rsg9dn74emwp





