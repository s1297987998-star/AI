#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='SARB-COCO-SupCon'
printFreq=1000

mode='SARB'
dataset='COCO2014'
prob=0.5 

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=10

batchSize=64
lr=3e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512
workers=8

mixupEpoch=5
contrastiveLossWeight=0.05
supConTemp=0.1  # 温度系数，通常 0.07 或 0.1 效果最好

prototypeNum=10
recomputePrototypeInterval=5

isAlphaLearnable='True'
isBetaLearnable='True'

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python SARB.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --mixupEpoch ${mixupEpoch} \
    --contrastiveLossWeight ${contrastiveLossWeight} \
    --supConTemp ${supConTemp} \
    --prototypeNum ${prototypeNum} \
    --recomputePrototypeInterval ${recomputePrototypeInterval} \
    --isAlphaLearnable ${isAlphaLearnable} \
    --isBetaLearnable ${isBetaLearnable} \
