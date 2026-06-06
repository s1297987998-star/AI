# 切换到项目目录
cd "E:\Supermap_transfermor\HCP-MLR-PL"
if (-not $?) { exit 1 }  # 若上一条命令失败（cd失败），则退出

# 激活虚拟环境
. .venv/Scripts/activate.ps1

# 变量定义
$post='SARB-COCO-SupCon'
$printFreq=1000
$mode='SARB'
$dataset='COCO2014'
$prob=0.5
$pretrainedModel='./data/checkpoint/resnet101.pth'
$resumeModel='None'
$evaluate='False'
$epochs=20
$startEpoch=0
$stepEpoch=10
$batchSize=32
$lr=1e-5
$momentum=0.9
$weightDecay=5e-4
$cropSize=448
$scaleSize=512
$workers=8
$mixupEpoch=5
$contrastiveLossWeight=0.05
$supConTemp=0.1 # 温度系数，通常 0.07 或 0.1 效果最好
$prototypeNum=10
$recomputePrototypeInterval=5
$isAlphaLearnable='True'
$isBetaLearnable='True'

# 设置环境变量并执行脚本
$env:OMP_NUM_THREADS=8
$env:MKL_NUM_THREADS=8
$env:CUDA_VISIBLE_DEVICES=0

python SARB.py `
    --post $post `
    --printFreq $printFreq `
    --mode $mode `
    --dataset $dataset `
    --prob $prob `
    --pretrainedModel $pretrainedModel `
    --resumeModel $resumeModel `
    --evaluate $evaluate `
    --epochs $epochs `
    --startEpoch $startEpoch `
    --stepEpoch $stepEpoch `
    --batchSize $batchSize `
    --lr $lr `
    --momentum $momentum `
    --weightDecay $weightDecay `
    --cropSize $cropSize `
    --scaleSize $scaleSize `
    --workers $workers `
    --mixupEpoch $mixupEpoch `
    --contrastiveLossWeight $contrastiveLossWeight `
    --supConTemp $supConTemp `
    --prototypeNum $prototypeNum `
    --recomputePrototypeInterval $recomputePrototypeInterval `
    --isAlphaLearnable $isAlphaLearnable `
    --isBetaLearnable $isBetaLearnable