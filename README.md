# Multi-label Image Recognition with Partial Labels 

[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_Microsoft_COCO-blue)](https://paperswithcode.com/sota/multi-label-image-recognition-with-partial)

Implementation of the SARB paper: 

- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://aaai-2022.virtualchair.net/poster_aaai1134)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tao Pu*, Tianshui Chen*, Hefeng Wu, Liang Lin. (* equally contributed) 

## Preliminary
1. Donwload data.zip ([[One Drive](https://1drv.ms/u/s!Auj5G110nTE5gjeEHDh17tf_K0zp?e=GIvTvH)] [[Baidu Drive](https://pan.baidu.com/s/11hwhedvUePdGNvW3DSrqQA?pwd=5bxz)]), and unzip it.
3. Modify the lines 16-19 in config.py.
4. Create servel folders (i.e., "exp/log", "exp/code", "exp/checkpoint", "exp/summary") to record experiment details.

## Usage
```
cd HCP-MLR-PL-GMM_SupCon

# modify experiment settings for SARB
vim scripts/SARB.sh

./scripts/SARB.sh
```

## Web Frontend Demo
如果你希望通过网页浏览器查看模型预测结果，已新增一个简单的本地前端系统。

1. 安装 Flask 依赖：
```powershell
pip install -r requirements.txt
```
2. 启动应用：
```powershell
python app.py
```
3. 打开浏览器访问：
```text
http://127.0.0.1:5000
```

> 如果没有可用的训练检查点，系统仍会启动，但预测结果可能是随机初始化模型的输出。

## Common Issues
### 1. How to generate the partial labels?
Since all the datasets have complete labels, we randomly drop a certain proportion of positive and negative labels to create partially annotated datasets. To control the remaining labels' proportion, we can modify the variable **'prob'** in each file of the directory **'scripts'**. Specifically, we provide the partial labels generating function in **'datasets/coco2014.py'**, **'datasets/vg.py'**, **'datasets/voc2007.py'**. 

As you can find, in each dataset class, we provide two elements of annotations: (1) **'labels'**: original ground truth annotations whose shape is $N * C$; (2) **'changeLabels'**: generated partial labels whose shape is $N * C$. For ease of reproducibility, we freeze the random seed of generating partial labels.

**Notes:** for convenience, we also provide partial labels of each dataset on all known label proportions. ([[One Drive]](https://1drv.ms/u/s!Auj5G110nTE5gjUkjiTzJkgSHOJ3?e=mXmRqe) [[Baidu Drive]](https://pan.baidu.com/s/19R-tWBtsOTbSUphihLXr_g))

## Citation
```
@inproceedings{Pu2022SARB,
  author={Pu, Tao and Chen, Tianshui and Wu, Hefeng and Lin, Liang},
  title={Semantic-aware representation blending for multi-label image recognition with partial labels},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022},
  volume={36},
  number={2},
  pages={2091--2098}
}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* tianshuichen@gmail.com
* putao537@gmail.com
