# 0 写在前面  
队伍名：xxxxx  
复赛分数：0.8997（第二）  
所有运行过程文件可参考：链接: https://pan.baidu.com/s/1jrdarunuUaG3mJwljAdeBQ 提取码: nm5h  
显卡要求：A100*7  

# 1 环境配置  
建立虚拟环境  
```shell
conda create -n ccks2 python=3.7
conda activate ccks2
```
安装pytorch  
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
安装其他依赖库  
```shell
pip install -r requirements.txt
```

# 权重下载  
https://drive.google.com/file/d/18Fd3vGvj0Dz8rPlxROxugjZaF8Z4jf7g/view?usp=sharing 文件存放于checkpoints/r2d2文件夹下  
或者 在 https://pan.baidu.com/s/1jrdarunuUaG3mJwljAdeBQ 提取码: nm5h 中checkpoints/r2d2文件夹下下载  

# 2 运行  
```python
sh run.sh
```

# 3 文件说明  
----checkpoints 存放R2D2预训练权重，R2D2 config，roberta config  
----data 数据目录  
--------build_text_input.py 数据集生成，训练用文本数据生成  
----R2D2_cross  
--------1.r2d2_pretrain.py R2D2 pretrain，结果权重存放于pretrain_checkpoints文件夹  
--------2.train.py 训练交互式模型，结果权重存放于align_checkpoints文件夹  
----R2D2_distill  
--------2.train.py 训练蒸馏模型，结果权重存放于align_checkpoints文件夹  
--------2.get_result.py 得出提交文件，结果存放于outputs文件夹  
