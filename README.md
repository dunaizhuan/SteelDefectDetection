# SteelDefectDetection
kaggle上的比赛的代码整理上传
第一次用github不是很熟练 也没有文件结构 
缺少了input 文件夹 还有croped_df.csv 文件--太大了 53MB 无法上传。
应该是先运行make244.py 然后运行train.py 
大体流程 -》裁减图片 然后训练网络
后面改变了模型 FPAv2
使之可以直接使用 1,600px × 256 进行upsample downsample
Unet -34  + SCSE
LOSS BCE
