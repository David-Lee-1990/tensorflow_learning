Sharing some experience and experiments on tensorflow
1. eager file 包含一些学习tensorflow eager模式时的代码，目前只有线性回归的代码
2. tf.group 是在研究meta learning的MAML和Reptile算法时用到的，所以学习一下
3. tf_save_restore 是在研究模型加速时，考虑将参数导出，然后用C或者C++将网络重新写一遍，然后用C网络来做前向传播。文件中的C文件夹是练习C写了卷积，pooling等操作
