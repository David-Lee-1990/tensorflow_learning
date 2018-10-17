#ifndef __FUNC__
#define __FUNC__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

float* Relu_1(float* array, int length);// 一维数组的激活
float** Relu_2(float** matrix, int matrixSize, float b); // 二维矩阵激活
float** addmat(float** matrix1, float** matrix2, int matrixSize); //矩阵相加
float** padding_conv(float** matrix, int matrixSize);// 卷积之前，矩阵四边都增加一行
float** padding_pool(float** matrix, int matrixSize);// 池化之前，矩阵右边和下边都增加一行
float** conv(float** featureMap, float** kernel, int featureMapSize, int kernelSize);//卷积操作
float max(float* array, int num);// 取出一维数组中的最大值
float** pooling(float** matrix, int matrixSize);// 2x2的池化操作，步长都为2
float* mlp(float* array, float** w, float* bias,int inSize, int outSize);// 全连接操作
float* flat(float*** matrix, int channelNum, int width); // 将三维矩阵打平，用于接下来的全连接
int argmax(float* array, int length); // 替换最终的softmax,直接输出最大值对应的坐标

//卷积层
typedef struct convolutional_layer{
    int inputWidth; // 输入图像的宽度
    int kernelSize; // 卷积核的长(宽)
    
    int inChannels; //输入图像的通道数
    int outChannels; //输出图像的通道数
    
    float**** kernelData; //卷积核的数据
    
    float* basicData; //偏置项
    
    float*** v; //未激活函数的神经元的输出
    float*** y; // Relu激活后的神经元输出
}ConvLayer;

//采样层 pooling, 这里pooling只用 2x2, 且为max pooling
typedef struct pooling_layer{
    int inputWidth; // 输入图像的宽度
    int kernelSize; // pooling窗口的大小
    int outputWidth; // 输出图像的宽度
    int channelNum;
    float*** y; // 采样后神经元的输出
}PoolLayer;

//全连接层
typedef struct nn_layer{
    int inputNum; // 输入数据的数目
    int outputNum; // 输出数据的数目
    
    float** w; // 权重数据，大小为inputNum*outputNum
    float* basicData; //偏置项，大小为outoutNum
    
    float* v; // 激活前，神经元的输出
    float* y; // 激活后，神经元的输出
}OutLayer;

typedef struct cnn_network
{
    int layerNum;
    ConvLayer* C1;
    PoolLayer* P1;
    ConvLayer* C2;
    PoolLayer* P2;
    OutLayer*  O1;
    OutLayer*  O2;
}CNN_MNIST;

typedef struct MinstImg{
    int c;           // 图像宽
    int r;           // 图像高
    float** ImgData; // 图像数据二维动态数组
}MinstImg;

typedef struct MinstImgArr{
    int ImgNum;        // 存储图像的数目
    MinstImg* ImgPtr;  // 存储图像数组指针
}*ImgArr;              // 存储图像数据的数组

#endif




