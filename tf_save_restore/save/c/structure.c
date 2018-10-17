#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "func.h"

int cnnforward(CNN_MNIST* cnn, float** inputData)
{
    // 第一层卷积
    
    int inSizeW = cnn->C1->inputWidth; //由于所有的矩阵都是方阵，所以只取一个就好
    int outSizeW = cnn->P1->inputWidth;
    
    for(int i=0;i<(cnn->C1->outChannels);i++)
    {   // 先做加和
        for(int j=0;j<(cnn->C1->inChannels);j++)
        {
            float** mapout = conv(inputData[j],
                                  cnn->C1->kernelData[i][j], // 第i个卷积核的第j个通道；
                                  inSizeW,
                                  cnn->C1->kernelSize);
            addmat(cnn->C1->v[i],mapout);
            
            // 释放mapout占用的内存
            for(int r=0;r<outSizeW;r++)
            {
                free(mapout[r]);
            }
            
            free(mapout);
        }
        //再做Relu激活
        cnn->C1->y[i] = Relu_2(cnn->C1->v[i], outSizeW, cnn->C1->basicData[i]);
    }
    
    // 第一层池化
    cnn->P1->y = pooling(cnn->C1->y, outSizeW);
    free(inSizeW);
    free(outSizeW);
    
    // 第二层卷积
    
    int inSizeW = cnn->C2->inputWidth; //由于所有的矩阵都是方阵，所以只取一个就好
    int outSizeW = cnn->P2->inputWidth;
    
    for(int i=0;i<(cnn->C2->outChannels);i++)
    {   // 先做加和
        for(int j=0;j<(cnn->C2->inChannels);j++)
        {
            float** mapout = conv(cnn->P1->y[j],
                                  cnn->C1->kernelData[i][j], // 第i个卷积核的第j个通道；
                                  inSizeW,
                                  cnn->C2->kernelSize);
            addmat(cnn->C2->v[i],mapout);
            
            // 释放mapout占用的内存
            for(int r=0;r<outSizeW;r++)
            {
                free(mapout[r]);
            }
            
            free(mapout);
        }
        //再做Relu激活
        cnn->C2->y[i] = Relu_2(cnn->C2->v[i], outSizeW, cnn->C2->basicData[i]);
    }
    
    // 第二层池化
    cnn->P2->y = pooling(cnn->C2->y, outSizeW);
    
    // 第三层全连接
    float* flat_mlp_1 = flat(cnn->P2->y,cnn->P2->channelNum,cnn->P2->inputWidth);
    cnn->O1->v = mlp(flat_mlp_1, cnn->O1->w, cnn->O1->basicData,
                     cnn->O1->inputNum, cnn->O1->outputNum);
    cnn->O1->y = Relu_1(cnn->O1->v,cnn->O1->outputNum);
    free flat_mlp_1;
    
    // 第四层全连接
    cnn->O2->v = mlp(cnn->O1->y, cnn->O2->w, cnn->O2->basicData,
                     cnn->O2->inputNum, cnn->O2->outputNum);
    int result = argmax(cnn->O2->v, cnn->O2->outputNum);
    return result;
}


int main()
{
    // 第一层卷积的各参数设定
    
    struct ConvLayer C1;
    
    C1.inputWidth = (int)28;
    C1.kernelSize = (int)3;
    C1.inChannels = (int)1;
    C1.outChannels = (int)32;
    C1.kernelData = ??;
    C1.basicData = ??;
    
    C1.v = (float***)malloc(C1.outChannels*sizeof(float**));
    for(int n=0;n<C1.inputWidth;n++)
    {
        C1.v[n] = (float**)malloc(C1.inputWidth*sizeof(float*));
        for(int m=0;m<C1.inputWidth;m++)
        {
            C1.v[n][m] = (float*)calloc(C1.inputWidth*sizeof(float));
        }
    }
    
    C1.y = (float***)malloc(C1.outChannels*sizeof(float**));
    for(int n=0;n<C1.inputWidth;n++)
    {
        C1.y[n] = (float**)malloc(C1.inputWidth*sizeof(float*));
        for(int m=0;m<C1.inputWidth;m++)
        {
            C1.y[n][m] = (float*)calloc(C1.inputWidth*sizeof(float));
        }
    }
    
    // 第一层池化的各参数设定
    
    struct PoolLayer P1;
    
    P1.inputWidth = C1.inputWidth; // 因为卷积都是用的SAME，所以，这里pool层的输入大小也是卷积层的输入大小
    P1.kernelSize = (int)2;
    P1.outputWidth = (int)14;
    P1.channelNum = C1.outChannels;
    
    P1.y = (float***)malloc(P1.channelNum*sizeof(float**));
    for(int n=0;n<P1.outputWidth;n++)
    {
        P1.y[n] = (float**)malloc(P1.outputWidth*sizeof(float*));
        for(int m=0;m<P1.outputWidth;m++)
        {
            P1.y[n][m] = (float*)calloc(P1.outputWidth*sizeof(float));
        }
    }
    
    // 第二层卷积的各参数设定
    
    struct ConvLayer C2;
    
    C2.inputWidth = P1.outputWidth;
    C2.kernelSize = (int)3;
    C2.inChannels = P1.channelNum;
    C2.outChannels = (int)64;
    C2.kernelData = ??;
    C2.basicData = ??;
    
    C2.v = (float***)malloc(C2.outChannels*sizeof(float**));
    for(int n=0;n<C2.inputWidth;n++)
    {
        C2.v[n] = (float**)malloc(C2.inputWidth*sizeof(float*));
        for(int m=0;m<C2.inputWidth;m++)
        {
            C2.v[n][m] = (float*)calloc(C2.inputWidth*sizeof(float));
        }
    }
    
    C2.y = (float***)malloc(C2.outChannels*sizeof(float**));
    for(int n=0;n<C2.inputWidth;n++)
    {
        C2.y[n] = (float**)malloc(C2.inputWidth*sizeof(float*));
        for(int m=0;m<C2.inputWidth;m++)
        {
            C2.y[n][m] = (float*)calloc(C2.inputWidth*sizeof(float));
        }
    }
    
    // 第二层池化的各参数设定
    
    struct PoolLayer P2;
    
    P2.inputWidth = C2.inputWidth; // 因为卷积都是用的SAME，所以，这里pool层的输入大小也是卷积层的输入大小
    P2.kernelSize = (int)2;
    P2.outputWidth = (int)7;
    P2.channelNum = C2.outChannels;
    
    P2.y = (float***)malloc(P2.channelNum*sizeof(float**));
    for(int n=0;n<P2.outputWidth;n++)
    {
        P2.y[n] = (float**)malloc(P2.outputWidth*sizeof(float*));
        for(int m=0;m<P2.outputWidth;m++)
        {
            P2.y[n][m] = (float*)calloc(P2.outputWidth*sizeof(float));
        }
    }
    
    //第一层全连接
    struct OutLayer  O1;
    O1.inputNum = (int)7*7*64;
    O1.outputNum = (int)1024;
    O1.w = ??;
    O1.basicData = ?? ;
    O1.v = (float*)calloc(O1.outputNum*sizeof(float));
    O1.y = (float*)calloc(O1.outputNum*sizeof(float));
    
    // 第二层全连接
    struct OutLayer  O2;
    O2.inputNum = O1.outputNum;
    O2.outputNum = (int)10;
    O2.w = ??;
    O2.basicData = ?? ;
    O2.v = (float*)calloc(O2.outputNum*sizeof(float));
    O2.y = (float*)calloc(O2.outputNum*sizeof(float));
    
    
    
    // 总的模型架构
    struct CNN_MNIST* cnn;
    
    cnn->C1 = &C1;
    cnn->P1 = &P1;
    cnn->C2 = &C2;
    cnn->P2 = &P2;
    cnn->O1 = &O1;
    cnn->O2 = &O2;
    
    printf("This picture is %d\n",cnnforward(cnn,inputData));
}



