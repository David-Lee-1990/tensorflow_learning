#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// 一维矩阵relu激活
float* Relu_1(float* array, int length)
{
    float* outPut = (float*)malloc(length*sizeof(float));
    for(int n=0;n<length;n++)
    {
        outPut[n] = (array[n]>0)?array[n]:0;
    }
    return outPut;
}


// 二维矩阵relu激活,同时加入了偏置项

float** Relu_2(float** matrix,int matrixSize,float b)
{
    float** outPut = (float**)malloc(matrixSize*sizeof(float*));
    for(int n=0;n<matrixSize;n++)
        outPut[n] = (float*)calloc(matrixSize,sizeof(float));
    for(int m=0;m<matrixSize;m++)
        for(int n=0;n<matrixSize;n++)
        {
            float v = *((float*)matrix+m*matrixSize+n) + b;
            outPut[m][n] = (v>0)?v:(float)0.0;
        }
    return outPut;
}



// 矩阵相加

float** addmat(float** matrix1, float** matrix2, int matrixSize)
{
    float** out = (float**)malloc(matrixSize*sizeof(float*));
    for(int m=0;m<matrixSize;m++)
    {
        out[m] = (float*)calloc(matrixSize,sizeof(float));
    }
    for(int m=0;m<matrixSize;m++)
        for(int n=0;n<matrixSize;n++)
        {
            out[m][n] = *((float*)matrix1 + m*matrixSize + n) +
                        *((float*)matrix2 + m*matrixSize + n);
        }
    return out;
}

// 假设卷积核都为3，所以对featuremap左边上边都增加一行0，右边和下边增加一行0

float** padding_cov(float** matrix,int matrixSize)
{
    
    int outSize = matrixSize + 2;
    
    float** outPut = (float**)malloc(outSize*sizeof(float*));
    for(int i=0;i<outSize;i++)
        outPut[i] = (float*)malloc(outSize*sizeof(float));
    
    for(int m=0;m<outSize;m++)
    {
        for(int n=0;n<outSize;n++)
        {
            if(m==0 || m==outSize-1 || n==0 || n==outSize-1)
            {
                outPut[m][n]=(float)0.0;
            }
            else{
                outPut[m][n] = *((float*)matrix + (m-1)*matrixSize + n - 1);
            }
//            printf("Padding:out[%d][%d]=%f\n",m,n,outPut[m][n]);
        }
    }
    return outPut;
}

// 如果featruemap长度为奇数，假设maxpool,step都是2，所以对featuremap右边下边都增加一行0

float** padding_pool(float** matrix,int matrixSize)
{
    
    int outSize = matrixSize + 1;
    
    float** outPut = (float**)malloc(outSize*sizeof(float*));
    for(int i=0;i<outSize;i++)
        outPut[i] = (float*)malloc(outSize*sizeof(float));
    
    for(int m=0;m<outSize;m++)
    {
        for(int n=0;n<outSize;n++)
        {
            if(m==outSize-1 || n==outSize-1)
            {
                outPut[m][n]=(float)0.0;
            }
            else{
                outPut[m][n] = *((float*)matrix + (m)*matrixSize + n);
            }
            //            printf("Padding:out[%d][%d]=%f\n",m,n,outPut[m][n]);
        }
    }
    return outPut;
}

// 卷积步长都是1,只实现padding为SAME的情形

float** conv(float** featureMap, float** kernel, int featureMapSize, int kernelSize)
{
    
    float** featureMap_padding = padding_cov(featureMap, featureMapSize);// 先完成Padding；
    int featureMapSize_new = featureMapSize + 2;
    
    int outSize = featureMapSize;
    float** outPut = (float**)malloc(outSize*sizeof(float*));
    for(int i=0;i<outSize;i++)
        outPut[i] = (float*)calloc(outSize,sizeof(float));// 经过这样的定义，output就可以按照数组指标的格式来取数了
            // calloc 申请内存并初始化为零；
    
    for(int r=0;r<outSize;r++)
        for(int c=0;c<outSize;c++)
        {
            for (int m=0;m<kernelSize;m++)
                for(int n=0;n<kernelSize;n++)
                    outPut[r][c] = outPut[r][c] + (*((float*)kernel + m*kernelSize + n))*featureMap_padding[r+m][c+n];
//            printf("out[%d][%d]=%f\n",r,c,outPut[r][c]);
        }
    free(featureMap_padding);
    return outPut;
}

float max(float* array, int num)
{
    float max = (float)0.0;
    for(int m=0;m<num;m++)
        if(*(array+m)>max)
        {
            max = *(array+m);
        }
    return max;
}

//只实现了2X2的max pooloing，步长都是2.

float** pooling(float** matrix,int matrixSize)
{
    // 偶数情形照常操作
    if(matrixSize%2 == 0)
    {
        int outSize = matrixSize/2;
        float** outPut = (float**)malloc(outSize*sizeof(float*));
        for(int n=0;n<outSize;n++)
            outPut[n] = (float*)malloc(outSize*sizeof(float));
    
        for(int m=0;m<outSize;m++)
            for(int n=0;n<outSize;n++)
            {
                float s[4] = {matrix[2*m][2*n],
                    matrix[2*m][2*n+1],
                    matrix[2*m+1][2*n],
                    matrix[2*m+1][2*n+1]
                    
                };
                outPut[m][n] = max(s,4);
//                printf("a[%d][%d]=%f\n",m,n,outPut[m][n]);
            }
        return outPut;
    }

    else{
        int outSize = (matrixSize+1)/2;
        
        float** outPut = (float**)malloc(outSize*sizeof(float*));
        for(int n=0;n<outSize;n++)
            outPut[n] = (float*)malloc(outSize*sizeof(float));
        
        float** matrix_new = padding_pool(matrix,matrixSize);
        for(int m=0;m<outSize;m++)
            for(int n=0;n<outSize;n++){
                float s[4] = {matrix_new[2*m][2*n],
                    matrix_new[2*m][2*n+1],
                    matrix_new[2*m+1][2*n],
                    matrix_new[2*m+1][2*n+1]};
                outPut[m][n] = max(s,4);
//                printf("a[%d][%d]=%f\n",m,n,outPut[m][n]);
            }
        return outPut;
    }
}

// 全连接层

float* mlp(float* array,float** w, float* bias, int inSize, int outSize)
{
    float* outPut = (float*)malloc(outSize*sizeof(float));
    for(int n=0;n<outSize;n++)
    {
        outPut[n] = bias[n]; // 加入偏置项
        for(int m=0;m<inSize;n++)
            outPut[n] += *((float*)w+n*inSize+m)*(array[m]);
    }
    return outPut;
}

// 将矩阵打平，变成一维向量

float* flat(float*** matrix, int channelNum, int width)
{
    float* outPut = (float*)malloc(channelNum*width*width*sizeof(float));
    int num = (int)0;
    for(int m=0;m<channelNum;m++)
        for(int n=0;n<width;n++)
            for(int j=0;j<width;j++)
            {
                outPut[num] = *((float*)matrix + m*channelNum + n*width + j);
                printf("outPut[%d]=%f\n",num,outPut[num]);
                num += 1;
            }
    return outPut;
}

// 取出一维数组中最大值对应的位置

int argmax(float* array, int length)
{
    int arg = (int)0;
    float max = array[0];
    for(int n=0;n<length;n++)
    {
        arg = (array[n]>max)?n:arg;
        max = (array[n]>max)?array[n]:max;
    }
    return arg;
}


//int main()
//{
//    float a[2][2][2] = {{{1,2},{3,4}},{{5,6},{7,8}}};
//    float w[3][3] = {{1,0,1},{0,1,0},{1,0,1}};
//    float m[10] = {1,2,3,4,5,6,7,8,9,0};
//    printf("Maximum of m is %f\n",max(m,10));
//    int featureMapSize = 3;
//    int kernelSize = 3;
//    wo = conv((float**)a, (float**)w, featureMapSize, kernelSize);
//    wo.outSize = 3;
//    pooling((float**)a,(int)3);
//    float*sum = flat((float***)a,2,2);
//    for(int n=0;n<8;n++)
//    {
//        printf("sum[%d]=%f\n",n,sum[n]);
//    }
//    float b[4] = {6,2,8,0};
//    printf("最大值在第%d个位置\n",argmax(b,4)+1);
//    char filename[] = "train-images-idx3-ubyte";
//    return 0;
//}

