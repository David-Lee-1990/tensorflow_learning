#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct Mat2DSize{
    int c; // 列（宽）
    int r; // 行（高）
}nSize;

float** rotate180(float** mat, nSize matSize)// 矩阵翻转180度
{
    int i,c,r;
    int outSizeW=matSize.c;
    int outSizeH=matSize.r;
    float** outputData=(float**)malloc(outSizeH*sizeof(float*));
    for(i=0;i<outSizeH;i++)
        outputData[i]=(float*)malloc(outSizeW*sizeof(float));
    
    for(r=0;r<outSizeH;r++)
        for(c=0;c<outSizeW;c++)
        {
//            outputData[r][c]=mat[outSizeH-r-1][outSizeW-c-1];
            outputData[r][c] = *(*mat+r*outSizeH+c);
            printf("a[%d][%d]=%f\n",r,c,outputData[r][c]);
        }
    return outputData;
}

int Dayin(int** array, int m, int n){
    for (int i=0;i<m;i++)
    {
        for (int j=0;j<n;j++)
        {
            printf("%d\n",*((int*)array+i*n+j)); // c写成array[i][j]会报错！！
        }
    }
    return 0;
}

int main(){
    float a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
    float f =2;
    printf("size of float is :%lu\n",sizeof(f));
    int m = 3;
    int n = 3;
    nSize shape;
    shape.c = m;
    shape.r = n;
    float* path1;
    path1 = a[0];
    float** path2;
    path2 = &path1;
//    rotate180((float**)a,shape);
    rotate180(path2,shape);
    return 0;
};
