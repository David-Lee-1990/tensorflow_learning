#include "func.h"
#include <assert.h>

//英特尔处理器和其他低端机用户必须翻转头字节。
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// 读取mnist数据集数据
ImgArr read_Img(const char* filename) // 共60，000张图片
{
    FILE *fp = NULL;
    fp = fopen(filename,"rb");

    if (fp==NULL)
    {
        printf("open file failed.\n");
    }
    assert(fp);
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number
    fread((char*)&magic_number,sizeof(magic_number),1,fp);
    magic_number = ReverseInt(magic_number);
    //获取训练或测试image的个数number_of_images
    fread((char*)&number_of_images,sizeof(number_of_images),1,fp);
    number_of_images = ReverseInt(number_of_images);
    //获取训练或测试图像的高度Heigh
    fread((char*)&n_rows,sizeof(n_rows),1,fp);
    n_rows = ReverseInt(n_rows);
    //获取训练或测试图像的宽度Width
    fread((char*)&n_cols,sizeof(n_cols),1,fp);
    n_cols = ReverseInt(n_cols);
    //获取第i幅图像，保存到vec中
    int i,r,c;
    
    ImgArr imgarr = (ImgArr)malloc(sizeof(ImgArr));
    imgarr->ImgNum = number_of_images;
    imgarr->ImgPtr = (MinstImg*)malloc(number_of_images*sizeof(MinstImg));

    for(i=0;i<number_of_images;i++)
    {
//        printf("第%d张图片！",i);
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (float**)malloc(n_rows*sizeof(float*));
        for(r=0;r<n_rows;r++)
        {
            imgarr->ImgPtr[i].ImgData[r] = (float*)malloc(n_cols*sizeof(float));
            for(c=0;c<n_cols;c++)
            {
                unsigned char temp = 0;
                fread((char*)&temp, sizeof(temp), 1, fp);
                imgarr->ImgPtr[i].ImgData[r][c] = (float)temp/255.0;
            }
        }
    }
    fclose(fp);
    return imgarr;
}


int main()
{
    char filename[] = "train-images-idx3-ubyte";
    read_Img(filename);
    return 0;
}
