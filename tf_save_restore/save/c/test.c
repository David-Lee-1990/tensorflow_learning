//  判断一个数是奇偶数

#include <stdio.h>

void judge_sd(int a)
{
    if ((a & 1))
    {
        printf("是奇数\n");
        return;
    }
    else
    {
        printf("是偶数\n");
        return;
    }
}

int main()
{
    judge_sd(0);
    judge_sd(1);
    judge_sd(4);
    return 0;
}
