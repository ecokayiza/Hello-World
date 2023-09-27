//（编写程序，读入一个三位整数，计算并输出三个数字之和。
//（如输入234，输出9（2+3+4）））

#include <cstdio>
#include <iostream>
using namespace std;
int main()
{
 int a;
 cin>>a;
 int sum=0;
 while(a>0)
 {
  sum+=a%10;
  a=a/10;
 }
 cout << sum << endl;
 return 0;
}