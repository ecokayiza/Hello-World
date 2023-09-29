#include<iostream>
using namespace std;

int main()
{
    int a,i=-1;
    int digits[4];
    cin >> a;
    if(a<0){a = -a;}
    while(a>0)
    {
        i = i-1;
        digits[i] = a % 10;
        cout << digits[i]<<",";
        a = a / 10;
        
    }
    system("pause");
}