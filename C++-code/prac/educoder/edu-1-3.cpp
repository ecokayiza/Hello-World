//编写一个程序，输入一个字母，如果是小写，输出它的大写；如果是大写，输出它的小写。（利用条件运算符实现）

#include <cstdio>
#include <iostream>
using namespace std;

int main()
{
    char ch;
    cin >> ch;
    char result = (ch >= 'a' && ch <= 'z') ? (ch - 'a' + 'A') : ((ch >= 'A' && ch <= 'Z') ? (ch - 'A' + 'a') : ch);
    cout << result << endl;
    return 0;
}