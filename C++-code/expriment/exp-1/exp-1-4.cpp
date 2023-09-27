//<1> 根据随机从键盘输入的圆半径值，求圆的周长和半径并输出。

#include<iostream>
using namespace std;

int main()
{
    float r;
    const float pi = 3.14159;
    cin >> r;
    cout << "the perimeter is " << 2 * pi * r << endl;
    cout << "the radius is " << r << endl;
}