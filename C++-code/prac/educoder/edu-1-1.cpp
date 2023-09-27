//编写程序，读入圆柱体的半径和高，计算并输出圆柱体的表面积和体积）

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
const double PI = 3.14;
int main()
{
     double r, h;
    cin >> r;
    cin >> h;
    double S = 2 * PI * r * (r + h);
    double V = PI * h * pow(r, 2);
    cout << setiosflags(ios::fixed) << setprecision(2);
    cout << S << " " << V << endl;
    return 0;
}