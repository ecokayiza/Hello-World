//编写程序，读入秒数，计算秒数对应的时分秒(如读入3700秒，输出1h 1min 40s

#include <cstdio>
#include <iostream>
using namespace std;

int main()
{
    int seconds;
    cin >> seconds;
    int hours = seconds / 3600; 
    seconds %= 3600; 
    int minutes = seconds / 60; 
    seconds %= 60; 
    cout << hours << "h " << minutes << "min " << seconds << "s" << endl;
    return 0;
}