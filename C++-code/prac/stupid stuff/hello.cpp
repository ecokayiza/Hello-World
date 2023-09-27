#include<iostream>
using namespace std;

int main()
{
    int a, b, c;
    cin >> a >> b >> c;
    int temp = a;
    a = b;
    b = c;
    c = temp;
    cout <<  a << " " << b << " "<<  c << endl;
    return 0; 
}