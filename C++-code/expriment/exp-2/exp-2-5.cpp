#include<iostream>
#include<cmath>
using namespace std;

int main()
{
    float z, x, y;
    cout << "x=";
    cin >> x;
    cout << "y=";
    cin >> y;
    if(x<0,y<0)   
    {
        z=exp(x+y);
    }
    else if (1<=x+y&&x+y<10)
    {
        z=log(x+y);
    }
    else
    {
        z=log10(abs(x+y)+1);
    }
    cout<<"z="<<z<<endl;
}
