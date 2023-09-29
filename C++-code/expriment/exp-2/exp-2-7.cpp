#include<iostream>
#include<cmath>
#include<iomanip>
using namespace std;
int main()
{
    double x, y, term, n = 1.0;
    cout << "X=";
    cin >> x;
    if(abs(x)-3.14159>1e-3)
    {
        cout<<"erro!!";
    }
    else{
        term = x;
        y = x;
        while(abs(term)>1e-7)
        {
            term = (term * (-1.0) * pow(x, 2) )/ (2.0*n*(2.0*n + 1.0));
            y += term;
            n++;
        }
        cout << "Y = " << fixed << setprecision(6)<<y<<endl;
    }
    
}


