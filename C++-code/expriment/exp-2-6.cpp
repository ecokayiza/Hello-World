#include<iostream>
using namespace std;

int main()
{
    int i=1, S=0;
    while(i<=100)
    {
        S=S+i;
        i=i+1;
    }
    cout<<"S="<<S<<endl;
}