#include<iostream>
using namespace std;

int main()
{
    int m, n;
    m = 1000;
    n = 850;
    cout<<"\n ("<<m<<","<<n<<")";
    while (m!=n)
    {
        while(m>n)
        {
            m = m - n;
            cout<<"\n ("<<m<<","<<n<<")";
        }
        while(m<n)
        {
            n = n - m;
            cout<<"\n ("<<m<<","<<n<<")";
        }
    }
    cout << "\n" << m;

}