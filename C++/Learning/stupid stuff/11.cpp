#include <iostream>
using namespace std;

bool prime(int n);
int main()
{
    int i = 2;
    do{
        if (prime(i)){
            cout << i << endl;
        }
        i++;
    }while(i <= 100)
}

bool prime(int n)
{
    for (int i = 2; i < n; i++){
        if (n%i == 0){
            return false;
        }
    }
    return true;
}