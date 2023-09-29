（1）阅读下列程序，写出（由指定的输入）所产生的运行结果，并指出其功能。
<1> 
	#include <iostream.h>

	void main()
	{
int m,n;
m = 1000;
n = 850;
		
cout << “\n(“ << m << ‘,’ << n << ‘)’;
while ( m != n ) {
while ( m > n ) {
m = m – n ;
cout << ‘(‘ << m << ‘,’ << n << ‘)’;
}
while ( m < n ) {
n = n – m ;
cout << ‘(‘ << m << ‘,’ << n << ‘)’; 
}
		}
cout << “\n” << m ;
}

<2>
	#include <iostream.h>

	void main()
	{
		int m,n,k ;
	
m = 1000 ;
n = 45 ;
cout << “\n(“ << m << ‘:’ << n << ‘)’ ;
k=0;
while ( m >=n ){
m = m – n ;
k = k + 1 ;
		}
    	cout << k << “---” << m << endl ;
}

<3>
	#include <iostream.h>

	void main()
	{
		int i;

		for ( i = 1 ; i <= 5 ; i ++ ){
			if ( i % 2 ) 
cout << ‘*’;
			else 
continue;
			cout << ‘#’ ;
		}
		cout << “$\n” ;
}

<4> 
#include <iostream.h>

void main()
{
	int a = 1,b = 10;

	do{
		b -= a ;
		a ++ ;
	}while ( b -- <= 0 ) ;
	cout << “a=” << a << “b=” << b <<endl ;
}

（2）编写程序实现下列问题的求解。
<1> 求解下面函数的值。
需要包含头文件 #include <math.h> 或<cmath>
函数名: log
功能：log() 函数返回以 e 为底的对数值，其原型为：
double log (double x);
log()用来计算以e为底的 x 的对数值，然后将结果返回。
函数名: log10
功 能: 对数函数log,函数返回以10为底的对数值，其原型为：
double log10(double x);
Log10()用来计算以10为底的 x 的对数值，然后将结果返回。
函数名： exp
功能：exp()用来计算以e 为底的x 次方值，然后将结果返回。其原型为：
 double exp(double x);
exp()用来返回 e 的x 次方计算结果。

               ex+y                     x<0,y<0
         z=   ln(x+y)                 1≤x+y〈10
              log10|x+y|+1               其它情况

<2> 编程求解下列各计算式:
 1)      Ｓ＝  ＝１＋２＋３＋…＋１００
 4)　　　Y=X－ ＋ － ＋…＋(-1)n+1 ＋…的值,精确到10-6。
