#include<bits/stdc++.h>
using namespace std;

void fun1(int x){
	x ++;
}

int main(){
	int a = 1;
	thread td1(fun1, ref(a)); 
	td1.join();
	cout << a << endl;
	return 0;
}
