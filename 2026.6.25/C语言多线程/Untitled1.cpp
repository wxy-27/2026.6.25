#include<bits/stdc++.h>
using namespace std;

void fun1(string s){
	cout << s << endl;
}

void fun2(){
	cout << "222" << endl;
}

void fun3(){
	for(int i = 1; i <= 10; i ++) cout << i << endl;  
}

//用join（）来完成线程 
int main(){
	//thread线程 
	thread thread1(fun1, "!!!"); //传递参数直接“， C ” 
	thread1.join(); //主线程等待结束 
	
	thread thread2(fun2);
	thread2.detach(); // 在操作系统后台运行，没有join（）等待结束
	cout << "PP" << endl;
	
	thread thread3(fun3);
	bool f = thread3.joinable();  
	if(f){
		cout << "***" << endl;
		thread3.join(); //阻塞程序 
	}
	cout << "&&&" << endl; //需要等待线程3执行完毕后才打印 
	return 0;
}
