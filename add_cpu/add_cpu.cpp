#include <stdio.h>
#include <iostream>
#include <vector>

const size_t SIZE = 1 <<20;
void add_cpu(std::vector<float> &c,const std::vector<float> &a,const std::vector<float> &b){
    for (size_t i =0;i< a.size();i++){
        c[i] = a[i] +b[i];
    }

    std::cout << "执行完毕" << '\n'; 
    std::cout <<"c[SIZE-1]:"<< c[SIZE-1] << '\n'; 
}

int main(){
    std::vector<float> a(SIZE,1);
    std::vector<float> b(SIZE,2);
    std::vector<float> c(SIZE,0);

    add_cpu(c,a,b);
    return 0;
}