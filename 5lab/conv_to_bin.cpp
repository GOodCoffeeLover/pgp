#include <iostream>

using namespace std;

int main(){
  int n;
  cin>>n;
  while(cin.good()){
    cout.write((char*)(&n), sizeof(n));
    cin>>n;
  }
  return 0;
}