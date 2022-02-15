#include <iostream>
#include <string>

using namespace std;

int main(int argc, char ** argv){

  int size;
  if(argc<2)
    size=1000;
  else
   size = stoi(string(argv[1]));
  int equ[5][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {31, 12, 99, 14, 15}, {50, 17, 18, 10, 75}, {43, 22, 23, 55, 77}};
  int free[5] = {26, 27, 28, 29, 30};
  cout<<size<<endl;
  for(int i=0; i<size; ++i ){
    for(int j=0; j<size; ++j){
      if( i/5 == j/5)
        cout<<equ[i%5][j%5]<<' ';
      else
        cout<<0<<' ';
    }
    cout<<endl;
  }
    for(int i=0; i<size; ++i)
      cout<<free[i%5]<<' ';
    cout<<endl;

  return 0;
}