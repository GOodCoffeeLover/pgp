#include <iostream>
#include <time.h>

using namespace std;

#define HIST_SIZE 16777216 

void print_arr(uint32_t *data, uint32_t size){
  for(uint32_t i=0; i<size && i < 1000; ++i)
    cout<<data[i]<<' ';
  cout<<endl;
}

void hist(uint32_t *data, uint32_t size, uint32_t *h){
  for(uint32_t i=0; i<size; ++i)
    h[data[i]] +=1;
}


void scan(uint32_t *data, uint32_t size){
  for(uint32_t i=1; i<size; ++i)
    data[i] += data[i-1];
}


void place_elem(uint32_t* data, uint32_t* h, uint32_t h_size){
  int32_t prev =0;
  for(uint32_t i=0; i<h_size; ++i){
     if(i)
      prev=h[i-1];
    for(int32_t j=h[i]-1; j>=prev; --j )
      data[j] = i; 
  }
}

void count_sort(uint32_t *data, uint32_t size){

  uint32_t *h = new uint32_t[HIST_SIZE]();
  if(h == nullptr){
    cerr<<"cant alloc hist \\w size = "<<HIST_SIZE<<endl;
    return;
  }
  hist(data, size, h);
  scan(h, HIST_SIZE);
  place_elem(data, h, HIST_SIZE);
 
  delete [] h;
}

void read_arr(uint32_t*& arr, uint32_t& size ){
  cin.read((char*)(&size), sizeof(uint32_t));
  
  arr = new uint32_t[size]();
  
  cin.read((char*)arr, sizeof(uint32_t)*size);
}


void write_arr(uint32_t* arr, const uint32_t& size ){

  cout.write((char*)arr, size*sizeof(uint32_t));
}



int main(){
  
  uint32_t size, *arr;
  read_arr(arr, size);

  clock_t start = clock();
  count_sort(arr, size);
  clock_t end = clock();
  double miliseconds = (double)(end - start) / CLOCKS_PER_SEC*1000.0;
  printf("The time: %f miliseconds\n", miliseconds);

  //print_arr(arr, size);  
  // print_arr(arr, size);
  // cout<<"deleting arr"<<endl;
  delete [] arr;
  
  return 0;
}