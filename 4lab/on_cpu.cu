#include <iostream>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <time.h>

using namespace std;


#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    cerr<<"ERROR in "<<__FILE__<<':'<<__LINE__\
    <<". Message: "<<cudaGetErrorString(res)<<endl;\
    exit(0);                          \
  }                               \
} while(0);


struct comparator
{  __device__ __host__
  bool operator ()(double lhs, double rhs){
    return lhs < rhs;
  }
};

int arg_max(double* mtrx, int l, int r){
  int i_max=l;

  for(int i=l+1; i<r; ++i)
    
    if(fabs(mtrx[i_max])<fabs(mtrx[i]))
      i_max=i;
  
  return i_max;
}

void swap(double* mtrx, int n, int ii, int jj){
  int tmp;
  for(int i=0; i<n+1; ++i){
    tmp = mtrx[n*i + ii];
    mtrx[n*i + ii] = mtrx[n*i + jj];
    mtrx[n*i + jj] = tmp;
  }
}

void gauss_step(double *matrix, int n, int i){
  for(int k = i+1; k<n+1; ++k ){
    for(int j = i+1; j<n; ++j ){
      matrix[ k*n + j ] -= matrix[ k*n + i ] * matrix[ i*n + j ] / matrix[ i * n + i ];
    }
  }
}

double *read_slau(int n){
  double *matrix = (double *) new double[n*(n+1)];
  for(int i=0; i<n; ++i)
    for(int j=0; j<n; ++j)
      cin>>matrix[ n*j + i ];

  for(int j=0; j<n; ++j)
    cin>>matrix[ n*n + j ];
  
  return matrix;
}

__host__
int main(){
  int n;
  cin>>n;
  
  double* matrix=read_slau(n);

  int* swaped = new int[n];
  for(int i=0; i<n; ++i)
    swaped[i]=i;


  clock_t start = clock();

  for(int i=0; i<n-1; ++i){

    int i_max=arg_max(matrix, n*i+i, n*(i+1));
    i_max -= n*i;

    if(i_max != i){
      swap(matrix, n, i, i_max);
      swap(swaped[i], swaped[i_max]);
    }



    if(fabs(matrix[n*i+i])>=0.0000001)
      gauss_step(matrix, n, i);
  }
  clock_t end = clock();
  double miliseconds = (double)(end - start) / CLOCKS_PER_SEC*1000.0;
  printf("The time: %f miliseconds\n", miliseconds);

  double* ans = (double*) new double[n];

  for(int i=n-1; i>=0; --i){
    ans[i] = matrix[n*n+i];
    for(int j=n-1; j>i; --j){
      ans[i] -= ans[j]*matrix[j*n+i];
    }
    ans[i]/=matrix[i*n+i];
  }

  // std::cout.precision(10);
  // std::cout.setf(std::ios::scientific);
  // for(int i=0; i<n; ++i)
  //   cout<<ans[i]<<' ';
  // cout<<endl;

  delete [] matrix;
  delete [] ans;
  delete [] swaped;

  return 0;
}