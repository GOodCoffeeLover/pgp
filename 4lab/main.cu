#include <iostream>
#include <cmath>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define endl '\n'

#define GY 64
#define GX 32

#define BY 32
#define BX 32


#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    cerr<<"ERROR in "<<__FILE__<<':'<<__LINE__\
    <<". Message: "<<cudaGetErrorString(res)<<endl;\
    exit(0);                          \
  }                               \
} while(0);


struct comparator{  
  __device__ __host__
  bool operator ()(double lhs, double rhs){
    return fabs(lhs) < fabs(rhs);
  }
};

using namespace std;

__global__
void kernel_swap(double* mtrx, int n, int ii, int jj){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int offsetx = blockDim.x*gridDim.x;
  double tmp;
  for(int i=idx; i<n+1; i+=offsetx){
    tmp = mtrx[n*i + ii];
    mtrx[n*i + ii] = mtrx[n*i + jj];
    mtrx[n*i + jj] = tmp;
  }
}

__global__
void kernel_gauss_step(double *matrix, int n, int i){
  
  if(fabs(matrix[n*i+i])<0.0000001)
    return;
  
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int offsetx = blockDim.x*gridDim.x;
  int offsety = blockDim.y*gridDim.y;


  for(int k = i+1 + idy; k<n+1; k+=offsety ){
    for(int j = i+1 + idx; j<n; j+=offsetx ){
      matrix[ k*n + j ] -= matrix[ k*n + i ] * matrix[ i*n + j ] / matrix[ i * n + i ];
    }
  }
}

__host__
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
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  int n;
  cin>>n;
  
  double* matrix=read_slau(n);
  double* dev_matrix;
  CSC(cudaMalloc(&dev_matrix, sizeof(double)*n*(n+1)));
  CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double)*n*(n+1), cudaMemcpyHostToDevice));

  comparator cmp;
  thrust::device_ptr<double> i_ptr, i_max_ptr;
  
  for(int i=0; i<n-1; ++i){

    int i_max=i;
    i_ptr = thrust::device_pointer_cast(dev_matrix + i * n);
    i_max_ptr = thrust::max_element(i_ptr + i, i_ptr + n, cmp);
    i_max = i_max_ptr - i_ptr;
    
    if(i_max != i)
      kernel_swap<<<GX*GY,BX*BY>>>(dev_matrix, n, i, i_max);
    CSC(cudaGetLastError());
    
    kernel_gauss_step<<<dim3(GX, GY), dim3(BX, BY)>>>(dev_matrix, n, i);
    CSC(cudaGetLastError());
  }

  CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double)*n*(n+1), cudaMemcpyDeviceToHost));

  double* ans = (double*) new double[n];

  for(int i=n-1; i>=0; --i){
    ans[i] = matrix[n*n+i];
    for(int j=n-1; j>i; --j){
      ans[i] -= ans[j]*matrix[j*n+i];
    }
    ans[i]/=matrix[i*n+i];
  }

  std::cout.precision(10);
  std::cout.setf(std::ios::scientific);
  for(int i=0; i<n; ++i)
    cout<<ans[i]<<' ';
  cout<<endl;

  delete [] matrix;
  delete [] ans;
  CSC(cudaFree(dev_matrix));
  return 0;
}