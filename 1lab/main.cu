#include <stdio.h>

__global__ 
void revers_kernel(double* arr, unsigned long long n){
  unsigned long long i, pos =  blockIdx.x * blockDim.x + threadIdx.x;
  double tmp;
  for(i=pos; i<n/2; i+=blockDim.x*gridDim.x){
    tmp = arr[i];
    arr[i]=arr[n-1-i];
    arr[n-1-i]=tmp;
  }

}

// Trofimov Max lab 1 var 8
int main(){
  unsigned long long i, n;
  scanf("%llu", &n);
  double *arr=(double*)malloc(sizeof(double)*n), *dev_arr;
  for(i=0; i<n; ++i)
    scanf("%lf", &arr[i]);
    
  cudaMalloc(&dev_arr, sizeof(double)*n);
  cudaMemcpy(dev_arr, arr, sizeof(double)*n, cudaMemcpyHostToDevice);

  revers_kernel<<<2,2>>>(dev_arr, n);

  cudaMemcpy(arr, dev_arr, sizeof(double)*n, cudaMemcpyDeviceToHost);
  cudaFree(dev_arr);

  for(i=0; i<n; ++i)
    printf("%.10le ", arr[i]);
  printf("\n");
  free(arr);

  return 0;
}


  1 2 3 4 5 6 7 
  4 3 4 2 2 3 4
              4

  0 |  1 |  2 |  3 | 4
  0    0    2    4   7
   
