#include <iostream>
#include <cmath>
#include <stdio.h>

#define THREAD_NUM 1024
#define BLOCK_NUM 2048

#define HIST_SIZE 16777216 

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5 
#define OFFSET(n) \
( (n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    cerr<<"ERROR in "<<__FILE__<<':'<<__LINE__\
    <<". Message: "<<cudaGetErrorString(res)<<endl;\
    exit(0);                          \
  }                               \
} while(0);

using namespace std;



__global__
void kernel_hist(uint32_t* hist, uint32_t* arr, uint32_t n){
  uint32_t idx = blockDim.x*blockIdx.x + threadIdx.x;
  uint32_t offsetx = blockDim.x*gridDim.x;

  for(uint32_t i=idx; i<n; i+=offsetx)
    atomicAdd(hist + arr[i], 1);

}

__device__ 
void scan_block(uint32_t *data, uint32_t size){ 
  __shared__ 
  uint32_t tmp[THREAD_NUM<<1]; 
  uint32_t id = threadIdx.x, 
           offset = 1,   
           l = id, //int ai = thid; 
           r = id + (size>>1), 
           sum;


  tmp[l + OFFSET(l)] = data[l]; 
  tmp[r + OFFSET(r)] = data[r]; 

  for (uint32_t i = size>>1; i > 0; i >>= 1, offset<<=1){                   
    __syncthreads();    
    if(id < i){ 
      uint32_t l = offset*(2*id+1)-1, 
               r = offset*(2*id+2)-1;  
      
      
      tmp[r + OFFSET(r)] += tmp[l + OFFSET(l)];    
    }    

  }
  

  if(id==0){
    sum = tmp[size-1 + OFFSET(size-1)];
    tmp[size-1 + OFFSET(size-1)] = 0;
  }

  for (uint32_t i = 1; i < size; i <<= 1){ 
    offset >>= 1;
    __syncthreads();
    if (id < i){
      uint32_t l = offset*(2*id+1)-1, 
               r = offset*(2*id+2)-1;  
      uint32_t t = tmp[l + OFFSET(l)]; 
      tmp[l + OFFSET(l)] = tmp[r + OFFSET(r)]; 
      tmp[r + OFFSET(r)] += t;
    } 
  }  
  
  __syncthreads();


  if(l == 0 ){
    data[r-1] =tmp[r + OFFSET(r)];
    data[size -1] =sum; 
  }else{
    data[l-1] = tmp[l + OFFSET(l)]; 
    data[r-1] = tmp[r + OFFSET(r)]; 

  }

} 


__global__
void kernel_scan_step(uint32_t* data, uint32_t size){
  uint32_t block_size = 2*THREAD_NUM,
           block_beg = blockIdx.x*block_size,
           block_offset = gridDim.x*block_size;

  for(uint32_t cur_pos=block_beg; cur_pos < size; cur_pos +=block_offset){
    scan_block(data+cur_pos, block_size);
  }
}


__host__ 
void print_arr(uint32_t *arr, uint32_t size){
  for(uint32_t i =0; i<size && i<1000; ++i)
    cout<<arr[i]<<' ';
  cout<<endl;

}
__global__
void kernel_add(uint32_t* data, uint32_t *sums, uint32_t s_size){
  uint32_t thread_id = threadIdx.x,
           block_id = blockIdx.x,
           block_size = 2*THREAD_NUM,
           offset = gridDim.x;
  for(uint32_t i = block_id; i < s_size-1; i += offset){
    data[(i+1)*block_size + 2*thread_id   ] += sums[i];
    data[(i+1)*block_size + 2*thread_id+1 ] += sums[i];
  }

}



__host__
uint32_t make_full_block(uint32_t size){
  if(size%(2*THREAD_NUM))
    size = (size/(2*THREAD_NUM) +1) * 2*THREAD_NUM;
  return size;
}

__host__
void scan(uint32_t *data, uint32_t size){
  uint32_t blocks_num=size/(2*THREAD_NUM);

  kernel_scan_step<<<min(BLOCK_NUM, blocks_num), THREAD_NUM>>>(data, size);
  CSC(cudaGetLastError());  
  
  if(blocks_num==1)
    return;

  uint32_t *block_sums,
            full_blocks_num = make_full_block(blocks_num);
  CSC(cudaMalloc(&block_sums, sizeof(uint32_t)*full_blocks_num));
  CSC(cudaMemcpy2D(block_sums, sizeof(uint32_t), data + 2*THREAD_NUM -1, 2*THREAD_NUM*sizeof(uint32_t), sizeof(uint32_t), blocks_num, cudaMemcpyDeviceToDevice));
  
  if(full_blocks_num >= blocks_num)
    CSC(cudaMemset(block_sums + blocks_num, 0, sizeof(uint32_t) * (full_blocks_num - blocks_num) ));
  
  scan(block_sums, full_blocks_num);

  kernel_add<<<BLOCK_NUM, THREAD_NUM>>>(data, block_sums, blocks_num);
  CSC(cudaGetLastError());

  CSC(cudaFree(block_sums));

}

__global__
void place_element(uint32_t* arr, uint32_t* hist, uint32_t size){
  int32_t block_idx = blockIdx.x,
          block_offset = gridDim.x,
          thr_idx = threadIdx.x,
          thr_offset = blockDim.x;
  
  int prev=0; 
  for(uint32_t i=block_idx; i<size; i+=block_offset){
    if(i>0)
      prev = hist[i-1];

    for(int j=int(hist[i])-1 - thr_idx; j>= prev ; j-=thr_offset){
      arr[j]=i;
    }
  }
}

__host__
void count_sort(uint32_t* arr, const uint32_t size){
  
  if( size == 0)
    return;

  uint32_t *dev_hist, *dev_arr;
  
  CSC(cudaMalloc(&dev_hist, sizeof(uint32_t) * HIST_SIZE ));
  CSC(cudaMalloc(&dev_arr, sizeof(uint32_t) * size ));

  CSC(cudaMemset(dev_hist, 0, sizeof(uint32_t) * HIST_SIZE ));
  CSC(cudaMemcpy(dev_arr, arr, sizeof(uint32_t) * size, cudaMemcpyHostToDevice));

  kernel_hist<<<BLOCK_NUM, THREAD_NUM>>>(dev_hist, dev_arr, size);
  CSC(cudaGetLastError());

  scan(dev_hist, HIST_SIZE);
  CSC(cudaGetLastError());

  
  place_element<<<BLOCK_NUM, THREAD_NUM>>>(dev_arr, dev_hist, HIST_SIZE);
  CSC(cudaGetLastError());

  CSC(cudaMemcpy(arr, dev_arr, sizeof(uint32_t)*size, cudaMemcpyDeviceToHost));
  
  
  CSC(cudaFree(dev_hist));
  CSC(cudaFree(dev_arr));
}



void read_arr(uint32_t** arr, uint32_t& size ){
  cin.read((char*)(&size), sizeof(uint32_t));
  
  uint32_t *tmp = new uint32_t[size]();
  *arr = tmp;
  cin.read((char*)*arr, sizeof(uint32_t)*size);
}

void write_arr(uint32_t* arr, const uint32_t& size ){
  cout.write((char*)arr, size*sizeof(uint32_t));
}

__host__
int main(){
  
  uint32_t size, *arr;
  read_arr(&arr, size);

  count_sort(arr, size);

  print_arr(arr, size);  
  //write_arr(arr, size);

  delete [] arr;
  
  return 0;
}