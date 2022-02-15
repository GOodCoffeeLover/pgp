#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KX 32
#define KY 32

#define BX 32
#define BY 32


#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0);

texture<uchar4,  2, cudaReadModeElementType> tex;
texture<float4, 2, cudaReadModeElementType> tex1;


__global__ void kernel_gauss_first_iter(float4* out, int w, int h, int r, float* f){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;
  int x=0, y=0;
  uchar4 p;
  for(y=idy; y < h; y+=offsety){
    for(x=idx; x < w; x+=offsetx){
      out[w*y+x]=make_float4(0,0,0,0);
      //out[h*x+y]=make_float4(0,0,0,0);
      for(int k=-r; k<=r; k+=1){
        p = tex2D(tex, x+k,y);
        out[w*y+x].x +=((float)p.x)*f[r+k];
        out[w*y+x].y +=((float)p.y)*f[r+k];
        out[w*y+x].z +=((float)p.z)*f[r+k];
        out[w*y+x].w +=((float)p.w)*f[r+k];
        // out[h*x+y].x +=((float)p.x)*f[r+k];
        // out[h*x+y].y +=((float)p.y)*f[r+k];
        // out[h*x+y].z +=((float)p.z)*f[r+k];
        // out[h*x+y].w +=((float)p.w)*f[r+k];
      }
    }  
  }
}

__global__ void kernel_gauss_second_iter(uchar4* out, int w, int h, int r, float* f){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;
  int x=0, y=0;
  float4 p;
  for(y=idy; y < h; y+=offsety)
    for(x=idx; x < w; x+=offsetx){
      //out[w*y+x]=make_uchar4(0,0,0,0);
      float4 tmp=make_float4(0,0,0,0);
      for(int k=-r; k<=r; k+=1){
        p = tex2D(tex1, x, y+k);
        tmp.x += p.x*f[r+k];
        tmp.y += p.y*f[r+k];
        tmp.z += p.z*f[r+k];
        tmp.w += p.w*f[r+k];
      }
      out[w*y+x]= make_uchar4(tmp.x, tmp.y, tmp.z, tmp.w );
      
    }
}

float* createFilterOnDevice(int r){
  float *filter = (float *) malloc(sizeof(float)*(2*r+1));
  float sum=0.0;  
  if(r!=0){
    for(int i=0; i<=r; i+=1){
      filter[r-i]=exp(-((float)i*i)/(2*r*r));
      filter[r+i] = filter[r-i];
      if(i!=0)
        sum+=2*filter[r-i];
      else
        sum+=filter[r-i];
    }
    for(int i=0; i<2*r+1; i+=1){
      filter[i]/=sum;

    }
  }else{
    filter[0]=1;
  }

  float* dev_filter;
  CSC(cudaMalloc(&dev_filter, sizeof(float)*(2*r+1)));
  CSC(cudaMemcpy(dev_filter, filter, sizeof(float)*(2*r+1), cudaMemcpyHostToDevice));
  free(filter);
  return dev_filter;
}

int main(){
  int w=0, h=0, r=0;
  char file_name[256];
  scanf("%s", file_name);
  FILE *fp = fopen(file_name, "rb");
  fread(&w, sizeof(unsigned int), 1, fp);
  fread(&h, sizeof(unsigned int), 1, fp);
  uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
  fread(data, sizeof(uchar4), w * h, fp);
  fclose(fp);
  scanf("%s", file_name);
  scanf("%d", &r);

  cudaArray *arr;
  
  cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
  CSC(cudaMallocArray(&arr, &ch, w, h));
  CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
  
  
  float* dev_filter = createFilterOnDevice(r);
  // Подготовка текстурной ссылки, настройка интерфейса работы с данными
  tex.addressMode[0] = cudaAddressModeClamp;  // Политика обработки выхода за границы по каждому измерению
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.channelDesc = ch;
  tex.filterMode = cudaFilterModePoint;      // Без интерполяции при обращении по дробным координатам
  tex.normalized = false;                    // Режим нормализации координат: без нормализации

  CSC(cudaBindTextureToArray(tex, arr, ch));

  float4 *intermediate;
  CSC(cudaMalloc( &intermediate, sizeof(float4)*w*h));
  
  kernel_gauss_first_iter<<<dim3(KX, KY), dim3(BX,BY)>>>(intermediate, w,h,r,dev_filter);
  CSC(cudaGetLastError());
 
 

  CSC(cudaUnbindTexture(tex));
  CSC(cudaFreeArray(arr));
 


  cudaChannelFormatDesc dbl = cudaCreateChannelDesc<float4>();
  
  // Подготовка текстурной ссылки, настройка интерфейса работы с данными
  tex1.addressMode[0] = cudaAddressModeClamp;  // Политика обработки выхода за границы по каждому измерению
  tex1.addressMode[1] = cudaAddressModeClamp;
  tex1.channelDesc = dbl;
  tex1.filterMode = cudaFilterModePoint;      // Без интерполяции при обращении по дробным координатам
  tex1.normalized = false;                    // Режим нормализации координат: без нормализации

  cudaArray *arr1;
  CSC(cudaMallocArray(&arr1, &dbl, w, h));
  CSC(cudaMemcpyToArray(arr1, 0, 0, intermediate, sizeof(float4) * w * h, cudaMemcpyDeviceToDevice));
  CSC(cudaBindTextureToArray(tex1, arr1, dbl));
  
  uchar4 *dev_out;
  CSC(cudaMalloc(&dev_out, sizeof(uchar4)*w*h));

  kernel_gauss_second_iter<<<dim3(KX, KY), dim3(BX,BY)>>>(dev_out, w,h,r,dev_filter);
  CSC(cudaGetLastError());

  CSC(cudaUnbindTexture(tex));
  cudaMemcpy(data, dev_out, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost);
  
  fp = fopen(file_name, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(data, sizeof(uchar4), w * h, fp);
  fclose(fp);

  CSC(cudaFreeArray(arr1));
  CSC(cudaFree(intermediate));
  CSC(cudaFree(dev_out));
  CSC(cudaFree(dev_filter));
  free(data);
  

  return 0;
}