#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define uchar unsigned char

int main(){
  int w=0, h=0, r=0;
  char file_name[256];
  scanf("%s", file_name);
  FILE *fp = fopen(file_name, "rb");
  fread(&w, sizeof(unsigned int), 1, fp);
  fread(&h, sizeof(unsigned int), 1, fp);
  uchar *data = (uchar *)malloc(sizeof(uchar)* 4 * w * h);
  fread(data, sizeof(uchar), 4*w * h, fp);
  fclose(fp);
  scanf("%s", file_name);
  scanf("%d", &r);
  printf("w:%d, h:%d\n", w,h);


  float *intrm = (float *)malloc(sizeof(float)* 4 * w * h);
  

  float *filter = (float *) malloc(sizeof(float)*(2*r+1));
  float sum=0.0;  
  
  for(int i=0; i<=r; i+=1){
      filter[r-i]=exp(-((float)(i*i))/((float)2*r*r));
      filter[r+i] = filter[r-i];
      if(i!=0)
        sum+=2*filter[r-i];
      else
        sum+=filter[r-i];
    }
    for(int i=0; i<2*r+1; i+=1){
      filter[i]/=sum;
     // printf("%f\n", filter[i] );
    }
    // for(int y=0; y<h; y+=1){
    //   for(int x=0; x<w; x+=1){
    //     for(int i=0; i<4; i+=1){
    //       printf("%d ", data[4*w*y+4*x+i]);
    //     }
    //     printf("|");
    //   }
    //   printf("\n");
    // }


    clock_t start = clock();
    for(int y=0; y<h; y+=1)
      for(int x=0; x<w; x+=1){
        for(int i=0; i<4; i+=1){
          
          intrm[4*w*y+4*x+i]=0.0;
          for(int k=-r; k<=r; k+=1){
            int xx=fmax(fmin(x+k, w-1),0);
            int yy=fmax(fmin(y, h-1),0);
            intrm[4*w*y+4*x+i]+=(float)((float)data[4*w*yy+4*xx+i])*filter[r+k];
          }

        }
      }
    

    // for(int y=0; y<h; y+=1){
    //   for(int x=0; x<w; x+=1){
    //     for(int i=0; i<4; i+=1){
    //       printf("%2.2f ", intrm[4*w*y+4*x+i]);
    //     }
    //     printf("|");
    //   }
    //   printf("\n");
    // }

    for(int x=0; x<w; x+=1)
      for(int y=0; y<h; y+=1){
        for(int i=0; i<4; i+=1){
          float tmp=0.0;
          for(int k=-r; k<=r; k+=1){
            int xx=fmax(fmin(x, w-1),0);
            int yy=fmax(fmin(y+k, h-1),0);
            tmp+=intrm[4*w*yy+4*xx+i]*filter[r+k];
          }
          data[4*w*y+4*x+i]=tmp;
        }
      }
    
  // for(int y=0; y<h; y+=1){
  //     for(int x=0; x<w; x+=1){
  //       for(int i=0; i<4; i+=1){
  //         printf("%d ", data[4*w*y+4*x+i]);
  //       }
  //       printf("|");
  //     }
  //     printf("\n");
  //   }

  clock_t end = clock();
  double seconds = (double)(end - start) / CLOCKS_PER_SEC*1000.0;
  printf("The time: %f miliseconds\n", seconds);
  

  fp = fopen(file_name, "wb");
  fwrite(&w, sizeof(int), 1, fp);
  fwrite(&h, sizeof(int), 1, fp);
  fwrite(data, sizeof(unsigned char), 4 * w * h, fp);
  fclose(fp);
  free(data);
  free(filter);

  return 0;
}