#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>

#include <mpi.h>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define file_name_len 128

#define GRID_DIM_3D  dim3(2, 2, 2 )
#define BLOCK_DIM_3D dim3(8, 8, 8 )

#define GRID_DIM_2D  dim3(4, 4)
#define BLOCK_DIM_2D dim3(16, 16)


#define _bid(i,j,k) (((k) + 1) * (arg.bx + 2)*(arg.by + 2) + ((j) + 1) * (arg.bx + 2) + (i) + 1)
#define _pid(i,j,k) ((k) * arg.x*arg.y + (j) * arg.x + (i))

#define _x(id) ((id) % arg.x)
#define _y(id) ((id) %(arg.y*arg.x) /arg.x)
#define _z(id) ((id)/(arg.x*arg.y))

#define _bx(bid) ( ( (bid) % (arg.bx +2) ) - 1 )
#define _by(bid) ( ( (bid) % ( (arg.bx+2) * (arg.by+2) ) / (arg.bx +2) ) - 1 )
#define _bz(bid) ( (bid) / ( (arg.bx+2) * (arg.by+2) )  - 1)

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0);

struct Args{
  int x,y,z;
  int bx, by, bz;
  char name[file_name_len];
  double eps;
  double lx, ly, lz;
  double ud, uu, ul, ur, uf, ub;
  double u0;
};

struct Buffers{
  double 
  *down, *up, 
  *left, *right, 
  *front, *back;
  double *XOY, *XOZ, *YOZ;
};

struct comparator{  
  __device__ __host__
  bool operator ()(double lhs, double rhs){
    return fabs(lhs) < fabs(rhs);
  }
};

void read_Args(Args& arg){
  scanf("%d", &arg.x);
  scanf("%d", &arg.y);
  scanf("%d", &arg.z);

  scanf("%d", &arg.bx);
  scanf("%d", &arg.by);
  scanf("%d", &arg.bz);

  scanf("%s", arg.name);
  
  scanf("%lf", &arg.eps);

  scanf("%lf", &arg.lx);
  scanf("%lf", &arg.ly);
  scanf("%lf", &arg.lz);

  scanf("%lf", &arg.ud);
  scanf("%lf", &arg.uu);

  scanf("%lf", &arg.ul);
  scanf("%lf", &arg.ur);

  scanf("%lf", &arg.uf);
  scanf("%lf", &arg.ub);
  
  scanf("%lf", &arg.u0);
}

void print_Args(const Args& arg){
  fprintf(stderr,"arg.x = %d\n", arg.x);
  fprintf(stderr,"arg.y = %d\n", arg.y);
  fprintf(stderr,"arg.z = %d\n", arg.z);

  fprintf(stderr,"arg.bx = %d\n", arg.bx);
  fprintf(stderr,"arg.by = %d\n", arg.by);
  fprintf(stderr,"arg.bz = %d\n", arg.bz);

  fprintf(stderr,"arg.name = %s\n", arg.name);
  
  fprintf(stderr,"arg.eps = %le\n", arg.eps);

  fprintf(stderr,"arg.lx = %lf\n", arg.lx);
  fprintf(stderr,"arg.ly = %lf\n", arg.ly);
  fprintf(stderr,"arg.lz = %lf\n", arg.lz);

  fprintf(stderr,"arg.ud = %lf\n", arg.ud);
  fprintf(stderr,"arg.uu = %lf\n", arg.uu);

  fprintf(stderr,"arg.ul = %lf\n", arg.ul);
  fprintf(stderr,"arg.ur = %lf\n", arg.ur);

  fprintf(stderr,"arg.uf = %lf\n", arg.uf);
  fprintf(stderr,"arg.ub = %lf\n", arg.ub);
  
  fprintf(stderr,"arg.u0 = %lf\n", arg.u0);
  fflush(stderr);
}

void bcast_Args(Args& arg){
  // Bcast(* from/where_to, count, type, root, comm)
  MPI_Bcast(&arg.x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.y, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.z, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.bx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.by, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.bz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(arg.name, file_name_len, MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.ud, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.uu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.ul, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.ur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.uf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&arg.ub, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&arg.u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void init(int id, double* data, const Args& arg){
  for(int k=0; k<arg.bz; ++k)
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        data[_bid(i,j,k)] = arg.u0;
  
  int x=_x(id),
      y=_y(id),
      z=_z(id);

  //down or z==0
  for(int j=0; j<arg.by; ++j)
    for(int i=0; i<arg.bx; ++i)
      if( z == 0 )
        data[_bid(i,j,-1)] = arg.ud;
      else
        data[_bid(i,j,-1)] = arg.u0;

  //upper or z==arg.z-1
  for(int j=0; j<arg.by; ++j)
    for(int i=0; i<arg.bx; ++i)
      if( z == (arg.z-1) )
        data[_bid(i,j,arg.bz)] = arg.uu;
      else
        data[_bid(i,j,arg.bz)] = arg.u0;

  //front or y==0
  for(int j=0; j<arg.bz; ++j)
    for(int i=0; i<arg.bx; ++i)
      if( y == 0 )
        data[_bid(i,-1,j)] = arg.uf;
      else
        data[_bid(i,-1,j)] = arg.u0;
  
  //bottom or y==arg.y-1
  for(int j=0; j<arg.bz; ++j)
    for(int i=0; i<arg.bx; ++i)
      if( y == (arg.y-1) )
        data[_bid(i,arg.by,j)] = arg.ub;
      else
        data[_bid(i,arg.by,j)] = arg.u0;

  //left or x==0
  for(int j=0; j<arg.bz; ++j)
    for(int i=0; i<arg.by; ++i)
      if( x == 0 )
        data[_bid(-1,i,j)] = arg.ul;
      else
        data[_bid(-1,i,j)] = arg.u0;
  
  //right or x==arg.x-1
  for(int j=0; j<arg.bz; ++j)
    for(int i=0; i<arg.by; ++i)
      if( x == (arg.x-1) )
        data[_bid(arg.bx,i,j)] = arg.ur;
      else
        data[_bid(arg.bx,i,j)] = arg.u0;
}

void swap(double*& l, double*& r){
  double *tmp;
  tmp = l;
  l=r;
  r=tmp;
}

void print_data_blocks(int id, double* data, const Args& arg){
  
  double *buff = (double*)malloc(sizeof(double)*(arg.bx+2));
  
  if (id != 0) {
   
    for(int k =-1; k <= arg.bz; ++k)
      for(int j = -1; j <= arg.by; ++j) {
        for(int i = -1; i <= arg.bx; ++i) 
          buff[i + 1] = data[_bid(i, j, k)];
        MPI_Send(buff, arg.bx+2, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
      }
  
  } else {
    FILE* out = fopen(arg.name, "w");
    for(int kb = 0; kb < arg.z; ++kb)
      for(int k = -1; k <= arg.bz; ++k){

        for(int jb = 0; jb < arg.y; ++jb)//nomer stroki processov
          for(int j = -1; j <= arg.by; ++j)//nomer stroki setki odnogo processa
            
            for(int ib = 0; ib < arg.x; ++ib) {//nomer stolbca processov
              
              if (_pid(ib, jb, kb) == 0)
                for(int i = -1; i <= arg.bx; ++i)
                  buff[i + 1] = data[_bid(i, j, k)];
              else
               MPI_Recv(buff,  arg.bx+2, MPI_DOUBLE, _pid(ib, jb, kb), _pid(ib, jb, kb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              
              for(int i = -1; i <=  arg.bx; ++i)
                fprintf(out, "%.2f ", buff[i + 1]);
               

              if(ib + 1 ==  arg.x) {
                fprintf(out, "\n");
                if (j == arg.by)
                  fprintf(out, "\n");
              } else 
                fprintf(out, " ");
            }
        fprintf(out, "\n");
      }
    fclose(out);
  }
  fflush(stdout);
  free(buff); 
}

void print_data(int id, double* data, const Args& arg){
  // int buffer_size;
  // MPI_Pack_size((arg.x), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
  
  // buffer_size = (arg.y)*(arg.z)*(buffer_size + MPI_BSEND_OVERHEAD);
  // double *buffer = (double *)malloc(buffer_size);
  // MPI_Buffer_attach(buffer, buffer_size);

  int n_size = 15;
  char *buff = (char*)malloc(sizeof(char)*n_size*arg.bx*arg.by*arg.bz);

  memset(buff, ' ', sizeof(char)*n_size*arg.bx*arg.by*arg.bz);
  for(int k=0; k<arg.bz; ++k ){

    for(int j=0; j<arg.by; ++j ){
      for(int i=0; i<arg.bx; ++i )
        sprintf(buff + (k*arg.bx*arg.by + j*arg.bx + i)*n_size, "%.7le", data[_bid(i,j,k)]);
      if(_x(id) == arg.x-1)
        buff[(k*arg.by + j + 1)*arg.bx*n_size -1 ] = '\n';
    }
    if(_y(id) == arg.y-1 && _x(id) == arg.x-1)
      buff[(k + 1)*arg.by*arg.bx*n_size-2 ] = '\n';
  }
  for(int i=0; i<arg.bx*arg.by*arg.bz*n_size; ++i)
    if(buff[i] == '\0')
      buff[i] =' ';


  MPI_File fp;
  MPI_Datatype ftype;

  MPI_Datatype num;
  MPI_Type_contiguous(n_size, MPI_CHAR, &num);
  MPI_Type_commit(&num);


  int sizes[3]    = {arg.bx*arg.x, arg.by*arg.y, arg.bz*arg.z};
  int subsizes[3] = {arg.bx, arg.by, arg.bz};
  int starts[3]   = {arg.bx*_x(id), arg.by*_y(id), arg.bz*_z(id)};

  //int 
  //MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[],         int order, MPI_Datatype oldtype, MPI_Datatype *newtype)
  MPI_Type_create_subarray  (        3,                      sizes,                      subsizes,                      starts, MPI_ORDER_FORTRAN,                  num,                &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_delete(arg.name, MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, arg.name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp );

  MPI_File_set_view(fp, 0, num, ftype, "native", MPI_INFO_NULL);

  MPI_File_write_all(fp, buff, arg.bx*arg.by*arg.bz*n_size, MPI_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&fp);

  
  
  free(buff); 
}

__global__
void do_math(double* next, double* cur, const Args arg){
  double hx = double(arg.lx)/arg.x/arg.bx,
         hy = double(arg.ly)/arg.y/arg.by,
         hz = double(arg.lz)/arg.z/arg.bz;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsety = blockDim.y*gridDim.y;
  int offsetz = blockDim.z*gridDim.z;

  for(int k=idz; k<arg.bz; k+=offsetz)
    for(int j=idy; j<arg.by; j+=offsety)
      for(int i=idx; i<arg.bx; i+=offsetx){

        next[_bid(i,j,k)] = 
         (( cur[_bid(i+1,j,k)] + cur[_bid(i-1,j,k)] ) / (hx*hx) +
          ( cur[_bid(i,j+1,k)] + cur[_bid(i,j-1,k)] ) / (hy*hy) +
          ( cur[_bid(i,j,k+1)] + cur[_bid(i,j,k-1)] ) / (hz*hz) ) /
            ( 2.0 * ( 1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz) ) );
      }
}

void init_buff(Buffers& buff, const Args& arg){
  buff.down = (double*) malloc(sizeof(double)*arg.bx*arg.by);
  buff.up   = (double*) malloc(sizeof(double)*arg.bx*arg.by);

  buff.left  = (double*) malloc(sizeof(double)*arg.by*arg.bz);
  buff.right = (double*) malloc(sizeof(double)*arg.by*arg.bz);
  
  buff.back  = (double*) malloc(sizeof(double)*arg.bz*arg.bx);
  buff.front = (double*) malloc(sizeof(double)*arg.bz*arg.bx);

  CSC(cudaMalloc(&buff.XOY, sizeof(double)*arg.bx*arg.by));
  CSC(cudaMalloc(&buff.XOZ, sizeof(double)*arg.bx*arg.bz));
  CSC(cudaMalloc(&buff.YOZ, sizeof(double)*arg.by*arg.bz));
}

void free_buff(Buffers& buff){
  free(buff.down);
  free(buff.up);
  
  free(buff.left);
  free(buff.right);
  
  free(buff.front);
  free(buff.back);

  CSC(cudaFree(buff.XOY));
  CSC(cudaFree(buff.XOZ));
  CSC(cudaFree(buff.YOZ));
}

__global__
void get_XOY_slice(double *data, double *buf, int k, const Args arg){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsety = blockDim.y*gridDim.y;
  
  for(int j=idy; j<arg.by; j+=offsety)
    for(int i=idx; i<arg.bx; i+=offsetx)
      buf[arg.bx*j + i] = data[_bid(i,j,k)];
}

__global__
void get_XOZ_slice(double *data, double *buf, int j, const Args arg){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idz = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsetz = blockDim.y*gridDim.y;
  
  for(int k=idz; k<arg.bz; k+=offsetz)
    for(int i=idx; i<arg.bx; i+=offsetx)
      buf[arg.bx*k + i] = data[_bid(i,j,k)];
}

__global__
void get_YOZ_slice(double *data, double *buf, int i, const Args arg){
  int idy = blockIdx.x*blockDim.x + threadIdx.x;
  int idz = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsety = blockDim.x*gridDim.x;
  int offsetz = blockDim.y*gridDim.y;
  
  for(int k=idz; k<arg.bz; k+=offsetz)
    for(int j=idy; j<arg.by; j+=offsety)
      buf[arg.by*k + j] = data[_bid(i,j,k)];
}

__global__
void set_XOY_slice(double *data, double *buf, int k, const Args arg){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsety = blockDim.y*gridDim.y;
  
  for(int j=idy; j<arg.by; j+=offsety)
    for(int i=idx; i<arg.bx; i+=offsetx)
      data[_bid(i,j,k)] = buf[arg.bx*j + i];
}

__global__
void set_XOZ_slice(double *data, double *buf, int j, const Args arg){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idz = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsetz = blockDim.y*gridDim.y;
  
  for(int k=idz; k<arg.bz; k+=offsetz)
    for(int i=idx; i<arg.bx; i+=offsetx)
      data[_bid(i,j,k)] = buf[arg.bx*k + i];
}

__global__
void set_YOZ_slice(double *data, double *buf, int i, const Args arg){
  int idy = blockIdx.x*blockDim.x + threadIdx.x;
  int idz = blockIdx.y*blockDim.y + threadIdx.y;
  
  int offsety = blockDim.x*gridDim.x;
  int offsetz = blockDim.y*gridDim.y;
  
  for(int k=idz; k<arg.bz; k+=offsetz)
    for(int j=idy; j<arg.by; j+=offsety)
      data[_bid(i,j,k)] = buf[arg.by*k + j];
}

__global__
void pre_proc_max(double *next, double *cur, const Args arg){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;
  
  int offsetx = blockDim.x*gridDim.x;
  int offsety = blockDim.y*gridDim.y;
  int offsetz = blockDim.z*gridDim.z;

  for(int k=idz-1; k<=arg.bz; k+=offsetz)
    for(int j=idy-1; j<=arg.by; j+=offsety)
      for(int i=idx-1; i<=arg.bx; i+=offsetx)
        if(i==-1 || j==-1 || k==-1 || i==arg.bx || j==arg.by || k==arg.bz){
          cur[_bid(i,j,k)] = 0.0;
        }else{
          cur[_bid(i,j,k)] = fabs(next[_bid(i,j,k)] - cur[_bid(i,j,k)]);
        }
}

double calc_eps(double *next, double *cur, const Args& arg){
  double cur_eps=0, eps=0;

  pre_proc_max<<<GRID_DIM_3D, BLOCK_DIM_3D>>>(next, cur, arg);
  CSC(cudaGetLastError());

  comparator cmp;
  thrust::device_ptr<double> i_ptr = thrust::device_pointer_cast(cur); 

  cur_eps = *thrust::max_element(i_ptr , i_ptr + (arg.bx+2)*(arg.by+2)*(arg.bz+2), cmp);

  //MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
    MPI_Allreduce(           &cur_eps,          &eps,         1,            MPI_DOUBLE,   MPI_MAX, MPI_COMM_WORLD);
  return eps;
}

void sync_edges(double * data, double *prev, Buffers buff[2], int id,  const Args& arg){
  // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
  // int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
  // int MPI_Waitall(int count, MPI_Request array_of_requests[],MPI_Status *array_of_statuses)
  // MPI_ANY_TAG
  // MPI_REQUEST_NULL
   
  MPI_Request req_send[6], req_recv;
  for(int i=0; i<6; ++i){
    req_send[i] = MPI_REQUEST_NULL;
  }
  req_recv = MPI_REQUEST_NULL;


  int x=_x(id),
      y=_y(id),
      z=_z(id);
  
  //-----------------------------------------
  // send
  //-----------------------------------------
  
  if(z>0){
    get_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].XOY, 0, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].down, buff[0].XOY, arg.bx*arg.by*sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Isend(buff[0].down, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z-1), 0, MPI_COMM_WORLD, req_send+0);
  }

  if(z<arg.z-1){
    get_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].XOY, arg.bz-1, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].up, buff[0].XOY, arg.bx*arg.by*sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Isend(buff[0].up, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z+1), 0, MPI_COMM_WORLD, req_send+1);
  }

  if(y>0){
    get_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].XOZ, 0, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].front, buff[0].XOZ, arg.bx*arg.bz*sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Isend(buff[0].front, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y-1,z), 0, MPI_COMM_WORLD, req_send+2);
  }

  if(y<arg.y-1){
    get_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].XOZ, arg.by-1, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].back, buff[0].XOZ, arg.bx*arg.bz*sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Isend(buff[0].back, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y+1,z), 0, MPI_COMM_WORLD, req_send+3);
  }

  if(x>0){
    get_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].YOZ, 0, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].left, buff[0].YOZ, arg.by*arg.bz*sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Isend(buff[0].left, arg.by*arg.bz, MPI_DOUBLE, _pid(x-1,y,z), 0, MPI_COMM_WORLD, req_send+4);
  }

  if(x<arg.x-1){
    get_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[0].YOZ, arg.bx-1, arg);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(buff[0].right, buff[0].YOZ, arg.by*arg.bz*sizeof(double), cudaMemcpyDeviceToHost));
    
    MPI_Isend(buff[0].right, arg.by*arg.bz, MPI_DOUBLE, _pid(x+1,y,z), 0, MPI_COMM_WORLD, req_send+5);
  }

  //-----------------------------------------
  // recive
  //-----------------------------------------


  if(z<arg.z-1){
    MPI_Irecv(buff[1].up, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z+1), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

    CSC(cudaMemcpy(buff[1].XOY, buff[1].up, arg.bx*arg.by*sizeof(double), cudaMemcpyHostToDevice));
    set_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOY, arg.bz, arg);
  }else{
    get_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOY, arg.bz, arg);
    set_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].XOY, arg.bz, arg); 
  }

  if(z>0){
    MPI_Irecv(buff[1].down, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z-1), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

    CSC(cudaMemcpy(buff[1].XOY, buff[1].down, arg.bx*arg.by*sizeof(double), cudaMemcpyHostToDevice));
    set_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOY, -1, arg);
  }else{//z == 0
    get_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOY, -1, arg);
    set_XOY_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].XOY, -1, arg); 
  }

  if(y<arg.y-1){
    MPI_Irecv(buff[1].back, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y+1,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

    CSC(cudaMemcpy(buff[1].XOZ, buff[1].back, arg.bx*arg.bz*sizeof(double), cudaMemcpyHostToDevice));
    set_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOZ, arg.by, arg);
  }else{
    get_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOZ, arg.by, arg);
    set_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].XOZ, arg.by, arg); 
  }

  if(y>0){
    MPI_Irecv(buff[1].front, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y-1,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    
    CSC(cudaMemcpy(buff[1].XOZ, buff[1].front, arg.bx*arg.bz*sizeof(double), cudaMemcpyHostToDevice));
    set_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOZ, -1, arg);
  }else{
    get_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].XOZ, -1, arg);
    set_XOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].XOZ, -1, arg); 
  }

  if(x<arg.x-1){
    MPI_Irecv(buff[1].left, arg.by*arg.bz, MPI_DOUBLE, _pid(x+1,y,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    
    CSC(cudaMemcpy(buff[1].YOZ, buff[1].left, arg.by*arg.bz*sizeof(double), cudaMemcpyHostToDevice));
    set_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].YOZ, arg.bx, arg);
  }else{
    get_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].YOZ, arg.bx, arg);
    set_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].YOZ, arg.bx, arg); 
  }

  if(x>0){
    MPI_Irecv(buff[1].right, arg.by*arg.bz, MPI_DOUBLE, _pid(x-1,y,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    
    CSC(cudaMemcpy(buff[1].YOZ, buff[1].right, arg.by*arg.bz*sizeof(double), cudaMemcpyHostToDevice));
    set_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].YOZ, -1, arg);
  }else{
    get_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(data, buff[1].YOZ, -1, arg);
    set_YOZ_slice<<<GRID_DIM_2D, BLOCK_DIM_2D>>>(prev, buff[1].YOZ, -1, arg); 
  }

  MPI_Waitall(6, req_send, MPI_STATUS_IGNORE);
}

int main(int argc, char* argv[]){

  Args arg;
  Buffers buff[2];
  
  int id, numproc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  
  int dev_cnt;
  CSC(cudaGetDeviceCount(&dev_cnt));
  CSC(cudaSetDevice(id%dev_cnt));

  if(id == 0){
    read_Args(arg);
    print_Args(arg);
  }
    
  bcast_Args(arg);
  MPI_Barrier(MPI_COMM_WORLD);

  double *data = (double*) calloc(((arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)), sizeof(double)), 
         *prev = (double*) calloc(((arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)), sizeof(double));
  
  init(id, data, arg);
  init(id, prev, arg);


  double *dev_data,
         *dev_prev;

  CSC(cudaMalloc(&dev_data, (arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)*sizeof(double) ));
  CSC(cudaMalloc(&dev_prev, (arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)*sizeof(double) ));

  CSC(cudaMemcpy(dev_data, data, (arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)*sizeof(double), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(dev_prev, prev, (arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)*sizeof(double), cudaMemcpyHostToDevice));

  init_buff(buff[0], arg);
  init_buff(buff[1], arg);
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  cudaEvent_t events[2];
  cudaEventCreate(&events[0]);
  cudaEventCreate(&events[1]);
  
  cudaEventRecord(events[0], 0);
  double eps;
  do{
    MPI_Barrier(MPI_COMM_WORLD);
    

    do_math<<< GRID_DIM_3D, BLOCK_DIM_3D>>>(dev_prev, dev_data, arg);
    CSC(cudaGetLastError());

    eps = calc_eps(dev_prev, dev_data, arg);
    if(id==0){
      fprintf(stderr, "eps = %le\n", eps);
      fflush(stderr);
    }
    swap(dev_prev, dev_data);
    sync_edges(dev_data, dev_prev, buff, id, arg);
  }while(eps>=arg.eps);
  
  cudaEventRecord(events[1], 0);
  cudaDeviceSynchronize(); 
  float timeValue;
  cudaEventElapsedTime( &timeValue, events[0], events[1] ); 
  if(id==0){
    fprintf(stderr,"Elapsed time:  %f  ms \n", timeValue);
    fflush(stderr);
  }

  std::chrono::steady_clock::time_point end   = std::chrono::steady_clock::now();
  
  if(id==0)
    std::cerr<< "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << std::endl;

  CSC(cudaMemcpy(data, dev_data, (arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)*sizeof(double), cudaMemcpyDeviceToHost));
  
  MPI_Barrier(MPI_COMM_WORLD);
  print_data(id, data, arg);
  MPI_Barrier(MPI_COMM_WORLD);

  free(data);
  free(prev);
  free_buff(buff[0]);
  free_buff(buff[1]);
  CSC(cudaFree(dev_data));
  CSC(cudaFree(dev_prev));

  MPI_Finalize();

  return 0;
}