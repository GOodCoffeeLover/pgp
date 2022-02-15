#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>

#include <mpi.h>

#define file_name_len 128

#define _bid(i,j,k) (((k) + 1) * (arg.bx + 2)*(arg.by + 2) + ((j) + 1) * (arg.bx + 2) + (i) + 1)
#define _pid(i,j,k) ((k) * arg.x*arg.y + (j) * arg.x + (i))

#define _x(id) ((id) % arg.x)
#define _y(id) ((id) %(arg.y*arg.x) /arg.x)
#define _z(id) ((id)/(arg.x*arg.y))

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
};


double f_max(double l, double r){
  if(l<r)
    return r;
  else 
    return l;
}

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

  double *buff = (double*)malloc(sizeof(double)*(arg.bx+2));
  
  if (id != 0) {
   
    for(int k =0; k < arg.bz; ++k)
      for(int j = 0; j < arg.by; ++j) {
        for(int i = 0; i < arg.bx; ++i) 
          buff[i] = data[_bid(i, j, k)];
        MPI_Send(buff, arg.bx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
      }
  
  } else {
    FILE* out = fopen(arg.name, "w");
  
    for(int kb = 0; kb < arg.z; ++kb)
      for(int k = 0; k < arg.bz; ++k){

        for(int jb = 0; jb < arg.y; ++jb)//nomer stroki processov
          for(int j = 0; j < arg.by; ++j)//nomer stroki setki odnogo processa
            
            for(int ib = 0; ib < arg.x; ++ib) {//nomer stolbca processov
              
              if (_pid(ib, jb, kb) == 0)
                for(int i = 0; i < arg.bx; ++i)
                  buff[i] = data[_bid(i, j, k)];
              else
               MPI_Recv(buff,  arg.bx, MPI_DOUBLE, _pid(ib, jb, kb), _pid(ib, jb, kb), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              
              
              for(int i = 0; i <  arg.bx; ++i)
                fprintf(out, "%e ", buff[i]);

              if(ib + 1 ==  arg.x)
                fprintf(out, "\n");
            }
        fprintf(out, "\n");
      }
    fclose(out);
  }
  //fflush(stdout);
  free(buff); 
}

void do_math(double* next, double* cur, const Args& arg){
  double hx = double(arg.lx)/arg.x/arg.bx,
         hy = double(arg.ly)/arg.y/arg.by,
         hz = double(arg.lz)/arg.z/arg.bz;

  for(int k=0; k<arg.bz; ++k)
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        next[_bid(i,j,k)] = 
         ((cur[_bid(i+1,j,k)] + cur[_bid(i-1,j,k)])/(hx*hx) +
          (cur[_bid(i,j+1,k)] + cur[_bid(i,j-1,k)])/(hy*hy) +
          (cur[_bid(i,j,k+1)] + cur[_bid(i,j,k-1)])/(hz*hz)) /
          (2.0*( 1.0/(hx*hx) + 1.0/(hy*hy) + 1.0/(hz*hz) ));
}

double calc_eps(double *next, double *cur, const Args& arg){
  double cur_eps=0, eps=0;

  for(int k=0; k<arg.bz; ++k)
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        cur_eps = f_max(cur_eps, fabs(next[_bid(i,j,k)] - cur[_bid(i,j,k)]));
      
   
  //MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
    MPI_Allreduce(           &cur_eps,          &eps,         1,            MPI_DOUBLE,   MPI_MAX, MPI_COMM_WORLD);
  
  return eps;
}

void init_buff(Buffers& buff, const Args& arg){
  buff.down = (double*) malloc(sizeof(double)*arg.bx*arg.by);
  buff.up   = (double*) malloc(sizeof(double)*arg.bx*arg.by);

  buff.left  = (double*) malloc(sizeof(double)*arg.by*arg.bz);
  buff.right = (double*) malloc(sizeof(double)*arg.by*arg.bz);
  
  buff.back  = (double*) malloc(sizeof(double)*arg.bz*arg.bx);
  buff.front = (double*) malloc(sizeof(double)*arg.bz*arg.bx);
}

void free_buff(Buffers& buff){
  free(buff.down);
  free(buff.up);
  
  free(buff.left);
  free(buff.right);
  
  free(buff.front);
  free(buff.back);
}

void sync_edges(double * data, Buffers buff[2], int id,  const Args& arg){
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
  x=x;
  y=y;
  //-----------------------------------------
  // send
  //-----------------------------------------
  
  if(z>0){
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        buff[0].down[j*arg.bx + i] = data[_bid(i,j, 0)];
    MPI_Isend(buff[0].down, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z-1), 0, MPI_COMM_WORLD, req_send+0);
  }

  if(z<arg.z-1){
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        buff[0].up[j*arg.bx + i] = data[_bid(i,j, arg.bz-1)];
    MPI_Isend(buff[0].up, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z+1), 0, MPI_COMM_WORLD, req_send+1);
  }

  if(y>0){
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.bx; ++i)
        buff[0].front[j*arg.bx + i] = data[_bid(i, 0, j)];
    MPI_Isend(buff[0].front, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y-1,z), 0, MPI_COMM_WORLD, req_send+2);
  }

  if(y<arg.y-1){
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.bx; ++i)
        buff[0].back[j*arg.bx + i] = data[_bid(i, arg.by-1, j)];
    MPI_Isend(buff[0].back, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y+1,z), 0, MPI_COMM_WORLD, req_send+3);
  }

  if(x>0){
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.by; ++i)
        buff[0].left[j*arg.by + i] = data[_bid(0, i, j)];
    MPI_Isend(buff[0].left, arg.by*arg.bz, MPI_DOUBLE, _pid(x-1,y,z), 0, MPI_COMM_WORLD, req_send+4);
  }

  if(x<arg.x-1){
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.by; ++i)
        buff[0].right[j*arg.by + i] = data[_bid(arg.bx-1, i, j)];
    MPI_Isend(buff[0].right, arg.by*arg.bz, MPI_DOUBLE, _pid(x+1,y,z), 0, MPI_COMM_WORLD, req_send+5);
  }
 //-----------------------------------------
 // recive
 //-----------------------------------------


  if(z<arg.z-1){
    MPI_Irecv(buff[1].up, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z+1), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        data[_bid(i,j, arg.bz)] = buff[1].up[j*arg.bx + i];
  }

  if(z>0){
    MPI_Irecv(buff[1].down, arg.bx*arg.by, MPI_DOUBLE, _pid(x,y,z-1), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.by; ++j)
      for(int i=0; i<arg.bx; ++i)
        data[_bid(i,j, -1)] = buff[1].down[j*arg.bx + i];
  }

  if(y<arg.y-1){
    MPI_Irecv(buff[1].back, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y+1,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.bx; ++i)
        data[_bid(i, arg.by, j)] = buff[1].back[j*arg.bx + i];
  }

  if(y>0){
    MPI_Irecv(buff[1].front, arg.bx*arg.bz, MPI_DOUBLE, _pid(x,y-1,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.bx; ++i)
        data[_bid(i, -1, j)] = buff[1].front[j*arg.bx + i];
  }

  if(x<arg.x-1){
    MPI_Irecv(buff[1].left, arg.by*arg.bz, MPI_DOUBLE, _pid(x+1,y,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.by; ++i)
        data[_bid(arg.bx, i, j)] = buff[1].left[j*arg.by + i];
  }

  if(x>0){
    MPI_Irecv(buff[1].right, arg.by*arg.bz, MPI_DOUBLE, _pid(x-1,y,z), MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
    for(int j=0; j<arg.bz; ++j)
      for(int i=0; i<arg.by; ++i)
        data[_bid(-1, i, j)] = buff[1].right[j*arg.by + i];
  }

  MPI_Waitall(6, req_send, MPI_STATUS_IGNORE);
}

int main(int argc, char* argv[]){

  Args arg;
  Buffers buff[2];
  
  int id, numproc, proc_name_len;
  char proc_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Get_processor_name(proc_name, &proc_name_len);

  if(id == 0){
    read_Args(arg);
    print_Args(arg);
  }
    
  bcast_Args(arg);
  MPI_Barrier(MPI_COMM_WORLD);

  double *data = (double*) calloc(((arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)), sizeof(double)), 
         *prev = (double*) calloc(((arg.bx + 2)*(arg.by + 2)*(arg.bz + 2)), sizeof(double));
  
  MPI_Barrier(MPI_COMM_WORLD);
  init(id, data, arg);
  init(id, prev, arg);
 
  init_buff(buff[0], arg);
  init_buff(buff[1], arg);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  double eps;
  do{
    MPI_Barrier(MPI_COMM_WORLD);
    do_math(prev, data, arg);
    eps = calc_eps(prev, data, arg);
    swap(prev, data);
    sync_edges(data, buff, id, arg);

  }while(eps>=arg.eps);

  std::chrono::steady_clock::time_point end   = std::chrono::steady_clock::now();
  

  std::cerr<< "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  print_data_blocks(id, data, arg);
  MPI_Barrier(MPI_COMM_WORLD);

  free(data);
  free(prev);
  free_buff(buff[0]);
  free_buff(buff[1]);

  MPI_Finalize();

  return 0;
}