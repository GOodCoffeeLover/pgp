#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <cmath>
#include <float.h>
#include <stdio.h>
#include <time.h>
// #define CSC(call)                           \
// do {                                \
//   cudaError_t res = call;                     \
//   if (res != cudaSuccess) {                   \
//     fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
//         __FILE__, __LINE__, cudaGetErrorString(res));   \
//     exit(0);                          \
//   }                               \
// } while(0);

#define classes_numder 32

using namespace std;

struct uchar4{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
  
};

class ClassData{
public:
  double avg[3];
  double inv_cov[3][3];
  double det_cov;
};

std::ostream& operator<<(std::ostream& os, const ClassData& class_data){
  os << "avg : \n( ";
  for(int i=0; i<3; ++i)
    os<<class_data.avg[i]<<(i<2? ", ": " )");
  os<<'\n';
  os<<"cov : \n";
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j)
      os<<class_data.inv_cov[i][j]<<' ';
    os<<'\n';
  }
  os<<"det = "<<class_data.det_cov<<endl;
  return os;
}


ClassData metrics[classes_numder]  = {};

void read_file(uchar4*& data, 
               int& w, 
               int& h, 
               const string& file_name){
  ifstream input_file(file_name, ios::in|ios::binary);
  
  input_file.read(reinterpret_cast<char*>(&w), sizeof(int));
  input_file.read(reinterpret_cast<char*>(&h), sizeof(int));
  
  data = new uchar4[w*h];

  input_file.read(reinterpret_cast<char*>(data), sizeof(uchar4)*w*h);

  input_file.close();
  return;
}

void write_file(const uchar4*  data, 
                const int& w, 
                const int& h, 
                const string& file_name){
  ofstream output_file(file_name, ios::out|ios::binary);
  
  output_file.write(reinterpret_cast<const char*>(&w), sizeof(int));
  output_file.write(reinterpret_cast<const char*>(&h), sizeof(int));
  output_file.write(reinterpret_cast<const char*>(data), sizeof(uchar4)*w*h);

  output_file.close();
  return;
}

void calc_metrics(uchar4* data, 
                  const int& w, 
                  const int& h, 
                  int& NC){
  cin>>NC;
  
  for(int k=0; k<NC; ++k){
    metrics[k].avg[0] =0.0;
    metrics[k].avg[1] =0.0;
    metrics[k].avg[2] =0.0;

    metrics[k].inv_cov[0][0]=0.0;
    metrics[k].inv_cov[0][1]=0.0;
    metrics[k].inv_cov[0][2]=0.0;

    metrics[k].inv_cov[1][0]=0.0;
    metrics[k].inv_cov[1][1]=0.0;
    metrics[k].inv_cov[1][2]=0.0;

    metrics[k].inv_cov[2][0]=0.0;
    metrics[k].inv_cov[2][1]=0.0;
    metrics[k].inv_cov[2][2]=0.0;


    metrics[k].det_cov=0.0;
    unsigned long long np, j,i;
    cin>>np;
    vector<array<double,3>> pixels(np);
    
    for( auto& pixel: pixels){
      cin>>j>>i;
    
      uchar4& p = data[i*w+j];
    
      pixel[0] = p.x;
      pixel[1] = p.y;
      pixel[2] = p.z;

      for(i=0; i<3; ++i)    
        metrics[k].avg[i]+=pixel[i];

    }
    for(i=0; i<3; ++i)    
      metrics[k].avg[i]/=np;

    array<array<double, 3>, 3> cov={{{0,0,0}, {0,0,0}, {0,0,0}}};
    for( auto& pixel: pixels){
      for(i=0; i<3; ++i)
        pixel[i]-=metrics[k].avg[i];
      
      
      for(i=0; i<3; ++i)
        for(j=0; j<3; ++j)
          cov[i][j]+=pixel[i]*pixel[j];
      }

      for(i=0; i<3; ++i)
        for(j=0; j<3; ++j)
          cov[i][j]/=np-1;
      
      
      double cov_star[3][3];

      cov_star[0][0] = cov[1][1]*cov[2][2] - cov[2][1]*cov[1][2];
      cov_star[0][1] = cov[1][0]*cov[2][2] - cov[2][0]*cov[1][2];
      cov_star[0][2] = cov[1][0]*cov[2][1] - cov[2][0]*cov[1][1];

      cov_star[1][0] = cov[0][1]*cov[2][2] - cov[2][1]*cov[0][2];
      cov_star[1][1] = cov[0][0]*cov[2][2] - cov[2][0]*cov[0][2];
      cov_star[1][2] = cov[0][0]*cov[2][1] - cov[2][0]*cov[0][1];

      cov_star[2][0] = cov[0][1]*cov[1][2] - cov[1][1]*cov[0][2];
      cov_star[2][1] = cov[0][0]*cov[1][2] - cov[1][0]*cov[0][2];
      cov_star[2][2] = cov[0][0]*cov[1][1] - cov[1][0]*cov[0][1];

      
      for(i=0; i<3; ++i)
        metrics[k].det_cov +=(i%2?-1:1)*cov[0][i]*cov_star[0][i];

      for(i=0; i<3; ++i)
        for(j=0; j<3; ++j)
          metrics[k].inv_cov[i][j] = (((i+j)%2)?-1:1) * cov_star[j][i] / metrics[k].det_cov;
    
      
  }
  return;
}


double compute_mmp(const uchar4& pp, const ClassData& metric){
  double 
    p[3] = {
      (double)pp.x - metric.avg[0], 
      (double)pp.y - metric.avg[1], 
      (double)pp.z - metric.avg[2]}, 
    
    intrm[3] = {0,0,0};

  for(int i=0; i<3; ++i)
    for(int j=0; j<3; ++j)
      intrm[i]+= p[j] * metric.inv_cov[i][j];
  
  double value=0.0;

  for(int i=0; i<3; ++i)
    value +=intrm[i]*p[i];

  return - value - log(abs(metric.det_cov));
}

void classify(uchar4* data, const int size, const int NC){
  int j, max_j=-1, i;
  double cur_value_mmp, max_mmp=-DBL_MAX;
  uchar4 p;
  
  for(i=0; i<size; i += 1 ){
    
    max_j=-1;
    max_mmp = -DBL_MAX;
    p=data[i];
    
    for(j=0; j<NC; ++j){
     
      cur_value_mmp = compute_mmp(p, metrics[j]);
     
      if(( max_j == -1) || (cur_value_mmp > max_mmp) ){
        max_j = j;
        max_mmp = cur_value_mmp;
      }
    
    }
    
    data[i].w = max_j;
  }
  return;
}


int main(){
  string file_name;
  uchar4* data;
  int w,h;
  
  cin>>file_name;
  read_file(data, w, h, file_name);
  cout<< w << ' ' << h << endl;
  cin>>file_name;
   
  int NC=0;
  calc_metrics(data, w,h, NC);

  clock_t start = clock();
  classify(data, w*h, NC);
  clock_t end = clock();
  double miliseconds = (double)(end - start) / CLOCKS_PER_SEC*1000.0;
  printf("The time: %f miliseconds\n", miliseconds);

  write_file(data, w,h, file_name);
  delete [] data;

  return 0;
}