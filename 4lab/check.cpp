#include <iostream>
#include <vector>

using namespace std;

int main(){
  int n;
  cin>>n;
  vector<vector<double>> mtrx(n, vector<double>(n+1,0));
  for(int i=0; i<n; ++i)
    for(int j=0; j<n; ++j)
      cin>>mtrx[i][j];
  for(int j=0; j<n; ++j)
    cin>>mtrx[j][n];
  vector<double> ans(n,0);
    for(auto& a: ans)
      cin>>a;

  for(int i=0; i<n; ++i){
    double sum=-mtrx[i][n];
    for(int j=0; j<n; ++j)
      sum+=ans[j]*mtrx[i][j];
    cout<<sum<<endl;
  }
  
}