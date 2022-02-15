make

for (( i=1; i<=4; ++i ))
do 
  nvprof --log-file out.txt ./2lab < test$i.txt 
  cat out.txt |  grep kernel
done