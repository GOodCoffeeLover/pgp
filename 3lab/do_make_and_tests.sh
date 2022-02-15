make

for size in "320x320" "500x500" "889x908" ; do
   nvprof --log-file log.txt ./3lab < test${size}.txt
   cat log.txt | grep kernel
done
rm log.txt