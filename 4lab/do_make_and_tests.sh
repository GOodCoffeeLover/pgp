
make && \
for size in "10" "100" "500" "1000"  ; do
   nvprof --log-file log.txt ./4lab < test${size}.txt > /dev/null
   cat log.txt | grep Time\(\%\) 
   cat log.txt | grep kernel_gauss_step 
   cat log.txt | grep kernel_swap
   echo ""

done
rm log.txt