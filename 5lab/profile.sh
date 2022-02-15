/usr/local/cuda-11.4/bin/nvprof -m branch_efficiency,global_hit_rate,local_hit_rate -e global_store,global_load,shared_ld_bank_conflict,shared_st_bank_conflict ./$1 <$2 

