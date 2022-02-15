who=`whoami`
student="schizophrenia"
ifeq ($(who), $(student))
	CC=nvcc
else
	CC=/usr/local/cuda/bin/nvcc
endif

CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = main.cu
BIN = 1lab
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

clean:
	if [ -e $(BIN) ]; then \
		rm $(BIN); \
	fi