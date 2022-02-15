who=`whoami`
student="schizophrenia"
ifeq ($(who), $(student))
	CC=nvcc
else
	CC=/usr/local/cuda/bin/nvcc
endif

CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -w
SOURSES = main.cu
BIN = 2lab
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

clean:
	if [ -e $(BIN) ]; then \
		rm $(BIN); \
	fi