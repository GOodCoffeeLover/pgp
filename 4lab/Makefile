who=${USER}
student=schizophrenia
ifeq (${who}, ${student})
  CC=nvcc
else
  CC=/usr/local/cuda/bin/nvcc
endif


CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -w
SOURSES = main.cu
BIN = 4lab

all: compile

compile:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)


compress:
	if [ -e $(BIN) ]; then \
		rm $(BIN); \
	fi
	mkdir ${BIN}
	cp ${SOURSES} Makefile ${BIN} 
	tar -cvf ${BIN}.tar ${BIN}
	gpg -ab ${BIN}.tar
	rm -rf ${BIN}



clean:
	if [ -e $(BIN) ]; then \
		rm $(BIN); \
	fi
	if [ -e $(BIN).tar ]; then \
		rm $(BIN).tar; \
	fi
	if [ -e $(BIN).tar.asc ]; then \
		rm $(BIN).tar.asc; \
	fi

