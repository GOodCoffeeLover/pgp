who=${USER}
student=schizophrenia
ifeq (${who}, ${student})
  CC=nvcc
else
  CC=/usr/local/cuda/bin/nvcc
endif

CFLAGS = -ccbin=mpic++ --std=c++14 -Werror cross-execution-space-call -lm
SOURSES = main.cu
BIN = 8lab


all: $(BIN)

$(BIN): $(SOURSES)
	$(CC) $(CFLAGS)  -o $(BIN) $(SOURSES)


compress:
	if [ -e $(BIN) ]; then \
		rm -rf $(BIN); \
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

