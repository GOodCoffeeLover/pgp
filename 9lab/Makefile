who=${USER}
student=schizophrenia
ifeq (${who}, ${student})
  CC=mpic++
else
  CC=/usr/local/bin/mpic++
endif


CFLAGS = --std=c++11 -fopenmp -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm
SOURSES = main.cpp
BIN = 9lab

all: compile

compile:
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

