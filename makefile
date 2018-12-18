MEX:=mex
SRC:=$(wildcard *.c)
OBJS=$(SRC:%.c=%.mexa64)

$(OBJS): %.mexa64: %.c
	$(MEX) $<

all: $(OBJS)
