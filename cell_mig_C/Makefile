CC = gcc
CFLAGS = -g -Wall -Wextra -std=c11
DFLAGS = -DM_PI=3.14159265358979323846
EFLAGS = -DM_E=2.718281828459045
MFLAGS = -lm
#GSLFLAGS = -lgsl -lgslcblas
#OFLAGS = -O3
#OMPFLAGS = -fopenmp
RENAME = -o
name ?= p1

all: p1

debug: p1.c
	$(CC) $(DFLAGS) $(CFLAGS) $(IFLAGS) $(OFLAGS) $(EFLAGS) $(STACKFLAGS) -DDEBUG p1.c $(RENAME) debug_$(name) $(MFLAGS) $(GSLFLAGS)

p1: p1.c
	$(CC) $(DFLAGS) $(CFLAGS) $(IFLAGS) $(OFLAGS) $(EFLAGS) $(STACKFLAGS) p1.c $(RENAME) $(name) $(MFLAGS) $(GSLFLAGS) $(OMPFLAGS)



