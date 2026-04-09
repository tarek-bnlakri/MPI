lab 3 how to run it 
mpicc -o sample3array_fixed sample3array_fixed.c -lm
time mpirun -np 4 ./sample3array_fixed
