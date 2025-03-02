all:
#nvcc --resource-usage bank.cu -o bank.x
	nvcc --resource-usage banks_shared.cu -o bank.x

multik:

	nvcc --resource-usage banks_multi_kernel.cu -o bank.x

multis:

	nvcc --resource-usage banks_multi_SM.cu -o bank.x

clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite



#export PATH=$PATH:/usr/local/cuda/bin
#export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/usr/local/cuda/lib64
#nsys nvprof