all:
#nvcc --resource-usage bank.cu -o bank.x
	nvcc --resource-usage bank.cu -o bank.x
	nvcc --resource-usage banks_shared.cu -o bankS.x
	nvcc --resource-usage banks_multi_kernel.cu -o bankMK.x
	nvcc --resource-usage banks_multi_SM.cu -o bankMS.x

default: 

	nvcc --resource-usage bank.cu -o bank.x

shared:

	nvcc --resource-usage banks_shared.cu -o bankS.x

multik:

	nvcc --resource-usage banks_multi_kernel.cu -o bankMK.x

multis:

	nvcc --resource-usage banks_multi_SM.cu -o bankMS.x

clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite



#export PATH=$PATH:/usr/local/cuda/bin
#export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/usr/local/cuda/lib64
#nsys nvprof