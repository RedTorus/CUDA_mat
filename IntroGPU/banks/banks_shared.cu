#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;


__global__ void kernel_call(int N, float *in, float* out)
{
   __shared__ float share_buf[64*64];

   //DO NOT CHANGE ANY CODE ABOVE THIS COMMENT
   
   int id = threadIdx.x; //get my id;
   //64*64/blocdim.x = number of elements to be processed per Thread

   //read 2 rows at a time, write 2 columns at a time
   // for (int i = 0; i != 64*64/blockDim.x; ++i)
   //    share_buf[i*2 + (id % 64)*64 + (id / 64)] = in[id + blockDim.x*i]; 

   int offset = (id/32)*16;
   int fac= 64; //blockDim.x/4; //32
   int col;
   int row;
   for (int i = 0; i != 64*64/blockDim.x; ++i){
      //share_buf[offset + (id % 32) + i * fac] = in[id + blockDim.x * i];
      col = (i%2)*32;
      row= i/2;
      share_buf[row + offset + fac*( (id%32) +col)] = in[id%32 + col + fac*(row +offset)];
      //share_buf[offset + row + (id % 32) + i * fac] = in[id + blockDim.x * i];
   }   

   // int total = 64 * 64;

   // for (int x = id; x < total; x += blockDim.x) {
   //    int r, c;
   //    if (x < 64) {
   //       // Main diagonal: x -> (x, x)
   //       r = x;
   //       c = x;
   //    } else {
   //       int t = x - 64;
   //       if (t < 2016) {
   //          // Upper triangle: elements where r < c.
   //          int sum = 0;
   //          int r0 = 0;
   //          // For row r0, number of off-diagonals = (63 - r0)
   //          while (r0 < 63 && sum + (63 - r0) <= t) {
   //             sum += (63 - r0);
   //             r0++;
   //          }
   //          r = r0;
   //          c = r0 + 1 + (t - sum);
   //       } else {
   //          // Lower triangle: elements where r > c.
   //          int t2 = t - 2016;
   //          int sum = 0;
   //          int r0 = 1;
   //          // For row r0, number of lower off-diagonals = r0
   //          while (r0 < 64 && sum + r0 <= t2) {
   //             sum += r0;
   //             r0++;
   //          }
   //          r = r0;
   //          c = t2 - sum;
   //       }
   //    }
   //    // Write the transposed element:
   //    // We want share_buf[r*64 + c] = in[c*64 + r]
   //    share_buf[r * 64 + c] = in[c * 64 + r];
   // }



 
   __syncthreads();  //wait till everyone is done

   //copy everything to main memory
   for (int i = 0; i != 64*64/blockDim.x; ++i)
      out[id + blockDim.x * i] = share_buf[id + blockDim.x*i]; 
   }


int main(){

    float *host_in, *host_out;
    float *dev_in, *dev_out;

    size_t N = 64;
		
    //create buffer on host	
    host_in = (float*) malloc(N * N * sizeof(float));
    host_out = (float*) malloc(N * N * sizeof(float));

    //creates a matrix stored in row major order
    for (int i = 0; i != N; ++i)
        for (int j = 0; j != N; ++j)
     	  host_in[i*N + j] = i*N + j;   


    //create buffer on device
    cudaError_t err = cudaMalloc(&dev_in, N*N*sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    err = cudaMalloc(&dev_out, N*N*sizeof(float));
    if (err != cudaSuccess){
       cout<<"Dev Memory not allocated"<<endl;
       exit(-1);
    }
     
    cudaMemcpy(dev_in, host_in, N * N * sizeof(float), cudaMemcpyHostToDevice);

    //create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);        
     
    cudaEventRecord(st2);
    kernel_call<<<1, 128>>>(N, dev_in, dev_out);
    cudaEventRecord(et2);
        
    //host waits until et2 has occured     
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    cout<<"Kernel time: "<<milliseconds<<"ms"<<endl;

    //copy data out
    cudaMemcpy(host_out, dev_out, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i != N; ++i)
       for (int j = 0; j != N; ++j)
          correct &= (host_out[i*N+j] == host_in[j*N+i]);
    cout<<(correct ? "Yes" : "No")<<endl;	 
   
    cudaEventDestroy(st2);
    cudaEventDestroy(et2);

    free(host_in);
    free(host_out);
    cudaFree(dev_in);
    cudaFree(dev_out);

  return 0;
}
