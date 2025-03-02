#include <iostream>

using namespace std;


__global__ void kernel_call(float *c)
{

  __shared__ float buffer[12*1024];

  float* s_c = buffer;
  float* s_a = buffer + 4096;
  float* s_b = buffer + 4096 * 2;
  
  //1 threadblock only
  int id = threadIdx.x;
  int p = blockDim.x;

  for (int i = id; i<64*64; i += p)
    {
      s_c[i] = threadIdx.x + 1.0;
      s_a[i] = threadIdx.x + 2.0;
      s_b[i] = threadIdx.x + 1.0;
    }

  //ensure all threads are done initializing the buffer
  __syncthreads();

  /**** Do Not Change Code Above This ****/

  float loc;
  //Computes C += A * B using only 1 thread
  //A is column major order, the other 2 matrices are row major order
  
  // for (int i = 0; i < 64; ++i){      //64 rows of C
  //   for (int j = 0; j < 64; ++j){    //64 columns of C
  //     loc=s_c[i*64+j];
  //     for (int p = 0; p < 64; ++p){  //64 columns of A
  //       loc += s_a[p*64 + i] * s_b[p*64 + j];

  //     }
  //     s_c[i*64+j] = loc;
  //   }

  // }

  int num_t = blockDim.x * blockDim.y * blockDim.z;
  int thread_number = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + 1;
  //printf("Num threads is %d\n", num_t);

  
  int i;
  int j;
  for(int idx=thread_number; idx<64*64; idx+= num_t){
    i = idx/64;
    j= idx%64;
    loc=s_c[i*64+j];
    for (int p = 0; p < 64; ++p){  //64 columns of A
      loc += s_a[p*64 + i] * s_b[p*64 + j];

    }
    s_c[i*64+j] = loc;
  }

  //bank conflic minimizing approach (works only for 32 threads)

  // int row = thread_number - 1;

  // if (row<32){
  // for(int col=0; col<64; col++){

  //   loc = s_c[row*64 + col];
  //   for(int p=0; p<64; p++){

  //     loc += s_a[p*64 + row]* s_b[p*64 + col];

  //   }

  //   s_c[row*64 + col] = loc;

  // }

  // __syncthreads();

  // row = row+32;

  // for(int col=0; col<64; col++){

  //   loc = s_c[row*64 + col];
  //   for(int p=0; p<64; p++){

  //     loc += s_a[p*64 + row]* s_b[p*64 + col];

  //   }

  //   s_c[row*64 + col] = loc;

  // }}

  //other approach 

  // int row = thread_number - 1;
  // if (row<64){
  // for(int col=0; col<64; col++){

  //   loc = s_c[row*64 + col];
  //   for(int p=0; p<64; p++){

  //     loc += s_a[p*64 + row]* s_b[p*64 + col];

  //   }

  //   s_c[row*64 + col] = loc;

  // }}
  
 
  /**** Do Not Change Code Below This ****/
  
  //copy C out such that C is in row major order
  for (int i = id; i < 64*64; i += p)
    c[i] = s_c[i] ;

  }
  



int main(){

    float *host_out;
    float *dev_out;

    cudaEvent_t st1, et1, st2, et2;
    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);
    
    float ms1, ms2;
    
    //create buffer on host
    host_out = (float*) malloc(64 * 64 * sizeof(float));
    
    //create buffer on device
    cudaError_t err = cudaMalloc(&dev_out, 64 * 64 * sizeof(float));
    if (err != cudaSuccess){
       cout<<"Dev Memory not allocated"<<endl;
       exit(-1);
     }

    //record time at start
    cudaEventRecord(st2);

    //change number of threads here
    kernel_call<<<1, 1024>>>(dev_out);

    //wait until kernel is done start timing
    cudaDeviceSynchronize();       
    cudaEventRecord(et2);

    cudaEventElapsedTime(&ms2, st2, et2);
    cout<<"Kernel:\t\t\t"<<ms2<<"ms"<<endl;
    
    cudaMemcpy(host_out, dev_out, sizeof(float)*64*64, cudaMemcpyDeviceToHost);   
    
    free(host_out);
    cudaFree(dev_out);

  return 0;
}