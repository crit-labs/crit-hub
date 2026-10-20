#include "gemm.cuh"
#include <cstdio>
#include <vector>
using namespace fnl;
int main(){
  struct Case{WFmt f; int M,N,K;} cases[]={{WFmt::mx8,2600,10240,2560},{WFmt::bf16,256,248320,2560},{WFmt::bf16,2600,10240,320},{WFmt::fp4,2600,2560,640},{WFmt::mx8,2600,48,2560}};
  for(auto c: cases){
    void *A,*W,*S,*C; size_t wb = c.f==WFmt::bf16? size_t(c.N)*c.K*2 : c.f==WFmt::mx8? size_t(c.N)*c.K : size_t(c.N)*c.K/2;
    cudaMalloc(&A,size_t(c.M)*c.K*2); cudaMalloc(&W,wb); cudaMalloc(&S,size_t(c.N)*c.K/16+16); cudaMalloc(&C,size_t(c.M)*c.N*4);
    cudaMemset(A,0,size_t(c.M)*c.K*2); cudaMemset(W,0,wb); cudaMemset(S,0x40,size_t(c.N)*c.K/16+16);
    GemmArgs p; p.A=(const __half*)A; p.lda=c.K; p.M=c.M; p.N=c.N; p.K=c.K; p.W=W; p.Ws=(const uint8_t*)S; p.C=(float*)C; p.ldc=c.N;
    gemm(c.f,Epi::f32,p,0,0); cudaDeviceSynchronize();
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for(int i=0;i<5;i++) gemm(c.f,Epi::f32,p,0,0);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); ms/=5;
    printf("fmt %d M %d N %d K %d: %.3f ms, %.1f TFLOPS\n",(int)c.f,c.M,c.N,c.K,ms,2.0*c.M*c.N*c.K/ms/1e9);
    cudaFree(A);cudaFree(W);cudaFree(S);cudaFree(C);
  }
}
