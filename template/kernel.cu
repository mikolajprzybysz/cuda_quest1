//*Side_Colision_Force(p);
//Top_Colision_Force(p);
//Floor_Colision_Force(p,0.90f);
//Ball_Colision_Force(p,1.0f);
//ParticleDerivative(p,temp1,mask);
//ScaleVector(temp1,DeltaT,p);*/
//
//*void Floor_Colision_Force(ParticleSystem *p,float loss_param){
//
//	for(int i=0;i<p->n;i++){
//		if(p->p[i].x[1]-p->p[i].R< -20.0f && p->p[i].v[1]<-0.2f){
//			p->p[i].v[1]= -p->p[i].v[1]*loss_param;
//			p->p[i].x[1]=-20.0f+p->p[i].R;
//		}
//	}
//}
//
//void Top_Colision_Force(ParticleSystem *sys){
//		for(int i=0;i<sys->n;i++){
//			if(sys->p[i].x[1]+sys->p[i].R> 20.0f){
//				sys->p[i].v[1]= -sys->p[i].v[1];
//				sys->p[i].x[1]= 20.0f - sys->p[i].R;
//			}
//	}
//
//}
//
//void Side_Colision_Force(ParticleSystem *sys){
//	for(int i=0;i<sys->n;i++){
//		if(sys->p[i].x[0]+sys->p[i].R> 20.0f){
//			sys->p[i].v[0]= -sys->p[i].v[0];
//			sys->p[i].x[0]= 20.0f-sys->p[i].R;
//		}
//		if(sys->p[i].x[0]-sys->p[i].R< -20.0f){
//			sys->p[i].v[0]= -sys->p[i].v[0];
//			sys->p[i].x[0]= -20.0f+sys->p[i].R;
//		}
//	}
//}
//
//void Gravity_Force(ParticleSystem *p,float gravity){
//	for(int i=0;i<p->n;i++){
//		p->p[i].f[0] += 0.0f;
//		p->p[i].f[1] += - (p->p[i].m*gravity);
//	}
//}*/

#include "generalCuda.h"
__device__ inline int getutid()
{
int threadsPerBlock = blockDim.x * blockDim.y;
int tidWithinBlock = threadIdx.x + threadIdx.y * blockDim.x;
int gid = blockIdx.x + blockIdx.y * gridDim.x;
return gid * threadsPerBlock + tidWithinBlock;
}

int ParticleGetStateAll(ParticleSystem *p, float *dst){
	int i;
	for(i=0; i < p->n; i++){
		*(dst++) = p->p[i].x[0];
		*(dst++) = p->p[i].x[1];
		*(dst++) = p->p[i].v[0];
		*(dst++) = p->p[i].v[1];
		*(dst++) = p->p[i].f[0];
		*(dst++) = p->p[i].f[1];
		*(dst++) = p->p[i].m;
		*(dst++) = p->p[i].R;
	}
	return i;
}

///* scatter state from src into the particles */
void ParticleSetStateAll(ParticleSystem *p, float *src){
int i;
	for(i=0; i < p->n; i++){
		p->p[i].x[0] = *(src++);
		p->p[i].x[1] = *(src++);
		p->p[i].v[0] = *(src++);
		p->p[i].v[1] = *(src++);
		p->p[i].f[0] = *(src++);
		p->p[i].f[1] = *(src++);
		p->p[i].m = *(src++);
		p->p[i].R = *(src++);
	}
}

__global__ void gravityForceKernel(float *d_system,float gravity,int no){
	int thIndex = getutid();

	int d_system_index = thIndex*8;
	if(thIndex<no){
		d_system[d_system_index+5] += - (d_system[d_system_index+6]*gravity);
	}
}
extern "C" void callGravityForceKernel(float *d_system,ParticleSystem *sys, float gravity){
	dim3 dimBlock(16,16);
	dim3 dimGrid(32,32);
	if(sys->n > dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y){
		printf("\n to many objects. limit is: %d",dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y);
		return;
	}
	gravityForceKernel<<<dimGrid,dimBlock>>>(d_system,gravity,sys->n);
}
/*
void Top_Colision_Force(ParticleSystem *sys,float loss_param){
		for(int i=0;i<sys->n;i++){
			if(sys->p[i].x[1]+sys->p[i].R> 20.0f){
				sys->p[i].v[1]= -sys->p[i].v[1]*loss_param;
				sys->p[i].x[1]= 20.0f - sys->p[i].R;
			}
	}

}
*/
__global__ void topColisionKernel(float *d_system,float loss_param,int no,float maxY){
	int thIndex = getutid();

	int d_system_index = thIndex*8;
	if(thIndex<no){
		if(d_system[d_system_index+1] + d_system[d_system_index+7]> maxY){
			d_system[d_system_index+3]= -d_system[d_system_index+3]*loss_param;
			d_system[d_system_index+1]= maxY-d_system[d_system_index+7];
		}
	}
}
extern "C" void callTopColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float maxY){
	dim3 dimBlock(16,16);
	dim3 dimGrid(32,32);
	if(sys->n > dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y){
		printf("\n to many objects. limit is: %d",dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y);
		return;
	}
	topColisionKernel<<<dimGrid,dimBlock>>>(d_system,loss_param,sys->n ,maxY);

}


__global__ void sideColisionKernel(float *d_system,float loss_param,int no,float maxX){
	int thIndex = getutid();

	int d_system_index = thIndex*8;
	if(thIndex<no){
		if(d_system[d_system_index] + d_system[d_system_index+7]> maxX){
			d_system[d_system_index+2]= -d_system[d_system_index+2]*loss_param;
			d_system[d_system_index]= maxX-d_system[d_system_index+7];
		}
		if(d_system[d_system_index] - d_system[d_system_index+7]< -maxX){
			d_system[d_system_index+2]= -d_system[d_system_index+2]*loss_param;
			d_system[d_system_index]= -maxX+d_system[d_system_index+7];
		}
		if(abs(d_system[d_system_index+2])<=3.0f && d_system[d_system_index+4]==0.00f) d_system[d_system_index+2]=d_system[d_system_index+2]*(-0.1f);
	}
}

extern "C" void callSideColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float maxX){
	dim3 dimBlock(16,16);
	dim3 dimGrid(32,32);
	if(sys->n > dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y){
		printf("\n to many objects. limit is: %d",dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y);
		return;
	}
	sideColisionKernel<<<dimGrid,dimBlock>>>(d_system,loss_param,sys->n ,maxX);
}
/*
	if(p->p[i].x[1]-p->p[i].R< -20.0f){
			p->p[i].x[1]=-20.0f+p->p[i].R;
			if(abs(p->p[i].v[1]) >3.0f) p->p[i].v[1]= -p->p[i].v[1]*loss_param;
			else
				p->p[i].f[1]=0.0f;
		}
*/
__global__ void floorColisionKernel(float *d_system,float loss_param,int no,float minY){
	int thIndex = getutid();

	int d_system_index = thIndex*8;
	if(thIndex<no){
		if(d_system[d_system_index+1] - d_system[d_system_index+7]< minY){
			d_system[d_system_index+1]=minY+d_system[d_system_index+7];
			if(abs(d_system[d_system_index+3])>3.0f) d_system[d_system_index+3]= -d_system[d_system_index+3]*loss_param;
			else
			d_system[d_system_index+5]= 0.0f;
		}
	}
}
extern "C" void callFloorColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float minY){
	dim3 dimBlock(16,16);
	dim3 dimGrid(32,32);
	if(sys->n > dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y){
		printf("\n to many objects. limit is: %d",dimBlock.x*dimBlock.y*dimGrid.x*dimGrid.y);
		return;
	}
	floorColisionKernel<<<dimGrid,dimBlock>>>(d_system,loss_param,sys->n ,minY);
	

}
extern "C" void callMallocSystemMemory(ParticleSystem *sys,float **h_system, float **d_system){
	float *d_temp =NULL;
	cudaMalloc((void**) d_system,sizeof(float)* 8*sys->n);
	//cudaMalloc((void**) &d_temp,sizeof(float)* 8*sys->n);
	cudaError err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf( "Cuda error: %s: %s.\n", "error", 
                                  cudaGetErrorString( err) );
    } 
	//float *temp= (float *)malloc(sizeof(float)* 8*sys->n);
	*h_system = (float *)malloc(sizeof(float)* 8*sys->n);
	//cudaFree(d_temp);
	//free(temp);
	
//minY = -20.0f
	//floorColisionKernel<<<>>>(d_system,loss_param,sys->n);
}
extern "C" void callClearMemory(float *d_system, float *h_system){
	cudaFree(d_system);
	free(h_system);
}
extern "C" void uploadMemorySystem(ParticleSystem *sys, float *h_system, float *d_system){
	ParticleGetStateAll(sys,h_system);
	cudaMemcpy(d_system,h_system,sizeof(float)* 8*sys->n,cudaMemcpyHostToDevice);
}
extern "C" void downloadMemorySystem(ParticleSystem *sys, float *h_system, float *d_system){
	cudaMemcpy(h_system,d_system,sizeof(float)* 8*sys->n,cudaMemcpyDeviceToHost);
	ParticleSetStateAll(sys,h_system);
}