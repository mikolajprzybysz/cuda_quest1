#include "generalCuda.h"
extern "C" void callTopColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float maxY);
extern "C" void callSideColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float maxX);
extern "C" void callGravityForceKernel(float *d_system,ParticleSystem *sys, float gravity);
extern "C" void callFloorColisionKernel(float *d_system,float *h_system, ParticleSystem *sys, float loss_param, float minY);
extern "C" void uploadMemorySystem(ParticleSystem *sys, float *h_system, float *d_system);
extern "C" void downloadMemorySystem(ParticleSystem *sys, float *h_system, float *d_system);

/* length of state derivative, and force vectors */
int ParticleDims(ParticleSystem *p){
	return(4 * p->n);
}

/* gather state from the particles into dst */
int ParticleGetState(ParticleSystem *p, float *dst){
	int i;
	for(i=0; i < p->n; i++){
		*(dst++) = p->p[i].x[0];
		*(dst++) = p->p[i].x[1];
		*(dst++) = p->p[i].v[0];
		*(dst++) = p->p[i].v[1];
	}
	return i;
}

///* scatter state from src into the particles */
void ParticleSetState(ParticleSystem *p, float *src){
int i;
	for(i=0; i < p->n; i++){
		p->p[i].x[0] = *(src++);
		p->p[i].x[1] = *(src++);
		p->p[i].v[0] = *(src++);
		p->p[i].v[1] = *(src++);
	}
}
float distance(Particle a,Particle b){
	float dist = sqrtf( (b.x[0]-a.x[0])*(b.x[0]-a.x[0]) +(b.x[1]-a.x[1])*(b.x[1]-a.x[1]) );
	return dist;
}
float valueOfVector(float * vector){
	return sqrtf(vector[0]*vector[0]+vector[1]*vector[1]);
}

void Ball_Colision_Force(ParticleSystem *sys,float lossParam){
	//float x1,x2,y1,y2;
	float m1,m2;
	float v1[2], v2[2];
	float mi1,mi2;
	float u1[2],u2[2];
	float p[2],p1[2],p2[2];
	float dp[2];
	float p1after[2];
	float p2after[2];
	float vector[2];
	float dpVal,pVal,vectorVal;
	int h=0;

	for(int i=0;i<sys->n;i++){
		for(int j=h;j<sys->n;j++){
			if(j==i)continue;
			//if(distance(sys->p[i],sys->p[j]) - sys->p[i].R - sys->p[j].R <0) printf("\n %f ",distance(sys->p[i],sys->p[j])-sys->p[i].R-sys->p[j].R);
			if(distance(sys->p[i],sys->p[j])-0.1f <= (sys->p[i].R+sys->p[j].R) ){
				m1 = sys->p[i].m;
				m2 = sys->p[j].m;
				
				v1[0]=sys->p[i].v[0];
				v1[1]=sys->p[i].v[1];

				v2[0]=sys->p[j].v[0];
				v2[1]=sys->p[j].v[1];
				
				p1[0]=m1*v1[0];
				p1[1]=m1*v1[1];

				p2[0]=m2*v2[0];
				p2[1]=m2*v2[1];

				p[0]=p2[0]-p1[0];
				p[1]=p2[1]-p1[1];
				
				

				vector[0]=sys->p[i].x[0] - sys->p[j].x[0];
				vector[1]=sys->p[i].x[1] - sys->p[j].x[1];

				if(vector[0]==0 && vector[1]==0){
					sys->p[i].x[0]-= (sys->p[i].R* sys->p[i].v[0]/valueOfVector(sys->p[i].v));
					sys->p[i].x[1]-= (sys->p[i].R* sys->p[i].v[1]/valueOfVector(sys->p[i].v));

					sys->p[j].x[0]-= (sys->p[j].R* sys->p[j].v[0]/valueOfVector(sys->p[j].v));
					sys->p[j].x[1]-= (sys->p[j].R* sys->p[j].v[1]/valueOfVector(sys->p[j].v));

					vector[0]=sys->p[i].x[0] - sys->p[j].x[0];
					vector[1]=sys->p[i].x[1] - sys->p[j].x[1];
				}

				dpVal = (p[0]*vector[0]+p[1]*vector[1])/valueOfVector(vector);

				dp[0]=vector[0]*dpVal/valueOfVector(vector);
				dp[1]=vector[1]*dpVal/valueOfVector(vector);
				
				p1after[0]=p1[0]+dp[0];
				p1after[1]=p1[1]+dp[1];

				p2after[0]=p2[0]-dp[0];
				p2after[1]=p2[1]-dp[1];

				sys->p[i].v[0]=lossParam*p1after[0]/m1;
				sys->p[i].v[1]=lossParam*p1after[1]/m1;

				sys->p[j].v[0]=lossParam*p2after[0]/m2;
				sys->p[j].v[1]=lossParam*p2after[1]/m2;

				int counter=0;
				while(distance(sys->p[i],sys->p[j]) - sys->p[i].R - sys->p[j].R <0){
					sys->p[i].x[0]+=vector[0]*0.02f;
					sys->p[i].x[1]+=vector[1]*0.02f;

					sys->p[j].x[0]-=vector[0]*0.02f;
					sys->p[j].x[1]-=vector[1]*0.02f;
					counter++;
					if(counter> (sys->p[i].R + sys->p[j].R)/0.02f)break;
				}
				/*mi1=m1/(m1+m2);
				mi2=m2/(m1+m2);

				u1[0]=(mi1-mi2)*v1[0]+2*mi2*v2[0];
				u2[0]=(mi2-mi1)*v2[0]+2*mi1*v1[0];

				u1[1]=(mi1-mi2)*v1[1]+2*mi2*v2[1];
				u2[1]=(mi2-mi1)*v2[1]+2*mi1*v1[1];

				p->p[i].v[0]=u1[0];
				p->p[i].v[1]=u1[1];

				p->p[j].v[0]=u2[0];
				p->p[j].v[1]=u2[1];*/
			}
		}
		h++;
	}
}

void Floor_Colision_Force(ParticleSystem *p,float loss_param){

	for(int i=0;i<p->n;i++){
		if(p->p[i].x[1]-p->p[i].R< -20.0f /*&& p->p[i].v[1]<-0.2f*/){
			p->p[i].x[1]=-20.0f+p->p[i].R;
			if(abs(p->p[i].v[1]) >3.0f) p->p[i].v[1]= -p->p[i].v[1]*loss_param;
			else
				p->p[i].f[1]=0.0f;
		}
	}
}

void Top_Colision_Force(ParticleSystem *sys,float loss_param){
		for(int i=0;i<sys->n;i++){
			if(sys->p[i].x[1]+sys->p[i].R> 20.0f){
				sys->p[i].v[1]= -sys->p[i].v[1]*loss_param;
				sys->p[i].x[1]= 20.0f - sys->p[i].R;
			}
	}

}

void Side_Colision_Force(ParticleSystem *sys,float loss_param){
	for(int i=0;i<sys->n;i++){
		if(sys->p[i].x[0]+sys->p[i].R> 20.0f){
			sys->p[i].v[0]= -sys->p[i].v[0]*loss_param;
			sys->p[i].x[0]= 20.0f-sys->p[i].R;
			
		}
		if(sys->p[i].x[0]-sys->p[i].R< -20.0f){
			sys->p[i].v[0]= -sys->p[i].v[0]*loss_param;
			sys->p[i].x[0]= -20.0f+sys->p[i].R;
		}
		if(abs(sys->p[i].v[0])<=3.0f && sys->p[i].f[0]==0.00f) sys->p[i].v[0]*=-0.1f;
	}
}

void Gravity_Force(ParticleSystem *p,float gravity){
	for(int i=0;i<p->n;i++){
		//p->p[i].f[0] += 0.0f;
		p->p[i].f[1] += - (p->p[i].m*gravity);
	}
}

/*applies forces to system*/
void Compute_Forces(ParticleSystem *p,int mask,float *h_system,float *d_system){
	if(mask & CUDASIMULATION){
		uploadMemorySystem(p,h_system,d_system);
		
		if(mask&GRAVITY) callGravityForceKernel(d_system,p,9.81f);
		callSideColisionKernel(d_system,h_system,p,0.80f,20.0f);
		callTopColisionKernel(d_system,h_system,p,0.80f,20.0f);
		callFloorColisionKernel(d_system,h_system,p,0.80f,-20.0f);

		downloadMemorySystem(p,h_system,d_system);
		Ball_Colision_Force(p,1.0f);

	}else{
		if(mask&GRAVITY) Gravity_Force(p,9.81f);
		Side_Colision_Force(p,0.80f);
		Top_Colision_Force(p,0.80f);
		Floor_Colision_Force(p,0.80f);
		Ball_Colision_Force(p,1.0f);
	}

}

/*zero the force accumulators*/
void Clear_Forces(ParticleSystem *p){
	for(int i=0;i<p->n;i++){
		p->p[i].f[0]=0.0f;
		p->p[i].f[1]=0.0f;
	}
}

/* calculate derivative, place in dst */
void ParticleDerivative(ParticleSystem *p, float *dst,int mask,float *h_system,float *d_system){
	int i;
	Clear_Forces(p); /* zero the force accumulators */
    Compute_Forces(p,mask,h_system,d_system); /* magic force function */

	for(i=0; i < p->n; i++){
		*(dst++) = p->p[i].v[0]; /* xdot = v */
		*(dst++) = p->p[i].v[1];
		*(dst++) = p->p[i].f[0] / p->p[i].m; /* vdot = f/m */
		*(dst++) = p->p[i].f[1] / p->p[i].m;
	}
}
/* add vectors temp1 and temp2 then store result in temp3. // temp3 = temp1+temp2
  n number of particles
*/
void AddVectors(float *temp1, float *temp2, float *temp3,ParticleSystem *p){
	for(int i=0;i< ParticleDims(p) ;i++){
		*(temp3++)= *(temp1++)+ *(temp2++);
	}
}

/*scale vector by DeltaT where DeltaT is 1s /fps number*/
void ScaleVector(float *temp1,float DeltaT, ParticleSystem *p){
	for(int i=0;i<ParticleDims(p);i++){
		*(temp1++) = *(temp1)*DeltaT;
	}
}

void EulerStep(ParticleSystem *p, float DeltaT,int mask,float *h_system,float *d_system){
	float *temp1 = (float*) malloc(sizeof(float)*ParticleDims(p));
	float *temp2 = (float*) malloc(sizeof(float)*ParticleDims(p));
	//DeltaT = 1/fps
	ParticleDerivative(p,temp1,mask,h_system,d_system); /* get deriv */
	ScaleVector(temp1,DeltaT,p); /* scale it */
	ParticleGetState(p,temp2); /* get state */
	AddVectors(temp1,temp2,temp2,p); /* add -> temp2 */
	ParticleSetState(p,temp2); /* update state */
	p->t += DeltaT; /* update time */

	free(temp1);
	free(temp2);
}
void cleanupParticleSystem(ParticleSystem *sys){
	int n = sys->n;
	for(int i=0;i<n;i++){
		free(sys->p[i].f);
		free(sys->p[i].v);
		free(sys->p[i].x);
		
	}
	free(sys->p);
	free(sys);

}

ParticleSystem * createColisionTestingParticleSystem(int no,float radiusBall,float radius){
	//int no = side;
	//float radius = 3.0f;
	ParticleSystem * sys = createParticleSystem(no,radiusBall,3.0f);
	//for(int i=0;i<side;i++){
	//	for(int j=0;j<side;j++){
	//		sys->p[j*side+i].x[0] = i*0.3f - 2.0f;
	//		sys->p[j*side+i].x[1] = 19.0f - j*0.3f;
	//	}
	//}

	for (int i = 0; i < no; i++)
	{
		float degInRad = (360.00f/(float)no)*i * 3.14159/180;
		sys->p[i].x[0] = cos(degInRad)*radius;
		sys->p[i].x[1] =	10.0f+sin(degInRad)*radius;
	}
	
	
	//sys->p[(side/2)*side+side/2].v[0] = -2.0f;
	/*sys->p[2].x[0] = 10.0f;
	sys->p[2].v[0] = -6.0f;
	sys->p[2].v[1] = 6.0f;*/

	return sys;
}
ParticleSystem * createParticleSystem(int noOfParticles, float radius,float mass){

	ParticleSystem * sys = (ParticleSystem *) malloc(sizeof(ParticleSystem));
	sys->p =  (Particle *) malloc(sizeof(Particle)*noOfParticles);

	/*for(int i=0;i<noOfParticles;i++){
		sys->p[i] = (Particle)malloc(sizeof(Particle));
	}*/

	for(int i=0;i<noOfParticles;i++){

		sys->p[i].f = (float*)malloc(sizeof(float)*2);
		sys->p[i].f[0] = 0.0f;
		sys->p[i].f[1] = 0.0f;

		sys->p[i].v = (float*)malloc(sizeof(float)*2);
		sys->p[i].v[0] = 0.0f;
		sys->p[i].v[1] = 0.0f;

		sys->p[i].x = (float*)malloc(sizeof(float)*2);
		sys->p[i].x[0] = 0.0f;
		sys->p[i].x[1] = 0.0f;

		sys->p[i].R = radius;
		sys->p[i].m = mass;
		
	}

	sys->n=noOfParticles;
	sys->t = 0.0f;
	return  sys;
}