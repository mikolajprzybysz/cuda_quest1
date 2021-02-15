#ifndef GENERALCUDA_H
#define GENERALCUDA_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#define GRAVITY        0x00000001
#define CUDASIMULATION 0x10000000
typedef struct {
float m; /* mass */
float *x; /* position vector */
float *v; /* velocity vector */
float *f; /* force accumulator */
float R;
} Particle;

typedef struct{
Particle *p; /* array of pointers to particles */
int n; /* number of particles */
float t; /* simulation clock */
} ParticleSystem;

ParticleSystem * createParticleSystem(int noOfParticles, float radius,float mass);
int ParticleDims(ParticleSystem *p);
int ParticleGetState(ParticleSystem *p, float *dst);
void ParticleSetState(ParticleSystem *p, float *src);
float distance(Particle a,Particle b);
float valueOfVector(float * vector);
void Ball_Colision_Force(ParticleSystem *sys,float lossParam);
void Floor_Colision_Force(ParticleSystem *p,float loss_param);
void Top_Colision_Force(ParticleSystem *sys,float loss_param);
void Side_Colision_Force(ParticleSystem *sys,float loss_param);
void Gravity_Force(ParticleSystem *p,float gravity);
void Compute_Forces(ParticleSystem *p,int mask,float *h_system,float *d_system);
void Clear_Forces(ParticleSystem *p);
void ParticleDerivative(ParticleSystem *p, float *dst,int mask,float *h_system,float *d_system);
void AddVectors(float *temp1, float *temp2, float *temp3,ParticleSystem *p);
void ScaleVector(float *temp1,float DeltaT, ParticleSystem *p);
void EulerStep(ParticleSystem *p, float DeltaT,int mask,float *h_system,float *d_system);
void cleanupParticleSystem(ParticleSystem *sys);
ParticleSystem * createColisionTestingParticleSystem(int no,float radiusBall,float radius);
ParticleSystem * createParticleSystem(int noOfParticles, float radius,float mass);



#endif