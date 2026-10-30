#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "Bindings.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

constexpr unsigned particles = 32, stride = 8, words = particles * stride;
using State = std::array<float, words>;
extern "C" {
size_t md_cpu_gid;
int md_init(const float *), md_close(), md_read(float *), md_contract_tests();
int md_step(unsigned,float,float,float,float,float,float,float,float,float,float,float,float,float);
void md_drift(const float*,float*,float,float,float,float,float,float,float,float,float,float,float,float,float,float);
void md_kick(const float*,float*,float,float,float,float,float,float,float,float,float,float,float,float,float,float);
}
void check(int status) { if(status) throw std::runtime_error("OpenCL error "+std::to_string(status)); }
double minimumImage(double x,double box){return x-box*std::round(x/box);}

State initialState(float &box) {
  State state{}; constexpr float r0=1.122462048f, lattice=1.587401052f;
  box=2*lattice; const float basis[4][3]={{0,0,0},{0,.5f,.5f},{.5f,0,.5f},{.5f,.5f,0}};
  unsigned i=0; double momentum[3]{};
  for(unsigned iz=0;iz<2;++iz)for(unsigned iy=0;iy<2;++iy)for(unsigned ix=0;ix<2;++ix)for(auto &b:basis){
    state[stride*i]=(ix+b[0])*lattice; state[stride*i+1]=(iy+b[1]+.25f)*lattice; state[stride*i+2]=(iz+b[2])*lattice;
    state[stride*i+4]=.012f*std::sin(float(i)); state[stride*i+5]=.012f*std::cos(float(i)); state[stride*i+6]=.012f*std::sin(float(i)*1.618f);
    for(unsigned a=0;a<3;++a)momentum[a]+=state[stride*i+4+a]; ++i;
  }
  for(i=0;i<particles;++i)for(unsigned a=0;a<3;++a)state[stride*i+4+a]-=float(momentum[a]/particles);
  return state;
}
double energy(const State&s,double box){
  constexpr double rc=1.5,ic2=1/(rc*rc),ic6=ic2*ic2*ic2,vc=4*(ic6*ic6-ic6),fc=24*(2*ic6*ic6-ic6)/rc;
  double e=0; for(unsigned i=0;i<particles;++i){
    for(unsigned a=0;a<3;++a)e+=.5*double(s[stride*i+4+a])*s[stride*i+4+a];
    for(unsigned j=i+1;j<particles;++j){double r2=0;for(unsigned a=0;a<3;++a){double d=minimumImage(double(s[stride*i+a])-s[stride*j+a],box);r2+=d*d;}
      double r=std::sqrt(r2);if(r<rc){double q=1/r2,q6=q*q*q;e+=4*(q6*q6-q6)-vc+(r-rc)*fc;}}
  }return e;
}
std::array<double,3> momentum(const State&s){std::array<double,3>p{};for(unsigned i=0;i<particles;++i)for(unsigned a=0;a<3;++a)p[a]+=s[stride*i+4+a];return p;}
double temperature(const State&s){double k=0;for(unsigned i=0;i<particles;++i)for(unsigned a=0;a<3;++a)k+=.5*double(s[stride*i+4+a])*s[stride*i+4+a];return 2*k/(3*particles-3);}
double meanY(const State&s){double y=0;for(unsigned i=0;i<particles;++i)y+=s[stride*i+1];return y/particles;}
double associatedLegendre6(unsigned m,double x){
  double pmm=1; if(m){double root=std::sqrt(std::max(0.0,1-x*x)),factor=1;for(unsigned k=1;k<=m;++k){pmm*=-factor*root;factor+=2;}}
  if(m==6)return pmm; double pm1=x*(2*m+1)*pmm;if(m==5)return pm1;
  double previous=pmm,current=pm1;for(unsigned l=m+2;l<=6;++l){double next=((2*l-1)*x*current-(l+m-1)*previous)/(l-m);previous=current;current=next;}return current;
}
double bondOrder6(const State&s,double box){
  std::array<std::complex<double>,7> q{}; unsigned bonds=0; constexpr double pi=3.14159265358979323846;
  for(unsigned i=0;i<particles;++i)for(unsigned j=i+1;j<particles;++j){double d[3],r2=0;for(unsigned a=0;a<3;++a){d[a]=minimumImage(double(s[stride*j+a])-s[stride*i+a],box);r2+=d[a]*d[a];}
    if(r2>=1.4*1.4)continue;double r=std::sqrt(r2),phi=std::atan2(d[1],d[0]),x=d[2]/r;
    for(unsigned m=0;m<=6;++m){double ratio=1;for(unsigned k=6-m+1;k<=6+m;++k)ratio/=k;double norm=std::sqrt(13/(4*pi)*ratio);
      q[m]+=norm*associatedLegendre6(m,x)*std::polar(1.0,double(m)*phi);}++bonds;}
  if(!bonds)return 0;double sum=std::norm(q[0]);for(unsigned m=1;m<=6;++m)sum+=2*std::norm(q[m]);return std::sqrt(4*pi/13*sum)/bonds;
}
const char*phaseName(double t,double q6){if(t<.12&&q6>.42)return "ice";if(t>.35||q6<.25)return "liquid";return "melting";}
double localKinetic(const State&s,float cx,float cy,float cz,bool near,float radius){double k=0;unsigned n=0;for(unsigned i=0;i<particles;++i){double r2=0;float c[3]={cx,cy,cz};for(unsigned a=0;a<3;++a){double d=s[stride*i+a]-c[a];r2+=d*d;}if((r2<radius*radius)!=near)continue;for(unsigned a=0;a<3;++a)k+=.5*double(s[stride*i+4+a])*s[stride*i+4+a];++n;}return n?k/n:0;}

int main(int argc,char**argv){try{
  if(argc!=4&&argc!=5)throw std::runtime_error("usage: md-crystal drift-shader kick-shader vertex-shader [--interactive]");
  native_opencl::Session session(deviceSource(),manifestBindings(argv[1],argv[2]),argv[3]); constexpr float dt=.001f;float box;State state=initialState(box),oracle{},temp{},actual{};
  auto step=[&](unsigned n,float h0x,float h0y,float h0z,float h1x,float h1y,float h1z,float radius,float amplitude,float cooling){check(md_step(n,dt,box,box,box,h0x,h0y,h0z,h1x,h1y,h1z,radius,amplitude,cooling));};
  if(argc==5&&std::string(argv[4])=="--interactive"){
    unsigned frame=0;check(md_init(state.data()));auto publish=[&]{check(md_read(actual.data()));double t=temperature(actual),q=bondOrder6(actual,box);std::cout<<"{\"frame\":"<<frame<<",\"box\":["<<box<<","<<box<<","<<box<<"],\"temperature\":"<<t<<",\"bond_order_6\":"<<q<<",\"phase\":\""<<phaseName(t,q)<<"\",\"state\":[";for(unsigned i=0;i<words;++i){if(i)std::cout<<',';std::cout<<actual[i];}std::cout<<"]}"<<std::endl;};publish();
    std::string command;while(std::getline(std::cin,command)){std::istringstream in(command);std::string name;in>>name;if(name=="quit")break;if(name=="reset"){check(md_init(state.data()));frame=0;}else if(name=="step"){unsigned n;float x0,y0,z0,x1,y1,z1,amp,r=1.7f,c=1.25f;if(!(in>>n>>x0>>y0>>z0>>x1>>y1>>z1>>amp)||n>100)throw std::runtime_error("invalid simulation command");if(in>>r){if(!(in>>c))throw std::runtime_error("radius requires cooling rate");}step(n,x0,y0,z0,x1,y1,z1,r,amp,c);frame+=n;}else throw std::runtime_error("unknown simulation command");publish();}check(md_close());return 0;
  }
  // Independent finite-difference force check on all three axes.
  State displaced=state;displaced[stride*27]+=.043f;displaced[stride*27+1]+=.027f;displaced[stride*27+2]-=.031f;
  for(md_cpu_gid=0;md_cpu_gid<words;++md_cpu_gid)md_drift(displaced.data(),temp.data(),dt,box,box,box,-100,-100,-100,-100,-100,-100,1.7f,0,0,0);
  for(unsigned axis=0;axis<3;++axis){State plus=displaced,minus=displaced;unsigned c=stride*27+axis;plus[c]+=.001f;minus[c]-=.001f;double expected=-(energy(plus,box)-energy(minus,box))/(plus[c]-minus[c]);double force=(temp[c+4]-displaced[c+4])/(.5*dt);if(std::abs(force-expected)>.004+std::abs(expected)*.003){std::cerr<<"axis="<<axis<<" expected="<<expected<<" force="<<force<<"\n";throw std::runtime_error("force differs from potential derivative");}}
  oracle=state;check(md_init(state.data()));check(md_contract_tests());float maxError=0;
  for(unsigned s=0;s<40;++s){for(md_cpu_gid=0;md_cpu_gid<words;++md_cpu_gid)md_drift(oracle.data(),temp.data(),dt,box,box,box,box/2,box/2,box/2,-100,-100,-100,1.7f,.01f,0,s);for(md_cpu_gid=0;md_cpu_gid<words;++md_cpu_gid)md_kick(temp.data(),oracle.data(),dt,box,box,box,box/2,box/2,box/2,-100,-100,-100,1.7f,.01f,0,s);step(1,box/2,box/2,box/2,-100,-100,-100,1.7f,.01f,0);check(md_read(actual.data()));for(unsigned w=0;w<words;++w){float d=actual[w]-oracle[w];if(w%stride<3)d-=box*std::round(d/box);maxError=std::max(maxError,std::abs(d));}}
  if(maxError>=.003f)throw std::runtime_error("CPU/GPU trajectory mismatch");
  check(md_init(state.data()));double e0=energy(state,box),maxEDrift=0,maxPDrift=0;auto p0=momentum(state);
  for(unsigned chunk=0;chunk<10;++chunk){step(30,-100,-100,-100,-100,-100,-100,1.7f,0,0);check(md_read(actual.data()));maxEDrift=std::max(maxEDrift,std::abs(energy(actual,box)-e0)/std::abs(e0));auto p=momentum(actual);double pd=0;for(unsigned a=0;a<3;++a)pd+=(p[a]-p0[a])*(p[a]-p0[a]);maxPDrift=std::max(maxPDrift,std::sqrt(pd));}
  if(maxEDrift>3e-4||maxPDrift>2e-3){std::cerr<<"energy_drift="<<maxEDrift<<" momentum_drift="<<maxPDrift<<"\n";throw std::runtime_error("NVE conservation failed");}
  double control=localKinetic(actual,box/2,box/2,box/2,true,1.7f);step(800,box/2,box/2,box/2,-100,-100,-100,1.7f,.04f,0);check(md_read(actual.data()));double heated=localKinetic(actual,box/2,box/2,box/2,true,1.7f);if(heated<=control*1.5)throw std::runtime_error("3D heater did not heat crystal");
  check(md_init(state.data()));check(md_read(actual.data()));double iceT=temperature(actual),iceQ=bondOrder6(actual,box),iceY=meanY(actual);
  step(2000,box/2,box/2,box/2,box/2,box/2,box/2,100,.04f,0);check(md_read(actual.data()));double liquidT=temperature(actual),liquidQ=bondOrder6(actual,box),liquidY=meanY(actual);
  step(6000,-100,-100,-100,-100,-100,-100,1.7f,0,1);check(md_read(actual.data()));double frozenT=temperature(actual),frozenQ=bondOrder6(actual,box),frozenY=meanY(actual);
  if(std::string(phaseName(iceT,iceQ))!="ice"||std::string(phaseName(liquidT,liquidQ))!="liquid"||std::string(phaseName(frozenT,frozenQ))!="ice"||iceY-liquidY<.2){std::cerr<<"ice="<<iceT<<","<<iceQ<<" liquid="<<liquidT<<","<<liquidQ<<" frozen="<<frozenT<<","<<frozenQ<<" flow_drop="<<iceY-liquidY<<"\n";throw std::runtime_error("3D ice-liquid-flow-refreeze cycle failed");}
  std::cout<<"NVE_steps=300 relative_energy_drift="<<maxEDrift<<" momentum_drift="<<maxPDrift<<" local_control_KE="<<control<<" local_heated_KE="<<heated<<"\n";
  std::cout<<"phase_cycle=ice-liquid-flow-ice ice_T="<<iceT<<" ice_Q6="<<iceQ<<" ice_mean_y="<<iceY<<" liquid_T="<<liquidT<<" liquid_Q6="<<liquidQ<<" liquid_mean_y="<<liquidY<<" flow_drop="<<iceY-liquidY<<" refrozen_T="<<frozenT<<" refrozen_Q6="<<frozenQ<<" refrozen_mean_y="<<frozenY<<"\n";
  check(md_close());if(udoncl_cl_live_objects())throw std::runtime_error("OpenCL object leak");std::cout<<"renderer="<<session.deviceName()<<" oracle_steps=40 draws="<<session.drawCount()<<" max_cpu_gpu_error="<<maxError<<"\n";return 0;
}catch(const std::exception&e){md_close();std::cerr<<e.what()<<"\n";return 1;}}
