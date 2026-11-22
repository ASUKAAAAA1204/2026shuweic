// Finite-horizon stochastic Bellman recursion for the 16-task nominal case.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
constexpr int N=16, Z=5, F=3;
struct Value {
    int ticks;
    double dt,H,cw,late;
    std::array<int,N> family,zone,parent;
    std::array<double,N> release,due,w;
    std::array<double,Z*Z> dist;
    std::vector<int> index;
    std::vector<float> values;
    Value(double horizon,double step,double weight,double penalty,const int* fam,const int* zones,const int* pred,
          const double* rel,const double* deadlines,const double* weights,const double* geo,int root=0):
        ticks(int(std::ceil(horizon/step))+1),dt(step),H(horizon),cw(weight),late(penalty),index(1<<N,-1) {
        std::copy(fam,fam+N,family.begin());std::copy(zones,zones+N,zone.begin());std::copy(pred,pred+N,parent.begin());
        std::copy(rel,rel+N,release.begin());std::copy(deadlines,deadlines+N,due.begin());
        std::copy(weights,weights+N,w.begin());std::copy(geo,geo+Z*Z,dist.begin());
        int count=0;
        for(int m=0;m<(1<<N);m++){
            if((m&root)!=root)continue;
            bool ok=true;
            for(int j=0;j<N;j++)if((m&(1<<j)) && (m&parent[j])!=parent[j]){ok=false;break;}
            if(ok)index[m]=count++;
        }
        values.resize(size_t(count)*Z*ticks);
    }
    float* row(int m,int z){return values.data()+(size_t(index[m])*Z+z)*ticks;}
    double interpolate(const float* v,double t) const {
        if(t>=H)return 0.;
        double g=std::max(0.,t/dt);int k=int(g);
        return (1-(g-k))*v[k]+(g-k)*v[k+1];
    }
    double reward(int j,double end)const {
        return w[j]*((end<=due[j]+1e-10)+cw+late*(std::max(0.,H-due[j])-std::max(0.,end-due[j])));
    }
    void build(int count,const double* support,bool per_job=false){
        // First integrate over service at a given service-start time. This is shared across all origin zones.
        std::vector<float> q(N*ticks);
        for(int m=(1<<N)-2;m>=0;m--){
            if(index[m]<0)continue;
            std::array<int,N> ready{};int nr=0;
            for(int j=0;j<N;j++)if(!(m&(1<<j)) && (m&parent[j])==parent[j]){
                ready[nr++]=j;
                const float* child=row(m|(1<<j),zone[j]);float* base=&q[j*ticks];
                for(int t=0;t<ticks;t++){
                    double score=0;
                    for(int s=0;s<count;s++){
                        double end=t*dt+support[(per_job?j:family[j])*count+s];
                        if(end<=H+1e-10)score+=reward(j,end)+interpolate(child,end);
                    }
                    base[t]=float(score/count);
                }
            }
            for(int z=0;z<Z;z++){
                float* v=row(m,z);
                for(int t=ticks-2;t>=0;t--){
                    double now=t*dt,best=0;
                    for(int k=0;k<nr;k++){
                        int j=ready[k];
                        double begin=std::max(now,release[j])+dist[z*Z+zone[j]]+.5;
                        if(begin+.2<=H)best=std::max(best,interpolate(&q[j*ticks],begin));
                    }
                    v[t]=float(best);
                }
            }
        }
    }
    int first(int mask,int loc,double now,double budget,int count,const double* samples,double pace,double shift) {
        if(mask<0||mask>=(1<<N)||index[mask]<0)return -1;
        double best=-1.;int first=-1;
        for(int j=0;j<N;j++){
            if(mask&(1<<j) || (mask&parent[j])!=parent[j])continue;
            double d=dist[loc*Z+zone[j]],begin=std::max(release[j],now)+d+.5;
            if(begin+.2>H+1e-12 || 2.02+.2*d>budget+1e-12)continue;
            double score=0;
            const float* child=row(mask|(1<<j),zone[j]);
            for(int s=0;s<count;s++){
                double end=begin+samples[j*count+s];
                if(end>H+1e-10)continue;
                double virtual_end=std::clamp(H-(H-end)/pace+shift,0.,H);
                score+=reward(j,end)+interpolate(child,virtual_end);
            }
            if(score>best+1e-10){best=score;first=release[j]>now+1e-12?-2:j;}
        }
        return first;
    }
};
}
extern "C" __declspec(dllexport) void* fit_value(double H,double dt,double cw,double late,const int* fam,
const int* zone,const int* pred,const double* rel,const double* due,const double* w,const double* dist,
int count,const double* support){
    if(dt<.25||dt>5.||H<1||count<1)return nullptr;
    auto* v=new Value(H,dt,cw,late,fam,zone,pred,rel,due,w,dist);
    v->build(count,support);return v;
}
extern "C" __declspec(dllexport) int best_job(void* v,int mask,int loc,double now,double budget,int count,
const double* samples,double pace,double shift){return static_cast<Value*>(v)->first(mask,loc,now,budget,count,samples,pace,shift);}
extern "C" __declspec(dllexport) void free_value(void* v){delete static_cast<Value*>(v);}
extern "C" __declspec(dllexport) void* refit_value(double H,double dt,double cw,double late,const int* fam,
const int* zone,const int* pred,const double* rel,const double* due,const double* w,const double* dist,
int count,const double* support,int root){
    auto* v=new Value(H,dt,cw,late,fam,zone,pred,rel,due,w,dist,root);
    v->build(count,support,true);return v;
}
