#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
using Order=std::array<int,16>;
struct Problem {
    int n, known, scenarios;
    double time,budget,horizon,completion_weight;
    const double *release,*due,*weight,*dist,*duration;
    const int *parent;
    int next(const Order& o,int mask,double now,int loc,double left,double& wake) const {
        wake=horizon+1;
        for(int i=0;i<n;i++){
            int j=o[i];
            if(mask&(1<<j))continue;
            if(release[j]>now+1e-12){wake=std::min(wake,release[j]);continue;}
            if((mask&parent[j])!=parent[j])continue;
            double d=dist[loc*n+j];
            if(now+d+.7<=horizon+1e-12 && left+1e-12>=2.02+.2*d)return j;
        }
        return -1;
    }
    double score(const Order& o) const {
        double value=0;
        for(int s=0;s<scenarios;s++){
            int mask=known,loc=n;
            double now=time,left=budget;
            while(now<horizon-1e-12){
                double wake;
                int j=next(o,mask,now,loc,left,wake);
                if(j<0){if(wake>horizon)break;now=wake;continue;}
                double d=dist[loc*n+j],start=now+d+.5;
                double service=duration[s*n+j],end=start+service;
                if(end>horizon+1e-12)break;
                value+=weight[j]*((end<=due[j]+1e-12)+completion_weight
                    +.00065*(std::max(0.,horizon-due[j])-std::max(0.,end-due[j])));
                now=start+std::max(.2,service);
                left-=2.02+.2*d;loc=j;mask|=1<<j;
            }
        }
        return value;
    }
};
}

extern "C" __declspec(dllexport) int rolling(int n,int done,int ns,double now,double left,double horizon,
    double cw,const double* release,const double* due,const double* weight,const double* distance,
    const double* samples,const int* parent,const double* typical,int steps,uint32_t seed){
    Problem p{n,done,ns,now,left,horizon,cw,release,due,weight,distance,samples,parent};
    auto draw=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};
    Order best{};
    for(int i=0;i<n;i++)best[i]=i;
    double best_value=-1e100;
    int active=0;
    for(int i=0;i<n;i++)active+=!(done&(1<<i));
    if(!active)return -1;
    for(int restart=0;restart<4;restart++){
        Order cur=best;
        if(restart<3){
            std::array<double,16> priority{};
            for(int i=0;i<n;i++){
                double slack=std::max(0.,due[i]-now-typical[i]);
                priority[i]=restart==0 ? -due[i]+.8*weight[i] :
                    (restart==1 ? weight[i]/(3+slack) : weight[i]/(1+typical[i])+.08*(draw()%10));
            }
            std::stable_sort(cur.begin(),cur.begin()+n,[&](int a,int b){return priority[a]>priority[b];});
        }
        std::stable_partition(cur.begin(),cur.begin()+n,[&](int j){return !(done&(1<<j));});
        double value=p.score(cur);
        if(value>best_value){best=cur;best_value=value;}
        for(int k=0;k<steps && active>1;k++){
            Order trial=cur;
            int i=draw()%active,j=draw()%active;
            if(i==j)continue;
            if(k%3==0)std::swap(trial[i],trial[j]);
            else if(i<j)std::rotate(trial.begin()+i,trial.begin()+i+1,trial.begin()+j+1);
            else std::rotate(trial.begin()+j,trial.begin()+i,trial.begin()+i+1);
            double v=p.score(trial);
            if(v>=value-1e-10){cur=trial;value=v;}
            if(v>best_value+1e-10){best=trial;best_value=v;}
        }
    }
    double wake;
    return p.next(best,done,now,n,left,wake);
}
