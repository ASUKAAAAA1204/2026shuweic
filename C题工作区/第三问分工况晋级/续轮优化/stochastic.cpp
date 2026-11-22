// Stochastic beam search over feasible open-loop routes; replan after each completion.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {
constexpr int max_samples=32;
struct Path {
    int mask,loc,first;
    double cost,value,rank;
    std::array<double,max_samples> clock;
};
}
extern "C" __declspec(dllexport) int stochastic_route(
    int n,int done,int ns,int width,double time,double budget,double H,
    const double* release,const double* due,const double* weight,const double* dist,
    const double* service,const int* pred,double cw,double optimism,double softness) {
    if(n<1||n>16||ns<1||ns>max_samples||width<1)return -1;
    Path initial{}; initial.mask=done; initial.loc=n; initial.first=-1;
    initial.clock.fill(time);
    std::vector<Path> live{initial}, next;
    next.reserve(width*n);
    double best=-1;
    int action=-1;
    for(int depth=0;depth<n;depth++){
        next.clear();
        for(const auto& a:live){
            for(int j=0;j<n;j++){
                int bit=1<<j;
                if((a.mask&bit)||(pred[j]&a.mask)!=pred[j])continue;
                double move=dist[a.loc*n+j],cost=a.cost+2.02+.2*move;
                if(cost>budget+1e-10)continue;
                if(depth==0 && std::max(time,release[j])+move+.7>H+1e-12)continue;
                Path b=a;
                b.loc=j;b.mask|=bit;b.cost=cost;
                if(depth==0)b.first=release[j]>time+1e-12?-2:j;
                double earned=0,mean_clock=0;
                int reached=0;
                for(int s=0;s<ns;s++){
                    double end=std::max(a.clock[s],release[j])+move+.5+service[s*n+j];
                    b.clock[s]=end;
                    mean_clock+=std::min(end,H+10.);
                    if(end>H+1e-12)continue;
                    reached++;
                    double punctual=softness<=0 ? double(end<=due[j]) :
                        std::clamp(.5+(due[j]-end)/(2*softness),0.,1.);
                    earned+=weight[j]*(punctual+cw+.0008*(std::max(0.,H-due[j])-std::max(0.,end-due[j])));
                }
                if(reached==0)continue;
                b.value+=earned/ns;mean_clock/=ns;
                if(b.value>best){best=b.value;action=b.first;}
                // Relax all outstanding prerequisites for an optimistic ranking bound.
                double bound=0;
                for(int k=0;k<n;k++)if(!(b.mask&(1<<k))){
                    double end=std::max(mean_clock,release[k])+dist[j*n+k]+.5+service[ns*n+k];
                    bound+=weight[k]*(std::clamp(.5+(due[k]-end)/24.,0.,1.)
                                         +cw*std::clamp(.5+(H-end)/26.,0.,1.));
                }
                b.rank=b.value+optimism*bound-.014*mean_clock-.003*cost;
                next.push_back(b);
            }
        }
        if(next.empty())break;
        std::stable_sort(next.begin(),next.end(),[](const auto& a,const auto& b){return a.rank>b.rank;});
        std::unordered_set<int> seen;
        live.clear();
        for(const auto& b:next){
            if(!seen.insert(b.mask*17+b.loc).second)continue;
            live.push_back(b);
            if(int(live.size())==width)break;
        }
    }
    return action;
}
