// The two value tables are allocated by this same module. Values are in task-weight units.
#include "../dynamic.cpp"
extern "C" __declspec(dllexport) int guarded_best(void* main_ptr,void* guard_ptr,
    int mask,int loc,double now,double budget,int count,const double* samples,double pace,double tolerance,double strength){
    Value& main=*static_cast<Value*>(main_ptr); Value& guard=*static_cast<Value*>(guard_ptr);
    std::array<double,N> utility{},completion{};
    std::array<bool,N> valid{};
    double maximum=-1.;
    for(int j=0;j<N;j++){
        if(mask&(1<<j)||(mask&main.parent[j])!=main.parent[j])continue;
        double d=main.dist[loc*Z+main.zone[j]],start=std::max(main.release[j],now)+d+.5;
        if(start+.2>main.H+1e-12 || 2.02+.2*d>budget+1e-12)continue;
        valid[j]=true;
        const float* a=main.row(mask|(1<<j),main.zone[j]);
        const float* b=guard.row(mask|(1<<j),guard.zone[j]);
        for(int s=0;s<count;s++){
            double end=start+samples[j*count+s];
            if(end>main.H+1e-10)continue;
            double t=std::clamp(main.H-(main.H-end)/pace,0.,main.H);
            utility[j]+=main.reward(j,end)+main.interpolate(a,t);
            completion[j]+=(guard.reward(j,end)+guard.interpolate(b,end))/guard.cw;
        }
        utility[j]/=count;completion[j]/=count;
        maximum=std::max(maximum,completion[j]);
    }
    int first=-1;double best=-1e30;
    for(int j=0;j<N;j++)if(valid[j]){
        double rank=utility[j]-strength*std::max(0.,maximum-completion[j]-tolerance);
        if(rank>best+1e-10){best=rank;first=main.release[j]>now+1e-12?-2:j;}
    }
    return first;
}
