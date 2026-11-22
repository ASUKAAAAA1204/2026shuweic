// Fast training replay of legal public priority policies. The decision function
// has no access to a scenario's realized durations, source identities or seed.
#include <cmath>
#include <algorithm>
#include <limits>

struct Job { double release, due, weight, x, y, typical; int parents; };

static int select_job(const Job* jobs, int n, const double* p, int done,
                      double now, const double* distances, double budget,
                      double horizon, double& wake) {
    int best=-1;
    double top=-1e100;
    wake=horizon+1;
    for(int j=0;j<n;j++) {
        const Job& a=jobs[j];
        if(done & (1<<j)) continue;
        if(a.release>now+1e-12) { wake=std::min(wake,a.release); continue; }
        if((a.parents&done)!=a.parents) continue;
        double d=distances[j];
        if(now+d+.7>horizon+1e-12 || 2.02+.2*d>budget+1e-12) continue;
        double duration=d+.5+p[n+4]*a.typical;
        double slack=a.due-now-duration;
        double late=std::max(0.,std::min(2.,-slack/8.));
        double urgency=1./(1.+std::max(0.,slack)/12.);
        double score=p[j]+p[n]*urgency-p[n+1]*late-p[n+2]*d
                   +p[n+3]*a.weight/(1.+duration/5.);
        if(score>top) {top=score;best=j;}
    }
    return best;
}

extern "C" __declspec(dllexport) void evaluate(const Job* jobs,int n,
    double horizon,double budget,const double* samples,int count,
    const double* params,int candidates,double* output) {
    double total=0;
    for(int j=0;j<n;j++)total+=jobs[j].weight;
    double dist[17][16]{};
    for(int i=0;i<=n;i++)for(int j=0;j<n;j++)
        dist[i][j]=std::hypot((i==n?0:jobs[i].x)-jobs[j].x,(i==n?0:jobs[i].y)-jobs[j].y);
    for(int m=0;m<candidates;m++) {
        double on=0,comp=0,tardy=0;
        for(int s=0;s<count;s++) {
            double now=0,left=budget;
            int location=n;
            int done=0;
            double loss=0;
            while(now<horizon-1e-12) {
                double wake;
                int j=select_job(jobs,n,params+m*(n+5),done,now,dist[location],left,horizon,wake);
                if(j<0) {if(wake>horizon)break; now=wake;continue;}
                const Job& job=jobs[j];
                double distance=dist[location][j];
                double duration=samples[s*n+j];
                double start=now+distance+.5;
                double finish=start+duration;
                left-=2.02+.2*distance;
                location=j;
                if(finish>horizon+1e-12)break;
                done|=1<<j;
                comp+=job.weight;
                if(finish<=job.due+1e-12)on+=job.weight;
                loss+=job.weight*std::max(0.,finish-job.due);
                now=std::min(horizon,start+std::max(.2,duration));
            }
            for(int j=0;j<n;j++)if(!(done&(1<<j)))
                loss+=jobs[j].weight*std::max(0.,horizon-jobs[j].due);
            tardy+=loss;
        }
        output[3*m]=on/(total*count);
        output[3*m+1]=comp/(total*count);
        output[3*m+2]=tardy/count;
    }
}
