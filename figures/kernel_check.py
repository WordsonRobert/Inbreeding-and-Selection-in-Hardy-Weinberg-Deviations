"""Accuracy of the projected Gaussian transition kernel and of the grid likelihood
(numbers quoted in Section 4.1 of the manuscript).
usage (from the repository root):  python figures/kernel_check.py   (~5 min)"""
import os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bastion import model, sims
print('kernel variance ratio (interior rows, 6 sd from boundary)')
allr=[]
for Ne in [30, 100, 300, 1000, 3000, 10000, 31623]:
    K = model.choose_K(Ne); g, pr = model.make_grid(K)
    for (F,s,h) in [(0.1,0.05,0.5),(0.0,0.2,0.0),(0.4,-0.1,1.0)]:
        A = np.zeros((K,K)); lo=np.zeros(K,np.int64); hi=np.full(K,K-1,np.int64)
        model.build_transition(g, F, s, h, 1e-5, 1e-5, float(Ne), A, lo, hi)
        r=[]; me=[]
        for j in range(K):
            m = model.gamete_mean(g[j],F,s,h,1e-5,1e-5); V = m*(1-m)/(2*Ne)
            mu = A[j]@g; var = A[j]@(g**2)-mu**2
            if 6*np.sqrt(V) < min(m,1-m): r.append(var/V); me.append(abs(mu-m))
        allr += r
    print(f'Ne={Ne:6d} K={K}: ratio median {np.median(r):.3f} range [{min(r):.3f},{max(r):.3f}] max|mean err| {max(me):.1e}')
allr=np.array(allr); print('overall', np.percentile(allr,[0,1,50,99,100]).round(3))
# likelihood accuracy vs fine grid
rng=np.random.default_rng(3); diffs=[]; ddiffs=[]
for rep in range(30):
    Ne=float(10**rng.uniform(2,4)); p0=rng.uniform(0.05,0.95); F=rng.uniform(0,0.4); s=rng.uniform(-0.2,0.2); h=rng.uniform(0,1)
    P,X,A=sims.simulate(p0,F,s,h,40,Ne=Ne,seed=int(rng.integers(1e6)))
    gens=np.arange(0,41,5); C=sims.observe(P,X,A,gens,300,rng).astype(float)
    Ka=model.choose_K(Ne); out=[]
    for K in [Ka,1500]:
        g,pr=model.make_grid(K)
        l1,_=model.hmm_forward(g,pr,F,s,h,1e-5,1e-5,np.full(40,Ne),gens,C,0,False)
        l2,_=model.hmm_forward(g,pr,F+0.05,s*0.8,h*0.7,1e-5,1e-5,np.full(40,Ne),gens,C,0,False)
        out.append((l1,l2-l1))
    diffs.append(abs(out[0][0]-out[1][0])); ddiffs.append(abs(out[0][1]-out[1][1]))
print('|loglik(K) - loglik(1500)|: median %.4f max %.4f ; |difference of loglik between two thetas|: median %.4f max %.4f'%(np.median(diffs),max(diffs),np.median(ddiffs),max(ddiffs)))
