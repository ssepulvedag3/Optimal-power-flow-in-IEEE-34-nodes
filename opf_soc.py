import numpy as np
import networkx as nx
import cvxpy as cv
import pandas as pd 


feeder = pd.read_csv("FEEDER34.csv")
G=nx.DiGraph()
G.add_node(0,name='slack',s=10,p=10,d=0)


for k in range(len(feeder)):
  zkm = feeder['Rpu'][k]+1j*feeder['Xpu'][k]
  dk = feeder['Ppu'][k] + 1j*feeder['Qpu'][k]
  smax=feeder['SGmax']
  pmax=feeder['PGmax']
  G.add_node(feeder['To'][k],name=feeder['To'][k],d=dk,s=smax[k],p=pmax[k])
  G.add_edge(feeder['From'][k],feeder['To'][k],y=1/zkm,thlim=1)
  
n=G.number_of_nodes()
A=nx.incidence_matrix(G,oriented=True)
Yp=np.diag([G.edges[k]['y'] for k in G.edges])
Ybus=A@Yp@A.T

d=np.array([G.nodes[k]['d'] for k in G.nodes])
smax=np.array([G.nodes[k]['s'] for k in G.nodes])
pmax=np.array([G.nodes[k]['p'] for k in G.nodes])


h=cv.Variable(n)
s=cv.Variable(n,complex=True)
W=cv.Variable((n,n),complex=True)
M=Ybus@W
res=[h[0]==1]
for k in range(n):
    res +=[cv.conj(s[k]-d[k])==M[k,k]]
    res +=[cv.abs(s[k])<=smax[k]]
    res +=[cv.real(s[k])<=pmax[k]]
    res +=[h[k]>=0.9025]
    res +=[h[k]<=1.1025]
    res +=[W[k][k]==h[k]]
    for m in range (n):
        res +=[W[m,k]==cv.conj(W[k,m])]
        res +=[cv.SOC(h[k]+h[m],cv.vstack([2*W[k,m],h[k]-h[m]]))]
for (k,m) in G.edges:
    ylin=np.abs(G.edges[(k,m)]['y'])
    slin=G.edges[(k,m)]['thlim']
    res+=[cv.abs(h[k]-W[k,m])<=slin/ylin]
    res+=[cv.abs(h[m]-W[m,k])<=slin/ylin]

obj=cv.Minimize(cv.sum(cv.real(s-d)))
OPFSOC=cv.Problem(obj,res)
OPFSOC.solve()
print('pL',obj.value)



import pandas as pd 
results=pd.DataFrame()
results['name']=[G.nodes[k]['name'] for k in G.nodes]
results['vpu']=np.sqrt(h.value)
results['pnode']=np.round(s.value.real,4)
results['qnode']=np.round(s.value.imag,4)
results.head(n)
print(results)
