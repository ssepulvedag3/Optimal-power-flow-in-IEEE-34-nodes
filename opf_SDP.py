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
pmax=np.array([G.nodes[k]['p'] for k in G.nodes])
smax = np.array([G.nodes[k]['s'] for k in G.nodes])

E = cv.Variable((n,n),symmetric=True)
F = cv.Variable((n,n),symmetric=True)
Z = cv.Variable((n,n))
s = cv.Variable(n,complex=True)
W = cv.Variable((n,n),complex=True)
obj = cv.Minimize(cv.trace(Ybus.real@E+Ybus.real@F))
M = Ybus@W
res = [E[0,0] == 1]
res += [cv.bmat([[E,Z],[Z.T,F]]) >> 0]
res += [cv.trace(Ybus.real@E+Ybus.real@F) == cv.sum(cv.real(s-d))]
for k in range(n):
  res += [cv.conj(s[k]-d[k]) == M[k,k]]
  res +=[cv.abs(s[k])<=smax[k]]
  res += [F[k,0] == 0]
  res += [F[0,k] == 0]
  res += [cv.bmat([[smax[k],0,cv.real(s[k])],[0,smax[k],cv.imag(s[k])],[cv.real(s[k]),cv.imag(s[k]),smax[k]]]) >> 0]
  for m in range(n):
    res+=[W[k,m]==E[k,m]+F[k,m]+1j*(Z[k,m]-Z[m,k])]

OPFSDP = cv.Problem(obj,res)
OPFSDP.solve()
X=E+1j*F
a,b=(np.linalg.eig(X.value))
  

results=pd.DataFrame()
results['name']=[G.nodes[k]['name'] for k in G.nodes]
results['vpu']=np.abs(np.sqrt(a.real[0])*b[:,0])
results['pnode']=np.round(s.value.real,4)
results['qnode']=np.round(s.value.imag,4)
results.head(n)
print(results)
