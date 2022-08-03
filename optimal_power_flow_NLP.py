import pandas as pd
import pyomo.environ as pe
import pyomo.opt as po
import networkx as nx
import numpy as np

file = 'C:/Users/Usuario/Desktop/Important/34nodes.xlsx'

feeder = pd.read_excel(file)
G = nx.DiGraph()
G.add_node(0,name='slack',d=0,s=10,p=0,c=10)
print(feeder)

for k in range(len(feeder)):
  zkm = feeder['R'][k]+1j*feeder['X '][k]
  dk = feeder['P'][k] + 1j*feeder['Q'][k]
  smax=feeder['SG']
  G.add_node(feeder['To'][k],name=feeder['To'][k],d=dk,s=smax[k],c=5)
  G.add_edge(feeder['From '][k],feeder['To'][k],y=1/zkm,thlim=1)
  

nx.draw(G,with_labels=True,pos=nx.spectral_layout(G))

A=nx.incidence_matrix(G,oriented=True)
Yp=np.diag([G.edges[k]['y'] for k in G.edges])
Ybus=A@Yp@A.T
Gbus=Ybus.real
Xbus=Ybus.imag

n=G.number_of_nodes()
d=np.array([G.nodes[k]['d'] for k in G.nodes])
smax=np.array([G.nodes[k]['s'] for k in G.nodes])
c=np.array([G.nodes[k]['c'] for k in G.nodes])
d_real=d.real
d_imag=d.imag

#-----------------  Dictionaries -------------------------------

d_real_dict = dict(enumerate(d_real))
d_imag_dict = dict(enumerate(d_imag))
smax_dict = dict(enumerate(smax))
c_dict = dict(enumerate(c))

Gbus_dict={}
Xbus_dict={}
for k in range(Ybus.shape[0]):
    for m in range(Ybus.shape[0]):
        Gbus_dict.update({(k,m): Gbus[k,m]})
        Xbus_dict.update({(k,m): Xbus[k,m]}) 
        
 
#---------------- Optimization model ---------------

nodes=set(range(0,n))
model = pe.ConcreteModel()

#---------------- set ----------------------------
model.nodes = pe.Set(initialize=nodes)

#--------------- parameters ------------------------
model.gkm = pe.Param(model.nodes, model.nodes, initialize=Gbus_dict, default=1000) 
model.xkm = pe.Param(model.nodes, model.nodes, initialize=Xbus_dict, default=1000) 
model.pd =pe.Param(model.nodes, initialize=d_real_dict, default=1000) 
model.qd =pe.Param(model.nodes, initialize=d_imag_dict, default=1000) 
model.smax=pe.Param(model.nodes, initialize=smax_dict, default=1000) 
model.c=pe.Param(model.nodes, initialize=c_dict, default=1000) 

#----------------- Variables -------------------------
model.u = pe.Var(model.nodes, domain=pe.Reals)
model.z = pe.Var(model.nodes, domain=pe.Reals)
model.pg = pe.Var(model.nodes, domain=pe.Reals)
model.qg = pe.Var(model.nodes, domain=pe.Reals)

#--------------- Objective function ----------

#expr = sum(model.c[k] * model.pg[k] for k in model.nodes)     
expr = sum(sum(model.u[k]*model.u[m]*model.gkm[k,m]-model.u[m]*model.z[k]*model.xkm[k,m]+model.z[m]*model.u[k]*model.xkm[k,m]+model.z[k]*model.z[m]*model.gkm[k,m] for m in model.nodes) for k in model.nodes)     

model.objective = pe.Objective(sense=pe.minimize, expr=expr)

#-------------- Constraints -------------

model.pmax = pe.ConstraintList()
model.pmin = pe.ConstraintList()
model.qmin = pe.ConstraintList()
for k in model.nodes:
    lhs = (model.pg[k]**2+model.qg[k]**2)**0.5
    rhs = model.smax[k]
    model.pmax.add(lhs <= rhs)
    model.pmin.add(model.pg[k] >= 0 )
    model.qmin.add(model.qg[k] >= 0 )
     
model.tensions = pe.ConstraintList()
for k in model.nodes:
    model.tensions.add(model.u[k] <= 1.05)   
    model.tensions.add(model.u[k] >= 0.95)  
    model.tensions.add(model.z[k] <= 0.4)   
    model.tensions.add(model.z[k] >= -0.4)  

model.balance_real = pe.ConstraintList()
model.balance_imag = pe.ConstraintList()
for k in model.nodes:
    lhs_r = sum(model.u[k]*model.u[m]*model.gkm[k,m]-model.u[k]*model.z[m]*model.xkm[k,m]+model.z[k]*model.u[m]*model.xkm[k,m]+model.z[k]*model.z[m]*model.gkm[k,m] for m in model.nodes)
    rhs_r = model.pg[k]-model.pd[k]
    lhs_i = sum(-model.z[k]*model.u[m]*model.gkm[k,m]+model.z[k]*model.z[m]*model.xkm[k,m]+model.u[k]*model.u[m]*model.xkm[k,m]+model.u[k]*model.z[m]*model.gkm[k,m] for m in model.nodes)
    rhs_i = model.qg[k]-model.qd[k]
    model.balance_real.add(lhs_r == rhs_r)
    model.balance_imag.add(lhs_i == -rhs_i)



model.con_u = pe.Constraint(expr=model.u[0]==1)
model.con_z = pe.Constraint(expr=model.z[0]==0)

opt = po.SolverFactory('gams')
results = opt.solve(model, solver='MINOS')

#solver = po.SolverFactory('ipopt')
#results = solver.solve(model, tee=True)

for n in model.nodes:
    print(n, pe.value(model.pg[n]))


print(pe.value(model.objective))

