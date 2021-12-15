import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt

conversion = 21.2766 #ng/mL to nM

def fat_func_maker(par):
    l0 = par[0]
    l2 = par[1]
    x2 = par[2]
    cf = par[3]
    xcl = par[4]
    def lipo_func(x_try):
        return l0 + l2/(1+(x_try/x2)**2)
    def clear_func(x_try):
        y = (x_try/xcl)**2
        return cf*(1 + y/(1+y))
    return lipo_func,clear_func
##    return lipo_func,cf

def xf(parx,ins_interp,h=0.5):
    cx = parx[0]
    Ibx = parx[1]
    n = np.shape(ins_interp)[0]
    e = np.ones(n)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n,n).todense()/2/h
    F[0,0],F[0,1]=-1/h,0.5/h
    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
    F = F + cx*np.diag(e)
    x = np.dot(np.linalg.inv(F),cx*(ins_interp-Ibx)).reshape(ins_interp.shape)
    x = np.maximum(x,np.zeros(n))
    return np.array(x).reshape(ins_interp.shape)

def cminusif(parg,parcmi,ins_interp,g,h=0.5):
    gb = parg[2]
    kcmi,h0,h1 = parcmi[0],parcmi[1],parcmi[2]
    n = np.shape(ins_interp)[0]
    e = np.ones(n)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n,n).todense()/2/h
    F[0,0],F[0,1]=-1/h,0.5/h
    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
    F = F + kcmi*np.diag(e)
    x = np.dot(np.linalg.inv(F),-kcmi*(ins_interp)+h0*gb/(gb+h1*(g-gb))).reshape(ins_interp.shape)
#     x = np.maximum(x,np.zeros(n))
    return np.array(x).reshape(ins_interp.shape)

def ff(par,f_init,x,h=0.5):
    lip_fun,clear_fun = fat_func_maker(par)
    n = np.shape(x)[0]-1
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0],F[0,1]=-1/h,0.5/h
    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
    F = F + np.diag(clear_fun(x))
    f= np.maximum(np.dot(np.linalg.inv(F),(lip_fun(x)-f_init*clear_fun(x))).reshape(x.shape)+f_init,np.zeros(n+1))
##    plt.plot(np.arange(n+1),f.T)
##    plt.show()
    return np.array(f).reshape(x.shape)
    
#def gf(parg,g_init,x,rg,h=0.5):
def gf(parg,g_init,x,h=0.5):
    sg = parg[0]
    si = parg[1]
    gb = parg[2]
    n = np.shape(x)[0]-1
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0],F[0,1]=-1/h,0.5/h
    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
    F = F + np.diag((sg+si*x))
#    g = np.maximum(np.dot(np.linalg.inv(F),(sg*(gb-g_init)-si*x*g_init + rg)).reshape(x.shape)+g_init,np.zeros(n+1))
    g = np.maximum(np.dot(np.linalg.inv(F),(sg*(gb-g_init)-si*x*g_init)).reshape(x.shape)+g_init,np.zeros(n+1))
##    plt.plot(np.arange(n+1),g.T)
##    plt.show()
    return np.array(g).reshape(x.shape)

#def rgf(t_interp,parr,h=0.5):
#    tr = parr[0]
#    dr = parr[1]
#    mr = parr[2]
#    rg = np.zeros(t_interp.shape)
##    rg[1:] = np.exp(- 0.5*(np.log(t_interp[1:]/tr)/mr)**2)*dr/(mr*t_interp[1:])
##    n = np.shape(x)[0]-1
##    e = np.ones(n+1)
##    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
##    F[0,0],F[0,1]=-1/h,0.5/h
##    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
##    F = F + (sg+si*x)
##    rg = (sg*e+si*x)*g_interp + np.dot(F,g_interp).reshape(n+1)-sg*gb*e
##    rg = np.maximum(rg,np.zeros(n+1))
##    plt.plot(t_interp,rg)
##    plt.show()
##    plt.close()
#    return np.array(rg).reshape(t_interp.shape)


def interp(t_obs,f_obs,h=0.5):
    T = t_obs[-1]-t_obs[0]
    T_f = t_obs[-1]
#    print(T,t_obs[-1],t_obs[0])
    n = int(np.floor(T/h-1))
#    print("n", n)
    t = np.arange(t_obs[0]+h,T_f+h,h)
#    print(t[:4],t[-3:])
    time_indices = [np.argwhere(np.abs(t-time)<1e-8)[0,0] for time in t_obs if time>t_obs[0]]
#    print len(time_indices),len(t_obs)
    projection = np.eye(n+1)[time_indices]
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0],F[0,1]=-1/h,0.5/h
    F[1,0],F[1,1],F[1,2]=-.8/h,.6/h,.2/h
    invert = np.linalg.inv(np.dot(np.dot(projection,np.linalg.inv(np.dot(F.T,F))),projection.T))
    total = np.dot(np.dot(np.linalg.inv(np.dot(F.T,F)),projection.T),invert)
    return t,f_obs[0]+np.array(np.dot(total,f_obs[1:]-f_obs[0]).T).reshape(n+1)

def estimate_sigma(t_obs,f_obs,h=0.5,num_reps=100):
    t_interp,f_interp = interp(t_obs,f_obs,h)
    sigma_squared = 0.0
    for rep in range(num_reps):
        choice = np.random.choice(np.arange(1,t_interp.shape[0]-2),t_obs.shape[0]-2,replace=False)
        choice = np.sort(choice)
##        print(t_obs,t_interp[choice])
        t_sigma,f_sigma = interp([t_obs[0]]+list(t_interp[choice])+[t_obs[-1]],\
                                 [f_obs[0]]+list(f_interp[choice])+[f_obs[-1]])
##        plt.plot(t_sigma,f_sigma)
##        plt.plot(t_interp,f_interp)
##        plt.plot(t_obs,f_obs)
##        plt.show()
        sigma_squared += np.linalg.norm(f_sigma-f_interp)**2
    return f_interp.shape[0]*sigma_squared/float(num_reps*f_interp.shape[0]-2)

def parameters(num_data):
    t_obs = num_data[:,0]
    g_obs = num_data[:,1]
    i_obs = num_data[:,2]
    c_obs = num_data[:,3]
    f_obs = num_data[:,4]
    t_interp,ins = interp(t_obs,i_obs)    
    #cx,Ibx,l0,l2,x2,cf,xcl,SG,SI,Gb
    par0 = np.array([0.076, 3.02, 0.0026, 0.12, 12.7, 0.035, 23.32, 0.0046, 3.1e-4, 186.5]) #initial parameter values
    par = np.copy(par0)
#    cost,par,rg = optimize(par,t_obs,g_obs,f_obs,ins)
    cost,par = optimize(par,t_obs,g_obs,f_obs,ins)
    return cost,par


def parameters_cmi(num_data,par):
    t_obs = num_data[:,0]
    g_obs = num_data[:,1]
    i_obs = num_data[:,2]
    c_obs = num_data[:,3]*conversion
    f_obs = num_data[:,4]
    obese = par[0]
    age = par[1]
    #cx,Ibx,l0,l2,x2,cf,xcl,SG,SI,Gb
    if obese >= 30:
        a = -np.log(2)/4.55
        fraction = 0.78 #short half life
    else:
        a = -np.log(2)/4.95
        fraction = 0.76
    b = -np.log(2)/(0.14*age + 29.2) #long half life
    k2 = -(fraction*b + (1-fraction)*a)
    k3 = a*b/k2
    k1 = -a-b-k2-k3
    h1=0.3
    h0 = (k1+k3)*c_obs[0]
#     par0 = np.array([0.076, 3.02, 0.0026, 0.12, 12.7, 0.035, 23.32, 0.0046, 3.1e-4, 186.5, 106., 66.,.78,k1+k3,h0,h1]) #initial parameter values
    par0 = np.array([0.076, 3.02, 0.0026, 0.12, 12.7, 0.035, 23.32, 0.0046, 3.1e-4, 186.5, k1+k3,h0,h1]) #initial parameter values
    par = np.copy(par0)
#     cost,par,rg = HIE(par,t_obs,g_obs,f_obs,i_obs,c_obs)
    cost,par = HIE(par,t_obs,g_obs,f_obs,i_obs,c_obs)
    return cost,par

def HIE_only(par_input,t_obs,g_obs,f_obs,ins_obs,c_obs,h=0.5,eps=1e-3,iterations=2.5e5,beta=1e2):
    T = t_obs[-1]-t_obs[0]
    n = int(np.floor(T/h-1))
    t = np.arange(t_obs[0]+h,T+h,h)
    time_indices = [np.argwhere(np.abs(t-time)<1e-8)[0,0] for time in t_obs if time>t_obs[0]]
    projection = np.eye(n+1)[time_indices]
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0]=1/h
#    sigma_g = estimate_sigma(t_obs,g_obs)
#    sigma_f = estimate_sigma(t_obs,f_obs)
    sigma_cmi = estimate_sigma(t_obs,c_obs-ins_obs)
    par_old = np.copy(par_input)
    par = np.copy(par_input)
    #par_old[4]=1e4 #just to get into the loop
#    parx = par[:2]
#    parf = par[2:7]
#    parg = par[7:10]
#    parr = par[10:13]
#    parcmi = par[13:16] #parameters for c minus insulin (kcmi,h0,h1)
    parcmi = par[0:3] #parameters for c minus insulin (kcmi,h0,h1)
    num_par=16
    t_interp,ins = interp(t_obs,ins_obs)    
    t_interp2,gluc = interp(t_obs,g_obs) #added in v2   
#    x = xf(parx,ins)
#    g = gf(parg,g_obs[0],x,rgf(t,parr))
    cmi = cminusif(parg,parcmi,ins,gluc) #changed from g to gluc in v2 (interpolated rather than solved)
#    f = ff(parf,f_obs[0],x)
#    cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
#    cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
    cost = 0
    cost += np.linalg.norm(c_obs[1:]-ins_obs[1:]-np.dot(projection,cmi))**2/sigma_cmi
    old_cost,init_cost = cost,cost
    iteration_u = 0
    while iteration_u<iterations:
        par = np.abs(np.copy(par_old)+eps*par_input*(np.random.random(num_par)-0.5))
##        par = np.abs(np.copy(par_old)*(1+eps*(np.random.random(13)-0.5)))
        parx = par[:2]
        parf = par[2:7]
        parg = par[7:10]
        parr = par[10:13]
        parcmi = par[13:16]
        x = xf(parx,ins)
#         g = gf(parg,g_obs[0],x,rgf(t,parr))
        g = gf(parg,g_obs[0],x)
        cmi = cminusif(parg,parcmi,ins,g)
        f = ff(parf,f_obs[0],x)
        cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
        cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
        cost += np.linalg.norm(c_obs[1:]-ins_obs[1:]-np.dot(projection,cmi))**2/sigma_cmi
        if np.exp(beta*2.*(old_cost-cost)/(old_cost+cost)) > np.random.rand():
            par_old = np.copy(par)
            old_cost = cost
            iteration_u += 1
#    print(init_cost,cost)
#     return cost,par_old,rgf(t,par_old[10:13])

def HIE(par_input,t_obs,g_obs,f_obs,ins_obs,c_obs,h=0.5,eps=1e-3,iterations=2.5e5,beta=1e2):
    T = t_obs[-1]-t_obs[0]
    T_f = t_obs[-1]
    n = int(np.floor(T/h-1))
    t = np.arange(t_obs[0]+h,T_f+h,h)
    time_indices = [np.argwhere(np.abs(t-time)<1e-8)[0,0] for time in t_obs if time>t_obs[0]]
    projection = np.eye(n+1)[time_indices]
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0]=1/h
    sigma_g = estimate_sigma(t_obs,g_obs)
    sigma_f = estimate_sigma(t_obs,f_obs)
    sigma_cmi = estimate_sigma(t_obs,c_obs-ins_obs)
    par_old = np.copy(par_input)
    par = np.copy(par_input)
    #par_old[4]=1e4 #just to get into the loop
    parx = par[:2]
    parg = par[7:10]
    parcmi = par[10:13]
    parf = par[2:7]
#     parr = par[10:13]
#     parcmi = par[13:16] #parameters for c minus insulin (kcmi,h0,h1)
#     num_par=16
    num_par=13
    t_interp,ins = interp(t_obs,ins_obs)    
    t_interp2,gluc = interp(t_obs,g_obs) #added in v2   
    x = xf(parx,ins)
#     g = gf(parg,g_obs[0],x,rgf(t,parr))
    g = gf(parg,g_obs[0],x)
    cmi = cminusif(parg,parcmi,ins,gluc) #changed from g to gluc (interpolated rather than solved)
    f = ff(parf,f_obs[0],x)
    cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
    cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
    cost += np.linalg.norm(c_obs[1:]-ins_obs[1:]-np.dot(projection,cmi))**2/sigma_cmi
    old_cost,init_cost = cost,cost
    iteration_u = 0
    while iteration_u<iterations:
        par = np.abs(np.copy(par_old)+eps*par_input*(np.random.random(num_par)-0.5))
##        par = np.abs(np.copy(par_old)*(1+eps*(np.random.random(13)-0.5)))
        parx = par[:2]
        parg = par[7:10]
        parcmi = par[10:13]
        parf = par[2:7]
##        parr = par[10:13]
##        parcmi = par[13:16]
        x = xf(parx,ins)
##        g = gf(parg,g_obs[0],x,rgf(t,parr))
#         cmi = cminusif(parg,parcmi,ins,g)
        g = gf(parg,g_obs[0],x)
        cmi = cminusif(parg,parcmi,ins,gluc)
        f = ff(parf,f_obs[0],x)
        cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
        cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
        cost += np.linalg.norm(c_obs[1:]-ins_obs[1:]-np.dot(projection,cmi))**2/sigma_cmi
        if np.exp(beta*2.*(old_cost-cost)/(old_cost+cost)) > np.random.rand():
            par_old = np.copy(par)
            old_cost = cost
            iteration_u += 1
#    print(init_cost,cost)
#     return cost,par_old,rgf(t,par_old[10:13])
    return cost,par_old

def optimize(par_input,t_obs,g_obs,f_obs,ins,h=0.5,eps=1e-3,iterations=2.5e5,beta=3e2):
    T = t_obs[-1]-t_obs[0]
    T_f = t_obs[-1]
    n = int(np.floor(T/h-1))
    t = np.arange(t_obs[0]+h,T_f+h,h)
    time_indices = [np.argwhere(np.abs(t-time)<1e-8)[0,0] for time in t_obs if time>t_obs[0]]
    projection = np.eye(n+1)[time_indices]
    e = np.ones(n+1)
    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
    F[0,0]=1/h
    sigma_g = estimate_sigma(t_obs,g_obs)
    sigma_f = estimate_sigma(t_obs,f_obs)
    par_old = np.copy(par_input)
    par = np.copy(par_input)
    #par_old[4]=1e4 #just to get into the loop
    parx = par[:2]
    parf = par[2:7]
    parg = par[7:10]
#    parr = par[10:13]
    x = xf(parx,ins)
    f = ff(parf,f_obs[0],x)
#    g = gf(parg,g_obs[0],x,rgf(t,parr))
    g = gf(parg,g_obs[0],x)
    cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
    cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
    old_cost,init_cost = cost,cost
    iteration_u = 0
    while iteration_u<iterations:
#        par = np.abs(np.copy(par_old)+eps*par_input*(np.random.random(13)-0.5))
        par = np.abs(np.copy(par_old)+eps*par_input*(np.random.random(10)-0.5))
##        par = np.abs(np.copy(par_old)*(1+eps*(np.random.random(13)-0.5)))
        parx = par[:2]
        parf = par[2:7]
        parg = par[7:10]
#        parr = par[10:13]
        x = xf(parx,ins)
        f = ff(parf,f_obs[0],x)
#        g = gf(parg,g_obs[0],x,rgf(t,parr))
        g = gf(parg,g_obs[0],x)
        cost = np.linalg.norm(g_obs[1:]-np.dot(projection,g))**2/sigma_g
        cost += np.linalg.norm(f_obs[1:]-np.dot(projection,f))**2/sigma_f
        if np.exp(beta*2.*(old_cost-cost)/(old_cost+cost)) > np.random.rand():
            par_old = np.copy(par)
            old_cost = cost
            iteration_u += 1
#    print(init_cost,cost)
#    return cost,par_old,rgf(t,par_old[10:13])
    return cost,par_old

def secretion(subj,h=1):
    """
    h is the time step
    subj is the subject identifying string
    """

    p = subj[0] #subject_parameters[subj]
    data = subj[1] #num_data[subj]

    T = data[-1,0] # time length of data
    ts = data[:,0]
    n = int(np.floor(T/h-1))
    t = np.arange(h,T+h,h)
    times_indices = [np.argwhere(t==time)[0,0] for time in ts if time > 0]
    #indices corresponding to observations

    age = float(p['age_enrollment'])
    obese = float(p['obese'])
    a = -np.log(2)/(4.95 - 0.4*obese) #short half life
    b = -np.log(2)/(0.14*age + 29.2) #long half life
    fraction = 0.76 + 0.02*obese
    k2 = -(fraction*b + (1-fraction)*a)
    k3 = a*b/k2
    k1 = -a-b-k2-k3
    print(k1,k2,k3)

    A = np.array([[-(k1+k3),k2],[k1,-k2]])
    d,v = np.linalg.eig(A)
    sort_index = np.argsort(d)    
    v = v[:,sort_index] #columns are eigenvectors
    lam = d[sort_index] #ascending order eigenvalues
    denom = 1.0/np.linalg.det(v)

    c1 = v[0,0]*v[1,1]*denom
    c2 = -v[0,1]*v[1,0]*denom

    e = np.ones(n+1)
    E = sp.sparse.spdiags([e*h/2,e*h/2],[-1,0],n+1,n+1).todense()
    Cum = np.tril(np.ones((n+1,n+1)))

    e1m =np.diag(np.exp(-lam[0]*t))
    e1p =np.diag(np.exp(lam[0]*t))
    e2m =np.diag(np.exp(-lam[1]*t))
    e2p =np.diag(np.exp(lam[1]*t))

    M = np.eye(n+1)[times_indices]

    K = np.dot(M,np.dot(np.dot(c1*np.dot(e1p,Cum),E),e1m)+\
               np.dot(np.dot(c2*np.dot(e2p,Cum),E),e2m))

    F = sp.sparse.spdiags([1*e,-4*e,3*e],[-2,-1,0],n+1,n+1).todense()/2/h
#    print(F)
    F[0,0]=1/h

    f = np.dot(np.linalg.inv(np.dot(F.T,F)),K.T)
    f1 = np.linalg.inv(np.dot(K,f))
    S = np.dot(f1,data[1:,1]-data[0,1])
    S = np.dot(f,S.T)
    Stemp = np.ones(S.shape[0]+1)
    Stemp[0] = Stemp[0]*k3*data[0,1]
    Stemp[1:] = Stemp[0]+S.reshape(S.shape[0])
#    plt.plot(t,Stemp[1:],'k')
    Stemp,vartemp = montecarlo(Stemp,data[:,1],K,F)
#    print(np.dot(K,Stemp[1:]-Stemp[0]).shape)
#    del_Stemp = np.abs(np.dot(f,np.dot(f1,data[1:,1]-data[0,1]).T-np.dot(K,Stemp[1:]-Stemp[0]).T))
#    print(del_Stemp.shape)
##    plt.plot(t,Stemp[1:],'r')
##    plt.plot(ts[1:],(data[1:,1]-data[0,1]),'b')
##    y = np.copy(np.dot(K,Stemp[1:]-Stemp[0])).reshape(ts.shape[0]-1)
###    print(y.shape)
##    plt.plot(ts[1:],y,'g')
#    plt.show()
    return t,Stemp,vartemp

def montecarlo(s_orig,c_exp,k,f,delta=1e-3,iterations=int(5e6),power=8,beta=.05,mcupdate=1e-2):
    s = np.copy(s_orig)
    cpep_sigma = 0.1 #np.std(c_exp)
    s_max = np.max(s_orig)*cpep_sigma/np.max(c_exp)
    c_exp_temp = (c_exp[1:]-c_exp[0])
    d8 = delta**power
    cum_weight=0.0
    cum_s = np.zeros(s.shape[0]-1)
    #calibrate costs
    s_temp = s[1:] + s_max*mcupdate*np.random.normal(size=s.shape[0]-1)
    zeros = s_temp<delta
    s_temp[zeros] = 2*delta
    smooth_cost0 = np.linalg.norm(np.dot(f,s_temp-s[0])/s_max)**2
    cpep_cost0 = np.linalg.norm(np.dot(k,s_temp-s[0])-c_exp_temp)**2/cpep_sigma**2
    s8 = np.power(s_temp,power)
    zero_cost0 = -np.log(np.prod(s8/(s8+d8)))
    accept = False
    s_prev,cum_s,cum_s2 = np.copy(s_temp),np.copy(s_temp),np.power(np.copy(s_temp),2)
    prev_cost = 0.5*(smooth_cost0/2.0+2*cpep_cost0)+zero_cost0
    acceptance = 1.0
    # first equilibration
    for step in range(iterations):
        if accept:
            accept = False
            s_prev = np.copy(s_temp)
            prev_cost = temp_cost
##            acceptance += 1.0
##            cum_s= cum_s+ s_temp
##            cum_s2 = cum_s2 + np.power((s_temp),2)
        s_temp = np.copy(s_prev) + s_max*mcupdate*np.random.normal(size=s.shape[0]-1)
        zeros = s_temp<delta
        s_temp[zeros] = 2*delta
        smooth_cost = np.linalg.norm(np.dot(f,s_temp)/s_max)**2
        cpep_cost = np.linalg.norm(np.dot(k,s_temp-s[0])-(c_exp_temp))**2/cpep_sigma**2
        s8 = np.power(s_temp,power)
        zero_cost = -np.sum(np.log(s8/(s8+d8)))
        temp_cost = 0.5*(smooth_cost/2.0+2*cpep_cost)+zero_cost
        if np.exp(beta*(prev_cost-temp_cost)) > np.random.rand():
            accept = True
    # then recording
    for step in range(10*iterations):
        if accept:
            accept = False
            s_prev = np.copy(s_temp)
            prev_cost = temp_cost
            acceptance += 1.0
            cum_s= cum_s+ s_temp
            cum_s2 = cum_s2 + np.power((s_temp),2)
        s_temp = np.copy(s_prev) + s_max*mcupdate*np.random.normal(size=s.shape[0]-1)
        zeros = s_temp<delta
        s_temp[zeros] = 2*delta
        smooth_cost = np.linalg.norm(np.dot(f,s_temp)/s_max)**2
        cpep_cost = np.linalg.norm(np.dot(k,s_temp-s[0])-(c_exp_temp))**2/cpep_sigma**2
        s8 = np.power(s_temp,power)
        zero_cost = -np.sum(np.log(s8/(s8+d8)))
        temp_cost = 0.5*(smooth_cost/2.0+2*cpep_cost)+zero_cost
        if np.exp(beta*(prev_cost-temp_cost)) > np.random.rand():
            accept = True
#        print(accept)
    Stemp = np.ones(s.shape[0])
    Stemp[0] = s[0]
    Stemp[1:] = Stemp[0]+cum_s.reshape(s.shape[0]-1)/acceptance
    return Stemp,np.sqrt(np.abs(-np.power(cum_s/acceptance,2)+cum_s2/acceptance))
    
def pearson(v):
    return (v-np.mean(v))/np.std(v)
