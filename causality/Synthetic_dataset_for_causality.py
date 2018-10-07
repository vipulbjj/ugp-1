n=10000
m_i=100
x=np.zeros(shape=(10000,100))
for i in range (n):
    k_i=random.randint(1,6)
    params=[]
    r_i = np.random.uniform(0,5)
    s_i = np.random.uniform(0,5)
    for u in range (k_i):

        #params=pd.DataFrame(columns=['mu','sigma'])
        #Constructing k_ith gaussian

        mu = np.random.normal(0, r_i)
        sigma = np.abs(np.random.normal(0, s_i))
        array=[mu,sigma]
        params.append(array)
        
    
    df=pd.DataFrame(params,columns=['mu','sigma'])
    
    weights = np.random.normal(0, 1, k_i)
    weights = np.absolute(weights)
    inv_dict={}
    for v in range(k_i):
        inv_dict[weights[v]]=v
        
    normalized_weights = weights/(np.sum(weights))
    

    
    for j in range(m_i):
        mixture_idx_weight = numpy.random.choice(weights, replace=True, p=normalized_weights)
        mixture_idx = inv_dict[mixture_idx_weight]
        x[i][j]=np.random.normal(df['mu'][mixture_idx],df['sigma'][mixture_idx])
        
    x[i] = preprocessing.scale(x[i])