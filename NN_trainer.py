
import numpy as np
from scipy.optimize import curve_fit

from analyse_DCT import *


import tensorflow as tf

from fbm import fbm


def plateau(x,t1,t2):
    t1,t2 = np.sort([t1,t2])
    return np.heaviside(x-t1, 1)-np.heaviside(x-t2, 0)


def std_bridge(t, T, H):
    return (t**(2*H) + T**(2*H) - np.abs(t-T)**(2*H)) / (2*T**(2*H)) 

def normalised_bridge(t, t1,t2, H):
    T = t2 - t1

    return plateau(t,t1,t2) * std_bridge(plateau(t,t1,t2)*(t-t1), T, H) +  np.heaviside(t-t2, 0)

def walk_gen(N):
    walk = np.zeros((N,3))
    walk[1:,:] = np.random.normal(size = (N-1,3))
    return np.cumsum( walk, axis = 0 )
    
def bridge_gen( N, l, L ):
    walk = walk_gen(N)
    Rg = calc_Rg(polymer)

    V = walk[L+l] - walk[l]
    r0 = np.mean( polymer, axis = -2)[:,None,:]

    polymerrescaled = (polymer - r0)/Rg[:,None, None]

    bridge = normalised_bridge(np.arange(0,N), l,l+L, 1/2)
    return polymerrescaled.reshape(Ns,3*N )



def calc_Rg(polymer):
    _,N,_ = np.shape(polymer)

    r0 = np.mean(polymer, axis=-2)[:,None,:]

    polymerrescaled = polymer - r0
    
    return np.sqrt( 1/N * np.sum( np.sum(np.square(polymerrescaled),axis=-1), axis=1 ) )

def prepare(polymer):
    Ns, N, _ =np.shape(polymer)

    Rg = calc_Rg(polymer)

    r0 = np.mean( polymer, axis = -2)[:,None,:]

    polymerrescaled = (polymer - r0)/Rg[:,None, None]

    return polymerrescaled.reshape(Ns,3*N )



def fbm_gen(N, H):
    return np.array( [ fbm(n=N-1, hurst= H, length=N),fbm(n=N-1, hurst= H, length=N) ,fbm(n=N-1, hurst= H, length=N)] ).T

def bridge_gen( N, l, L, H ):
    walk = fbm_gen(N, H)

    V = walk[L+l-1] - walk[l]


    bridge = normalised_bridge(np.arange(0,N), l,l+L, H)

    return walk - np.array([ bridge* V[0], bridge* V[1], bridge* V[2] ]).T


def make_data(Ns, N, std_loopsize):

    startpos = np.random.randint(0, std_loopsize, Ns)
    endpos = np.random.randint(N-std_loopsize-1,N-1, Ns)
    loopsizes = endpos - startpos

    X1 = np.array( [ bridge_gen(N, startpos[j], loopsizes[j], 1/2  ) for j in range(Ns) ] )
    X2 = np.array( [ walk_gen(N) for j in range(Ns) ] )
    X = np.concatenate((X1,X2), axis = 0)   
    return X 


def make_CoM_data(Ns, N, std_loopsize, resolution):
    
    mu0 = 1.34767

    poss_start_loop = int(np.round(( N - N/mu0)/2))

    startpos = np.random.randint(resolution*poss_start_loop, resolution*poss_start_loop+resolution*std_loopsize, Ns)
    endpos = np.random.randint((resolution*N-resolution*std_loopsize-1-resolution*poss_start_loop),(resolution*N-1)-resolution*poss_start_loop, Ns)
    loopsizes = endpos - startpos

    highres1 = np.array( [ bridge_gen(resolution*N, startpos[j], loopsizes[j], 1/2  ) for j in range(Ns) ] )
    highres2 = np.array( [ walk_gen(resolution*N) for j in range(Ns) ] )
    
    CoM1 = np.mean( np.reshape(highres1, (Ns, -1, resolution, 3) ), axis = 2 ) # CoM of consecutive subchains of length resolution
    CoM2 = np.mean( np.reshape(highres2, (Ns, -1, resolution, 3) ), axis = 2 ) # CoM of consecutive subchains of length resolution


    CoM = np.concatenate((CoM1,CoM2), axis = 0)   


    return CoM


from tensorflow.keras import *

def NNmaker(N, std_loopsize, resolution,  learn_samples_per_category, validation_samples_per_category, test_samples):
    model = Sequential()
    model.add(layers.Dense(N, input_dim=3*N, activation='relu'))
    #model.add(Dense(n//2, activation='relu'))

    #model.add(Dense(40, activation='relu'))
    
    model.add(layers.Dense(30, activation='relu'))

    #model.add(Dense(10, activation='relu'))

    #model.add(Dense(8, activation='relu'))

    model.add(layers.Dense(5, activation='relu'))

    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))



    callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'], )


    Xlearn = prepare(make_CoM_data(learn_samples_per_category, N, std_loopsize, resolution))
    Ylearn = np.concatenate(([1]*learn_samples_per_category,[0]*learn_samples_per_category), axis = 0)


    Xvalid = prepare(make_CoM_data(validation_samples_per_category, N, std_loopsize, resolution))
    Yvalid = np.concatenate(([1]*validation_samples_per_category,[0]*validation_samples_per_category), axis = 0)


    
    # Train the model
    print("train model")
    history = model.fit(Xlearn, Ylearn, epochs=200, batch_size=20, verbose=0, validation_data=(Xvalid, Yvalid), callbacks=[callback])



    print("Testing model")
    X =  prepare(make_CoM_data(test_samples, N, std_loopsize, resolution))


    results = model.predict( X, verbose=0 )


    return model, history, results


    




