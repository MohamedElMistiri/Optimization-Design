from scipy import optimize
import numpy as np
np.set_printoptions(precision=3)
# Written by Mohamed El Mistiri as a part of MAE 598: Design Optimization (2022 Fall) @ ASU

def objective(x):
    # Defines the objective function
    obj = (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2
    return obj

# 1st method to define the constraints
def con1(x):
    # defines the first constraint (C1)
    return x[0] + 3*x[1]

def con2(x):
    # defines the second constraint (C2)
    return x[2] + x[3] - 2*x[4]

def con3(x):
    # defies the third constraint (C3)
    return x[1] - x[4]



x_0 = [-2, -8, 4, 9.5, -3.5] # defines initial conditions

# 2nd method to define constraints
cons = ({'type':'eq','fun': lambda x: x[0] + 3*x[1]}, # C1
        {'type':'eq','fun': lambda x: x[2] + x[3] - 2*x[4]}, # C2
        {'type':'eq','fun': lambda x: x[1] - x[4]} # C3
        )

bounds = [(-10,10) for i in range(0,len(x_0))] # defines the upper and lower bounds for x 

# Solve the formulated constrained optimization problem 
results = optimize.minimize(objective,x_0, constraints=cons, bounds= bounds,options={'disp': True})

print("For x_0 = {} initial conditions".format(x_0))
print("minimal value for the objective function: {:.3f}".format(results.fun))
print("Exists at x* = {}".format(results.x))

