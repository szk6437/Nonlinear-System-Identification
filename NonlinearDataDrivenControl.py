import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import os

#%% Hyperparameters, Matrices, Variables, ...

n = 2  # Number of States
m = 1  # Number of Inputs

w_bounds = 0.004
U_bounds = 2
init_bounds = 0.9

# Fixing the seed value
# %%
fixed_seed = 6
N_s = 100 # Number of Samples
itr = 90
Col = 8
N_list = []
M_list = []

A = np.array([1, 0.01, 0.01, 1])
B = np.array([0.001, 0.009, -0.004, 0.009])


vector = np.array([A[0],A[1],A[2],A[3],B[0],B[1],B[2],B[3]]).reshape(8,1)

#%% Input Excitation and Process Noise

def generate_vectors(n, m, N_s, seed):
    np.random.seed(seed)  # Set the random seed
    
    X_0 = np.random.uniform(-init_bounds, init_bounds, (n, 1))  # Initial Condition    
    U = np.random.uniform(-U_bounds, U_bounds, (m, N_s))  # Input Excitation
    
    return X_0, U

# Generate vectors with the fixed seed
X_0, U = generate_vectors(n, m, N_s, fixed_seed)

#%% Produce the Measurements

# Initialize the state matrix X_m
X_m = np.zeros((n, N_s+1))
X_m[:, 0] = X_0.flatten()

def Data(X_m):
    for k in range(N_s):
        w = np.random.uniform(-w_bounds, w_bounds, (X_m.shape[0], 1))
        
        X_m[0, k+1] = np.array([X_m[0,k], X_m[1,k],0,0,X_m[0,k]*U[0,k],U[0,k],0,0])@vector+w[0,0]

        X_m[1, k+1] = np.array([0,0,X_m[0,k],X_m[1,k],0,0,X_m[1,k]*U[0,k],U[0,k]])@vector+w[1,0]
    return X_m

X_m = Data(X_m)


#%% Shape the Matrices

def NV(X_m, U):
    N_list = []
    
    # Loop through each sample to build and append the matrix N_k
    for k in range(X_m.shape[1]-1):
        N_k = np.array([[X_m[0,k], X_m[1,k],0,0,X_m[0,k]*U[0,k],U[0,k],0,0],
                        [0,0,X_m[0,k],X_m[1,k],0,0,X_m[1,k]*U[0,k],U[0,k]]])
        N_list.append(N_k)
    
    # Data Collection Matrices
    N_positive = np.vstack(N_list)
    N = np.vstack([N_positive, -N_positive])

    
    V = np.ravel(np.column_stack((X_m[0, 1:], X_m[1, 1:])))
    V = np.concatenate([V + w_bounds, -V + w_bounds])
    V = V.reshape(-1, 1)
    
    return V, N

# Call NV to get V and N
V, N = NV(X_m, U)

#%% Linear Programming Optimization Using Gurobi
def LP(X_m,N,V):
    model = gp.Model("Y_Uk_LP")
    
    # Define the decision variables Y, U_1, U_2, and Gamma
    Y = model.addVars(2*n, 2*N_s*n, lb=0, vtype=GRB.CONTINUOUS, name="Y")  # Y >= 0 is enforced by lb=0
    U_1 = model.addVar(lb=-2, ub=2, vtype=GRB.CONTINUOUS, name="U_1")  # Bound on U_1
    Gamma = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Gamma")  # Gamma should be non-negative
    
    # Construct the matrix M
    M_list = []
    M_k = np.array([[X_m[0,-1], X_m[1,-1],0,0,X_m[0,-1]*U_1,U_1,0,0],
                    [0,0,X_m[0,-1],X_m[1,-1],0,0,X_m[1,-1]*U_1,U_1]])
    M_list.append(M_k)
    M_positive = np.vstack(M_list)
    M = np.vstack([M_positive, -M_positive])
    
    # Define the Y * N = M constraint
    for i in range(2 * n):  # Loop over the rows of Y and M
        for j in range(Col):  # Loop over the columns of M (M has 6 columns)
            constraint_expr = gp.quicksum(Y[i, k] * N[k, j] for k in range(2 * N_s * n))
            model.addConstr(constraint_expr == M[i, j], name=f"Y_N_eq_M_{i}_{j}")
    
    # Define the Mu vector as a function of Gamma
    Mu = np.full((2 * n, 1), Gamma)
    
    # Define the Y * V <= Mu constraint
    for i in range(2 * n):  # Loop over the rows of Y and Mu
        constraint_expr = gp.quicksum(Y[i, j] * V[j, 0] for j in range(2 * N_s * n))
        model.addConstr(constraint_expr <= Mu[i, 0], name=f"Y_V_leq_Mu_{i}")
    
    # Objective Function: Minimize Gamma
    model.setObjective(Gamma, GRB.MINIMIZE)
    #model.setObjective(U_1**2+U_2**2+Gamma, GRB.MINIMIZE)
    # Optimize the model
    model.optimize()
    
    # Extract results
    if model.status == GRB.OPTIMAL:
        Y_opt = model.getAttr('X', Y)
        U_1_opt = U_1.X
        Gamma_opt = Gamma.X
        print("Optimal U_1:", U_1_opt)
        print("Optimal Gamma:", Gamma_opt)
    
        # Substitute the optimal values into the matrix M_k
        M_k_opt = np.array([[X_m[0,-1], X_m[1,-1],0,0,X_m[0,-1]*U_1_opt,U_1_opt,0,0],
                        [0,0,X_m[0,-1],X_m[1,-1],0,0,X_m[1,-1]*U_1_opt,U_1_opt]])
    
        # Update the M_list and M
        M_list[-1] = M_k_opt  # Replace the last matrix in M_list with the optimized one
        M_positive_opt = np.vstack(M_list)
        M_opt = np.vstack([M_positive_opt, -M_positive_opt])
        
        # Validate the infinity norm constraint
        Inf = np.linalg.norm(M_positive_opt @ vector, ord=np.inf)
        
        if Inf <= Gamma_opt:
            print("Satisfied.")
            print(f"||AX_k+BU_k||: {Inf}, Gamma: {Gamma_opt}")
        else:
            print("Not Satisfied.")
            print(f"||AX_k+BU_k||: {Inf}, Gamma: {Gamma_opt}")
            
        return U_1_opt, Gamma_opt # RETURN BOTH VALUES
        
    else:
        print(f"No optimal solution found. Model status: {model.status}")
        # Return zeros or handle error if needed
        return 0.0, 0.0 

#%% Iterative Forward Solving


X_n = np.zeros((n, itr+1))
# Initialize array to store Gamma values. 
# Size is itr+1 to match X_n structure, though the last value will remain 0.
Gamma_all = np.zeros(itr+1) 

X_n[:, 0] = X_m[:, N_s].flatten()
U_op = np.zeros((m, itr))

for i in range(itr):
    # Retrieve both Control Input and Gamma
    U_1_opt, Gamma_val = LP(X_m, N, V)
    
    U_op[:, i] = [U_1_opt]
    Gamma_all[i] = Gamma_val # Store Gamma
    
    # Generate process noise w1 from a uniform distribution
    w = np.random.uniform(-w_bounds, w_bounds, (n, 1))
    
    # Compute the next state based on the given equations
    X_n[0, i+1] = np.array([X_n[0,i], X_n[1,i],0,0,X_n[0,i]*U_op[0,i],U_op[0,i],0,0])@vector+w[0,0]

    X_n[1, i+1] = np.array([0,0,X_n[0,i],X_n[1,i],0,0,X_n[1,i]*U_op[0,i],U_op[0,i]])@vector+w[1,0]

    # Update X_m: Eliminate the first column and add the new state X_n
    X_m = np.hstack((X_m[:, 1:], X_n[:, i+1].reshape(n, 1)))
    
    # Update U: Eliminate the first column and add the new control input U_op
    U = np.hstack((U[:, 1:], U_op[:, i].reshape(m, 1)))
    
    # Recompute the matrices N and V based on the updated X_m and U
    V, N = NV(X_m, U)

#%% Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(range(N_s+1), X_m[0, :], marker='o', linestyle='-', color='black')
axs[0].set_ylabel('$x_1$')
axs[0].set_title('State $x_1$')
axs[0].grid(True)

axs[1].plot(range(N_s+1), X_m[1, :], marker='o', linestyle='-', color='black')
axs[1].set_ylabel('$x_2$')
axs[1].set_title('State $x_2$')
axs[1].grid(True)

axs[2].step(range(N_s), U[0, :], where='post', color='r')
axs[2].plot(range(N_s), U[0, :], 'r.')
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('Sample $k$')
axs[2].set_title('Input $u$')
axs[2].grid(True)

plt.tight_layout()
plt.savefig('simulation_plot.png')


iterations = np.arange(1, itr + 2)  # Iterations from 1 to itr+1

# Plot States Over Iterations
plt.figure()
plt.plot(iterations, X_n[0, :], label='State 1')
plt.plot(iterations, X_n[1, :], label='State 2')
plt.xlabel('Iterations')
plt.ylabel('State Values')
plt.title('State Evolution Over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# --- SAVE TO CSV SECTION ---

# 1. Define the Desktop path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "simulation_states.csv")

# 2. Combine Data
# Stack X_n (2 rows) and Gamma_all (1 row) vertically, then transpose to get columns
# Structure: Col 1 = State 1, Col 2 = State 2, Col 3 = Gamma
data_to_save = np.vstack((X_n, Gamma_all)).T

# 3. Save to CSV with new header
np.savetxt(file_path, data_to_save, delimiter=",", header="State_1,State_2,Gamma", comments="")

print(f"File successfully saved to: {file_path}")