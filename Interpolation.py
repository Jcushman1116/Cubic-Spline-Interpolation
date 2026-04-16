import numpy as np
import matplotlib.pyplot as plt

def Chebyshev_Points(n, interval=(-1, 1)):
    a, b = interval

    # Generate Chebyshev points in [-1,1]
    mesh = np.zeros(n)
    for i in range(n):
        val = ((2 * i + 1) * np.pi) / (2 * n)
        mesh[i] = np.cos(val)

    # Map to [a, b]
    mesh = 0.5 * (a + b) + 0.5 * (b - a) * mesh

    # Leja ordering
    ordered_mesh = np.zeros(n)

    # Start with point of largest magnitude
    index = np.argmax(np.abs(mesh))
    ordered_mesh[0] = mesh[index]

    # remove largest point
    remaining = list(mesh)
    remaining.pop(index)

    # Select next point in the sequence
    for i in range(1, n):
        best_index = 0
        best_val = -1

        #find the candidate that maximizes the product of distances
        for j, candidate in enumerate(remaining):
            prod = np.prod([abs(candidate - ordered_mesh[k]) for k in range(i)])
            if prod > best_val:
                best_val = prod
                best_index = j

        # move the best candidate to the ordered array 
        ordered_mesh[i] = remaining[best_index]
        remaining.pop(best_index)

    return ordered_mesh 

def Barycentric1_Coefficients(mesh):
    n = len(mesh)
    
    # storage for weights
    gamma = np.ones(n)

    # Compute the weights
    for i in range(n):
        # skip the term where i = j to avoid 0 mult
        for j in range(n): 
            if i != j: 
                # compute product
                gamma[i] *= (mesh[i] - mesh[j])

    # Take the reciporical     
    gamma = 1 / gamma
    
    return gamma

def Barycentric1_Interpolation(eval_pts, mesh, gamma, fx): 
    n = len(eval_pts)

    # storage for polynomial 
    px = np.zeros(n)

    # polynomial construction
    for i in range(n): 
        x_curr = eval_pts[i]
        diff = x_curr - mesh 

        # check for spots that could be 0
        if np.any(diff == 0):
            # find the first index where diff = 0 
            j = np.where(diff == 0)[0][0]
            px[i] = fx[j]
            continue 
            
        # compute product
        lx = np.prod(diff)
        # compute sum from formula
        total = np.sum((gamma * fx) / diff)
        px[i] = lx * total
        
    return px 

# partition global mesh into n subintervals
def Mesh_Partition(a, b, n): 
    return np.linspace(a, b, n+1)

# Define node distribution within subinterval 
def local_nodes(a_i, b_i, s, method): 
    if method == "uniform": 
        # standard spacing
        return np.linspace(a_i, b_i, s+1)

    # generate cheb. nodes    
    if method == "chebyshev": 
        local_mesh = np.zeros(s+1)
        for i in range(s+1):
            # cheb. node formula
            val = (np.pi * i) / s 
            local_mesh[i] = np.cos(val)
        
        # Map to [a, b]
        mesh = 0.5 * (a_i + b_i) + 0.5 * (b_i - a_i) * local_mesh
        
        # Leja ordering
        ordered_mesh = np.zeros(s+1)

        # Start with point of largest magnitude
        index = np.argmax(np.abs(mesh))
        ordered_mesh[0] = mesh[index]

        # remove largest point
        remaining = list(mesh)
        remaining.pop(index)

        for i in range(1, s+1):
            best_index = 0
            best_val = -1

            # select next point that maximizes prod of distances
            for j, candidate in enumerate(remaining):
                prod = np.prod([abs(candidate - ordered_mesh[k]) for k in range(i)])
                if prod > best_val:
                    best_val = prod
                    best_index = j

            # move best point to the ordered result
            ordered_mesh[i] = remaining[best_index]
            remaining.pop(best_index)

        return ordered_mesh
    
    
def divided_differences(x, y): 
    n = len(x)
    # initialize coefficients with function values
    coef = y.copy()

    # iterate through order of differences
    for i in range(1,n): 
        # iterate backwards to update 
        for j in range(n-1, i-1, -1): 
            # divided diff. formula
            coef[j] = (coef[j] - coef[j-1]) / (x[j] - x[j-i])

    return coef


def Newton_Polynomial(local_ordered_nodes, coef, x): 
    n = len(coef)
    # start nested eval. with highest order first
    px = coef[-1]

    #iterate backwards from second to last coeff.
    for i in range(n-2,-1,-1): 
        #update with running total
        px = px * (x - local_ordered_nodes[i]) + coef[i]
        
    return px 

def piecewise_interpolation(f, a, b, m, s, method, eval_pts): 
    
    # create boundaries for m subintervals
    partition = np.linspace(a, b, m+1)
    pieces = []

    # Precompute model for each subinterval 
    for i in range(m): 
        a_i, b_i = partition[i], partition[i+1]

        # generate local nodes and sample function
        x_nodes = local_nodes(a_i, b_i, s, method)
        y_nodes = f(x_nodes)

        # compute Newton div. diff. 
        coef = divided_differences(x_nodes, y_nodes)
        pieces.append((a_i, b_i, x_nodes, coef))

    # initalize eval. results
    result = np.zeros(len(eval_pts))

    # for all eval points, find the subinterval 
    for k, x in enumerate(eval_pts):
        for (a_i, b_i, x_nodes, coef) in pieces: 
            #check if x is within subinterval 
            if a_i <= x <= b_i: 
                # evaluate piece 
                result[k] = Newton_Polynomial(x_nodes, coef, x)
                # move to next point
                break 
    return result


def Hermite_Divided_Differences(x, f, df):
    #Initialize storage
    X = np.zeros(4) 
    DDT = np.zeros((4,4))
    
    # Set repeated roots X = [a_i, a_i, b_i, b_i]
    X[0] = X[1] = x[0]
    X[2] = X[3] = x[1]
    
    # Set First Column of table as function values
    # [f(a_i), f(a_i), f(b_i), f(b_i)]
    DDT[0,0] = f(x[0])
    DDT[1,0] = f(x[0])
    DDT[2,0] = f(x[1])
    DDT[3,0] = f(x[1])
    
    # First Div. Diff.
    DDT[1,1] = df(x[0])
    DDT[3,1] = df(x[1])
    
    # Normal Div. Diff. 
    DDT[2,1] = (DDT[2,0] - DDT[1,0]) / (X[2] - X[1])
    
    # Higher order Div. Diff. 
    for j in range(2, 4): 
        for i in range(j, 4): 
            DDT[i, j] = (DDT[i,j-1] - DDT[i-1,j-1]) / (X[i] - X[i-j])

    # return nodes and coefficients
    return X, DDT.diagonal()



def hermite_piecewise_interpolation(f, df, a, b, m, eval_pts): 
    # Create the boundaries for the m subintervals
    partition = np.linspace(a, b, m+1)
    pieces = []

    # precomputing 
    for i in range(m): 
        # define local interval
        a_i, b_i = partition[i], partition[i+1]

        # compute hermite coeff. and store it all 
        X, coef = Hermite_Divided_Differences([a_i, b_i], f, df)
        pieces.append((a_i, b_i, X, coef))

    # initalize eval. results
    result = np.zeros(len(eval_pts))

    # find subinterval for eval points 
    for k, x in enumerate(eval_pts):
        for (a_i, b_i, x_nodes, coef) in pieces:
            if a_i <= x <= b_i:
                # eval. using horners rule
                result[k] = Newton_Polynomial(x_nodes, coef, x)
                break
    return result


def cubic_spline_param(x, y, bc_type, bc_vals): 
    n = len(x) - 1
    h = np.diff(x) # x_i+1 - x_i 

    # compute div. diff
    delta = np.diff(y) / h
    size = n + 1
    
    # Create System of Equations
    A = np.zeros((size, size))
    b = np.zeros(size)

    # fill internal rows (1 through n-1)
    # enforce that second derivative is continuous
    for i in range(1,n): 
        A[i, i-1] = h[i]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i-1]
        b[i] = 3 * (h[i] * delta[i-1] + h[i-1] * delta[i])

    bc_1, bc_2 = bc_vals[0], bc_vals[1]

    # if s1 then specify exact derivative at the endpoints
    if bc_type == 's1': 
        A[0, 0] = 1
        A[n, n] = 1
        b[0] = bc_1
        b[n] = bc_2 

    # specify the second derivative at the endpoints
    elif bc_type == 's2': 
        A[0, 0] = 2 
        A[0, 1] = 1
        A[n, n] = 2 
        A[n, n-1] = 1 
        b[0] = 3 * delta[0] - (h[0] / 2) * bc_1
        b[n] = 3 * delta[n-1] + (h[n-1] / 2) * bc_2

    # solve the liner system
    d = np.linalg.solve(A, b)
    return d


def cubic_spline_eval(t, x, y, d):
    # t - eval points
    t = np.asarray(t)
    s_t = np.zeros(len(t))
    h = np.diff(x)

    # begin interval search
    for k, t_k in enumerate(t):
        i = len(x) - 2
        for j in range(len(x) - 1): 
            if x[j] <= t_k <= x[j+1]:
                i = j
                break 

        # local coordinate mapping
        h_i = h[i]
        mu = (t_k - x[i]) / h_i

        # Hermite basis functions
        phi_00 = 2 * mu**3 - 3 * mu**2 + 1
        phi_10 = mu**3 - 2 * mu**2 + mu
        phi_01 = -2 * mu**3 + 3 * mu**2 
        phi_11 = mu**3 - mu**2 

        # Spline construction
        s_t[k] = (phi_00*y[i]) + (phi_10*h_i*d[i]) + (phi_01*y[i+1]) + (phi_11*h_i*d[i+1])
    return s_t


def cubic_spline_deriv(t, x, y, d):
    
    t = np.asarray(t)
    s_prime = np.zeros(len(t))
    h = np.diff(x)

    # begin interval search
    for k, t_k in enumerate(t):
        i = len(x) - 2
        for j in range(len(x) - 1):
            if x[j] <= t_k <= x[j+1]:
                i = j
                break

        # local coordinate mapping
        h_i = h[i]
        mu = (t_k - x[i]) / h_i

        # Derivatives of the cubic Hermite basis functions
        phi_00_prime = 6*mu**2 - 6*mu
        phi_10_prime = 3*mu**2 - 4*mu + 1
        phi_01_prime = -6*mu**2 + 6*mu
        phi_11_prime = 3*mu**2 - 2*mu

        # spline construction
        s_prime[k] = (phi_00_prime * y[i] +
                      phi_10_prime * h_i * d[i] +
                      phi_01_prime * y[i+1] +
                      phi_11_prime * h_i * d[i+1]) / h_i   

    return s_prime


def bspline_basis(t, xi, h):
    # define support of the basis function
    xim2 = xi - 2*h
    xim1 = xi - h
    # xi is center
    xip1 = xi + h
    xip2 = xi + 2*h

    # piece1: left tail 
    if xim2 <= t <= xim1:
        return (t - xim2)**3 / h**3

    # piece2: left center
    elif xim1 < t <= xi:
        u = t - xim1
        return (h**3 + 3*h**2*u + 3*h*u**2 - 3*u**3) / h**3

    #piece3: right center 
    elif xi < t <= xip1:
        u = xip1 - t
        return (h**3 + 3*h**2*u + 3*h*u**2 - 3*u**3) / h**3

    #piece 4: right tail
    elif xip1 < t <= xip2:
        return (xip2 - t)**3 / h**3

    # outside support 
    else:
        return 0.0
    
def bspline_basis_deriv(t, xi, h):
    # Define the same support intervals as the original basis function
    xim2 = xi - 2*h
    xim1 = xi - h
    xip1 = xi + h
    xip2 = xi + 2*h

    # piece1 derivative 
    if xim2 <= t <= xim1:
        return 3*(t - xim2)**2 / h**3

    # piece2 derivative 
    elif xim1 < t <= xi:
        u = t - xim1
        return (3*h**2 + 6*h*u - 9*u**2) / h**3
        
    # piece3 derivative 
    elif xi < t <= xip1:
        u = xip1 - t
        return -(3*h**2 + 6*h*u - 9*u**2) / h**3
        
    # piece4 derivative 
    elif xip1 < t <= xip2:
        return -3*(xip2 - t)**2 / h**3

    else:
        return 0.0
    
    
def cubic_bspline_param(x, y, bc_type, bc_vals):
    n = len(x) - 1
    h = x[1] - x[0]  # uniform spacing

    xi_centers = np.concatenate([[x[0] - h], x, [x[-1] + h]])

    size = n + 3  # number of basis functions = number of unknowns
    A = np.zeros((size, size))
    r = np.zeros(size)

    # Interpolation constraints 
    # fill rows such that S(x_i) = y_i
    for j in range(n + 1):
        row = j + 1  # leave room for boundary condition
        col = j      
        A[row, col]     = 1.0  
        A[row, col + 1] = 4.0  
        A[row, col + 2] = 1.0  
        r[row] = y[j]

    bc_left, bc_right = bc_vals

    # Boundary condition handling 
    # enforce specified derivatives at x_0 and x_n
    if bc_type == 's1':
        # at x_0
        A[0, 1] =  0.0
        A[0, 2] =  3.0/h
        r[0] = bc_left

        # at x_n
        A[n+2, n]   = -3.0/h
        A[n+2, n+1] =  0.0
        A[n+2, n+2] =  3.0/h
        r[n+2] = bc_right

    # Enforce specified second derivatives at x_0 and x_n
    elif bc_type == 's2':
        # at x_0
        A[0, 0] =  6.0/h**2
        A[0, 1] = -12.0/h**2
        A[0, 2] =  6.0/h**2
        r[0] = bc_left

        # At xn:
        A[n+2, n]   =  6.0/h**2
        A[n+2, n+1] = -12.0/h**2
        A[n+2, n+2] =  6.0/h**2
        r[n+2] = bc_right

    alpha = np.linalg.solve(A, r)
    return xi_centers, alpha



def cubic_bspline_eval(t_arr, x, xi_centers, alpha):
    t_arr = np.asarray(t_arr)
    h = x[1] - x[0]
    n = len(x) - 1
    s_t = np.zeros(len(t_arr))

    # interval search 
    for k, tk in enumerate(t_arr):
        i = len(x) - 2
        for j in range(len(x) - 1):
            if x[j] <= tk <= x[j+1]:
                i = j
                break

        # local summation
        val = 0.0
        for idx in range(i, i + 4):
            # running sum 
            val += alpha[idx] * bspline_basis(tk, xi_centers[idx], h)
        s_t[k] = val

    return s_t


def Error_Statistics(true_vals, approx_vals):
    # Ensure inputs are numpy arrays 
    true_vals = np.asarray(true_vals)
    approx_vals = np.asarray(approx_vals)
    
    # Absolute Error
    abs_error = np.abs(true_vals - approx_vals)
    
    # Infinity Norm (Max Error)
    inf_norm = np.max(abs_error)
    
    # Mean Squared Error (MSE)
    mse = np.mean(abs_error**2)
    
    # Relative Error 
    # use a small epsilon for values where true_vals = 0
    relative_error = np.where(true_vals != 0, 
                              abs_error / np.abs(true_vals), 
                              abs_error)
    max_rel_error = np.max(relative_error)
    
    stats = {
        "Infinity Norm": inf_norm,
        "MSE": mse,
        "Max Relative Error": max_rel_error,
        "Mean Absolute Error": np.mean(abs_error)
    }
    
    return stats


# --- Test Functions ---
functions = {
    "Cubic Polynomial": (lambda x: x**3 + x**2 + x, None),
    "sin(x)":           (lambda x: np.sin(x),         None),
    "Runge Function":   (lambda x: 1/(1 + 25*x**2),   None),
}

hermite_func = (lambda x: np.exp(-x**2), lambda x: -2*x*np.exp(-x**2))
spline_func  = (lambda x: x**3 - 2*x**2 + x + 1, lambda x: 3*x**2 - 4*x + 1)

# define evaluation grids 
eval_pts_global   = np.linspace(-1, 1, 200)
eval_pts_piecewise = np.linspace(0, 2*np.pi, 500)
eval_pts_hermite  = np.linspace(-2, 2, 500)
eval_pts_spline   = np.linspace(0, 5, 300)
eval_pts_bspline  = np.linspace(0, 3, 300)

# Barycentric-1 Interpolation
print("=" * 55)
print("Barycentric-1 Interpolation")
print("=" * 55)

for name, (f, _) in functions.items():
    mesh  = Chebyshev_Points(10)
    fx    = f(mesh)
    gamma = Barycentric1_Coefficients(mesh)
    px    = Barycentric1_Interpolation(eval_pts_global, mesh, gamma, fx)
    true  = f(eval_pts_global)
    stats = Error_Statistics(true, px)

    print(f"\n  Function: {name}")
    for k, v in stats.items():
        print(f"    {k}: {v:.6e}")

    # visualizations
    plt.figure()
    plt.plot(eval_pts_global, true, label="True f(x)")
    plt.plot(eval_pts_global, px, '--', label="Barycentric-1")
    plt.scatter(mesh, fx, zorder=5, label="Nodes")
    plt.title(f"Barycentric-1 — {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

#  Piecewise Newton Interpolation
print("\n" + "=" * 55)
print("Piecewise Newton Interpolation")
print("=" * 55)

f_pw  = lambda x: np.sin(x)
m, d  = 3, 2
pw    = piecewise_interpolation(f_pw, 0, 2*np.pi, m, d, method="uniform", eval_pts=eval_pts_piecewise)
true  = f_pw(eval_pts_piecewise)
stats = Error_Statistics(true, pw)

print(f"\n  Function: sin(x),  m={m}, d={d}")
for k, v in stats.items():
    print(f"    {k}: {v:.6e}")
# visualizations
plt.figure()
plt.plot(eval_pts_piecewise, true, label="True f(x)")
plt.plot(eval_pts_piecewise, pw, '--', label=f"Piecewise Newton (m={m}, d={d})")
plt.title("Piecewise Newton Interpolation — sin(x)")
plt.legend()
plt.grid(True)
plt.show()

# Piecewise Hermite Interpolation
print("\n" + "=" * 55)
print("Piecewise Hermite Interpolation")
print("=" * 55)

f_h, df_h = hermite_func
m_h = 4
herm  = hermite_piecewise_interpolation(f_h, df_h, -2, 2, m_h, eval_pts=eval_pts_hermite)
true  = f_h(eval_pts_hermite)
stats = Error_Statistics(true, herm)

print(f"\n  Function: exp(-x^2),  m={m_h}")
for k, v in stats.items():
    print(f"    {k}: {v:.6e}")

# visualizations
plt.figure()
plt.plot(eval_pts_hermite, true, label="True f(x)")
plt.plot(eval_pts_hermite, herm, '--', label=f"Piecewise Hermite (m={m_h})")
plt.title("Piecewise Hermite Interpolation — exp(-x²)")
plt.legend()
plt.grid(True)
plt.show()

# Cubic Spline Interpolation
print("\n" + "=" * 55)
print("Cubic Spline Interpolation")
print("=" * 55)

f_s, df_s = spline_func
x_mesh = np.linspace(0, 5, 10)
y_mesh = f_s(x_mesh)
params = cubic_spline_param(x_mesh, y_mesh, bc_type='s1', bc_vals=(df_s(x_mesh[0]), df_s(x_mesh[-1])))
cs     = cubic_spline_eval(eval_pts_spline, x_mesh, y_mesh, params)
true   = f_s(eval_pts_spline)
stats  = Error_Statistics(true, cs)

print(f"\n  Function: x³ - 2x² + x + 1  (BC: S1)")
for k, v in stats.items():
    print(f"    {k}: {v:.6e}")

# visualizations
plt.figure()
plt.plot(eval_pts_spline, true, label="True f(x)")
plt.plot(eval_pts_spline, cs, '--', label="Cubic Spline (S1)")
plt.scatter(x_mesh, y_mesh, zorder=5, label="Nodes")
plt.title("Cubic Spline — x³ - 2x² + x + 1")
plt.legend()
plt.grid(True)
plt.show()

# Cubic B-Spline Interpolation
print("\n" + "=" * 55)
print("Cubic B-Spline Interpolation")
print("=" * 55)

x_mesh_b = np.linspace(0, 3, 6)
y_mesh_b  = f_s(x_mesh_b)
xi_centers, alpha = cubic_bspline_param(x_mesh_b, y_mesh_b, bc_type='s1',
                                         bc_vals=(df_s(x_mesh_b[0]), df_s(x_mesh_b[-1])))
bs    = cubic_bspline_eval(eval_pts_bspline, x_mesh_b, xi_centers, alpha)
true  = f_s(eval_pts_bspline)
stats = Error_Statistics(true, bs)

# visualizations
print(f"\n  Function: x³ - 2x² + x + 1  (BC: S1)")
for k, v in stats.items():
    print(f"    {k}: {v:.6e}")

plt.figure()
plt.plot(eval_pts_bspline, true, label="True f(x)")
plt.plot(eval_pts_bspline, bs, '--', label="Cubic B-Spline (S1)")
plt.scatter(x_mesh_b, y_mesh_b, zorder=5, label="Nodes")
plt.title("Cubic B-Spline — x³ - 2x² + x + 1")
plt.legend()
plt.grid(True)
plt.show()

# Discrete Data Testing & Validation
print("="*75)
print("TASK 2: Discrete Data Testing with Nonuniform Mesh")
print("="*75)

# Given discrete data for y(t)
t_data = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0])
y_data = np.array([0.0552, 0.06, 0.0682, 0.0801, 0.0843, 0.0931, 0.0912, 0.0857])

print("Given data points:")
for ti, yi in zip(t_data, y_data):
    print(f" t = {ti:5.1f} y = {yi:.5f}")

# Evaluation points
t_eval = np.arange(0.5, 40.01, 0.5)

# Natural Cubic Spline 
bc_type = 's2'
bc_vals = (0.0, 0.0) # Natural boundary conditions: s''(0) = s''(end) = 0

d_params = cubic_spline_param(t_data, y_data, bc_type=bc_type, bc_vals=bc_vals)

y_est = cubic_spline_eval(t_eval, t_data, y_data, d_params)
dy_est = cubic_spline_deriv(t_eval, t_data, y_data, d_params)   # ← Your proper analytic derivative
f_est = y_est + t_eval * dy_est
D_est = np.exp(-t_eval * y_est)

print("\nNatural Cubic Spline completed.")

# Piecewise Newton Interpolation (g_d) Comparison
degrees = [1, 2, 3]
pw_results = {}

for d in degrees:
    # Use your existing piecewise_interpolation function
    y_pw = piecewise_interpolation(lambda x: np.interp(x, t_data, y_data),
                                   t_data[0], t_data[-1],
                                   m=len(t_data)-1, s=d,
                                   method="uniform", eval_pts=t_eval)
    
    # Numerical derivative (necessary because piecewise Newton does not provide analytic derivative)
    dy_pw = np.gradient(y_pw, t_eval)
    f_pw = y_pw + t_eval * dy_pw
    D_pw = np.exp(-t_eval * y_pw)
    
    pw_results[d] = (y_pw, f_pw, D_pw)
    print(f"Piecewise Newton (degree d={d}) completed.")

# Plots - Spline vs Piecewise Newton
plt.figure(figsize=(14, 11))

# y(t)
plt.subplot(3, 1, 1)
plt.plot(t_eval, y_est, 'b-', linewidth=2.5, label='Natural Cubic Spline')
for d in degrees:
    plt.plot(t_eval, pw_results[d][0], '--', linewidth=1.3, label=f'Piecewise Newton d={d}')
plt.plot(t_data, y_data, 'ro', markersize=6, label='Given data points')
plt.title('y(t) Comparison: Cubic Spline vs Piecewise Newton')
plt.legend()
plt.grid(True)

# f(t)
plt.subplot(3, 1, 2)
plt.plot(t_eval, f_est, 'b-', linewidth=2.5, label='Natural Cubic Spline')
for d in degrees:
    plt.plot(t_eval, pw_results[d][1], '--', linewidth=1.3, label=f'Piecewise Newton d={d}')
plt.title('f(t) Comparison: Cubic Spline vs Piecewise Newton')
plt.legend()
plt.grid(True)

# D(t)
plt.subplot(3, 1, 3)
plt.plot(t_eval, D_est, 'b-', linewidth=2.5, label='Natural Cubic Spline')
for d in degrees:
    plt.plot(t_eval, pw_results[d][2], '--', linewidth=1.3, label=f'Piecewise Newton d={d}')
plt.title('D(t) Comparison: Cubic Spline vs Piecewise Newton')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()