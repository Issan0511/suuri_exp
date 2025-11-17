import numpy as np

def f (u,t =0):
    return u

def analytical_solution (t, u0 = 1):
    return u0 * np.exp(t)

def update_foward_euler (delta_t, u1, t1 ,f= f):
    return u1 + delta_t * f(u1, t1)

def update_adam_bashforth2 (delta_t, u1,u2, t1 ,t2,f= f):
    f1 = f(u1, t1)
    f2 = f(u2, t2)
    return u1 + delta_t * ( (3/2) * f1 - (1/2) * f2 )

def update_adam_bashforth3 (delta_t, u1,u2,u3, t1 ,t2,t3,f= f):
    f1 = f(u1, t1)
    f2 = f(u2, t2)
    f3 = f(u3, t3)
    return u1 + delta_t * ( (23/12) * f1 - (16/12) * f2 + (5/12) * f3 )

def update_heun (delta_t, u1, t1 ,f= f):
    f1 = f(u1, t1)
    u_predictor = u1 + delta_t * f1
    t = t1 + delta_t
    f2 = f(u_predictor, t )
    return u1 + (delta_t / 2) * (f1 + f2) 

def update_runge_kutta4 (delta_t, u1, t1 ,f= f):
    k1 = f(u1, t1)
    k2 = f(u1 + (delta_t / 2) * k1, t1 + (delta_t / 2))
    k3 = f(u1 + (delta_t / 2) * k2, t1 + (delta_t / 2))
    k4 = f(u1 + delta_t * k3, t1 + delta_t)
    return u1 + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def foward_euler (n,t_start = 0, t_end = 1, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_foward_euler (delta_t, u, t, f)
        t += delta_t
    return u

def adam_bashforth2 (n,t_start = 0, t_end = 1, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    # 最初の1ステップは真値を使う
    u2 = np.float64(u_start)
    t2 = np.float64(t_start)
    t1 = t2 + delta_t
    u1 = analytical_solution (t1, u_start)
    for i in range (1, n):
        u_new = update_adam_bashforth2 (delta_t, u1, u2, t1, t2, f)
        u2 = u1
        t2 = t1
        u1 = u_new
        t1 += delta_t
    return u1

def adam_bashforth3 (n,t_start = 0, t_end = 1, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    # 最初の2ステップは真値を使う
    u3 = np.float64(u_start)
    t3 = np.float64(t_start)
    t2 = t3 + delta_t
    u2 = analytical_solution (t2, u_start)
    t1 = t2 + delta_t
    u1 = analytical_solution (t1, u_start)
    for i in range (2, n):
        u_new = update_adam_bashforth3 (delta_t, u1, u2, u3, t1, t2, t3, f)
        u3 = u2
        t3 = t2
        u2 = u1
        t2 = t1
        u1 = u_new
        t1 += delta_t
    return u1

def heun (n,t_start = 0, t_end = 1, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_heun (delta_t, u, t, f)
        t += delta_t
    return u

def runge_kutta4 (n,t_start = 0, t_end = 1, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_runge_kutta4 (delta_t, u, t, f)
        t += delta_t
    return u

def main (n):
    # 更新方法を変えて誤差を確認
    
    print("Forward Euler:", abs ( foward_euler (n) - analytical_solution (1) ) )
    print("Adams-Bashforth 2nd order:", abs ( adam_bashforth2 (n) - analytical_solution (1) ) )
    print("Adams-Bashforth 3rd order:", abs ( adam_bashforth3 (n) - analytical_solution (1) ) )
    print("Heun method:", abs ( heun (n) - analytical_solution (1) ) )
    print("Runge-Kutta 4th order:", abs ( runge_kutta4 (n) - analytical_solution (1) ) )
    return


print("真の解:", analytical_solution (1) )
print("真の解との誤差:")
for pow in range(2, 10):
    n = 2 ** pow
    print("n =", n)
    main (n)
