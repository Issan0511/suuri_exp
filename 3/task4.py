import numpy as np

def f (u,alpha = 1,beta =0,t =0):
    return -alpha * u + beta


def update_clank_nicolson (delta_t, u1, t1 ,f= f,alpha = 1,beta =0):
    # u_new = u1 + (delta_t / 2) * ( f(u1, t1) + f(u_new, t1 + delta_t) )
    # => u_new + (delta_t / 2) * alpha * u_new = u1 + (delta_t / 2) * ( -alpha * u1 + beta ) + (delta_t / 2) * alpha * 0
    # => (1 + (delta_t / 2) * alpha) * u_new = u1 + (delta_t / 2) * ( -alpha * u1 + beta )
    return (u1 + (delta_t / 2) * ( -alpha * u1 + beta )) / (1 + (delta_t / 2) * alpha)

def update_forward_backward_euler (delta_t, u1, t1 ,f= f,alpha = 1,beta =0):
    # 1. Forward Euler で予測
    u_predictor = u1 + delta_t * f(u1, alpha, beta, t1)
    return (u1 + delta_t * f(u_predictor, alpha, beta, t1 + delta_t)) / 2  

def update_heun (delta_t, u1, t1 ,f= f,alpha = 1,beta =0):
    f1 = f(u1, alpha, beta, t1)
    u_predictor = u1 + delta_t * f1
    t = t1 + delta_t
    f2 = f(u_predictor, alpha, beta, t )
    return u1 + (delta_t / 2) * (f1 + f2) 



def clank_nicolson (n,t_start = 0, t_end = 10, u_start = 1,alpha = 1,beta =0):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_clank_nicolson (delta_t, u, t, f, alpha, beta)
        t += delta_t
    return u

def foward_backward_euler (n,t_start = 0, t_end = 10, u_start = 1,alpha = 1,beta =0):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_forward_backward_euler (delta_t, u, t, f, alpha, beta)
        t += delta_t
    return u

def heun (n,t_start = 0, t_end = 10, u_start = 1):
    delta_t = np.float64((t_end - t_start) / n)
    u = np.float64(u_start)
    t = np.float64(t_start)
    for i in range (n):
        u = update_heun (delta_t, u, t, f)
        t += delta_t
    return u




