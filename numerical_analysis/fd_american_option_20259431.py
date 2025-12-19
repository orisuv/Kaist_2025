#%%
import numpy as np 
import pandas as pd
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d

def fd_american_option(s, k, r, q, t, sigma, option_type, n, m):

    S_max = 4.0 * s
    S_min = 0.0

    dS = (S_max - S_min) / n
    dt = t / m

    theta = 0.5  # Crank–Nicolson
    one_minus_theta = 1.0 - theta

    # S 그리드
    idx = np.arange(n + 1)
    S = S_min + idx * dS

    # 만기 페이오프
    opt = option_type.lower()
    if opt == "call":
        payoff = np.maximum(S - k, 0.0)
    else: 
        payoff = np.maximum(k - S, 0.0)  # put

    # t = T 에서의 값 (만기조건)
    v = payoff.copy()    
    v_t1 = None            

    i_int = np.arange(1, n)
    S_i = S[i_int]

    sigma2 = sigma * sigma

    # a_i = dt * σ^2 S_i^2 / (2 ΔS^2)
    # b_i = dt * (r - q) S_i / (2 ΔS)
    a_i = dt * 0.5 * sigma2 * (S_i ** 2) / (dS ** 2)
    b_i = dt * (r - q) * S_i / (2.0 * dS)

    d_i = a_i - b_i
    m_i = -2.0 * a_i - dt * r
    u_i = a_i + b_i

    # A 행렬의 삼중대각
    lower_A = d_i.copy()   # 하부 대각
    diag_A  = m_i.copy()   # 주대각
    upper_A = u_i.copy()   # 상부 대각

    # θ-method용 Am, Ap 삼중대각 구성
    # Am = I - θA,  Ap = I + (1-θ)A
    lower_Am = -theta * lower_A
    diag_Am  = 1.0 - theta * diag_A
    upper_Am = -theta * upper_A

    lower_Ap = one_minus_theta * lower_A
    diag_Ap  = 1.0 + one_minus_theta * diag_A
    upper_Ap = one_minus_theta * upper_A

    # PSOR
    inv_diag_Am = 1.0 / diag_Am
    n_int = n - 1 

    # PSOR 파라미터
    omega = 1.2
    tol = 1e-8
    max_iter = 10000

    # 내부 페이오프
    payoff_int = payoff[1:n]

    # 시간 방향으로 뒤로 (j = m-1 ... 0) 진행
    for j in range(m - 1, -1, -1):
  
        t_j = j * dt
        tau = t - t_j    

        # 경계조건 (S = 0, S = S_max)
        if opt == "call":
            V_0 = 0.0
            V_N = S_max - k * np.exp(-r * tau)
        else:  # put
            V_0 = k * np.exp(-r * tau)
            V_N = 0.0

        v_next_int = v[1:n].copy()

        # RHS = Ap v^{j+1} + θ B v^j_∂Ω + (1-θ) B v^{j+1}_∂Ω
        # B는 첫 행에 d_1 * v_0, 마지막 행에 u_{N-1} * v_N만 가짐
        rhs = diag_Ap * v_next_int
        rhs[1:] += lower_Ap[1:] * v_next_int[:-1]
        rhs[:-1] += upper_Ap[:-1] * v_next_int[1:]

        # 경계값 (j+1 시점, j 시점)
        V0_next, VN_next = v[0], v[-1]

        # B 부분 (d_1 v_0, u_{N-1} v_N)
        d1 = d_i[0]
        u_last = u_i[-1]

        rhs[0]  += lower_Ap[0] * V0_next
        rhs[-1] += upper_Ap[-1] * VN_next

        # θ B v^j_∂Ω + (1-θ) B v^{j+1}_∂Ω
        rhs[0]  += theta * d1 * V_0      + one_minus_theta * d1 * V0_next
        rhs[-1] += theta * u_last * V_N  + one_minus_theta * u_last * VN_next

        # PSOR로 Am v^j = rhs를 풀되, American 조기행사 조건 적용
        # v^j_i >= payoff_i  (i = 1..n-1)
        x = v_next_int.copy()
        x = np.maximum(x, payoff_int) 

        for it in range(max_iter):
            x_old = x.copy()

            # Gauss–Seidel + Over-relaxation + Projection
            for i in range(n_int):

                sigma_gs = rhs[i]

                if i > 0:
                    sigma_gs -= lower_Am[i] * x[i - 1]
                if i < n_int - 1:
                    sigma_gs -= upper_Am[i] * x[i + 1]

                x_new = (1.0 - omega) * x[i] + omega * sigma_gs * inv_diag_Am[i]
                # American constraint (조기행사)
                x[i] = max(x_new, payoff_int[i])

            # 수렴 체크
            if np.max(np.abs(x - x_old)) < tol:
                break

        # 이번 스텝에서 얻은 v^j를 전체 벡터에 반영
        v[1:n] = x
        v[0] = max(V_0, payoff[0])
        v[-1] = max(V_N, payoff[-1])

        if j == 1:
            v_t1 = v.copy()

    # t = 0 에서의 가격 및 그릭스 계산 (S = s 지점)
    # 선형보간으로 정확히 S = s 위치의 값 얻기
    price_func_0 = interp1d(S, v, kind='linear')
    price = float(price_func_0(s))

    # Delta, Gamma: 공간 중앙차분
    idx_s = np.searchsorted(S, s)
    if idx_s == 0:
        idx_s = 1
    if idx_s == n:
        idx_s = n - 1

    S_im1, S_i, S_ip1 = S[idx_s - 1: idx_s + 2]
    V_im1, V_i, V_ip1 = v[idx_s - 1: idx_s + 2]

    delta = (V_ip1 - V_im1) / (S_ip1 - S_im1)
    gamma = (V_ip1 - 2.0 * V_i + V_im1) / ((S_ip1 - S_i) * (S_i - S_im1))
    # Theta: t=0 에서의 ∂V/∂t 를 (t=dt 값과 차분)으로 근사
    if v_t1 is not None:
        price_func_1 = interp1d(S, v_t1, kind='linear')
        price_t1 = float(price_func_1(s))
        theta = (price_t1 - price) / (dt * 365)
    else:
        theta = np.nan

    return float(price), float(delta), float(gamma), float(theta)


if __name__=="__main__":
    s = 100
    k = 100
    r = 0.03
    q = 0.01
    t = 1
    sigma = 0.3
    optionType = 'put'
    n, m = 500, 500
    print("="*50)
    print("American Option")
    print("="*50)

    price, delta, gamma, theta = fd_american_option(s, k, r, q, t, sigma, optionType, n, m)
    print(f"American(CN-FDM) = {price:0.6f} Delta = {delta:0.6f}, Gamma = {gamma:0.6f}, Theta = {theta:0.6f}")

# %%
