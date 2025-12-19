#%%
import QuantLib as ql
import numpy as np 
from ql_worst_of import ql_worst_of
from scipy.linalg import solve_banded

def osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    
    s1_max, s2_max = 2.0 * s1, 2.0 * s2
    dx, dy = s1_max / nx, s2_max / ny
    dt = t / nt

    x_axis = np.linspace(0.0, s1_max, nx + 1)
    y_axis = np.linspace(0.0, s2_max, ny + 1)

    X, Y = np.meshgrid(x_axis, y_axis)

    W = np.minimum(X, Y)
    vals = np.zeros_like(W)

    k_low = k - 1.0 / oh
    mask_itm = (W >= k)
    mask_lin = (W >= k_low) & (W < k)

    vals[mask_itm] = 10000.0
    vals[mask_lin] = 10000.0 * oh * (W[mask_lin] - k_low)

    def apply_bc(a):
        a[0, :]  = 2.0 * a[1, :]  - a[2, :]
        a[-1, :] = 2.0 * a[-2, :] - a[-3, :]
        a[:, 0]  = 2.0 * a[:, 1]  - a[:, 2]
        a[:, -1] = 2.0 * a[:, -2] - a[:, -3]
        a[0, 0]      = a[0, 1]    + a[1, 0]    - a[1, 1]
        a[-1, 0]     = a[-2, 0]   + a[-1, 1]   - a[-2, 1]
        a[0, -1]     = a[0, -2]   + a[1, -1]   - a[1, -2]
        a[-1, -1]    = a[-1, -2]  + a[-2, -1]  - a[-2, -2]

    apply_bc(vals)

    # OSM coefficients
    ax = dt * sigma1 ** 2 * X ** 2 / (2.0 * dx ** 2)
    bx = dt * (r - q1)   * X      / (2.0 * dx)

    ay = dt * sigma2 ** 2 * Y ** 2 / (2.0 * dy ** 2)
    by = dt * (r - q2)   * Y      / (2.0 * dy)

    sub_x  = -(ax[:, 1:-1] - bx[:, 1:-1])
    sup_x  = -(ax[:, 1:-1] + bx[:, 1:-1])
    diag_x = 1.0 + 2.0 * ax[:, 1:-1] + 0.5 * dt * r

    sub_y  = -(ay[1:-1, :] - by[1:-1, :])
    sup_y  = -(ay[1:-1, :] + by[1:-1, :])
    diag_y = 1.0 + 2.0 * ay[1:-1, :] + 0.5 * dt * r

    cross_coef = dt * corr * sigma1 * sigma2 * X * Y / (8.0 * dx * dy)
    vals_dt = None

    # Time stepping (OSM) 
    for step in range(nt):
        if step == nt - 1:
            vals_dt = vals.copy()

        # x-sweep
        apply_bc(vals)
        for j in range(1, ny):
            rhs = vals[j, 1:-1] + cross_coef[j, 1:-1] * (
                vals[j + 1, 2:] - vals[j - 1, 2:]
                - vals[j + 1, :-2] + vals[j - 1, :-2]
            )

            low = sub_x[j].copy()
            up  = sup_x[j].copy()
            mid = diag_x[j].copy()

            rhs[0]  -= low[0]   * vals[j, 0]
            rhs[-1] -= up[-1]   * vals[j, -1]
            low[0]  = 0.0
            up[-1]  = 0.0

            n_unknown = nx - 1
            ab = np.zeros((3, n_unknown))
            ab[0, 1:] = up[:-1]
            ab[1, :]  = mid
            ab[2, :-1] = low[1:]

            vals[j, 1:-1] = solve_banded((1, 1), ab, rhs)

        apply_bc(vals)

        # y-sweep
        apply_bc(vals)
        for i in range(1, nx):
            rhs = vals[1:-1, i] + cross_coef[1:-1, i] * (
                vals[2:, i + 1] - vals[:-2, i + 1]
                - vals[2:, i - 1] + vals[:-2, i - 1]
            )

            low = sub_y[:, i].copy()
            up  = sup_y[:, i].copy()
            mid = diag_y[:, i].copy()

            rhs[0]  -= low[0]   * vals[0, i]
            rhs[-1] -= up[-1]   * vals[-1, i]
            low[0]  = 0.0
            up[-1]  = 0.0

            m = ny - 1
            ab = np.zeros((3, m))
            ab[0, 1:] = up[:-1]
            ab[1, :]  = mid
            ab[2, :-1] = low[1:]

            vals[1:-1, i] = solve_banded((1, 1), ab, rhs)

        apply_bc(vals)

    # Bilinear interpolation & Greeks ----------------
    def interp_bilinear(x, y, G):
        i = int(np.clip(x / dx, 0, nx - 1))
        j = int(np.clip(y / dy, 0, ny - 1))

        x1, x2 = x_axis[i], x_axis[i + 1]
        y1, y2 = y_axis[j], y_axis[j + 1]

        wx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
        wy = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)

        v11 = G[j,     i    ]
        v21 = G[j,     i + 1]
        v12 = G[j + 1, i    ]
        v22 = G[j + 1, i + 1]

        return ((1 - wx) * (1 - wy) * v11 +
                wx * (1 - wy) * v21 +
                (1 - wx) * wy * v12 +
                wx * wy * v22)

    price = interp_bilinear(s1, s2, vals)

    p_xp = interp_bilinear(s1 + dx, s2, vals)
    p_xm = interp_bilinear(s1 - dx, s2, vals)
    p_yp = interp_bilinear(s1, s2 + dy, vals)
    p_ym = interp_bilinear(s1, s2 - dy, vals)

    delta1 = (p_xp - p_xm) / (2.0 * dx)
    delta2 = (p_yp - p_ym) / (2.0 * dy)

    gamma1 = (p_xp - 2.0 * price + p_xm) / (dx ** 2)
    gamma2 = (p_yp - 2.0 * price + p_ym) / (dy ** 2)

    p_pp = interp_bilinear(s1 + dx, s2 + dy, vals)
    p_pm = interp_bilinear(s1 + dx, s2 - dy, vals)
    p_mp = interp_bilinear(s1 - dx, s2 + dy, vals)
    p_mm = interp_bilinear(s1 - dx, s2 - dy, vals)
    crossgamma = (p_pp - p_pm - p_mp + p_mm) / (4.0 * dx * dy)
    price_dt = interp_bilinear(s1, s2, vals_dt)
    theta = (price_dt - price) / (dt * 365.0)

    return price, delta1, delta2, gamma1, gamma2, crossgamma, theta

def adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt):
    
    s1_max, s2_max = 2.0 * s1, 2.0 * s2
    dx, dy = s1_max / nx, s2_max / ny
    dt = t / nt

    x_axis = np.linspace(0.0, s1_max, nx + 1)
    y_axis = np.linspace(0.0, s2_max, ny + 1)

    X, Y = np.meshgrid(x_axis, y_axis)  # shape: (ny+1, nx+1)

    # worst-of binary with overhedge (notional 10,000)
    W = np.minimum(X, Y)
    grid = np.zeros_like(W)

    k_low = k - 1.0 / oh
    itm_mask = (W >= k)
    linear_mask = (W >= k_low) & (W < k)

    grid[itm_mask] = 10000.0
    grid[linear_mask] = 10000.0 * oh * (W[linear_mask] - k_low)

    def apply_bc(a):
        a[0, :]  = 2.0 * a[1, :]  - a[2, :]
        a[-1, :] = 2.0 * a[-2, :] - a[-3, :]
        a[:, 0]  = 2.0 * a[:, 1]  - a[:, 2]
        a[:, -1] = 2.0 * a[:, -2] - a[:, -3]
        a[0, 0]      = a[0, 1]    + a[1, 0]    - a[1, 1]
        a[-1, 0]     = a[-2, 0]   + a[-1, 1]   - a[-2, 1]
        a[0, -1]     = a[0, -2]   + a[1, -1]   - a[1, -2]
        a[-1, -1]    = a[-1, -2]  + a[-2, -1]  - a[-2, -2]

    apply_bc(grid)

    # ADI coefficients
    a_x = dt * sigma1 ** 2 * X ** 2 / (4.0 * dx ** 2)
    b_x = dt * (r - q1) * X      / (4.0 * dx)

    a_y = dt * sigma2 ** 2 * Y ** 2 / (4.0 * dy ** 2)
    b_y = dt * (r - q2) * Y      / (4.0 * dy)

    # tri-diagonal coeffs in x-direction
    sub_x  = -(a_x[:, 1:-1] - b_x[:, 1:-1])          # lower diag
    sup_x  = -(a_x[:, 1:-1] + b_x[:, 1:-1])          # upper diag
    diag_x = 1.0 + 2.0 * a_x[:, 1:-1] + 0.5 * dt * r

    # tri-diagonal coeffs in y-direction
    sub_y  = -(a_y[1:-1, :] - b_y[1:-1, :])
    sup_y  = -(a_y[1:-1, :] + b_y[1:-1, :])
    diag_y = 1.0 + 2.0 * a_y[1:-1, :] + 0.5 * dt * r

    cross_coef = dt * corr * sigma1 * sigma2 * X * Y / (8.0 * dx * dy)

    grid_dt = None  # grid at t = dt (for theta)

    # Time stepping (ADI)
    for step in range(nt):
        if step == nt - 1:
            grid_dt = grid.copy()

        # 1단계: x 방향 implicit, y & cross explicit
        apply_bc(grid)
        for j in range(1, ny):
            # explicit terms (y-direction diffusion & drift + cross term)
            cross_term = cross_coef[j, 1:-1] * (
                grid[j + 1, 2:] - grid[j - 1, 2:]
                - grid[j + 1, :-2] + grid[j - 1, :-2]
            )
            diff_y = a_y[j, 1:-1] * (grid[j + 1, 1:-1] - 2.0 * grid[j, 1:-1] + grid[j - 1, 1:-1])
            drift_y = b_y[j, 1:-1] * (grid[j + 1, 1:-1] - grid[j - 1, 1:-1])

            rhs = grid[j, 1:-1] + cross_term + diff_y + drift_y

            low = sub_x[j].copy()
            up  = sup_x[j].copy()
            mid = diag_x[j].copy()

            rhs[0]  -= low[0]   * grid[j, 0]
            rhs[-1] -= up[-1]   * grid[j, -1]
            low[0]  = 0.0
            up[-1]  = 0.0

            n_unknown = nx - 1
            ab = np.zeros((3, n_unknown))
            ab[0, 1:] = up[:-1]
            ab[1, :]  = mid
            ab[2, :-1] = low[1:]

            grid[j, 1:-1] = solve_banded((1, 1), ab, rhs)

        apply_bc(grid)

        # 2단계: y 방향 implicit, x & cross explicit
        apply_bc(grid)
        for i in range(1, nx):
            cross_term = cross_coef[1:-1, i] * (
                grid[2:, i + 1] - grid[:-2, i + 1]
                - grid[2:, i - 1] + grid[:-2, i - 1]
            )
            diff_x = a_x[1:-1, i] * (grid[1:-1, i + 1] - 2.0 * grid[1:-1, i] + grid[1:-1, i - 1])
            drift_x = b_x[1:-1, i] * (grid[1:-1, i + 1] - grid[1:-1, i - 1])

            rhs = grid[1:-1, i] + cross_term + diff_x + drift_x

            low = sub_y[:, i].copy()
            up  = sup_y[:, i].copy()
            mid = diag_y[:, i].copy()

            rhs[0]  -= low[0]   * grid[0, i]
            rhs[-1] -= up[-1]   * grid[-1, i]
            low[0]  = 0.0
            up[-1]  = 0.0

            m = ny - 1
            ab = np.zeros((3, m))
            ab[0, 1:] = up[:-1]
            ab[1, :]  = mid
            ab[2, :-1] = low[1:]

            grid[1:-1, i] = solve_banded((1, 1), ab, rhs)

        apply_bc(grid)

    # Bilinear interpolation & Greeks 
    def interp_bilinear(x, y, G):
        i = int(np.clip(x / dx, 0, nx - 1))
        j = int(np.clip(y / dy, 0, ny - 1))

        x1, x2 = x_axis[i], x_axis[i + 1]
        y1, y2 = y_axis[j], y_axis[j + 1]

        wx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
        wy = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)

        v11 = G[j,     i    ]
        v21 = G[j,     i + 1]
        v12 = G[j + 1, i    ]
        v22 = G[j + 1, i + 1]

        return ((1 - wx) * (1 - wy) * v11 +
                wx * (1 - wy) * v21 +
                (1 - wx) * wy * v12 +
                wx * wy * v22)

    price = interp_bilinear(s1, s2, grid)

    p_xp = interp_bilinear(s1 + dx, s2, grid)
    p_xm = interp_bilinear(s1 - dx, s2, grid)
    p_yp = interp_bilinear(s1, s2 + dy, grid)
    p_ym = interp_bilinear(s1, s2 - dy, grid)

    delta1 = (p_xp - p_xm) / (2.0 * dx)
    delta2 = (p_yp - p_ym) / (2.0 * dy)

    gamma1 = (p_xp - 2.0 * price + p_xm) / (dx ** 2)
    gamma2 = (p_yp - 2.0 * price + p_ym) / (dy ** 2)

    p_pp = interp_bilinear(s1 + dx, s2 + dy, grid)
    p_pm = interp_bilinear(s1 + dx, s2 - dy, grid)
    p_mp = interp_bilinear(s1 - dx, s2 + dy, grid)
    p_mm = interp_bilinear(s1 - dx, s2 - dy, grid)
    crossgamma = (p_pp - p_pm - p_mp + p_mm) / (4.0 * dx * dy)
    price_dt = interp_bilinear(s1, s2, grid_dt)
    theta = (price_dt - price) / (dt * 365.0)

    return price, delta1, delta2, gamma1, gamma2, crossgamma, theta


if __name__=="__main__":
    s1, s2 = 100,100
    r = 0.02
    q1, q2 = 0.015, 0.01
    k = 95
    t = 1
    sigma1, sigma2 = 0.25, 0.25
    corr = 0.3
    nx, ny, nt = 100, 100, 400
    oh = 10

    osm_price = osm_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt)
    print(f"OSM Price = {osm_price[0]:0.6f}, Delta1 = {osm_price[1]:0.6f}, Delta2 = {osm_price[2]:0.6f}, Gamma1 = {osm_price[3]:0.6f}, Gamma2 = {osm_price[4]:0.6f}, Theta = {osm_price[5]:0.6f}")

    adi_price = adi_worst_of(s1, s2, k, r, q1, q2, t, sigma1, sigma2, corr, oh, nx, ny, nt)
    print(f"ADI Price = {adi_price[0]:0.6f}, Delta1 = {adi_price[1]:0.6f}, Delta2 = {adi_price[2]:0.6f}, Gamma1 = {adi_price[3]:0.6f}, Gamma2 = {adi_price[4]:0.6f}, Theta = {adi_price[5]:0.6f}")

# %%
