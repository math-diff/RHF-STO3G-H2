import numpy as np
from scipy.special import erf

# ===================================================================
# 1. 基本设置和常数
# ===================================================================

# H2 分子的几何结构 (原子单位: Bohr)
R = 1.4
coords = np.array([
    [0.0, 0.0, 0.0],
    [R, 0.0, 0.0]
])
atomic_numbers = [1, 1]
num_atoms = len(atomic_numbers)
num_electrons = sum(atomic_numbers)
num_occupied_orbitals = num_electrons // 2

# STO-3G 基组参数 for H
sto3g_params = {
    1: {
        'exponents': np.array([3.42525091, 0.62391373, 0.16885540]),
        'coeffs': np.array([0.15432897, 0.53532814, 0.44463454])
    }
}


# ===================================================================
# 2. 基函数类
# ===================================================================

class BasisFunction:

    def __init__(self, atom_center, exponents, coeffs):
        self.center = np.array(atom_center)
        self.exponents = np.array(exponents)
        self.coeffs = np.array(coeffs)  # 原始收缩系数 d
        self.norm_coeffs = np.zeros_like(self.coeffs)
        for i in range(len(self.exponents)):
            alpha = self.exponents[i]
            N = (2 * alpha / np.pi) ** 0.75
            self.norm_coeffs[i] = self.coeffs[i] * N


# 构建基组
basis_set = []
for i in range(num_atoms):
    Z = atomic_numbers[i]
    params = sto3g_params[Z]
    basis_set.append(BasisFunction(coords[i], params['exponents'], params['coeffs']))

num_basis_functions = len(basis_set)


# ===================================================================
# 3. 积分计算函数
# ===================================================================

def boys_function(t):
    """Boys 函数 F0(t)"""
    if t < 1e-10:
        return 1.0 - t / 3.0
    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))


def gaussian_product_center(alpha1, A, alpha2, B):
    """计算两个高斯函数乘积的新中心"""
    return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)


def compute_overlap(basis_set):
    """计算重叠积分矩阵 S"""
    n = len(basis_set)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bf_i = basis_set[i]
            bf_j = basis_set[j]
            R_AB_sq = np.sum((bf_i.center - bf_j.center) ** 2)

            for mu in range(len(bf_i.exponents)):
                alpha1 = bf_i.exponents[mu]
                norm_d1 = bf_i.norm_coeffs[mu]
                for nu in range(len(bf_j.exponents)):
                    alpha2 = bf_j.exponents[nu]
                    norm_d2 = bf_j.norm_coeffs[nu]

                    p = alpha1 + alpha2
                    S_prim = (np.pi / p) ** 1.5 * np.exp(-alpha1 * alpha2 * R_AB_sq / p)
                    S[i, j] += norm_d1 * norm_d2 * S_prim
    return S


def compute_kinetic(basis_set):
    """计算动能积分矩阵 T"""
    n = len(basis_set)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bf_i = basis_set[i]
            bf_j = basis_set[j]
            R_AB_sq = np.sum((bf_i.center - bf_j.center) ** 2)

            for mu in range(len(bf_i.exponents)):
                alpha1 = bf_i.exponents[mu]
                norm_d1 = bf_i.norm_coeffs[mu]
                for nu in range(len(bf_j.exponents)):
                    alpha2 = bf_j.exponents[nu]
                    norm_d2 = bf_j.norm_coeffs[nu]

                    p = alpha1 + alpha2
                    S_prim = (np.pi / p) ** 1.5 * np.exp(-alpha1 * alpha2 * R_AB_sq / p)
                    T_prim = alpha1 * alpha2 / p * (3.0 - 2.0 * alpha1 * alpha2 * R_AB_sq / p) * S_prim
                    T[i, j] += norm_d1 * norm_d2 * T_prim
    return T


def compute_nuclear_attraction(basis_set, coords, atomic_numbers):
    """计算核-电子吸引积分矩阵 V"""
    n_bf = len(basis_set)
    n_atoms = len(coords)
    V = np.zeros((n_bf, n_bf))

    for i in range(n_bf):
        for j in range(n_bf):
            bf_i = basis_set[i]
            bf_j = basis_set[j]
            R_AB_sq = np.sum((bf_i.center - bf_j.center) ** 2)

            for mu in range(len(bf_i.exponents)):
                alpha1 = bf_i.exponents[mu]
                norm_d1 = bf_i.norm_coeffs[mu]

                for nu in range(len(bf_j.exponents)):
                    alpha2 = bf_j.exponents[nu]
                    norm_d2 = bf_j.norm_coeffs[nu]

                    p = alpha1 + alpha2
                    P = gaussian_product_center(alpha1, bf_i.center, alpha2, bf_j.center)

                    V_prim = 0.0
                    for k_atom in range(n_atoms):
                        C = coords[k_atom]
                        Zk = atomic_numbers[k_atom]
                        R_PC_sq = np.sum((P - C) ** 2)

                        V_prim_contrib = (2.0 * np.pi / p) * np.exp(-alpha1 * alpha2 * R_AB_sq / p) * boys_function(
                            p * R_PC_sq)
                        V_prim += -Zk * V_prim_contrib

                    V[i, j] += norm_d1 * norm_d2 * V_prim
    return V


def compute_eri(basis_set):
    """计算双电子排斥积分张量 ERI"""
    n = len(basis_set)
    ERI = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    bf_i, bf_j, bf_k, bf_l = basis_set[i], basis_set[j], basis_set[k], basis_set[l]

                    for g1_idx in range(len(bf_i.exponents)):
                        alpha1 = bf_i.exponents[g1_idx]
                        norm_d1 = bf_i.norm_coeffs[g1_idx]
                        for g2_idx in range(len(bf_j.exponents)):
                            alpha2 = bf_j.exponents[g2_idx]
                            norm_d2 = bf_j.norm_coeffs[g2_idx]
                            p = alpha1 + alpha2
                            P = gaussian_product_center(alpha1, bf_i.center, alpha2, bf_j.center)
                            R_AB_sq = np.sum((bf_i.center - bf_j.center) ** 2)
                            term_exp1 = np.exp(-alpha1 * alpha2 * R_AB_sq / p)

                            for g3_idx in range(len(bf_k.exponents)):
                                alpha3 = bf_k.exponents[g3_idx]
                                norm_d3 = bf_k.norm_coeffs[g3_idx]
                                for g4_idx in range(len(bf_l.exponents)):
                                    alpha4 = bf_l.exponents[g4_idx]
                                    norm_d4 = bf_l.norm_coeffs[g4_idx]
                                    q = alpha3 + alpha4
                                    Q = gaussian_product_center(alpha3, bf_k.center, alpha4, bf_l.center)
                                    R_CD_sq = np.sum((bf_k.center - bf_l.center) ** 2)
                                    term_exp2 = np.exp(-alpha3 * alpha4 * R_CD_sq / q)
                                    R_PQ_sq = np.sum((P - Q) ** 2)
                                    boys_arg = (p * q * R_PQ_sq) / (p + q)
                                    term_boys = boys_function(boys_arg)
                                    term_const = (2.0 * np.pi ** 2.5) / (p * q * np.sqrt(p + q))
                                    ERI_prim = term_const * term_exp1 * term_exp2 * term_boys
                                    ERI[i, j, k, l] += norm_d1 * norm_d2 * norm_d3 * norm_d4 * ERI_prim
    return ERI


# ===================================================================
# 4. 打印辅助函数
# ===================================================================

def print_matrix(matrix, name, fmt="{: >12.8f}"):
    """以可读格式打印一个2D矩阵"""
    print(f"\n--- {name} ---")
    dim = matrix.shape[0]
    header = " " * 5
    for i in range(dim):
        header += f"   AO {i + 1}    "
    print(header)
    print("-" * len(header))
    for i in range(dim):
        row_str = f"AO {i + 1:<2} |"
        for j in range(dim):
            row_str += fmt.format(matrix[i, j])
        print(row_str)
    print("")


def print_eri_tensor(tensor, name, threshold=1e-8):
    """打印4D ERI张量中大于阈值的非零唯一元素"""
    print(f"\n--- {name} (values > {threshold}) ---")
    n = tensor.shape[0]
    printed = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # 利用对称性减少重复打印 (ij|kl) = (ji|kl) = (ij|lk) ... etc.
                    # 创建一个排序后的索引元组作为键
                    key = tuple(sorted((tuple(sorted((i, j))), tuple(sorted((k, l))))))
                    if key not in printed:
                        val = tensor[i, j, k, l]
                        if abs(val) > threshold:
                            print(f"  ({i + 1} {j + 1} | {k + 1} {l + 1}) = {val:12.8f}")
                            printed.add(key)
    print("")


# ===================================================================
# 5. 主 SCF 程序
# ===================================================================

def run_rhf(max_iter=100, E_conv_tol=1e-8, D_conv_tol=1e-6):
    """执行RHF-SCF计算"""
    print("--- RHF/STO-3G for H2 (With Matrix Printout) ---")
    print(f"H-H distance: {R:.4f} Bohr")
    print("-" * 30)

    # 1. 计算原子核间的排斥能
    E_nuc = (atomic_numbers[0] * atomic_numbers[1]) / R
    print(f"Nuclear Repulsion Energy: {E_nuc:.8f} Hartree")

    # 2. 计算积分并打印
    print("\nCalculating integrals...")

    S = compute_overlap(basis_set)
    print_matrix(S, "Overlap Matrix (S)")

    T = compute_kinetic(basis_set)
    print_matrix(T, "Kinetic Energy Matrix (T)")

    V = compute_nuclear_attraction(basis_set, coords, atomic_numbers)
    print_matrix(V, "Nuclear Attraction Matrix (V)")

    H_core = T + V  # 核心哈密顿量
    print_matrix(H_core, "Core Hamiltonian Matrix (H_core = T + V)")

    ERI = compute_eri(basis_set)
    print_eri_tensor(ERI, "Electron Repulsion Integrals (ERI)")

    print("\nIntegrals calculation finished.")
    print("-" * 30)

    # 3. 正交化矩阵 S^{-1/2}
    S_eigvals, S_eigvecs = np.linalg.eigh(S)
    S_inv_sqrt_diag = np.diag(S_eigvals ** -0.5)
    X = S_eigvecs @ S_inv_sqrt_diag @ S_eigvecs.T
    print_matrix(X, "Orthogonalization Matrix (X = S^-1/2)")

    # 4. 初始猜测 (零密度矩阵)
    P = np.zeros((num_basis_functions, num_basis_functions))

    # 5. SCF 迭代循环
    E_total_old = 0.0
    print("Starting SCF iterations...")
    print("Iter | Total Energy (Hartree) | Delta E    | RMS(D)")

    for i in range(max_iter):
        J = np.einsum('ls,uvls->uv', P, ERI)
        K = np.einsum('ls,ulvs->uv', P, ERI)
        F = H_core + J - 0.5 * K

        F_prime = X.T @ F @ X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        P_old = P.copy()
        C_occ = C[:, :num_occupied_orbitals]
        P = 2.0 * (C_occ @ C_occ.T)

        E_elec = 0.5 * np.einsum('uv,uv', P, H_core + F)
        E_total = E_elec + E_nuc

        delta_E = E_total - E_total_old
        rms_D = np.sqrt(np.mean((P - P_old) ** 2))

        print(f"{i + 1:<4} | {E_total: >22.12f} | {delta_E: >10.2e} | {rms_D: >10.2e}")

        if abs(delta_E) < E_conv_tol and rms_D < D_conv_tol:
            print("-" * 30)
            print(f"SCF converged after {i + 1} iterations.")
            break

        E_total_old = E_total
    else:
        print("SCF did not converge!")

    # 结果输出
    print("\n--- Final Results ---")
    print(f"Electronic Energy: {E_elec:.12f} Hartree")
    print(f"Nuclear Repulsion: {E_nuc:.12f} Hartree")
    print(f"Total RHF Energy:  {E_total:.12f} Hartree")
    print("-" * 30)
    print("Orbital Energies (Hartree):")
    for i in range(len(eps)):
        occ_status = "(Occupied)" if i < num_occupied_orbitals else "(Virtual)"
        print(f"  E({i + 1}) = {eps[i]:.8f} {occ_status}")
    print("-" * 30)
    # 调整 MO 系数的符号以获得规范形式
    for i in range(C.shape[1]):
        max_abs_idx = np.argmax(np.abs(C[:, i]))
        if C[max_abs_idx, i] < 0:
            C[:, i] *= -1
    print("Molecular Orbital Coefficients (C):")
    header = " " * 6
    for i in range(len(eps)):
        header += f"   MO {i + 1}   "
    print(header)
    for j in range(num_basis_functions):
        row = f"AO {j + 1:<2} |"
        for i in range(C.shape[1]):
            row += f" {C[j, i]: >9.6f} "
        print(row)

    return E_total, eps, C


if __name__ == "__main__":
    run_rhf()

