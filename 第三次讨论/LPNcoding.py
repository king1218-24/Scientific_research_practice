import numpy as np
from scipy import sparse
import time
import sys

class TrapdooredMatrixLPN:
    """
    基于LPN假设的陷门矩阵实现
    矩阵结构: M = A * B + E (mod 2)
    其中A(rows×k), B(k×cols)是伪随机矩阵，E是稀疏噪声矩阵
    """
    
    def __init__(self, rows, cols=None, k=None, p=None, base_size=256):
        """
        初始化陷门矩阵
        :param rows: 矩阵行数
        :param cols: 矩阵列数（默认等于行数）
        :param k: 中间维度参数（默认为√min(rows,cols)）
        :param p: 噪声概率（默认为 (log₂k)² / k）
        :param base_size: 递归基础大小
        """
        # 处理列数默认值
        if cols is None:
            cols = rows
            
        self.rows = rows
        self.cols = cols
        
        # 设置k的默认值（不超过维度限制）
        self.k = min(k or max(1, int(np.sqrt(min(rows, cols)))), min(rows, cols))
        
        # 设置噪声概率p
        if p is None:
            self.p = np.log2(max(2, self.k))**2 / self.k  # 避免log(1)=0
        else:
            self.p = p
            
        self.base_size = base_size
        
        
        # 生成基础组件
        self.A = self._generate_component(self.rows, self.k)
        self.B = self._generate_component(self.k, self.cols)
        self.E = self._generate_sparse_noise(self.rows, self.cols)
        
    def _generate_component(self, rows, cols):
        """递归生成矩阵组件（A或B）"""
        # 基础情况：直接生成随机矩阵
        if rows <= self.base_size or cols <= self.base_size:
            return np.random.randint(0, 2, size=(rows, cols))
            
        # 递归情况：创建子矩阵（关键修改点）
        return TrapdooredMatrixLPN(
            rows, cols,  # 保持当前维度
            k=self.k,    # 继承父矩阵参数
            p=self.p,
            base_size=self.base_size
        )
    
    def _generate_sparse_noise(self, rows, cols):
        """生成稀疏噪声矩阵 (COO格式)"""
        nnz = int(self.p * rows * cols)  # 非零元素数量
        if nnz == 0:
            return sparse.coo_matrix((rows, cols), dtype=np.int8)
            
        rows_ind = np.random.randint(0, rows, size=nnz)
        cols_ind = np.random.randint(0, cols, size=nnz)
        data = np.ones(nnz, dtype=np.int8)
        return sparse.coo_matrix((data, (rows_ind, cols_ind)), shape=(rows, cols))
    
    def multiply(self, v):
        """快速矩阵-向量乘法: M·v = A(B·v) + E·v (mod 2)"""
        if v.shape[0] != self.cols:
            raise ValueError(f"向量长度{v.shape[0]} ≠ 矩阵列数{self.cols}")
        
        # 计算 B·v
        if isinstance(self.B, TrapdooredMatrixLPN):
            Bv = self.B.multiply(v)
        else:
            Bv = self.B @ v
        Bv %= 2
        
        # 计算 A·(B·v)
        if isinstance(self.A, TrapdooredMatrixLPN):
            ABv = self.A.multiply(Bv)
        else:
            ABv = self.A @ Bv
        ABv %= 2
        
        # 计算 E·v
        Ev = self.E.dot(v) % 2
        
        return (ABv + Ev) % 2

    def toarray(self):
        """转换为密集矩阵（用于验证）"""
        # 递归计算子矩阵
        A_dense = self.A.toarray() if isinstance(self.A, TrapdooredMatrixLPN) else self.A
        B_dense = self.B.toarray() if isinstance(self.B, TrapdooredMatrixLPN) else self.B
        
        # 计算核心乘积
        AB = np.dot(A_dense, B_dense) % 2
        
        # 添加噪声
        E_dense = self.E.toarray() if sparse.issparse(self.E) else self.E
        return (AB + E_dense) % 2

# 测试函数
def test_trapdoored_matrix(n):
    """测试陷门矩阵的生成和乘法效率"""
    np.random.seed(42)
    v = np.random.randint(0, 2, size=n)
    
    print(f"生成 {n}×{n} 矩阵 (k={int(np.sqrt(n))}, p={(np.log2(max(2, int(np.sqrt(n))))**2)/int(np.sqrt(n)):.4f})")
    
    # 生成陷门矩阵
    start = time.time()
    M = TrapdooredMatrixLPN(n)
    gen_time = time.time() - start
    print(f"生成时间: {gen_time:.10f}s")
    
    # 快速乘法
    start = time.time()
    result = M.multiply(v)
    trap_time = time.time() - start
    print(f"陷门乘法: {trap_time:.10f}s")
    
    # 直接乘法（仅小规模验证）
    
    
    dense_matrix = M.toarray()
    start = time.time()
    direct_result = dense_matrix @ v % 2
    direct_time = time.time() - start
        
        # 验证结果
    error_count = np.sum(result != direct_result)
    print(f"直接乘法: {direct_time:.10f}s | 误差数: {error_count}")
    print(f"加速比: {direct_time/trap_time:.2f}x")
    
    
    print("-" * 50)

# 运行测试
if __name__ == "__main__":
    
    
    # 测试不同规模
    for n in [1024,2048,4096,8192,16384,32768]:
        test_trapdoored_matrix(n)