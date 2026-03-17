# 암묵적 표현 (Implicit Representation)
# 공간의 자유도보다 높은 차원 공간에서 제약조건을 거는 방식
# 1차원 원 S1을 2차원 평면 R2에서 x^2 + y^2 - 1 = 0 으로 표현

import numpy as np
import matplotlib.pyplot as plt

def implicit_representation_S1():
    """S1(원)을 2D 평면에서 제약조건으로 암묵적 표현"""
    # 2D 그리드 생성
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)
    
    # 제약조건: g(x, y) = x^2 + y^2 - 1 = 0
    Z = X**2 + Y**2 - 1
    
    plt.figure(figsize=(6, 6))
    # 등고선에서 0인 부분만 그리기 (제약조건을 만족하는 점들)
    plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Implicit Representation of S¹: x² + y² - 1 = 0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    return X, Y, Z