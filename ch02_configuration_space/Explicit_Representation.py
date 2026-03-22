# 명시적 표현 (Explicit Representation)
# 자유도만큼의 파라미터로 직접 공간을 표현
# 예: S1을 각도 θ ∈ [0, 2π)로 표현

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from ch02_configuration_space.topology import *

def explicit_representation_S1():
    """S1(원)을 각도 파라미터로 명시적 표현"""
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1.0
    
    x, y = S1(theta, radius)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Explicit Representation of S¹: (cos θ, sin θ)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    return x, y