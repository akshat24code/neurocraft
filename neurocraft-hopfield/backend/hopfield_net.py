import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train using Hebbian rule: W = sum(x x^T)"""
        if not patterns:
            return
        self.weights = np.zeros((self.size, self.size))
        for p in patterns:
            p = np.array(p).flatten()
            self.weights += np.outer(p, p)
        
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, mode="sync", iterations=20):
        s = np.array(input_pattern).flatten().astype(float)
        
        for _ in range(iterations):
            if mode == "sync":
                new_s = np.sign(np.dot(self.weights, s))
                new_s[new_s == 0] = 1
            else:
                new_s = s.copy()
                indices = list(range(self.size))
                np.random.shuffle(indices)
                for i in indices:
                    activation = np.dot(self.weights[i], new_s)
                    new_s[i] = 1 if activation >= 0 else -1
            
            if np.array_equal(new_s, s):
                break
            s = new_s
            
        return s

def get_letter_patterns():
    # 12x12 grid patterns
    # 1 for black, -1 for white
    def create_empty():
        return np.full((12, 12), -1)

    # Pattern for A
    a = create_empty()
    a[2:11, 2:3] = 1   # Left stem
    a[2:11, 9:10] = 1  # Right stem
    a[2:3, 3:9] = 1    # Top bar
    a[6:7, 3:9] = 1    # Mid bar
    
    # Pattern for B
    b = create_empty()
    b[1:11, 2:3] = 1   # Left stem
    b[1:2, 3:9] = 1    # Top bar
    b[5:6, 3:9] = 1    # Mid bar
    b[10:11, 3:9] = 1  # Bottom bar
    b[2:5, 9:10] = 1   # Top curve
    b[6:10, 9:10] = 1  # Bottom curve

    # Pattern for C
    c = create_empty()
    c[2:10, 2:3] = 1   # Left stem
    c[1:2, 3:10] = 1   # Top bar
    c[10:11, 3:10] = 1 # Bottom bar
    c[2:3, 9:10] = 1   # Upper tip
    c[9:10, 9:10] = 1  # Lower tip

    # Pattern for D
    d = create_empty()
    d[1:11, 2:3] = 1   # Left stem
    d[1:2, 3:9] = 1    # Top bar
    d[10:11, 3:9] = 1  # Bottom bar
    d[2:10, 9:10] = 1  # Right curve

    # Pattern for E
    e = create_empty()
    e[1:11, 2:3] = 1   # Left stem
    e[1:2, 3:10] = 1   # Top bar
    e[5:6, 3:9] = 1    # Mid bar
    e[10:11, 3:10] = 1 # Bottom bar

    return {
        "A": a.flatten().tolist(),
        "B": b.flatten().tolist(),
        "C": c.flatten().tolist(),
        "D": d.flatten().tolist(),
        "E": e.flatten().tolist()
    }
