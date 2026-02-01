
import numpy as np

def check(name, fn):
    try:
        fn()
    except IndexError as e:
        print(f"{name}: {e}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")

print("\n--- Numpy ---")
n0 = np.zeros(0)
check("Numpy (0,)[0]", lambda: n0[0])

n01 = np.zeros((0, 1))
check("Numpy (0, 1)[:, 0]", lambda: n01[:, 0])
check("Numpy (0, 1)[0]", lambda: n01[0])
