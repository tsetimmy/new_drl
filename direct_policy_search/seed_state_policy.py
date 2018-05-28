import numpy as np
import pickle
from tf_bayesian_model import random_seed_state

def main():
    random_state = random_seed_state()
    random_policy = np.random.uniform(-2., 2., 100)

    pickle.dump(random_state, open("random_state.p", "wb"))
    pickle.dump(random_policy, open("random_policy.p", "wb"))

if __name__ == '__main__':
    main()
