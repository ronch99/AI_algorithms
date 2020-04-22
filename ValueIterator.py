import sys
import copy

class MDP(object):

    def __init__(self, state_file:str, transition_file:str):
        # Read data from input files
        with open(state_file) as file1:
            input1 = [line.rstrip().split(',') for line in file1]
        with open(transition_file) as file2:
            input2 = [line.rstrip().split(',') for line in file2]

        # Create attributes from simple data
        self.action = ('S', 'G', 'B')
        self.state = [x.pop(1) for x in input1]
        self.reward = [float(x.pop(2)) for x in input1]
        self.policy = ['Goal' for i in range(len(self.state))]

        # Construct the transition table: map of matrix of tuples
        m = {}
        for x in input2:
            i = int(x.pop(0)) - 1 # match zero-indexing
            a = self.action.index(x.pop(0))
            
            table = []
            while len(x) > 0:
                state = int(x.pop(0)) - 1 # match zero-indexing
                prob = float(x.pop(0))
                table.append((state, prob))
            
            if i not in m:
                m[i] = []
            m[i].insert(a, table)

        self.transition = m

def ValueIterator(model:'MDP', count:int, max_error:float)->list:
    # Initialize delta and argmax function
    argmax = lambda x: x.index(max(x))

    # Read MDP model
    R = model.reward
    T = model.transition
    Uprime = [0 for i in range(len(model.state))]
    U = []

    # Evaluation
    for i in range(count):
        U = copy.deepcopy(Uprime)
        delta = 0
        r = 1

        for s in range(len(U)):
            if s in T:
                v = []
                # Sum each row
                for table in T[s]:
                    v.append(sum([U[t[0]] * t[1] for t in table]))
                
                # Pick the max utility
                Uprime[s] = R[s] + r * max(v)

                # Update policy and delta
                if Uprime[s] > U[s]:
                    model.policy[s] = model.action[argmax(v)]
                if abs(Uprime[s] - U[s]) > delta:
                    delta = abs(Uprime[s] - U[s])
            # Goal states here
            else:
                Uprime[s] = R[s]

        # Escape when converge
        if delta < max_error:
            break
    
    return U

def main(argv:list):
    model = MDP(argv[1], argv[2])
    utility = ValueIterator(model, 100, 0.0000001)
    for i in range(len(utility)):
        print('State %d: Utility = %.3f, Policy = %s'
              % (i+1, utility[i], model.policy[i]))

main(sys.argv)
