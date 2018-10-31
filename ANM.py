import harmonic
import node
import numpy as np

class ANM(Harmonic_Analysis):

    def __init__(self, input_file):
        self.nodes
        self.__read_pdb(input_file)

    def __read_pdb(self, input_file):
        """
        Read an input file and generate network nodes
        The nodes are placed on the alpha carbon of each residue
        """
        with open(input_file) as pdb:
            for line in pdb:
                line = line.split()
                if line[0] == "ATOM" and line[2] == "CA":
                    #Node object -> Node(Acid_name, np.array([X, Y, Z]))
                    self.nodes.append(Node(line[3], np.array([line[6] \
                                                              line[7] \
                                                              line[8]])))

    def force_deriv_2(self, gamma, distance, A, B, cutoff):
        """
        Calculates second derivative of potential energy for hessian
        Required for hessian matrix
        | gamma - force constant (uniform across network)
        | distance - distance between two nodes
        | A, B - the potential energy is differentiated with respect to 
        |        2 coordinates. A and B are just generalised forms and are
        |        defined as such - A_ij = (A_j - A_i) where A is either 
        |        x, y or z.
        """
        if distance < cutoff:
            return -1 * gamma * A * B / (distance ** 2)
        else:
            return 0
    
    def build_hessian(self):
        """
        The hessian is an NxN matrix of 3x3 submatrices where N is the
        number of nodes. Interactions are only considered if the distance
        is lower than a cut-off (typically ~15A).

