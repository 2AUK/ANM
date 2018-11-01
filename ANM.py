from harmonic import Harmonic_Analysis
from node import Node
import numpy as np
from scipy.spatial import distance

class ANM(Harmonic_Analysis):

    def __init__(self, input_file):
        self.nodes = []
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
                    self.nodes.append(Node(line[3], np.array([float(line[6]), \
                                                              float(line[7]), \
                                                              float(line[8])]  \
                                                              )))

    def force_deriv_2_prod_gamma(self, gamma, distance, A, B, cutoff):
        """
        Calculates second derivative of potential energy for hessian
        With gamma function implicitly multiplied
        Required for hessian matrix
        | gamma - force constant (uniform across network)
        | distance - distance between two nodes
        | A, B - the potential energy is differentiated with respect to 
        |        2 coordinates. A and B are just generalised forms and are
        |        defined as such - A_ij = (A_j - A_i) where A is either 
        |        x, y or z.
        | cutoff - a cutoff for where to determine interactions up to.
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
        """
        #Faster to work with a list of np arrays and convert to array at the end
        Hessian = []
        #Build off-diagonal elements of Hessian first
        for node_i in self.nodes:
            for node_j in self.nodes:
                #np.array[ROW, COLUMN]
                super_element = np.zeros((3,3), dtype=float)
                if node_i != node_j:
                    dist = distance.euclidean(node_i.xyz, node_j.xyz)
                    for i in range(3):
                        for j in range(3):
                            A = node_i.xyz[i] - node_j.xyz[i]
                            B = node_i.xyz[j] - node_j.xyz[j]
                            super_element[i,j] = self.force_deriv_2_prod_gamma(1.0, dist, A, B, 15.00)
                Hessian.append(super_element)
        #Build diagonal super-elements now
        size = 3*(len(self.nodes))
        Hessian = np.resize(np.asarray(Hessian), (size, size))
        Hessian = np.asarray(Hessian)
        row,col = np.diag_indices_from(Hessian)
        Hessian[row,col] = Hessian.sum(axis=0)
        return np.asarray(Hessian)

if __name__ == "__main__":
    print(ANM("pdb4cms.pdb").build_hessian())