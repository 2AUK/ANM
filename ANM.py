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
    
    def __kirchoff_matrix(self, cut_off):
        """
        Build matrix of contacts
        if the distance between two nodes is less than cut-off then the element is set to -1
        otherwise it's set to 0
        """
        Kirchoff = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if node_i != node_j:
                    dist = distance.euclidean(node_i.xyz, node_j.xyz)
                    if dist <= cut_off:
                        Kirchoff[i, j] = -1
                    else:
                        Kirchoff[i, j] = 0
        row, col = np.diag_indices_from(Kirchoff)
        Kirchoff[row, col] = -1 * Kirchoff.sum(axis = 0)
        print(Kirchoff)
        return Kirchoff

    def build_hessian(self, cut_off, gamma):
        """
        Hessian matrix builds from Kirchoff matrix
        Atilgan, A.R., Durell, S.R., Jernigan, R.L., Demirel, M.C., Keskin, O. and Bahar, I., 2001. Anisotropy of fluctuation dynamics of proteins with an elastic network model. Biophysical journal, 80(1), pp.505-515.
        for details of 2nd derivative forms
        Elements for each dimension
        3Nx3N matrix
        """
        hessian = np.zeros((3*len(self.nodes), 3*len(self.nodes)))
        K_mat = self.__kirchoff_matrix(cut_off)
        #Off-Diagonals
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                s0_ij = distance.euclidean(node_i.xyz, node_j.xyz)
                if node_i != node_j:
                    for k in range(3):
                        for l in range(3):
                            A = node_i.xyz[k] - node_j.xyz[k]
                            B = node_i.xyz[l] - node_j.xyz[l]
                            hessian[i+k,j+l] = -1.0 * gamma * A * B / (s0_ij * s0_ij)
                            print(i, j, k, l, i+k, j+l)
                else:
                    #On-Diagonals
                    for i in range(3):
                        for j in range(3):
                            pass
        return hessian

    # def build_hessian(self):
    #     """
    #     The hessian is an NxN matrix of 3x3 submatrices where N is the
    #     number of nodes. Interactions are only considered if the distance
    #     is lower than a cut-off (typically ~15A).
    #     TODO: Mass-Weight Hessian. Need to figure out what masses to use? Full residue?
    #     """
    #     #Faster to work with a list of np arrays and convert to array at the end
    #     Hessian = []
    #     #Build off-diagonal elements of Hessian first
    #     for node_i in self.nodes:
    #         for node_j in self.nodes:
    #             #np.array[ROW, COLUMN]
    #             super_element = np.zeros((3,3), dtype=float)
    #             if node_i != node_j:
    #                 dist = distance.euclidean(node_i.xyz, node_j.xyz)
    #                 for i in range(3):
    #                     for j in range(3):
    #                         A = node_i.xyz[i] - node_j.xyz[i]
    #                         B = node_i.xyz[j] - node_j.xyz[j]
    #                         super_element[i,j] = self.force_deriv_2(1.0, dist, A, B, 15.00) 
    #             Hessian.append(super_element)
    #     #Build diagonal super-elements now
    #     size = 3*(len(self.nodes))
    #     Hessian = np.resize(np.asarray(Hessian), (size, size))
    #     # Hessian[row,col] = Hessian.sum(axis=0)
    #     return np.asarray(Hessian)

if __name__ == "__main__":
    print(ANM("4ake.pdb").build_hessian(15.00, 1))
