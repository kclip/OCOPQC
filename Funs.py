# Functions to support onlineconvex optimization of PQCs main code

from turtle import end_fill
from unicodedata import decimal
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
from torch import linalg


# Function to compute dagger(transpose of conjugate) of a matrix/vector
def dagger(vector):
    # temp = vector.T.conj()  #For vectors and matrices, this is sufficient

    # # if vector dimension is more than 2 (example: tensor 2*2*2), we need to use the following
    temp = torch.transpose(torch.conj(vector),0,1)    # This is more general

    return temp


# Function to multiply three matrices in the same order
def three_mat_mul(mat1, mat2, mat3):
    temp = torch.matmul(torch.matmul(mat1, mat2), mat3)

    return temp


# Function to compute square root of a positive semi-definite matrix
def square_root(Matrix_for_squareroot):
    # print(Matrix_for_squareroot)
    eig_value, eig_vector = torch.linalg.eigh( compx_mat_round( Matrix_for_squareroot, 10 ) )
    # Here eig_value is of dtype = float64 and eig_vector is of dtype = complex128
    # and we need to convert eig_value to complex128

    eig_value = eig_value.to(torch.complex128)

    # using temp1 we can check that we reconstruct the matrix from eigen decomposition
    temp1= torch.matmul( torch.matmul(eig_vector, torch.diag(eig_value)), dagger(eig_vector))
    
    temp = torch.matmul( torch.matmul(eig_vector, torch.diag(torch.sqrt(eig_value))), dagger(eig_vector))
    
    return temp



# Function to compute natural logarithm a positive semi-definite matrix
def matrix_log_for_PSD(Matrix_for_log):
    # print(Matrix_for_squareroot)
    eig_value, eig_vector = torch.linalg.eigh( compx_mat_round( Matrix_for_log, 10 ) )
    # Here eig_value is of dtype = float64 and eig_vector is of dtype = complex128
    # and we need to convert eig_value to complex128

    eig_value = eig_value.to(torch.complex128)
    
    temp = torch.matmul( torch.matmul(eig_vector, torch.diag(torch.log(eig_value))), dagger(eig_vector))
    
    return temp




# Function to compute eigen values and projection matrices of a PSD matrix
def proj_decomp(mat):
    dimen = mat.size(dim=0) # Finding the dimension of the square matrix 
    E_val_all = torch.linalg.eigvalsh(mat)

    E_val_all = torch.round(E_val_all, decimals=8) # As we need to divide with difference of eigen values 
    # in the calculation of projection matrices, we need to round of the tensors. Otherwise they will 
    # create numerical errors

    E_val_float = torch.unique(E_val_all) # finds the unique eigen values. Also float data type is used to find sign of eigven values in our gradient formula 
    E_val = E_val_float.to(torch.complex128) # converting float64 to complex128
    no_u_E_val = E_val.size(dim=0) # no of unique eigen values
    proj_mat = torch.zeros((dimen, dimen, no_u_E_val), dtype=torch.complex128 )

    for i in range(0,no_u_E_val,1):
        proj_mat[:,:,i] = torch.eye(dimen, dtype=torch.complex128 )
        temp = torch.tensor(1, dtype=torch.complex128 )

        for j in range(0,no_u_E_val,1):
            if (i !=j) :
                proj_mat[:,:,i] = torch.matmul( proj_mat[:,:,i], (mat - (E_val[j]* I_4)) )
                temp = temp * (E_val[i] - E_val[j])

        proj_mat[:,:,i] = proj_mat[:,:,i]/temp
        proj_mat[:,:,i] = compx_mat_round(proj_mat[:,:,i], 10)
        # We can verify that this satisfies the properties of projection matrices by rounding appropriate 
        # number of decimals to avoid numerical errors

    return no_u_E_val, E_val_float, proj_mat #,E_val
        


# Function to perform POVM and return output probability and post measurement state
def POVM(Matrix_of_POVM, state):
    # probability = torch.trace( torch.matmul(Matrix_of_POVM, state)) 
    temp = square_root(Matrix_of_POVM)
    prob_times_post_mes_state = torch.matmul( temp, torch.matmul(state, temp) ) # Unnormalized post-measurement state
    # We are not dividing the above state by probability to avoid dividing by zero in some cases
    return prob_times_post_mes_state



# Function to calculation of partial trace
def partial_trace_out_first_two_qubits(state):
    out_state= torch.zeros((2,2), dtype=torch.complex128 )
    for i in range(0,4,1):
        temp1 = torch.matmul( torch.kron( dagger(two_qubit_comp_basis[:,:,i]),I_2), state )
        out_state = out_state +  torch.matmul( temp1, torch.kron( two_qubit_comp_basis[:,:,i], I_2) )
    
    del temp1
    return out_state




# Function to calculate the output of dephasing channel
def dephasing_channel(prob_dephase,state_in):
    temp = (1-prob_dephase)*state_in + prob_dephase* torch.matmul( torch.matmul( Pauli_gates[:,:,3], state_in ), Pauli_gates[:,:,3])
    
    return temp


# Function to compute Choi matrix of the channel given by adversary. As we are only considering dephasing channel 
# the input parameter to this function is probability of dephase
def Choi_adversary(prob_dephase):
    Choi_adv = torch.zeros((4,4), dtype=torch.complex128 )

    for i in range(0,2,1):
        temp1 = one_qubit_comp_basis[:,:,i]
        
        for j in range(0,2,1):
            temp2 = one_qubit_comp_basis[:,:,j]
            temp3 = torch.matmul(temp1, dagger(temp2))
            Choi_adv = Choi_adv + torch.kron( temp3, dephasing_channel(prob_dephase, temp3))
    return Choi_adv





# Function to compute Choi matrix of the simulated channel
# Each input to first qubit is one of four matrices each obtained from a computational basis vector
def Choi_of_simulated_channel(prgm_state):

    Choi_simulated = torch.zeros((4,4), dtype=torch.complex128 )

    for i in range(0,2,1):
        temp1 = one_qubit_comp_basis[:,:,i]
    
        for j in range(0,2,1):

            temp2 = one_qubit_comp_basis[:,:,j]
            qubit_in_mat = torch.matmul(temp1, dagger(temp2))
            input_state_3_qubits = torch.kron(qubit_in_mat,  prgm_state) # State of all three qubits

            qubit_out_mat = generalized_teleportation(input_state_3_qubits)            
            
            Choi_simulated = Choi_simulated + torch.kron( qubit_in_mat, qubit_out_mat) # Using the definition of Choi matrix

    # return compx_mat_round(Choi_simulated, 10)
    return Choi_simulated


# Function for generalized teleportation processor (for teleportation covariant channel)
def generalized_teleportation(input_state_3_qubits):
    qubit_out_mat = torch.zeros((2,2), dtype=torch.complex128 )
    for k in range(0,4,1):
        prob_times_output_state_3_qubits = POVM( torch.kron(Bell_mat[:,:,k], I_2), input_state_3_qubits)

        temp3 = partial_trace_out_first_two_qubits(prob_times_output_state_3_qubits) # Tracing out first 
        # two qubits(which are now classical bits) of three qubits through partial trace operation
        
        temp4 = torch.matmul( torch.matmul( dagger(Pauli_gates[:,:,k]), temp3 ), Pauli_gates[:,:,k])
        # Unitary correction at Bob. For unitary matrices dagger=inverse
        # Moreover we are using Pauli matrices for unitary correction(as dephasing channel is a teleportation-covariant channel)
        # and Pauli matrices are involutory

        qubit_out_mat = qubit_out_mat + temp4 

    return qubit_out_mat



# Function to compute trace distance between two PSD matrices
def trace_norm(matrix1,matrix2):
    matrix = matrix1-matrix2
    temp1 = torch.matmul(dagger(matrix),matrix)
    temp2 = 0.5*torch.trace(square_root(temp1))

    return temp2


# Function to compute infidelity (1-fidelity^2) distance between two PSD matrices
def in_fidelity_norm(matrix1,matrix2):
    MATRIX1 = square_root(matrix1)
    temp1 = torch.matmul( MATRIX1, torch.matmul(matrix2,MATRIX1) )
    temp2 = torch.tensor(1, dtype=torch.complex128)- torch.square(torch.trace(square_root(temp1) ) )

    return temp2



# Function to round (real and imaginary)entries of a complex matrix (vector or matrix) to desired number of decimals
def compx_mat_round(x,deci):
    for i in range(0, x.size(dim=0), 1):
        for j in range(0, x.size(dim=1), 1):
            x[i,j] = complex( torch.round( x.real[i,j], decimals=deci), torch.round( x.imag[i,j], decimals=deci))
    return x



# Function to to calculate channel mapping
def channel_mapping(state):
    output = torch.zeros((4,4), dtype=torch.complex128 )
    for i in range(0,4,1):
        temp1 = torch.kron( dagger(Pauli_gates[:,:,i]).contiguous(), Pauli_gates[:,:,i] )
        # Here .contiguous() avoids a run time error, as dagger gives a non-contiguous tensor
        output = output + three_mat_mul(temp1, state, dagger(temp1) )

    return output



# Function to compute gradient for infidelity loss at the program state
def fn_grad_infidelity(prgm_state, Choi_fixed, loss):
    temp1 = -torch.sqrt(1-loss)
    sqrt_Choi = square_root(Choi_fixed)
    temp2 = channel_mapping(prgm_state)

    temp3 = three_mat_mul(sqrt_Choi, temp2, sqrt_Choi)
    temp4 = torch.inverse( square_root(temp3) )
    temp5  = three_mat_mul(sqrt_Choi, temp4, sqrt_Choi)
    temp6 = channel_mapping(temp5) # channel is self dual for generalized teleportation protocol

    return temp1*temp6





# Function to compute gradient for trace distance loss at the program state
def fn_grad_tr_dist(Choi_siml, Choi_fixed, loss):
    mat = Choi_siml - Choi_fixed
    no_u_E_val, E_val_float, proj_mat = proj_decomp(mat)
    dimen = mat.size(dim = 0)
    grad = torch.zeros( (dimen,dimen), dtype = torch.complex128 )

    for i in range(0, no_u_E_val, 1):
        temp1 = channel_mapping(proj_mat[:,:,i])
        grad = grad + ( torch.sign(E_val_float[i]) * temp1 )

    return grad




# Function to perform MEGD update
def MEGD_update(itr_no, grad, log_prgm_state, eta, num_const):
    dimen = log_prgm_state.size(dim=0)
    herm_grad = torch.zeros( ( dimen, dimen ), dtype=torch.complex128)

    for i in range(0, itr_no, 1):
        herm_grad = herm_grad + (grad[:,:,i] + dagger(grad[:,:,i]) )/2

    temp1 = torch.matrix_exp( (num_const*I_4) + log_prgm_state - eta*herm_grad )

    return temp1/torch.trace(temp1)



###########################


# Identity gates
I_2 = torch.eye(2, dtype=torch.complex128 )
I_4 = torch.eye(4, dtype=torch.complex128 )



# Pauli gates (all in one tensor)
Pauli_gates = torch.zeros((2,2,4), dtype=torch.complex128 )

# Identity gate: Pauli_gates[:,:,0]
Pauli_gates[0,0,0] = 1
Pauli_gates[1,1,0] = 1

# X_gate: Pauli_gates[:,:,1]
Pauli_gates[0,1,1] = 1
Pauli_gates[1,0,1] = 1

# Y_gate: Pauli_gates[:,:,2]
Pauli_gates[0,1,2] = -(1.j)
Pauli_gates[1,0,2] = (1.j)

# Z_gate: Pauli_gates[:,:,3]
Pauli_gates[0,0,3] = 1
Pauli_gates[1,1,3] = -1




########################
# One qubit computational basis in one tensor
one_qubit_comp_basis = torch.zeros((2,1,2), dtype=torch.complex128 )

one_qubit_comp_basis[0,0,0] = 1
one_qubit_comp_basis[1,0,1] = 1
########################


########################
# Two qubit computational basis in one tensor
two_qubit_comp_basis = torch.zeros((4,1,4), dtype=torch.complex128 )

two_qubit_comp_basis[0,0,0] = 1
two_qubit_comp_basis[1,0,1] = 1
two_qubit_comp_basis[2,0,2] = 1
two_qubit_comp_basis[3,0,3] = 1
########################




#######################
# Bell_vectors: maximally entanglled state (vector form) (in the order of definition of Bell measurement) (all in one tensor)
Bell_vec = torch.zeros((4,1,4), dtype=torch.complex128 )   

#Phi^+
Bell_vec[0,0,0] = torch.sqrt(torch.tensor(0.5))
Bell_vec[3,0,0] = torch.sqrt(torch.tensor(0.5))

#Psi^+
Bell_vec[1,0,1] = torch.sqrt(torch.tensor(0.5))
Bell_vec[2,0,1] = torch.sqrt(torch.tensor(0.5))

#Psi^-
Bell_vec[1,0,2] = torch.sqrt(torch.tensor(0.5))
Bell_vec[2,0,2] = -torch.sqrt(torch.tensor(0.5))

#Phi^-
Bell_vec[0,0,3] = torch.sqrt(torch.tensor(0.5))
Bell_vec[3,0,3] = -torch.sqrt(torch.tensor(0.5))


# Bell_matrices: maximally entanglled state (density matrix form). These define POVM of Bell measurement
Bell_mat = torch.zeros((4,4,4), dtype=torch.complex128 )

for i in range(0,4,1):
    Bell_mat[:,:,i] = torch.matmul( Bell_vec[:,:,i], dagger(Bell_vec[:,:,i]) )


# First two qubits are at Alice and last qubit is at Bob
# Measuring first two qubits (qubits at Alice) in the generalized teleportation processor

# If the POVM_matrix is Bell_mat[:,:,i] use Pauli_gates[:,:,i] as unitary correction(Uni_corr) where i \in \{0,1,2,3\}

POVM_matrix = Bell_mat[:,:,1]  # Bell measurement on two qubits at Alice
Uni_corr = Pauli_gates[:,:,1]  # Unitary correction on the qubit at Bob
###################################
