# Python code to implement online learning of a dephasing channel whose dephase probability
# (assumed to vary uniformly in [P_min,P_max)}) is given by an adversary
# using generalized teleportation protocol and 
# plot regret vs time


import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
from torch import linalg
from Funs import *
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='quantum')
    parser.add_argument('--T', type=int, default=150)
    parser.add_argument('--num_const', type=int, default=2)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--itr', type=int, default=120)
    parser.add_argument('--P_max', type=float, default=0.6)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('args:', args)
    T = args.T
    num_const = args.num_const
    eta = args.eta
    itr = args.itr
    P_max1 = args.P_max 
    P_max = torch.tensor(P_max1)
    # P_max1 is just to avoid "tensor" in the saved file name
    
    ########################

    ################################

    ####### FUNCTIONS

    def loss_with_adversarially_varying_channel():

        ###########################   with trace distance

        # program states and Choi matrix of simulated channels uisng trace distance
        prg_st_using_tr_dist = torch.zeros((4,4,T+1), dtype=torch.complex128 ) # for T iterations
        # (T+1)th iteration we are not using for calculation of loss

        Choi_siml_using_tr_dist = torch.zeros((4,4,T), dtype=torch.complex128 )

        # loss function and gradient of trace distance
        loss_tr_dist = torch.zeros(T, dtype=torch.complex128 )
        grad_tr_dist = torch.zeros((4,4,T), dtype=torch.complex128 )

        # Initialize the first program state with maximally mixed state
        prg_st_using_tr_dist[:,:,0] = I_4/4

        for i in range(0,T,1):
        
            Choi_siml_using_tr_dist[:,:,i] = Choi_of_simulated_channel(prg_st_using_tr_dist[:,:,i])

            loss_tr_dist[i] = trace_norm(Choi_adv[:,:,i], Choi_siml_using_tr_dist[:,:,i])
            
            grad_tr_dist[:,:,i] = fn_grad_tr_dist(Choi_siml_using_tr_dist[:,:,i], Choi_adv[:,:,i], loss_tr_dist[i])

            prg_st_using_tr_dist[:,:,i+1] = MEGD_update(i, grad_tr_dist, log_init_prg_st, eta, num_const) # we are passing i also
        
        return loss_tr_dist


    ####################



    def min_cumul_loss_with_adversarially_varying_channel():

        # Mininmum cumulative loss (the second term in regret)
        min_cumul_loss_tr_dist = torch.zeros(T, dtype=torch.complex128 )

        for t in range(0,T,1): # t is the variable of summation in second term of regret
        
            # sum of loss functions and sum of gradients (using trace distance)
            sum_loss_tr_dist = torch.zeros(itr, dtype=torch.complex128 )
            sum_grad_tr_dist = torch.zeros((4,4,itr), dtype=torch.complex128 )

            # Initialize the first program state with maximally mixed state
            prg_st = I_4/4
            
            for i in range(0,itr,1): # i is to find pi^* (minimizer of cumulative loss at time t) iteratively for each t. 

                # loss functions and gradients
                loss_tr_dist = torch.zeros(t, dtype=torch.complex128 )
                grad_tr_dist = torch.zeros((4,4,t), dtype=torch.complex128 )

                Choi_siml = Choi_of_simulated_channel(prg_st)

                for j in range(0,t,1):
                    loss_tr_dist[j] = trace_norm(Choi_adv[:,:,j], Choi_siml)
                    grad_tr_dist[:,:,j] = fn_grad_tr_dist(Choi_siml, Choi_adv[:,:,j], loss_tr_dist[j])
                
                sum_loss_tr_dist[i] = torch.sum(loss_tr_dist,0) # adds the entries of the tensor along the dimension 0
                sum_grad_tr_dist[:,:,i] = torch.sum(grad_tr_dist,2) # adds the entries of the tensor along the dimension 2

                prg_st = MEGD_update(i, sum_grad_tr_dist/t, log_init_prg_st, eta, num_const) # we are passing i also

            min_cumul_loss_tr_dist[t] = sum_loss_tr_dist[itr-1]

            if t == 40:
                sum_loss_last_T = sum_loss_tr_dist  # to check an sum_loss is converging for arbitrary T
                # print(sum_loss_last_T)
            
            print('t value, regret term 2:', t, min_cumul_loss_tr_dist[t])   

        return min_cumul_loss_tr_dist, sum_loss_last_T

    ########################### END OF FUNCTIONS



    ######################  REQUIRED VARIABLES


    # Identity gates
    I_2 = torch.eye(2, dtype=torch.complex128 )
    I_4 = torch.eye(4, dtype=torch.complex128 )


    #########################
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

    #######################



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

    log_init_prg_st = matrix_log_for_PSD(I_4/4)

    ################## Choi matrices of dehpase channels
    # prob_dephase = torch.rand((T))  # dephase probability selected by adversary for T iterations 
    # We assume that adversary selects the probability of dephase from uniform probability distribution over [0,1)

    P_min = torch.tensor(0.2)

    prob_dephase = (P_max - P_min) * torch.rand(T) + P_min   # all the prob's are uniformly selected from [P_min, P_max]

    Choi_adv = torch.zeros((4,4,T), dtype=torch.complex128 ) # Choi matrices of the dephase channel given by adversary

    for i in range(0,T,1):

        Choi_adv[:,:,i] = Choi_adversary(prob_dephase[i]) # Computing Choi matrix of dephasing channel given by adversary

    ###################

    ######################  END OF REQUIRED VARIABLES



    #####################
    regret_part_1 = torch.cumsum( loss_with_adversarially_varying_channel().real, dim=0)

    temp_regret, sum_loss_T = min_cumul_loss_with_adversarially_varying_channel()

    regret_part_2 = temp_regret.real

    regret = regret_part_1 - regret_part_2
    #####################


    # Saving results

    torch.save(regret, '~/Downloads/results/regret_saved_'+ str(T) + 'eta_' + str(eta) + 'itr_'+str(itr) + 'P_max_'+str(P_max1))
    # torch.save(sum_loss_T.real, '~/Downloads/results/sum_loss_saved_' + str(T) + 'eta_' + str(eta) + 'itr_'+str(itr) + 'P_max_'+str(P_max1))
 
