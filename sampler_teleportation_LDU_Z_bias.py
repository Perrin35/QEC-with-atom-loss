from typing import Optional,Tuple,Any
import stim
import numpy as np
from surface_code_teleportation_LDU import Rotated_Surface_Code_teleportation_LDU
import pymatching
import warnings
import pickle
import re
class sampler_teleportation_LDU:
    """
    Class to sample atom loss (and standard Pauli error) in the rotated surface code equipped with the teleportation LDU
    
    """
    
    def __init__(
        self,
        lattice:Rotated_Surface_Code_teleportation_LDU,
        flatten_loops:Optional[bool]=True,
        
        
    ):
        """
        Args:
            lattice (Class Rotated_Surface_Code)           
            
            flatten_loops(Boolean): Default to True. If the attribute circuit_without_loss should be implemented with or without loops.
        """
        self.lattice=lattice
        self.flatten_loops=flatten_loops
        self.circuit_without_loss=lattice.rotated_surface_code_circuit(flatten_loops=flatten_loops)
        
       
    def atoms_lost_position_list(self,num_shots):
        """
        Generates a list of atom loss positions for a given number of shots.
    
        Args:
            num_shots (int): Number of shots to simulate.
    
        Returns:
            list: A list containing the positions of lost atoms for each shot. Each entry is a list of tuples,
                  where each tuple specifies the qubit index, round index, and CZ gate position (-1 indicates
                  that the fresh atom has been lost at a previous round).
        """
        # Extract lattice properties
        lattice=self.lattice
        dw=lattice.width
        dh=lattice.height
        rounds=lattice.rounds
        error_rate=lattice.loss_rate
        position_atoms_lost=[]

        # Get qubit indices from the lattice
        corner_indices_data=lattice.corner_indices_data_qubits()
        edge_indices_data=lattice.horizontal_edge_indices_data_qubits()+lattice.vertical_edge_indices_data_qubits()
        bulk_indices_data=lattice.bulk_indices_data_qubits()    
        edge_indices_ancilla=lattice.edge_indices_ancilla_z_qubits()+lattice.edge_indices_ancilla_x_qubits()
        bulk_indices_ancilla=lattice.bulk_indices_ancilla_z_qubits()+lattice.bulk_indices_ancilla_x_qubits()

        # Generate probabilities for different qubit groups
        # Corner data probabilities
        proba_corner_data_CZ=[error_rate*(1-error_rate)**t for t in range(3)]
        p_tot_corner=1-sum(proba_corner_data_CZ)
        proba_corner_data_list=[p_tot_corner] + proba_corner_data_CZ 
        rand_corner_data=np.random.RandomState().choice([i for i in range(4)], (num_shots,4,rounds-1), p=proba_corner_data_list)

        # Edge data probabilities
        proba_edge_data_CZ=[error_rate*(1-error_rate)**t for t in range(4)]
        p_tot_edge=1-sum(proba_edge_data_CZ)
        proba_edge_data_list=[p_tot_edge] + proba_edge_data_CZ
        rand_edge_data=np.random.RandomState().choice([i for i in range(5)], (num_shots,2*(dw-2)+2*(dh-2),rounds-1), p=proba_edge_data_list)

        # Bulk data probabilities
        proba_bulk_data_CZ=[error_rate*(1-error_rate)**t for t in range(5)]
        p_tot_bulk=1-sum(proba_bulk_data_CZ)
        proba_bulk_data_list=[p_tot_bulk] + proba_bulk_data_CZ
        rand_bulk_data=np.random.RandomState().choice([i for i in range(6)], (num_shots,(dw-2)*(dh-2),rounds-1), p=proba_bulk_data_list)

        # Last round probabilities
        last_round_proba_corner_data_list=[(1-error_rate)**2]+[error_rate*(1-error_rate)**t for t in range(2)]
        last_round_rand_corner_data=np.random.RandomState().choice([i for i in range(3)], (num_shots,4), p=last_round_proba_corner_data_list)

        last_round_proba_edge_data_list=[(1-error_rate)**3]+[error_rate*(1-error_rate)**t for t in range(3)]
        last_round_rand_edge_data=np.random.RandomState().choice([i for i in range(4)], (num_shots,2*(dw-2)+2*(dh-2)), p=last_round_proba_edge_data_list)

        last_round_proba_bulk_data_list=[(1-error_rate)**4]+[error_rate*(1-error_rate)**t for t in range(4)]
        last_round_rand_bulk_data=np.random.RandomState().choice([i for i in range(5)], (num_shots,(dw-2)*(dh-2)), p=last_round_proba_bulk_data_list)
        
        #Generate random loss for ancilla qubits
        rand_loss_ancilla=np.random.RandomState().choice([i for i in range(2)], (num_shots,dw*dh,rounds-1), p=[1-error_rate,error_rate])

        # Ancilla edge and bulk probabilities
        proba_edge_ancilla_list=[(1-error_rate)**2]+[error_rate*(1-error_rate)**t for t in range(2)]
        rand_edge_ancilla=np.random.RandomState().choice([i for i in range(3)], (num_shots,len(edge_indices_ancilla),rounds), p=proba_edge_ancilla_list)

        proba_bulk_ancilla_list=[(1-error_rate)**4]+[error_rate*(1-error_rate)**t for t in range(4)]
        rand_bulk_ancilla=np.random.RandomState().choice([i for i in range(5)], (num_shots,len(bulk_indices_ancilla),rounds), p=proba_bulk_ancilla_list)

        # Aggregate all loss positions
        for loss_ancilla, corners_data , last_round_corners_data , edges_data , last_round_edges_data , bulk_data , last_round_bulk_data , edges_ancilla , bulk_ancilla in zip( rand_loss_ancilla, rand_corner_data , last_round_rand_corner_data , rand_edge_data , last_round_rand_edge_data , rand_bulk_data , last_round_rand_bulk_data , rand_edge_ancilla , rand_bulk_ancilla):
            position_temp=[]

            # Process corner data
            for i,positions in enumerate(corners_data):
                for ro,position in enumerate(positions):
                    if ro>0 and loss_ancilla[corner_indices_data[i],ro-1]==1:
                        position_temp.append((corner_indices_data[i],ro,-1))
                    elif position>0:
                        position_temp.append((corner_indices_data[i],ro,position-1))

                # Handle last round
                position=last_round_corners_data[i]
                if loss_ancilla[corner_indices_data[i],rounds-2]==1:
                    position_temp.append((corner_indices_data[i],rounds-1,-1))
                elif position>0:
                    position_temp.append((corner_indices_data[i],rounds-1,position-1))
                               
            # Process edge data
            for i,positions in enumerate(edges_data):
                for ro,position in enumerate(positions):
                    if ro>0 and loss_ancilla[edge_indices_data[i],ro-1]==1:
                        position_temp.append((edge_indices_data[i],ro,-1))
                    elif position>0:
                        position_temp.append((edge_indices_data[i],ro,position-1))

                # Handle last round
                position=last_round_edges_data[i]
                if loss_ancilla[edge_indices_data[i],rounds-2]==1:
                    position_temp.append((edge_indices_data[i],rounds-1,-1))
                elif position>0:
                    position_temp.append((edge_indices_data[i],rounds-1,position-1))

            # Process bulk data      
            for i,positions in enumerate(bulk_data):
                for ro,position in enumerate(positions):
                    if ro>0 and loss_ancilla[bulk_indices_data[i],ro-1]==1:
                        position_temp.append((bulk_indices_data[i],ro,-1))
                    elif position>0:
                        position_temp.append((bulk_indices_data[i],ro,position-1))

                # Handle last round
                position=last_round_bulk_data[i]
                if loss_ancilla[bulk_indices_data[i],rounds-2]==1:
                    position_temp.append((bulk_indices_data[i],rounds-1,-1))
                elif position>0:
                    position_temp.append((bulk_indices_data[i],rounds-1,position-1))
                            
            # Process edge ancilla
            for i,positions in enumerate(edges_ancilla):
                for ro,position in enumerate(positions):
                    if position!=0:
                        position_temp.append((edge_indices_ancilla[i],ro,position-1))

            # Process bulk ancilla
            for i,positions in enumerate(bulk_ancilla):
                for ro,position in enumerate(positions):
                    if position!=0:
                        position_temp.append((bulk_indices_ancilla[i],ro,position-1))
                        
            position_atoms_lost.append(position_temp)
        return position_atoms_lost 


    def heralded_z_errors(self,num_shots):
        """
        Simulates heralded Z errors for a given number of shots.
    
        Args:
            num_shots (int): Number of shots to simulate.
    
        Returns:
            numpy.ndarray: A 3D boolean array of shape (num_shots, rounds-1, n_data),
                           where each entry indicates whether a Z error is heralded (True) or not (False).
        """
        # Extract lattice parameters
        lattice=self.lattice
        n_data=lattice.n_data
        rounds=lattice.rounds

        # Generate random Z errors. Each qubit in each round has a 50% probability of error.
        return np.random.rand(num_shots, rounds-1, n_data) < 0.5

    @staticmethod
    def new_rec_detectors(instruction,offset_list):
        """
        Adjusts detector records with a given list of offsets.
    
        Args:
            instruction (stim.CircuitInstruction): An instruction from the circuit (must be a DETECTOR).
            offset_list (list): List of offsets to apply to each target index.
    
        Returns:
            list: New targets with the offsets applied.
    
        Raises:
            Exception: If the instruction is not a DETECTOR or if the lengths of offset_list
                       and target indices do not match.
        """

        # Validate the instruction type
        name=instruction.name
        targets=instruction.targets_copy()
        args=instruction.gate_args_copy()
        if name!='DETECTOR':
            raise Exception('This is not a detector instruction.')

        # Extract target indices and ensure the offset list matches in length
        index_list=[trgts.value for trgts in targets]
        if len(offset_list)!=len(index_list):
            raise Exception('offset_list is of length {} but must be of length {}'.format(len(offset_list),len(index_list)))

        # Apply offsets to the target indices
        reindex_list=[index-offset for index,offset in zip(index_list,offset_list)]

        new_targets=[stim.target_rec(index) for index in reindex_list]
        return new_targets

    def atoms_loss_and_heralded_z_errors_circuit(self,atoms_lost_position=[],heralded_z_errors_list = [],proba_z_error=1.):
        """
        Modifies the circuit by suppressing operations after qubit loss and incorporating heralded Z errors.
    
        Args:
            atoms_lost_position (list): Specifies which qubits were lost, when, and at which position.
                                        Each entry is a tuple (qubit_index, round_index, CZ_gate_position).
            heralded_z_errors_list (list): Boolean array indicating heralded Z errors (True for error, False otherwise).
                                           Shape is (round-1, n_data).
            proba_z_error (float): Probability of a Z error occurring after it is heralded. Defaults to 1.
        Returns:
            stim.Circuit: The modified circuit with operations suppressed for lost qubits and heralded Z errors applied.
    
        Raises:
            Exception: If `flatten_loops` is not enabled for the lattice.
        """
        # Ensure the circuit is flattened (no REPEAT blocks)
        if not self.flatten_loops:
            raise Exception('The flatten_loops option should be enabled for that method to work')

               
        else:
            # Extract lattice properties and convert lost positions to a NumPy array
            circuit=self.circuit_without_loss
            lattice=self.lattice
            rounds=lattice.rounds
            n_data=lattice.n_data
            np_atoms_lost_position=np.array(atoms_lost_position)

            # Get qubit indices from the lattice
            corner_indices_data=lattice.corner_indices_data_qubits()
            edge_indices_data=lattice.horizontal_edge_indices_data_qubits()+lattice.vertical_edge_indices_data_qubits()
            bulk_indices_data=lattice.bulk_indices_data_qubits()    

            # Default to no heralded Z errors if none are provided
            if len(heralded_z_errors_list)==0:
                np_heralded_z_errors=np.zeros((rounds-1,n_data))
            else:
                np_heralded_z_errors=np.array(heralded_z_errors_list)
            #List all the operations that the qubit lost can experienced (for now it is manually). Sort by the type operation (single/two qubit gate/noise channel).
            qubit_op1=['R','RX','H']
            qubit_noise1=['DEPOLARIZE1','PAULI_CHANNEL_1','X_ERROR','Y_ERROR','Z_ERROR']
            qubit_op2=['CZ']
            qubit_noise2=['DEPOLARIZE2','PAULI_CHANNEL_2']
            new_round_instruction=['SHIFT_COORDS']

           
            # Initialize a new circuit and state-tracking variables
            circuit2=stim.Circuit()
            current_round=-1
            new_round=True
            qubits_lost_now=[]
            det=0
            
            # Process each instruction in the original circuit
            for instruction in circuit:
                name=instruction.name
                targets=instruction.targets_copy()
                args=instruction.gate_args_copy()
                
                # Handle new round setup
                if new_round:
                    # Apply heralded Z errors for the previous round
                    if current_round>=0 and current_round<rounds-1:
                        heralded_z_errors_now=[data_qubits for data_qubits,erasure in enumerate(np_heralded_z_errors[current_round]) if erasure]
                        circuit2.append('Z_ERROR',heralded_z_errors_now,proba_z_error)
                        
                    # reset lost qubits in the previous round
                    lost_reset=[ind for ind,pos  in qubits_lost_now if ind<n_data]
                    lost_tel_LDU=[]
                    for ind, pos in qubits_lost_now:
                        if ind in corner_indices_data and pos==2:
                            lost_tel_LDU.append(ind)
                        elif ind in edge_indices_data and pos==3:
                            lost_tel_LDU.append(ind)
                        elif ind in bulk_indices_data and pos==4:
                            lost_tel_LDU.append(ind)
                    
                    if len(lost_reset)!=0:
                        circuit2.append('R',lost_reset)
                        circuit2.append('X_ERROR', lost_tel_LDU,0.5)

                    # Track ancilla losses
                    ancilla_lost_before2=[]
                    for index,pos in qubits_lost_now:
                        if index>=n_data:
                            if len(ancilla_lost_before)!=0 and index in ancilla_lost_before[:,0]:
                                j=np.where(ancilla_lost_before[:,0]==index)[0][0]
                                ancilla_lost_before2.append(ancilla_lost_before[j])
                            else:
                                ancilla_lost_before2.append([index,current_round])
                   
                    ancilla_lost_before=np.array(ancilla_lost_before2)

                    # Update round and lost qubits
                    new_round=False
                    current_round+=1
                    #create a table with only the qubits lost in that round
                    if len(np_atoms_lost_position)!=0:
                        qubits_lost_now=np.array([[index,pos] for index,ro,pos in  np_atoms_lost_position if ro==current_round])

                    else:
                        qubits_lost_now=np.array([])
                   
                    position_qubits=[0 for i in range(len(qubits_lost_now))]

                
                #if no qubits have been lost in that round
                if len(qubits_lost_now)==0:
                    if name in new_round_instruction:
                        det=0
                        circuit2.append(name,targets,args)
                        new_round=True 

                    #modify detectors if ancilla qubits have been lost in previous rounds
                    elif len(ancilla_lost_before)!=0 and name=='DETECTOR':
                        if det==0:
                            det=1
                            if current_round==rounds and lattice.readout_direction=='X':
                                index_det=n_data+lattice.n_ancilla_z
                            else:
                                index_det=n_data
                        else:
                            index_det+=1


                        if index_det in ancilla_lost_before[:,0]:
                            i=np.where(ancilla_lost_before[:,0]==index_det)[0][0]
                            lost_round=ancilla_lost_before[i,1]
                            if lost_round==0:
                                if (((lattice.initial_state=='|0>' or lattice.initial_state=='|1>') and index_det<lattice.n_data+lattice.n_ancilla_z)
                                or ((lattice.initial_state=='|+>' or lattice.initial_state=='|->') and index_det>=lattice.n_data+lattice.n_ancilla_z)):
                                    new_targets=targets[1:]
                                    
                                else:
                                    new_targets=[]
                            else:
                                offset=(current_round-lost_round)*(lattice.n_ancilla_z+lattice.n_ancilla_x)
                                new_targets=self.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

                            circuit2.append(name,new_targets,args)
                            
                        else:
                            circuit2.append(name,targets,args)
                            
                    else:
                        det=0
                        circuit2.append(name,targets,args)

                        
                    

                else:

                    #modify detectors if ancilla qubits have been lost
                    if name=='DETECTOR':
                        if det==0:
                            det=1
                            if ((current_round==0 and (lattice.initial_state=='|+>' or lattice.initial_state=='|->'))
                                or (current_round==rounds and lattice.readout_direction=='X')):
                                index_det=n_data+lattice.n_ancilla_z
                            else:
                                index_det=n_data                                
                            
                        else:
                            index_det+=1
                        if index_det in qubits_lost_now[:,0]:
                            circuit2.append(name,[],args)
                            
                        elif len(ancilla_lost_before)!=0 and index_det in ancilla_lost_before[:,0]:
                            i=np.where(ancilla_lost_before[:,0]==index_det)[0][0]
                            lost_round=ancilla_lost_before[i,1]
                            if lost_round==0:
                                if (((lattice.initial_state=='|0>' or lattice.initial_state=='|1>') and index_det>=lattice.n_data+lattice.n_ancilla_z) 
                                    or ((lattice.initial_state=='|+>' or lattice.initial_state=='|->') and index_det<lattice.n_data+lattice.n_ancilla_z)):
                                    new_targets=[]
                                    
                            
                                else:
                                    new_targets=targets[1:]
                                    
                            else:
                                offset=(current_round-lost_round)*(lattice.n_ancilla_z+lattice.n_ancilla_x)
                                new_targets=self.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

                            circuit2.append(name,new_targets,args)

                        else:
                            circuit2.append(name,targets,args)
                    else:
                        det=0
                    #if it is single qubit operation, erase it if the target qubit has been lost before the operation
                        if name in qubit_op1 or name in qubit_noise1:
                            indices=[trgts.value for trgts in targets]
                            for qubit_index in indices:
    
                                if qubit_index in qubits_lost_now[:,0]:
                                    j=np.where(qubits_lost_now[:,0]==qubit_index)[0][0]
                                    if position_qubits[j]>qubits_lost_now[j,1]:
                                        indices.remove(qubit_index)
                            circuit2.append(name,indices,args)
    
                        #if the target qubit has been lost before the operation, it erases the indices of the two qubits involves in the operation.
                        elif name in qubit_op2:
                            indices=[trgts.value for trgts in targets]
                            indices2=indices.copy()
                            remaining_qubits = []
                            for qubit_index in indices:
                                if qubit_index in qubits_lost_now[:,0]:
                                    j=np.where(qubits_lost_now[:,0]==qubit_index)[0][0]
                                    if qubit_index in indices2:
                                        q=indices2.index(qubit_index)
                                        if position_qubits[j]>=qubits_lost_now[j,1]:
                                            indices2.remove(qubit_index)
                                            if q%2==0:
                                                if position_qubits[j]==qubits_lost_now[j,1]:
                                                    remaining_qubits.append(indices2[q])
                                                #supress the other qubit index on which act on the CZ gate
                                                indices2.pop(q)
                                            else:
                                                if position_qubits[j]==qubits_lost_now[j,1]:
                                                    remaining_qubits.append(indices2[q-1])
                                                indices2.pop(q-1)

                                            
                                            
    
                                    position_qubits[j]+=1
    
                            circuit2.append(name,indices2,args)
                            circuit2.append('Z_ERROR',remaining_qubits,0.5)
    
                        #if the target qubit has been lost before the noise channel, it erases the indices of the two qubits involves in the operation and create a new 1-qubit noise channel corresponding to the original noise channel projected on the remaning qubit
                        elif name in qubit_noise2:
                            indices=[trgts.value for trgts in targets]
                            indices2=indices.copy()
                            remaining_qubits=[[],[]]
                            for qubit_index in indices:
                                if qubit_index in qubits_lost_now[:,0]:
                                    j=np.where(qubits_lost_now[:,0]==qubit_index)[0][0]
                                    if qubit_index in indices2:
                                        q=indices2.index(qubit_index)
                                        if position_qubits[j]>qubits_lost_now[j,1]:
                                            indices2.remove(qubit_index)
                                            if q%2==0:             #supress the other qubit index on which act on the CZ gate
                                                remaining_qubits[0].append(indices2[q])
                                                indices2.pop(q)
                                            else:
                                                remaining_qubits[1].append(indices2[q-1])
                                                indices2.pop(q-1)
    
                            circuit2.append(name,indices2,args)
                                            
                            #build the corresponding noise channel
                            if name=='DEPOLARIZE2':
                                error_model,proba=lattice.trace_noise_on_1qubit(('DEPOLARIZE2',args),qubit=0)
                                circuit2.append(error_model,remaining_qubits[0]+remaining_qubits[1],proba)
    
                            elif name=='PAULI_CHANNEL_2':
                                error_model1,proba1=lattice.trace_noise_on_1qubit(('PAULI_CHANNEL_2',args),qubit=0)
                                error_model2,proba2=lattice.trace_noise_on_1qubit(('PAULI_CHANNEL_2',args),qubit=1)
                                if np.all(proba1==proba2):
                                    circuit2.append(error_model1,remaining_qubits[0]+remaining_qubits[1],proba1)
    
                                else:
                                    circuit2.append(error_model1,remaining_qubits[0],proba1)
                                    circuit2.append(error_model2,remaining_qubits[1],proba2)
                         
                        elif name in new_round_instruction:
                            new_round=True
                            circuit2.append(name,targets,args)
                        
                        
    
                            
                        else:
                            circuit2.append(name,targets,args)
                        
                        
                    
            return circuit2


    def modify_detector_error_model(self,atoms_lost_position=None):
        """
        Changes the causal structure of the detectors due to the loss of ancilla qubits.
    
        Args:
            atoms_lost_position (list or numpy.ndarray): A list of tuples or a numpy array specifying:
                - Which qubits have been lost (qubit index),
                - When they were lost (round index),
                - Where in the CZ gate cycle the loss occurred (CZ gate position, -1 for undetected prior loss).
    
        Returns:
            stim.Circuit: A modified circuit reflecting the changes in detector error models due to qubit losses.
    
        Raises:
            Exception: If the `flatten_loops` attribute is not enabled for the circuit.
        """
        # Ensure the circuit uses flattened loops.
        if not self.flatten_loops:
            raise Exception('The flatten_loops option should be enabled for that method to work')

        # Start with the unmodified circuit
        circuit=self.circuit_without_loss

        # If no qubit loss information is provided, return the unmodified circuit.
        if atoms_lost_position==None:
            return circuit
        else:
            # Extract lattice properties and convert lost positions to a NumPy array
            lattice=self.lattice
            rounds=lattice.rounds
            n_data=lattice.n_data
            np_atoms_lost_position=np.array(atoms_lost_position)

            # Return the unmodified circuit if no qubits are lost
            if  len(np_atoms_lost_position)==0:
                return circuit
            
            # Define operations indicating the start of a new round
            new_round_instruction=['SHIFT_COORDS']

           
            # Create a new empty circuit to store modified instructions
            circuit2=stim.Circuit()
            
            # Initialize variables for tracking the current round, lost qubits, and detector state
            current_round=-1
            new_round=True
            qubits_lost_now=[]
            det=0

            # Loop through the original circuit's instructions
            for instruction in circuit:
                name=instruction.name
                targets=instruction.targets_copy()
                args=instruction.gate_args_copy()

                # Handle the start of a new round
                if new_round:

                    # Tracks ancilla qubits lost in earlier rounds
                    ancilla_lost_before2=[]
                    for index,pos in qubits_lost_now:
                        if index>=n_data:
                            if len(ancilla_lost_before)!=0 and index in ancilla_lost_before[:,0]:
                                j=np.where(ancilla_lost_before[:,0]==index)[0][0]
                                ancilla_lost_before2.append(ancilla_lost_before[j])
                            else:
                                ancilla_lost_before2.append([index,current_round])
                   
                    ancilla_lost_before=np.array(ancilla_lost_before2)

                    # Move to the next round
                    new_round=False
                    current_round+=1
                    
                  
                    # Identify qubits lost in the current round
                    qubits_lost_now=np.array([[index,pos] for index,ro,pos in  np_atoms_lost_position if ro==current_round])
                    position_qubits=[0 for i in range(len(qubits_lost_now))]

                
                # Handle the case where no qubits are lost in the current round
                if len(qubits_lost_now)==0:
                    if name in new_round_instruction:
                        det=0
                        circuit2.append(name,targets,args)
                        new_round=True 

                    # Modify detectors based on previously lost ancilla qubits
                    elif len(ancilla_lost_before)!=0 and name=='DETECTOR':
                        if det==0:
                            det=1
                            if current_round==rounds and lattice.readout_direction=='X':
                                index_det=n_data+lattice.n_ancilla_z
                            else:
                                index_det=n_data
                        else:
                            index_det+=1


                        if index_det in ancilla_lost_before[:,0]:
                            i=np.where(ancilla_lost_before[:,0]==index_det)[0][0]
                            lost_round=ancilla_lost_before[i,1]
                            if lost_round==0:
                                if (((lattice.initial_state=='|0>' or lattice.initial_state=='|1>') and index_det<lattice.n_data+lattice.n_ancilla_z)
                                or ((lattice.initial_state=='|+>' or lattice.initial_state=='|->') and index_det>=lattice.n_data+lattice.n_ancilla_z)):
                                    new_targets=targets[1:]
                                    
                                else:
                                    new_targets=[]
                            else:
                                offset=(current_round-lost_round)*(lattice.n_ancilla_z+lattice.n_ancilla_x)
                                new_targets=self.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

                            circuit2.append(name,new_targets,args)
                            
                        else:
                            circuit2.append(name,targets,args)
                            
                    else:
                        det=0
                        circuit2.append(name,targets,args)

                        
                    

                else:

                    # Handle cases where qubits are lost in the current round
                    if name=='DETECTOR':
                        if det==0:
                            det=1
                            if ((current_round==0 and (lattice.initial_state=='|+>' or lattice.initial_state=='|->'))
                                or (current_round==rounds and lattice.readout_direction=='X')):
                                index_det=n_data+lattice.n_ancilla_z
                            else:
                                index_det=n_data                                
                            
                        else:
                            index_det+=1
                        if index_det in qubits_lost_now[:,0]:
                            circuit2.append(name,[],args)
                            
                        elif len(ancilla_lost_before)!=0 and index_det in ancilla_lost_before[:,0]:
                            i=np.where(ancilla_lost_before[:,0]==index_det)[0][0]
                            lost_round=ancilla_lost_before[i,1]
                            if lost_round==0:
                                if (((lattice.initial_state=='|0>' or lattice.initial_state=='|1>') and index_det>=lattice.n_data+lattice.n_ancilla_z) 
                                    or ((lattice.initial_state=='|+>' or lattice.initial_state=='|->') and index_det<lattice.n_data+lattice.n_ancilla_z)):
                                    new_targets=[]
                                    
                            
                                else:
                                    new_targets=targets[1:]
                                    
                            else:
                                offset=(current_round-lost_round)*(lattice.n_ancilla_z+lattice.n_ancilla_x)
                                new_targets=self.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

                            circuit2.append(name,new_targets,args)

                        else:
                            circuit2.append(name,targets,args)
                    else:
                        det=0
                         
                        if name in new_round_instruction:
                            new_round=True
                            circuit2.append(name,targets,args)
                        
                    
                        else:
                            circuit2.append(name,targets,args)
                        
                        
                    
            return circuit2



    def sampler(self, position,heralded_z_errors):
        """
        Simulates the detection events and observable flips in a quantum error correction circuit,
        given a specific set of qubit losses and heralded Z errors.
    
        Args:
            position (list of tuples): A list where each tuple specifies:
                - The index of the lost qubit,
                - The round in which the qubit was lost,
                - The CZ gate position where it was lost.
                A CZ gate position of -1 indicates the qubit was lost in an earlier round.
            heralded_z_errors (list or numpy.ndarray): A boolean array indicating which qubits experience
                heralded Z errors in each round.
    
        Returns:
            tuple: A pair `(detection_events, observable_flips)`:
                - `detection_events` (list): Detection events recorded during the circuit execution.
                - `observable_flips` (list): Flips in the logical observable(s).
        """
        # Step 1: Generate a modified circuit based on qubit losses and heralded Z errors.
        # This suppresses operations on lost qubits and adds heralded Z errors to the circuit.
        new_circuit=self.atoms_loss_and_heralded_z_errors_circuit(atoms_lost_position=position,heralded_z_errors_list=heralded_z_errors)

        # Step 2: Modify the detector error model to account for lost ancilla qubits.
        # This adjusts the causal structure of the detectors to reflect the qubit losses.
        new_detector=self.modify_detector_error_model(position)

        # Step 3: Compile the modified detector model into a converter.
        # The converter maps measurement results to detection events and observable flips.
        converter=new_detector.compile_m2d_converter()

        # Step 4: Compile the modified circuit into a sampler for execution.
        sampler = new_circuit.compile_sampler()

        # Step 5: Sample measurements from the circuit.
        # The `shots=1` argument indicates that we want a single sample.
        meas2=sampler.sample(shots=1)

         # Step 6: Convert the sampled measurements into detection events and observable flips.
        detection_events,observable_flips=converter.convert(measurements=meas2,separate_observables=True)

        return detection_events,observable_flips
        
    