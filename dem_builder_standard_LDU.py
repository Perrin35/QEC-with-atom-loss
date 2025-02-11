from typing import Optional,Tuple,Any
import stim
import numpy as np
from surface_code_standard_LDU import Rotated_Surface_Code_standard_LDU
from sampler_standard_LDU import sampler_standard_LDU
import pymatching
import warnings
import pickle
class DEM_builder:
    """
    Class to construct the DEM corresponding to the rotated surface code equipped with the standard LDU
    
    """
    
    def __init__(
        self,
        lattice:Rotated_Surface_Code_standard_LDU,
        flatten_loops:Optional[bool]=True,
        no_observable:Optional[bool]=False
        
    ):
        """
        Args:
            lattice (Class Rotated_Surface_Code)           
            
            flatten_loops(Boolean): Default to True. If the attribute circuit_without_loss should be implemented with or without loops.
            
            no_observable(Boolean): Default to False. The attribute circuit_without_loss should be implemented with or without observable.
            
        """
        self.lattice=lattice
        self.flatten_loops=flatten_loops
        self.no_observable=no_observable
        self.circuit_without_loss=lattice.rotated_surface_code_circuit(flatten_loops=flatten_loops,no_observable=no_observable)
        
        
    def atom_loss_circuit_without_noise(self,atoms_lost_position=None):
        """
        Modifies the circuit to suppress operations after the loss of qubits without Pauli noise.
    
        Args:
            atoms_lost_position (list or numpy.ndarray): Specifies:
                - The index of the lost qubit (first position of the tuple),
                - The round in which the qubit was lost (second position),
                - The CZ gate position where it was lost (third position, -1 if lost in an earlier round).
    
        Returns:
            stim.Circuit: A modified circuit with operations suppressed for lost qubits and without noise.
    
        Raises:
            Exception: If `flatten_loops` is not enabled.
        """

        # Ensure that the circuit uses flattened loops to handle qubit losses.
        if not self.flatten_loops:
            raise Exception('The flatten_loops option should be enabled for that method to work')
        
        # Start with the circuit without noise.
        circuit=self.circuit_without_loss.without_noise()

        # If no atom loss information is provided, return the noise-free circuit.
        if atoms_lost_position==None:
            return circuit
        else:
            # Step 1: Initialize lattice properties and handle lost qubits.
            lattice=self.lattice
            sampler=sampler_standard_LDU(lattice)
            rounds=lattice.rounds
            n_data=lattice.n_data

            # If there are no lost qubits, return the original circuit.
            np_atoms_lost_position=np.array(atoms_lost_position)
            if  len(np_atoms_lost_position)==0:
                return circuit
            
            #Step 2: List all the operations that the qubit lost can experienced (for now it is manually). Sort by the type operation (single/two qubit gate/noise channel).
            qubit_op1=['R','RX','H']
            qubit_op2=['CZ']
            new_round_instruction=['SHIFT_COORDS']

            # Step 3: Initialize state-tracking variables for circuit modification.
            position_qubits=[0 for i in range(len(np_atoms_lost_position))]# Tracks qubit positions  in the round
            circuit2 = stim.Circuit()  # A new circuit to store modified instructions.
            current_round = -1  # Tracks the current round.
            new_round = True  # Indicates the start of a new round.
            qubits_lost_now = []  # Tracks qubits lost in the current round.
            det = 0  #Tracks whether this is the first detector instruction of the round

            
            # Step 4: Iterate over each instruction in the circuit.
            for instruction in circuit:
                name=instruction.name
                targets=instruction.targets_copy()
                args=instruction.gate_args_copy()

                # Handle the start of a new round.
                if new_round:
                    # Reset data qubits that were lost in the previous round.
                    lost_reset=[ind for ind,pos  in qubits_lost_now if ind<n_data]
                    if len(lost_reset)!=0:
                        circuit2.append('R',lost_reset)

                    # Update ancilla losses from previous rounds.
                    ancilla_lost_before2=[]
                    for index,pos in qubits_lost_now:
                        if index>=n_data:
                            if len(ancilla_lost_before)!=0 and index in ancilla_lost_before[:,0]:
                                j=np.where(ancilla_lost_before[:,0]==index)[0][0]
                                ancilla_lost_before2.append(ancilla_lost_before[j])
                            else:
                                ancilla_lost_before2.append([index,current_round])
                   
                    ancilla_lost_before=np.array(ancilla_lost_before2)

                    # Move to the next round and update tracking variables.
                    new_round=False
                    current_round+=1
                    
                  
                    #create a table with only the qubits lost in that round
                    qubits_lost_now=np.array([[index,pos] for index,ro,pos in  np_atoms_lost_position if ro==current_round])
                    position_qubits=[0 for i in range(len(qubits_lost_now))]

                
                # Handle instructions when no qubits are lost in the current round.
                if len(qubits_lost_now)==0:
                    if name in new_round_instruction:
                        det=0
                        circuit2.append(name,targets,args)
                        new_round=True 

                    # Modify detectors for previously lost ancilla qubits.
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
                                new_targets=sampler.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

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
                                new_targets=sampler.new_rec_detectors(instruction,[offset]+[0]*(len(targets)-1))

                            circuit2.append(name,new_targets,args)

                        else:
                            circuit2.append(name,targets,args)
                    else:
                        det=0
                    #if it is single qubit operation, erase it if the target qubit has been lost before the operation
                        if name in qubit_op1:
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
                            for qubit_index in indices:
                                if qubit_index in qubits_lost_now[:,0]:
                                    j=np.where(qubits_lost_now[:,0]==qubit_index)[0][0]
                                    if qubit_index in indices2:
                                        q=indices2.index(qubit_index)
                                        if position_qubits[j]>=qubits_lost_now[j,1]:
                                            indices2.remove(qubit_index)
                                            if q%2==0:             #supress the other qubit index on which act on the CZ gate
                                                indices2.pop(q)
                                            else:
                                                indices2.pop(q-1)
                                            
    
                                    position_qubits[j]+=1
    
                            circuit2.append(name,indices2,args)
                         
                        elif name in new_round_instruction:
                            new_round=True
                            circuit2.append(name,targets,args)
                        
                        
    
                            
                        else:
                            circuit2.append(name,targets,args)
                        
                        
                    
            return circuit2
    
    
    def flip_observable(self,i,round_position):
        """
        Determines whether a lost qubit induces a logical observable flip
    
        Args:
            i (int): The index of the qubit.
            round_position (int): The round index.
    
        Returns:
            bool: True if a flip occurs, False otherwise.
    
        Raises:
            Exception: If the qubit index is invalid or the readout direction is not 'X' or 'Z'.
        """
        lattice=self.lattice

        # Check the readout direction and apply rules based on 'Z' or 'X'.
        if lattice.readout_direction=='Z':
            if i<lattice.n_data: # Data qubits
                if i%lattice.width==0: # Leftmost qubits in the lattice
                    if round_position==0: # First round
                        if lattice.initial_state=='|0>' or lattice.initial_state=='|1>':
                            return True
                    elif round_position+1<=lattice.rounds: # Intermediate rounds
                        return True
                    else:
                        return False
                else:
                    return False
            
            elif i<lattice.n_data+lattice.n_ancilla_z: # Ancilla Z qubits
                if (i-lattice.n_data)%(lattice.width-1)==0: # Left edge qubits
                    return True
                else:
                    return False
                
            elif i<lattice.n_data+lattice.n_ancilla_z+lattice.n_ancilla_x: # Ancilla X qubits
                if (i-lattice.n_data-lattice.n_ancilla_z)%((lattice.width+1)/2)==0: # Left edge qubits
                    return True
                else:
                    return False
            else:
                raise Exception('Qubit index does not exist')
                

        elif lattice.readout_direction=='X':
            if i<lattice.n_data: # Data qubits
                if i<lattice.width: # Topmost qubits in the lattice
                    if round_position==0: # First round
                        if lattice.initial_state=='|+>' or lattice.initial_state=='|->': # Initial state in X-basis
                            return True
                    elif round_position+1<=lattice.rounds: # Intermediate rounds
                        return True
                    else:
                        return False
                else:
                    return False
                
            elif i<lattice.n_data+lattice.n_ancilla_z: # Ancilla Z qubits
                if (i-lattice.n_data)<(lattice.width-1): # Top edge qubits
                    return True
                else:
                    return False
                
            elif i<lattice.n_data+lattice.n_ancilla_z+lattice.n_ancilla_x:# Ancilla X qubits
                if (i-lattice.n_data-lattice.n_ancilla_z)<(lattice.width+1)/2:#Top edge qubits
                    return True
                else:
                    return False
            else:
                raise Exception('Qubit index does not exist')
            
        else:
            raise Exception('Readout direction is either "X" or "Z"')
    
   
    def detectors_targets_flip_observable(self):
        """
        Identifies detector targets associated with logical observable flips.
    
        Returns:
            list: A list of `stim.DemTarget` objects representing the detector indices
                  for logical observable flips across all rounds.
        """
        lattice=self.lattice
        dw=lattice.width
        det_indices=[]

        # Handle the 'Z' readout direction.
        if lattice.readout_direction=='Z':
            
            # Identify stabilizer indices for Z-basis
            stab_indices=[i for i in range(lattice.n_ancilla_z)]

            # Add initial state detectors for Z-basis
            if lattice.initial_state=='|0>' or lattice.initial_state=='|1>':
                det_indices+=stab_indices
                offset=lattice.n_ancilla_z
                offset0 = offset

        # Handle the 'X' readout direction.
        elif lattice.readout_direction == 'X':
            # Identify stabilizer indices for X-basis
            stab_indices=[i for i in range(lattice.n_ancilla_x)]

            # Add initial state detectors for X-basis
            if lattice.initial_state=='|+>' or lattice.initial_state=='|->':
                det_indices+=stab_indices
                offset0=lattice.n_ancilla_x
                offset=lattice.n_ancilla_x+lattice.n_ancilla_z

        # Add detector indices for intermediate rounds.
        for r in range(1,lattice.rounds):
            det_indices+=[stab_index+offset+(lattice.n_ancilla_x+lattice.n_ancilla_z)*(r-1) for stab_index in stab_indices]

        # Add final round detectors.
        det_indices+=[stab_index+offset0+(lattice.n_ancilla_x+lattice.n_ancilla_z)*(lattice.rounds-1) for stab_index in stab_indices]

        # Convert indices to `stim.DemTarget` objects.
        det_targets=[stim.DemTarget.relative_detector_id(index) for index in det_indices]

        return det_targets

    


    def add_weighted_dem(self,weighted_dems,add_observable=False,only_errors=True):
        """
        Combines multiple weighted detector error models (DEMs) into a single model. 
        The input DEMs should not contain repeat block instructions.
    
        Args:
            weighted_dems (list of tuples): A list of tuples where:
                - The first element is the Detector Error Model (DEM),
                - The second element is the probability of the DEM,
                - The third element is a boolean indicating if it flips the observable.
            add_observable (bool, optional): Defaults to False. If True, adds observable flips
                to error mechanisms.
            only_errors (bool, optional): Defaults to True. If True, only 'error' instructions
                are copied. Non-error instructions are ignored.
    
        Returns:
            stim.DetectorErrorModel: The combined weighted DEM.
    
        Raises:
            Exception: If a repeat block instruction is present in any DEM, or if logical observables
                already exist and `add_observable` is set to True.
        """
        if add_observable:
            det_targs=self.detectors_targets_flip_observable()

        # Initialize data structures for tracking instructions and detectors.
        instruction_dict={}
        detectors_list=[]
        shift_indices=[]
        max_shift=0

        # Process each weighted DEM
        for d,(dem, proba,flip_obs) in enumerate(weighted_dems):
            current_shift=0 #Reset shift for the current DEM.

            # Iterate through instructions in the DEM.
            for instr in dem:
                typ=instr.type
                if typ=='repeat':
                    raise Exception('No repeat block instruction allowed. Use flattened method before providing it as an argument.')

                args=instr.args_copy()
                targets=instr.targets_copy()

                # Handle non-error instructions if `only_errors` is False.
                if typ!='error' and not only_errors:
                    if d==0:# For the first DEM, initialize detector instructions.
                        if typ=='shift_detectors':
                            max_shift+=1
                            detectors_list.append((typ,args,targets))
                            shift_indices.append(len(detectors_list)-1)
                        elif typ=='detector'or typ=='logical_observable':
                            detectors_list.append((typ,args,targets))
                        else:
                            raise Exception('Instruction not recognized.')
                    else:# For subsequent DEMs, compare and insert instructions as needed.
                        if typ=='shift_detectors':
                            current_shift+=1
                            if current_shift>max_shift:
                                max_shift=current_shift
                                detectors_list.append((typ,args,targets))
                                shift_indices.append(len(detectors_list)-1)
                        elif typ=='detector' or typ=='logical_observable':
                            if (typ,args,targets) not in detectors_list:
                                if current_shift==max_shift:
                                    detectors_list.append((typ,args,targets))
                                else:
                                    detectors_list.insert(shift_indices[current_shift+1]-1,(typ,args,targets))
                        else:
                            raise Exception('Instruction not recognized.')

                # Handle 'error' instructions.
                elif typ=='error':
                    if add_observable and flip_obs:
                        if self.contain_logical_observable(targets):
                            raise Exception('Logical observable already implemented in DEM. add_observable should be set to False')
                        else:
                            # Break targets into separate components.
                            list_targets=self.break_targets_using_separators(targets)
                            for targs in list_targets:
                                t=0
                                for targ in targs:
                                    if targ in det_targs:
                                        t+=1
                                if t%2==1: # Add logical observable flip if needed.
                                    targs.append(stim.DemTarget.logical_observable_id(0))

                            targets=self.join_targets_using_separators(list_targets)
                            
                    # Accumulate probabilities for the same target set.
                    try:
                        instruction_dict[tuple(targets)]+=proba*args[0]

                    except KeyError:
                        instruction_dict[tuple(targets)]=proba*args[0]

        # Create the final combined DEM.
        final_dem=stim.DetectorErrorModel()

        # Add all accumulated 'error' instructions to the final DEM.
        for targets in instruction_dict.keys():
            final_dem.append('error',instruction_dict[targets],targets)

        # Add non-error instructions if `only_errors` is False.
        if not only_errors:
            for typ,args,targets in detectors_list:
                final_dem.append(typ,args,targets)

        return final_dem

    

    def build_dem_lost_data_qubits(self,save_path):
        """
        Constructs and saves detector error models (DEMs) for data qubits lost in a quantum error correction process.
    
        Args:
            save_path (str): Path where the DEM files will be saved. Each file corresponds to a specific
                             qubit and round position.
    
        Returns:
            None: The method generates and saves DEM files for all data qubits and their loss positions.
    
        Raises:
            Exception: If the class was not initialized with `no_observable=True` or `flatten_loops=True`.
        """
        
        # Validate class initialization settings.
        if not self.no_observable:
            raise Exception('when the class Rotated_Surface_Code_atoms_loss is initialized, the option no_observable should be set to True')
        if not self.flatten_loops:
            raise Exception('when the class Rotated_Surface_Code_atoms_loss is initialized, the option fallten_loops should be set to True')

        # Retrieve lattice properties.
        lattice=self.lattice
        p=lattice.loss_rate
        
        #probabilities of loss detection error in the measurement due to noise on CZ gates
        sampler=sampler_standard_LDU(lattice)
        err_det=sampler.detection_error()
        err_no_det=sampler.no_detection_error()
       
        # Retrieve indices of different qubit types.
        corner_indices_data=lattice.corner_indices_data_qubits()
        horizontal_edge_indices_data=lattice.horizontal_edge_indices_data_qubits()
        vertical_edge_indices_data=lattice.vertical_edge_indices_data_qubits()
        bulk_indices_data=lattice.bulk_indices_data_qubits() 

        # Generate all possible (qubit, round) positions for data qubits.
        positions_data=[(i,r) for i in range(lattice.n_data) for r in range(lattice.rounds)]

        # Process each qubit and its round position.
        for qubit,round_position in positions_data:
            flip_obs=self.flip_observable(qubit,round_position)
            final_dem_list=[]

            # Handle corner data qubits.
            if qubit in corner_indices_data:
                num_CZ=3
                proba_corner_data_standard_CZ=[p*(1-p)**t for t in range(2)]
                proba_corner_data_detection_CZ=[p*(2-p-(1-p)**3)*(1-p)**2/(1-(1-p)**2+(1-p)**4),p*(1-p)**5/(1-(1-p)**2+(1-p)**4)]
                
                p_tot_corner=1-sum(proba_corner_data_standard_CZ)-sum(proba_corner_data_detection_CZ)
                if round_position==lattice.rounds-1:
                    proba_corner_data_list_detected=[proba for proba in proba_corner_data_standard_CZ] + [proba_corner_data_detection_CZ[0]+ proba_corner_data_detection_CZ[1]/2]
                else:
                    proba_corner_data_list_detected=[proba*(1-err_no_det) for proba in proba_corner_data_standard_CZ] + [p_tot_corner*err_det+proba_corner_data_detection_CZ[0]*(1-err_no_det) + proba_corner_data_detection_CZ[1]/2]

                proba_corner_data_list_not_detected=[proba*err_no_det for proba in proba_corner_data_standard_CZ] + [proba_corner_data_detection_CZ[0]*err_no_det + proba_corner_data_detection_CZ[1]/2]

                
                dems_temp2=[]
                proba_list=[]
                for round_lost in range(round_position):
                    
                    for CZ_pos in range(num_CZ):
                        dems_temp1=[]
                        dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_lost,CZ_pos]]).detector_error_model(
                        decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        for rounds in range(round_lost+1,round_position+1):
                            dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,rounds,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        
                        dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                        if round_position!=lattice.rounds-1:
                            proba_list.append(proba_corner_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1) * (1-err_no_det))
                        else:
                            proba_list.append(proba_corner_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1))
                        

                for CZ_pos in range(num_CZ):
                    dems_temp1=[]
                    dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_position,CZ_pos]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                    dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                    
                    proba_list.append(proba_corner_data_list_detected[CZ_pos]) 

                for last_round_detected in range(-1,round_position):
                    proba_tot=sum([sum([proba_list[num_CZ*round_lost+CZ_pos] for CZ_pos in range(num_CZ)]) * (p_tot_corner*(1-err_det)) ** (round_lost-last_round_detected-1) for round_lost in range(last_round_detected+1,round_position+1)])

                    
                    if proba_tot!=0:
                        final_dem_list=[]
                        for round_lost in range(last_round_detected+1,round_position+1):
                            proba_not_lost=(p_tot_corner*(1-err_det))**(round_lost-last_round_detected-1)
                            for CZ_pos in range(num_CZ):
                                final_proba=proba_list[num_CZ*round_lost+CZ_pos]*proba_not_lost/proba_tot
                                if final_proba!=0:
                                    final_dem_list.append((dems_temp2[num_CZ*round_lost+CZ_pos],final_proba,False))
                            
                        if len(final_dem_list)!=0:
                            final_dem=self.add_weighted_dem(final_dem_list)
                            with open(save_path+'_qubit={}_lost_round_detected={}_earlier_round_detected={}'.format(qubit,round_position,last_round_detected)+'.dem', 'w') as f:
                                final_dem.to_file(f)
                    
            # Handle edge data qubits.
            elif qubit in horizontal_edge_indices_data or qubit in vertical_edge_indices_data:
                num_CZ=4
                proba_edge_data_standard_CZ=[p*(1-p)**t for t in range(3)]
                proba_edge_data_detection_CZ=[p*(2-p-(1-p)**3)*(1-p)**3/(1-(1-p)**2+(1-p)**4),p*(1-p)**6/(1-(1-p)**2+(1-p)**4)]
                
                p_tot_edge=1-sum(proba_edge_data_standard_CZ)-sum(proba_edge_data_detection_CZ)
                if round_position==lattice.rounds-1:
                    proba_edge_data_list_detected=[proba for proba in proba_edge_data_standard_CZ] + [proba_edge_data_detection_CZ[0]+ proba_edge_data_detection_CZ[1]/2]
                else:
                    proba_edge_data_list_detected=[proba*(1-err_no_det) for proba in proba_edge_data_standard_CZ] + [p_tot_edge*err_det+proba_edge_data_detection_CZ[0]*(1-err_no_det) + proba_edge_data_detection_CZ[1]/2]
                proba_edge_data_list_not_detected=[proba*err_no_det for proba in proba_edge_data_standard_CZ] + [proba_edge_data_detection_CZ[0]*err_no_det + proba_edge_data_detection_CZ[1]/2]


                dems_temp2=[]
                proba_list=[]
                for round_lost in range(round_position):
                    
                    for CZ_pos in range(num_CZ):
                        dems_temp1=[]
                        dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_lost,CZ_pos]]).detector_error_model(
                        decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        for rounds in range(round_lost+1,round_position+1):
                            dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,rounds,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        
                        dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                        if round_position==lattice.rounds-1:
                            proba_list.append(proba_edge_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1))
                        else:
                            proba_list.append(proba_edge_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1) * (1-err_no_det))
                        

                for CZ_pos in range(num_CZ):
                    dems_temp1=[]
                    dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_position,CZ_pos]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                    dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                    proba_list.append(proba_edge_data_list_detected[CZ_pos]) 

                for last_round_detected in range(-1,round_position):
                    proba_tot=sum([sum([proba_list[num_CZ*round_lost+CZ_pos] for CZ_pos in range(num_CZ)]) * (p_tot_edge*(1-err_det)) ** (round_lost-last_round_detected-1) for round_lost in range(last_round_detected+1,round_position+1)])
                    if proba_tot!=0:
                        final_dem_list=[]
                        for round_lost in range(last_round_detected+1,round_position+1):
                            proba_not_lost=(p_tot_edge*(1-err_det))**(round_lost-last_round_detected-1)
                            for CZ_pos in range(num_CZ):
                                final_proba=proba_list[num_CZ*round_lost+CZ_pos]*proba_not_lost/proba_tot
                                if final_proba!=0:
                                    final_dem_list.append((dems_temp2[num_CZ*round_lost+CZ_pos],final_proba,False))
                            
                        if len(final_dem_list)!=0:
                            final_dem=self.add_weighted_dem(final_dem_list)
                            with open(save_path+'_qubit={}_lost_round_detected={}_earlier_round_detected={}'.format(qubit,round_position,last_round_detected)+'.dem', 'w') as f:
                                final_dem.to_file(f)
                    
              
            # Handle bulk data qubits.
            elif qubit in bulk_indices_data:
                num_CZ=5
                proba_bulk_data_standard_CZ=[p*(1-p)**t for t in range(4)]
                proba_bulk_data_detection_CZ=[p*(2-p-(1-p)**3)*(1-p)**4/(1-(1-p)**2+(1-p)**4),p*(1-p)**7/(1-(1-p)**2+(1-p)**4)]
                
                p_tot_bulk=1-sum(proba_bulk_data_standard_CZ)-sum(proba_bulk_data_detection_CZ)
                if round_position==lattice.rounds-1:
                    proba_bulk_data_list_detected=[proba for proba in proba_bulk_data_standard_CZ] + [proba_bulk_data_detection_CZ[0]+ proba_bulk_data_detection_CZ[1]/2]
                else:
                    proba_bulk_data_list_detected=[proba*(1-err_no_det) for proba in proba_bulk_data_standard_CZ] + [p_tot_bulk*err_det+proba_bulk_data_detection_CZ[0]*(1-err_no_det) + proba_bulk_data_detection_CZ[1]/2]
                proba_bulk_data_list_not_detected=[proba*err_no_det for proba in proba_bulk_data_standard_CZ] + [proba_bulk_data_detection_CZ[0]*err_no_det + proba_bulk_data_detection_CZ[1]/2]

                
                dems_temp2=[]
                proba_list=[]
                for round_lost in range(round_position):
                    
                    for CZ_pos in range(num_CZ):
                        dems_temp1=[]
                        dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_lost,CZ_pos]]).detector_error_model(
                        decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        for rounds in range(round_lost+1,round_position+1):
                            dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,rounds,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                        
                        dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                        if round_position==lattice.rounds-1:
                            proba_list.append(proba_bulk_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1) )
                        else:
                            proba_list.append(proba_bulk_data_list_not_detected[CZ_pos] * err_no_det ** (round_position-round_lost-1) * (1-err_no_det))
                        

                for CZ_pos in range(num_CZ):
                    dems_temp1=[]
                    dems_temp1.append((self.atom_loss_circuit_without_noise([[qubit,round_position,CZ_pos]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True), 1,flip_obs))
                    dems_temp2.append(self.add_weighted_dem(dems_temp1,add_observable=True))
                    proba_list.append(proba_bulk_data_list_detected[CZ_pos]) 

                for last_round_detected in range(-1,round_position):
                    proba_tot=sum([sum([proba_list[num_CZ*round_lost+CZ_pos] for CZ_pos in range(num_CZ)]) * (p_tot_bulk*(1-err_det)) ** (round_lost-last_round_detected-1) for round_lost in range(last_round_detected+1,round_position+1)])
                    if proba_tot!=0:
                        final_dem_list=[]
                        for round_lost in range(last_round_detected+1,round_position+1):
                            proba_not_lost=(p_tot_bulk*(1-err_det))**(round_lost-last_round_detected-1)
                            for CZ_pos in range(num_CZ):
                                final_proba=proba_list[num_CZ*round_lost+CZ_pos]*proba_not_lost/proba_tot
                                if final_proba!=0:
                                    final_dem_list.append((dems_temp2[num_CZ*round_lost+CZ_pos],final_proba,False))
                            
                        if len(final_dem_list)!=0:
                            
                            # Combine all DEMs for this qubit and round.
                            final_dem=self.add_weighted_dem(final_dem_list)

                            #Save the resulting DEM to a file.
                            with open(save_path+'_qubit={}_lost_round_detected={}_earlier_round_detected={}'.format(qubit,round_position,last_round_detected)+'.dem', 'w') as f:
                                final_dem.to_file(f)
                    
      
            else:
                raise Exception('qubit_index must be a positive integer smaller than {}'.format(lattice.n_data))

           
    
    def build_dem_lost_ancilla_qubits(self,save_path):
        """
        Constructs and saves detector error models (DEMs) for ancilla qubits lost in a quantum error correction process.
    
        Args:
            save_path (str): Path where the DEM files will be saved. Each file corresponds to a specific
                             ancilla qubit and round position.
    
        Returns:
            None: The method generates and saves DEM files for all ancilla qubits and their loss positions.
    
        Raises:
            Exception: If the class was not initialized with `no_observable=True` or `flatten_loops=True`,
                       or if an invalid qubit index is encountered.
        """

        # Ensure proper class initialization settings.
        if not self.no_observable:
            raise Exception('when the class Rotated_Surface_Code_atoms_loss is initialized, the option no_observable should be set to True')
        if not self.flatten_loops:
            raise Exception('when the class Rotated_Surface_Code_atoms_loss is initialized, the option fallten_loops should be set to True')

        # Retrieve lattice properties and parameters.
        lattice=self.lattice
        p=lattice.loss_rate

        # Retrieve indices of ancilla qubits.
        edge_indices_ancilla=lattice.edge_indices_ancilla_z_qubits()+lattice.edge_indices_ancilla_x_qubits()
        bulk_indices_ancilla=lattice.bulk_indices_ancilla_z_qubits()+lattice.bulk_indices_ancilla_x_qubits()

         # Generate all possible (ancilla qubit, round) positions.
        positions_ancilla=[(i,r) for i in range(lattice.n_data,lattice.n_data+lattice.n_ancilla_z+lattice.n_ancilla_x) for r in range(lattice.rounds)]

        # Process each ancilla qubit and its round position.
        for qubit,round_position in positions_ancilla:
            flip_obs=self.flip_observable(qubit,round_position)
            final_dem=stim.DetectorErrorModel()
            dems=[]

            # Handle edge ancilla qubits.
            if qubit in edge_indices_ancilla:
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True), p/(1-(1-p)**2),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)/(1-(1-p)**2),flip_obs))
                
            # Handle bulk ancilla qubits.         
            elif qubit in bulk_indices_ancilla:
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p/(1-(1-p)**4),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)/(1-(1-p)**4),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)**2/(1-(1-p)**4),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)**3/(1-(1-p)**4),flip_obs))
            else:
                raise Exception('qubit_index must be a positive integer smaller than {}'.format(lattice.n_data))

            # Combine all weighted DEMs for the current ancilla qubit and round.
            final_dem=self.add_weighted_dem(dems,add_observable=True)
            
            # Add an edge between the detectors associated to this ancilla at this round and the next round.
            shift_det=stim.DetectorErrorModel()
            det1=self.get_detector(qubit,round_position)
            det2=self.next_detector(qubit,round_position)
            if det1!=None:
                if det2!=None:
                    shift_det.append('error',[0.5],[stim.target_relative_detector_id(det1),stim.target_relative_detector_id(det2)])
                else:
                    shift_det.append('error',[0.5],[stim.target_relative_detector_id(det1)])
            else:
                if det2!=None:
                    shift_det.append('error',[0.5],[stim.target_relative_detector_id(det2)])
                
            final_dem+=shift_det
            with open(save_path+'_qubit={}_round={}'.format(qubit,round_position)+'.dem', 'w') as f:
                final_dem.to_file(f)


    
    def get_detector(self,qubit,round_position):
        """
        Retrieves the detector index associated with a specific ancilla qubit at a given round.
    
        Args:
            qubit (int): The index of the ancilla qubit.
            round_position (int): The round index.
    
        Returns:
            int or None: The detector index if it exists, or None if no detector is associated with the given qubit and round.
    
        Raises:
            Exception: If the qubit index is not an ancilla or if the round index exceeds the allowed number of rounds.
        """
        lattice=self.lattice

        # Validate that the qubit is an ancilla qubit.
        if qubit>=lattice.n_data and qubit<lattice.n_data+lattice.n_ancilla_x+lattice.n_ancilla_z:

            # Determine the offset based on the initial state.
            if lattice.initial_state=='|0>' or lattice.initial_state=='|1>':
                # Handle the first round for Z-basis initialization.
                if round_position==0:
                    if qubit<lattice.n_data+lattice.n_ancilla_z:
                        return qubit-lattice.n_data
                    else:
                        return None
                    
                else:
                    offset=lattice.n_ancilla_z
                    
            elif lattice.initial_state=='|+>' or lattice.initial_state=='|->':
                # Handle the first round for X-basis initialization.
                if round_position==0:
                    if qubit>=lattice.n_data+lattice.n_ancilla_z:
                        return qubit-lattice.n_data-lattice.n_ancilla_z
                    else:
                        return None
                    
                else:
                    offset=lattice.n_ancilla_x

            # Compute the detector index for intermediate or final rounds.
            if round_position<lattice.rounds:
                return (round_position-1)*(lattice.n_ancilla_z+lattice.n_ancilla_x)+offset+qubit-lattice.n_data
            
            elif round_position==lattice.rounds:
                # Handle the last round (data qubit measurements) based on the readout direction.
                if lattice.readout_direction=='Z':
                    if qubit<lattice.n_data+lattice.n_ancilla_z:
                        return (round_position-1)*(lattice.n_ancilla_z+lattice.n_ancilla_x)+offset+qubit-lattice.n_data
                    else:
                        return None
                if lattice.readout_direction=='X':
                    if qubit>=lattice.n_data+lattice.n_ancilla_z:
                        return (round_position-1)*(lattice.n_ancilla_z+lattice.n_ancilla_x)+offset+qubit-lattice.n_data-lattice.n_ancilla_z
                    else:
                        return None
            else:
                raise Exception('Too many rounds')
            
        
        else:
            raise Exception('qubit index is not an ancilla')
                
                
    def next_detector(self,qubit,round_position):
        """
        Retrieves the detector index associated with the next round for a given ancilla qubit.
    
        Args:
            qubit (int): The index of the ancilla qubit.
            round_position (int): The round index (0-indexed).
    
        Returns:
            int or None: The detector index for the next round if it exists, or None if it does not exist.
    
        Raises:
            Exception: If the qubit index is not an ancilla.
        """
        lattice=self.lattice

        # Validate that the qubit is an ancilla qubit.
        if qubit>=lattice.n_data and qubit<lattice.n_data+lattice.n_ancilla_x+lattice.n_ancilla_z:

            # Determine the offset based on the initial state.
            if lattice.initial_state=='|0>' or lattice.initial_state=='|1>':
                offset=lattice.n_ancilla_z
                    
            elif lattice.initial_state=='|+>' or lattice.initial_state=='|->':
                offset=lattice.n_ancilla_x

            # Compute the detector index for the next round.
            if round_position<lattice.rounds-1:
                return round_position*(lattice.n_ancilla_x+lattice.n_ancilla_z)+offset+qubit-lattice.n_data

            # Handle the final round based on the readout direction.
            elif round_position==lattice.rounds-1:
                if lattice.readout_direction=='Z':
                    if qubit<lattice.n_data+lattice.n_ancilla_z:
                        return round_position*(lattice.n_ancilla_x+lattice.n_ancilla_z)+offset+qubit-lattice.n_data
                    else:
                        return None
                
                elif lattice.readout_direction=='X':
                    if qubit>=lattice.n_data+lattice.n_ancilla_z:
                        return round_position*(lattice.n_ancilla_x+lattice.n_ancilla_z)+offset+qubit-lattice.n_data-lattice.n_ancilla_z
                    else:
                        return None
            else:
                return None
            
        
        else:
            raise Exception('qubit index is not an ancilla')
                
             
    

    def num_detectors(self):
        """
        Computes the total number of detectors in the circuit.
    
        Returns:
            int: The total number of detectors, including initial and final state detectors,
                 as well as intermediate round detectors.
    
        Raises:
            Exception: If the initial state or readout direction is invalid.
        """
        lattice=self.lattice

        # Determine the number of initial state detectors based on the initial state.
        if lattice.initial_state=='|0>' or lattice.initial_state=='|1>':
            ini_det=lattice.n_ancilla_z
        elif lattice.initial_state=='|+>' or lattice.initial_state=='|->':
            ini_det=lattice.n_ancilla_x

        else:
            raise Exception('initial state does not exist, must be either [0>, |1>, |+> or |->')

         # Determine the number of final state detectors based on the readout direction.
        if lattice.readout_direction=='X':
            final_det=lattice.n_ancilla_x
        elif lattice.readout_direction=='Z':
            final_det=lattice.n_ancilla_z
        else:
            raise Exception('readout direction does not exist, must be either X or Z')
            
        return ini_det+final_det+(lattice.rounds-1)*(lattice.n_ancilla_x+lattice.n_ancilla_z)            


    def extract_erasure(self, positions: list[list[int]]) -> list[list[int]]:
        """
        Extracts the round where the lost qubit has been detected
    
        Args:
            positions (list[list[int]]): 
                A list of 3-element lists, where each inner list contains:
                    - `qubit` (int): The index of the lost qubit.
                    - `round_position` (int): The round in which the qubit was lost.
                    - `CZ_pos` (int): The position of the CZ gate where the qubit was lost. 
                      If `CZ_pos` is -1, the qubit was lost in an earlier round and not detected
                The list must be sorted based on the round and CZ position.
    
        Returns:
            list[list[int]]:
                A list of detection events, where each detection event is represented as:
                    - `qubit` (int): The index of the lost qubit.
                    - `detected_round_position` (int): The round position where the loss has been detected
                      indicated by `CZ_pos = -1`.
        """
        lattice=self.lattice
        detection_lost_qubits=[]
        for (p,[qubit,round_position,CZ_pos]) in enumerate(positions):
            if qubit>=lattice.n_data:
                detection_lost_qubits.append([qubit,round_position])
            else:
                if CZ_pos>=0:
                    k=1
                    while p+k<len(positions) and positions[p+k][2]==-1:
                        k+=1
                    detection_lost_qubits.append([qubit,round_position+k-1])

        return detection_lost_qubits
                
    
    def build_dem_lost_qubits(self,position,path_dem):
        """
        Builds a detector error model (DEM) for lost qubits based on their positions and rounds from the stored DEM generated for each indivdual loss
    
        Args:
            position (list of lists): A list where each element is a list containing:
                - `qubit` (int): Index of the lost qubit,
                - `round_position` (int): The round in which the qubit was lost,
                - `CZ_pos` (int): Position in the CZ gate where the loss occurred.
            path_dem (str): Path prefix for DEM files corresponding to qubit losses.
    
        Returns:
            stim.DetectorErrorModel: The combined DEM for all lost qubits in the provided positions.
    
        Raises:
            NotImplementedError: If the lattice uses an unsupported gate for noise modeling.
        """
        lattice=self.lattice
        erasure=self.extract_erasure(position)
        dem=stim.DetectorErrorModel()
        
        # Process each lost qubit position.
        for (p,[qubit,round_position]) in enumerate(erasure):
            # Ancilla qubits.
            if qubit>=lattice.n_data:
                dem+= stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_qubit={}_round={}'.format(lattice.loss_rate,qubit,round_position)+'.dem')

            else:
                # Data qubits.
                if p!=0:
                    if erasure[p-1][0]!=qubit:
                        r0=-1
                    else:
                        r0=erasure[p-1][1]

                else:
                    r0=-1
                    
                if lattice.after_CZ==None:
                    dem+=stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_depo_noise=0_qubit={}_lost_round_detected={}_earlier_round_detected={}'.format(lattice.loss_rate,qubit,round_position,r0)+'.dem')
                elif lattice.after_CZ[0]=='DEPOLARIZE2' or lattice.after_CZ[0]=='DEPOLARIZE1':
                    dem+=stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_depo_noise={}_qubit={}_lost_round_detected={}_earlier_round_detected={}'.format(lattice.loss_rate,lattice.after_CZ[1],qubit,round_position,r0)+'.dem')
                else:
                    raise NotImplementedError()

        return dem
        
    def logical_errors(self, num_shots: int,path_dem):
        """
        Simulates logical errors in a quantum error correction process over multiple shots.
    
        Args:
            num_shots (int): Number of shots to simulate.
            path_dem (str): Path prefix for DEM files corresponding to qubit losses.
    
        Returns:
            float: Fraction of shots with logical errors.
        """
        lattice=self.lattice
        surface_code_circuit=self.circuit_without_loss
        dem0=surface_code_circuit.detector_error_model(decompose_errors=True)
        sampler=sampler_standard_LDU(lattice)

        # Simulate losses
        pos_list=sampler.atoms_lost_position_list(num_shots)
        num_errors=0
        
        for position in pos_list:
            position.sort()
            
            #Simulate detection events and observable flips.
            detection_events,observable_flips=sampler.sampler(position)

            #Build DEM given the lost qubits
            dem1=self.build_dem_lost_qubits(position,path_dem)
            dem1+=dem0
            
            matcher = pymatching.Matching.from_detector_error_model(dem1)
            
            # Run the decoder.
            predictions = matcher.decode(detection_events[0])
            predicted_for_shot = predictions
            actual_for_shot = observable_flips[0]


            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return(num_errors/num_shots)



    

    def build_naive_dem_loss(self,save_path):
        """
        Constructs the naive DEM (without knowledge of loss locations) for qubit loss, including both data and ancilla qubits.
    
        Args:
            save_path (str): Path to save the generated DEM file.
    
        Returns:
            None: Saves the generated DEM to the specified file path.
        """
        lattice=self.lattice
        p=lattice.loss_rate

        # Retrieve indices for different types of data qubits.
        corner_indices_data=lattice.corner_indices_data_qubits()
        horizontal_edge_indices_data=lattice.horizontal_edge_indices_data_qubits()
        vertical_edge_indices_data=lattice.vertical_edge_indices_data_qubits()
        bulk_indices_data=lattice.bulk_indices_data_qubits() 

        # Generate positions for all data qubits.
        positions_data=[(i,r) for i in range(lattice.n_data) for r in range(lattice.rounds+1)]
        dems=[]
        for qubit,round_position in positions_data:
            flip_obs=self.flip_observable(qubit,round_position)

            # Handle corner data qubits for different rounds.
            if qubit in corner_indices_data:
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(2-p-(1-p)**3)*(1-p)**2/((1-(1-p)**2+(1-p)**4)),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**5,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**5,flip_obs))

            # Handle edge data qubits for different rounds.
            elif qubit in horizontal_edge_indices_data or qubit in vertical_edge_indices_data:
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(2-p-(1-p)**3)*(1-p)**3/((1-(1-p)**2+(1-p)**4)),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**6,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**6,flip_obs))

            # Handle bulk data qubits for different rounds.
            elif qubit in bulk_indices_data:
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(2-p-(1-p)**3)*(1-p)**4/((1-(1-p)**2+(1-p)**4)),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,5]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**7,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p/(2*(1-(1-p)**2+(1-p)**4))*(1-p)**7,flip_obs))
            else:
                raise Exception('qubit_index must be a positive integer smaller than {}'.format(lattice.n_data))


        # Retrieve indices for ancilla qubits.
        edge_indices_ancilla=lattice.edge_indices_ancilla_z_qubits()+lattice.edge_indices_ancilla_x_qubits()
        bulk_indices_ancilla=lattice.bulk_indices_ancilla_z_qubits()+lattice.bulk_indices_ancilla_x_qubits()

         # Generate positions for all ancilla qubits.
        positions_ancilla=[(i,r) for i in range(lattice.n_data,lattice.n_data+lattice.n_ancilla_z+lattice.n_ancilla_x) for r in range(lattice.rounds)]

        # Process each ancilla qubit and round.
        for qubit,round_position in positions_ancilla:
            flip_obs=self.flip_observable(qubit,round_position)
            if qubit in edge_indices_ancilla:
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True), p,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p),flip_obs))
                
                           
            elif qubit in bulk_indices_ancilla:
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p),flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)**2,flip_obs))
                
                dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True,ignore_decomposition_failures=True),p*(1-p)**3,flip_obs))
            else:
                raise Exception('qubit_index must be a positive integer smaller than {}'.format(lattice.n_data))

        # Combine all weighted DEMs.
        final_dem=self.add_weighted_dem(dems,add_observable=True)

        # Save the final DEM to a file.
        with open(save_path, 'w') as f:
            final_dem.to_file(f)

        
    def naive_logical_errors(self, num_shots: int,path_dem):
        """
        Simulates logical errors using a naive DEM for qubit loss.
    
        Args:
            num_shots (int): Number of shots to simulate.
            path_dem (str): Path to the DEM file for qubit loss.
    
        Returns:
            float: Fraction of shots with logical errors.
        """
        lattice=self.lattice
        surface_code_circuit=self.circuit_without_loss
        dem=stim.DetectorErrorModel.from_file(path_dem)
        dem+=surface_code_circuit.detector_error_model(decompose_errors=True)
        sampler=sampler_standard_LDU(lattice)
        
        # Simulate losses
        pos_list=sampler.atoms_lost_position_list(num_shots)
        num_errors=0
        for position in pos_list:
            position.sort()

            #Simulate detection events and observable flips.
            detection_events,observable_flips=sampler.sampler(position)
            
            matcher = pymatching.Matching.from_detector_error_model(dem)
            
            # Run the decoder.
            predictions = matcher.decode(detection_events[0])
            predicted_for_shot = predictions
            actual_for_shot = observable_flips[0]

            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
                
        return(num_errors/num_shots)

  

    @staticmethod
    def break_targets_using_separators(targets):
        """
        Splits a list of targets into sublists, separating them at designated separators.
    
        Args:
            targets (list): A list of `stim.DemTarget` objects, where separators 
                            (e.g., `stim.DemTarget.separator()`) indicate split points.
    
        Returns:
            list of lists: A list of sublists, where each sublist contains targets grouped
                           between separators.
    
        Example:
            Input: [target1, target2, separator, target3, target4]
            Output: [[target1, target2], [target3, target4]]
        """
        list_targets=[]
        temp_list=[]
        for target in targets:
            if not(target.is_separator()):
                temp_list.append(target)

            else:
                list_targets.append(temp_list)
                temp_list=[]
        list_targets.append(temp_list)
        return list_targets
    
    @staticmethod
    def join_targets_using_separators(list_targets):
        """
        Combines a list of sublists into a single list of targets, adding separators
        between each sublist.
    
        Args:
            list_targets (list of lists): A list where each sublist contains `stim.DemTarget` 
                                          objects to be joined.
    
        Returns:
            list: A single list of `stim.DemTarget` objects, with separators inserted
                  between the original sublists.
    
        Example:
            Input: [[target1, target2], [target3, target4]]
            Output: [target1, target2, separator, target3, target4]
        """
        targs=[]
        for targets in list_targets:
            targs+=targets
            targs.append(stim.DemTarget.separator())
        targs.pop()
        return targs

    @staticmethod
    def contain_logical_observable(targets):
        """
        Checks if a list of targets contains a logical observable.
    
        Args:
            targets (list): A list of `stim.DemTarget` objects.
    
        Returns:
            bool: True if the list contains at least one logical observable, False otherwise.
    
        Example:
            Input: [target1, logical_observable, target2]
            Output: True
        """
        for target in targets:
            if target.is_logical_observable_id():
                return True
        return False

    @staticmethod
    def num_logical_observable(targets):
        """
        Counts the number of logical observables in a list of targets.
    
        Args:
            targets (list): A list of `stim.DemTarget` objects.
    
        Returns:
            int: The total number of logical observables in the list.
    
        Example:
            Input: [target1, logical_observable1, target2, logical_observable2]
            Output: 2
        """
        s=0
        for target in targets:
            if target.is_logical_observable_id():
                s+=1
        return s