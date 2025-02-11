from typing import Optional,Tuple,Any
import stim
import numpy as np
from surface_code_teleportation_LDU import Rotated_Surface_Code_teleportation_LDU
from sampler_teleportation_LDU_Z_bias import sampler_teleportation_LDU
import pymatching
import warnings
import pickle
class DEM_builder:
    """
    Class to construct the DEM corresponding to the rotated surface code equipped with the teleportation LDU
    
    """
    
    def __init__(
        self,
        lattice:Rotated_Surface_Code_teleportation_LDU,
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
            sampler=sampler_teleportation_LDU(lattice)
            rounds=lattice.rounds
            n_data=lattice.n_data
            corner_indices_data=lattice.corner_indices_data_qubits()
            edge_indices_data=lattice.horizontal_edge_indices_data_qubits()+lattice.vertical_edge_indices_data_qubits()
            bulk_indices_data=lattice.bulk_indices_data_qubits()    
            np_atoms_lost_position=np.array(atoms_lost_position)

            # If there are no lost qubits, return the original circuit.
            if  len(np_atoms_lost_position)==0:
                return circuit
            
            #Step 2: List all the operations that the qubit lost can experienced (for now it is manually). Sort by the type operation (single/two qubit gate/noise channel).
            qubit_op1=['R','RX','H']
            qubit_op2=['CZ']
            new_round_instruction=['SHIFT_COORDS']

                        
            
            # Step 3: Initialize state-tracking variables for circuit modification.
            position_qubits=[0 for i in range(len(np_atoms_lost_position))] # Tracks qubit positions  in the round
            circuit2=stim.Circuit() # A new circuit to store modified instructions.
            current_round=-1 # Tracks the current round.
            new_round=True # Indicates the start of a new round.
            qubits_lost_now=[]  # Tracks qubits lost in the current round.
            det=0 #Tracks whether this is the first detector instruction of the round
            
            # Step 4: Iterate over each instruction in the circuit.
            for instruction in circuit:
                name=instruction.name
                targets=instruction.targets_copy()
                args=instruction.gate_args_copy()

                # Handle the start of a new round.
                if new_round:
                    # Reset data qubits that were lost in the previous round or during the teleportation LDU
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
                        if lattice.initial_state=='|0>' or lattice.initial_state=='|1>': # Initial state in Z-basis
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
                if (i-lattice.n_data)<(lattice.width-1): #Top edge qubits
                    return True
                else:
                    return False
                
            elif i<lattice.n_data+lattice.n_ancilla_z+lattice.n_ancilla_x:# Ancilla X qubits
                if (i-lattice.n_data-lattice.n_ancilla_z)<(lattice.width+1)/2: #Top edge qubits
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
            current_shift=0

            # Iterate through instructions in the DEM.
            for instr in dem:
                typ=instr.type
                if typ=='repeat':
                    raise Exception('No repeat block instruction allowed. Use flattened method before providing it as an argument.')
                    
                args=instr.args_copy()
                targets=instr.targets_copy()
                
                # Handle non-error instructions if `only_errors` is False.
                if typ!='error' and not only_errors:
                    if d==0: # For the first DEM, initialize detector instructions.
                        if typ=='shift_detectors':
                            max_shift+=1
                            detectors_list.append((typ,args,targets))
                            shift_indices.append(len(detectors_list)-1)
                        elif typ=='detector'or typ=='logical_observable':
                            detectors_list.append((typ,args,targets))
                        else:
                            raise Exception('Instruction not recognized.')
                    else: # For subsequent DEMs, compare and insert instructions as needed.
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
        rounds=lattice.rounds
        
              
        # Retrieve indices of different qubit types.
        corner_indices_data=lattice.corner_indices_data_qubits()
        horizontal_edge_indices_data=lattice.horizontal_edge_indices_data_qubits()
        vertical_edge_indices_data=lattice.vertical_edge_indices_data_qubits()
        bulk_indices_data=lattice.bulk_indices_data_qubits() 

        # Generate all possible (qubit, round) positions for data qubits.
        positions_data=[(i,r) for i in range(lattice.n_data) for r in range(rounds)]

        # Process each qubit and its round position.
        for qubit,round_position in positions_data:
            flip_obs=self.flip_observable(qubit,round_position) # Determine if observable flips 
            final_dem=stim.DetectorErrorModel() # Initialize a DEM for this qubit and round.
            dems=[] # List to store weighted DEMs for this qubit and round.

            # Handle corner data qubits.
            if qubit in corner_indices_data:
                if round_position==0: # Initial round
                    p_tot_corner=sum([p*(1-p)**t for t in range(3)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_corner,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_corner,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_corner,flip_obs))
                    
                elif round_position==rounds-1:# Final round
                    p_tot_corner=sum([p*(1-p)**t for t in range(3)]) 
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_corner,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_corner,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_corner,flip_obs))
                    

                else:# Intermediate rounds
                    p_tot_corner=sum([p*(1-p)**t for t in range(4)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_corner,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_corner,flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p*(1-p)**2/p_tot_corner,flip_obs))
                
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_corner,flip_obs))
                
                
                
            # Handle edge data qubits.    
            elif qubit in horizontal_edge_indices_data or qubit in vertical_edge_indices_data:
                
                if round_position==0:# Initial round
                    p_tot_edge=sum([p*(1-p)**t for t in range(4)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_edge,flip_obs))
                    
                elif round_position==rounds-1: # Final round
                    p_tot_edge=sum([p*(1-p)**t for t in range(4)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_edge,flip_obs))
                    

                else:# Intermediate rounds
                    p_tot_edge=sum([p*(1-p)**t for t in range(5)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_edge,flip_obs))

                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p*(1-p)/p_tot_edge,flip_obs))
                
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_edge,flip_obs))

                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_edge,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4/p_tot_edge,flip_obs))
                    
                
                
                
            # Handle bulk data qubits.    
            elif qubit in bulk_indices_data:
                
                if round_position==0:# Initial round
                    p_tot_bulk=sum([p*(1-p)**t for t in range(5)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4/p_tot_bulk,flip_obs))
                    
                elif round_position==rounds-1:# Final round
                    p_tot_bulk=sum([p*(1-p)**t for t in range(5)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4/p_tot_bulk,flip_obs))
                    

                else:# Intermediate rounds
                    p_tot_bulk=sum([p*(1-p)**t for t in range(6)])
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4/p_tot_bulk,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**5/p_tot_bulk,flip_obs))

            
            else:
                raise Exception('qubit_index must be a positive integer smaller than {}'.format(lattice.n_data))
                
            # Combine all DEMs for this qubit and round.
            final_dem=self.add_weighted_dem(dems,add_observable=True)

            #Save the resulting DEM to a file.
            with open(save_path+'_qubit={}_round={}'.format(qubit,round_position)+'.dem', 'w') as f:
                final_dem.to_file(f)
                

           
    
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

            # Save the final DEM for this ancilla qubit and round to a file.
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

        # Compute the total number of detectors, including intermediate rounds.
        return ini_det+final_det+(lattice.rounds-1)*(lattice.n_ancilla_x+lattice.n_ancilla_z)            


    
    
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
        dem=stim.DetectorErrorModel()

        # Process each lost qubit position.
        for [qubit,round_position,CZ_pos] in position:

            # Ancilla qubits.
            if qubit>=lattice.n_data:
                dem+= stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_qubit={}_round={}'.format(lattice.loss_rate,qubit,round_position)+'.dem')

            else:
                # Data qubits.
                if lattice.after_CZ==None:
                    dem+=stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_depo_noise=0_qubit={}_round={}'.format(lattice.loss_rate,qubit,round_position)+'.dem')
                elif lattice.after_CZ[0]=='DEPOLARIZE2' or lattice.after_CZ[0]=='DEPOLARIZE1':
                    dem+=stim.DetectorErrorModel.from_file(path_dem+'_loss_rate={}_depo_noise={}_qubit={}_round={}'.format(lattice.loss_rate,lattice.after_CZ[1],qubit,round_position)+'.dem')
                else:
                    raise NotImplementedError()
            
        return dem

           

    @staticmethod
    def heralded_z_errors_not_on_atoms_loss(atoms_lost,heralded_z_errors):
        """
        Excludes Z errors on qubits that are marked as lost.
    
        Args:
            atoms_lost (list): A list of lists where each element contains:
                - `qubit` (int): Index of the lost qubit,
                - `round_position` (int): The round in which the loss occurred,
                - `CZ_pos` (int): Position in the CZ gate.
            heralded_z_errors (list of lists): Boolean 2D array indicating heralded Z errors,
                where each row corresponds to a round, and each column corresponds to a qubit.
    
        Returns:
            list of lists: Updated heralded Z error array where Z errors on lost qubits are removed.
        """
        # If no qubits are lost, return the original Z error array.
        if len(atoms_lost)==0:
            return heralded_z_errors
        else:
            # Create a set of lost qubit and round pairs for easy lookup.
            atoms_lost_round=[[qb,round_pos] for qb, round_pos, CZ_pos in atoms_lost]

            # Update the heralded Z error array to exclude lost qubits.
            heralded_z_errors2=[ [ False if z_error and [qb,round_pos] in atoms_lost_round else z_error for qb, z_error in enumerate(z_errors_round)]for round_pos,z_errors_round in enumerate(heralded_z_errors)]
    
        return heralded_z_errors2

    @staticmethod
    def flip_detectors_observable(dem_heralded_error,detection_events,observable_flips):
        """
        Flips detectors and observable values based on the heralded Z error.
    
        Args:
            dem_heralded_error (stim.DetectorErrorModel): DEM for heralded errors.
            detection_events (list): Array of detection events for the current shot.
            observable_flips (list): Array of logical observable flips for the current shot.
    
        Returns:
            tuple: Updated detection events and observable flips after applying the DEM.
        """
        detection_events2=detection_events.copy()
        observable_flips2=observable_flips.copy()

        # Process each instruction in the DEM.
        for instr in dem_heralded_error:
            typ=instr.type
            if typ=='repeat':
                raise Exception('No repeat block instruction allowed. Use flattened method before providing it as an argument.')

            elif typ=='error':
                args=instr.args_copy()
                targets=instr.targets_copy()

                # Update observables and detectors based on the targets in the instruction.
                for target in targets:
                    if target.is_logical_observable_id():
                        observable_flips2[target.val]=(observable_flips2[target.val]+1)%2
                       
             
                    elif target.is_relative_detector_id():
                        detection_events2[target.val] = (detection_events2[target.val]+ 1)%2

        return detection_events2,observable_flips2
                        

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
        sampler=sampler_teleportation_LDU(lattice)

        # Create a circuit without noise
        lattice0=Rotated_Surface_Code_teleportation_LDU(width = lattice.width, height = lattice.height, rounds = lattice.rounds, initial_state = lattice.initial_state, readout_direction = lattice.readout_direction, measurement_order = lattice.measurement_order, loss_rate = lattice.loss_rate)
        sampler0=sampler_teleportation_LDU(lattice0)
    
        # Simulate losses and Z errors.
        atoms_loss_list=sampler.atoms_lost_position_list(num_shots)
        heralded_z_errors_list=sampler.heralded_z_errors(num_shots)
        num_errors=0
        for atoms_lost,z_errors in zip(atoms_loss_list, heralded_z_errors_list):
            atoms_lost.sort()

            #Build DEM given the lost qubits
            dem1=self.build_dem_lost_qubits(atoms_lost,path_dem)
            dem1+=dem0 
                        
            # Update Z errors to exclude those on lost qubits.
            heralded_z_errors2=self.heralded_z_errors_not_on_atoms_loss(atoms_lost,z_errors)

            #Simulate detection events and observable flips.
            detection_events,observable_flips=sampler.sampler(atoms_lost,heralded_z_errors2)

            # Generate a new circuit and DEM with the updated Z errors and without loss and noise
            new_circuit_without_loss_and_depo = sampler0.atoms_loss_and_heralded_z_errors_circuit(heralded_z_errors_list = heralded_z_errors2)
            dem_heralded=new_circuit_without_loss_and_depo.detector_error_model(decompose_errors=True)

            # Flip detectors and observable values associated to the heralded Z error
            detection_events2,observable_flips2 = self.flip_detectors_observable(dem_heralded,detection_events[0],observable_flips[0])

            # Decode the detection events to predict logical errors.
            matcher = pymatching.Matching.from_detector_error_model(dem1)
            predicted_for_shot = matcher.decode(detection_events2)
            actual_for_shot = observable_flips2

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
        rounds=lattice.rounds

        # Retrieve indices for different types of data qubits.
        corner_indices_data=lattice.corner_indices_data_qubits()
        horizontal_edge_indices_data=lattice.horizontal_edge_indices_data_qubits()
        vertical_edge_indices_data=lattice.vertical_edge_indices_data_qubits()
        bulk_indices_data=lattice.bulk_indices_data_qubits() 

         # Generate positions for all data qubits.
        positions_data=[(i,r) for i in range(lattice.n_data) for r in range(rounds)]
        dems=[]
        for qubit,round_position in positions_data:
            flip_obs=self.flip_observable(qubit,round_position)
            
            # Handle corner data qubits for different rounds.
            if qubit in corner_indices_data:
                if round_position==0:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                elif round_position==rounds-1:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                    
                else:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p*(1-p),flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                    

            # Handle edge data qubits for different rounds.
            elif qubit in horizontal_edge_indices_data or qubit in vertical_edge_indices_data:
                
                if round_position==0:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                    
                elif round_position==rounds-1:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                    
                else:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p*(1-p),flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))

                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4,flip_obs))
                
                
            # Handle bulk data qubits for different rounds.
            elif qubit in bulk_indices_data:
                
                if round_position==0:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4,flip_obs))
                    
                elif round_position==rounds-1:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p),flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4,flip_obs))
                    
                else:
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,-1]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p,flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,0]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True), p*(1-p),flip_obs))
                    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,1]]).detector_error_model(
                    decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**2,flip_obs))
    
                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,2]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**3,flip_obs))

                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,3]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**4,flip_obs))

                    dems.append((self.atom_loss_circuit_without_noise([[qubit,round_position,4]]).detector_error_model( decompose_errors=True,allow_gauge_detectors=True),p*(1-p)**5,flip_obs))

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
        
        sampler=sampler_teleportation_LDU(lattice)

        # Create a circuit without noise
        lattice0=Rotated_Surface_Code_teleportation_LDU(width = lattice.width, height = lattice.height, rounds = lattice.rounds, initial_state = lattice.initial_state, readout_direction = lattice.readout_direction, measurement_order = lattice.measurement_order, loss_rate = lattice.loss_rate)
        sampler0=sampler_teleportation_LDU(lattice0)
        
        # Simulate losses and errors.
        atoms_loss_list=sampler.atoms_lost_position_list(num_shots)
        heralded_z_errors_list=sampler.heralded_z_errors(num_shots)
        
        num_errors=0
        for atoms_lost,z_errors in zip(atoms_loss_list, heralded_z_errors_list):
            atoms_lost.sort()
            
            # Update Z errors to exclude those on lost qubits.
            heralded_z_errors2=self.heralded_z_errors_not_on_atoms_loss(atoms_lost,z_errors)

            #Simulate detection events and observable flips.
            detection_events,observable_flips=sampler.sampler(atoms_lost,heralded_z_errors2)

            # Generate a new circuit and DEM with the updated Z errors and without loss and noise
            new_circuit_without_loss_and_depo = sampler0.atoms_loss_and_heralded_z_errors_circuit(heralded_z_errors_list = heralded_z_errors2)
            dem_heralded=new_circuit_without_loss_and_depo.detector_error_model(decompose_errors=True)

             # Flip detectors and observable values associated to the heralded Z error
            detection_events2,observable_flips2 = self.flip_detectors_observable(dem_heralded,detection_events[0],observable_flips[0])

            

            # Run the decoder.
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predicted_for_shot = matcher.decode(detection_events2)
            actual_for_shot = observable_flips2
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
