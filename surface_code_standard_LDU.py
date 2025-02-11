from typing import Optional,Tuple,Any
import stim
import numpy as np
from scipy import linalg
class Rotated_Surface_Code_standard_LDU:
    """
    Class to construct the circuit corresponding to the surface code  with the effective noise mimicking the action of the standard LDU
    
    """
    
    def __init__(
        self,
        height:int,
        width:int,
        rounds:int,
        initial_state:Optional[str]=None,
        readout_direction:Optional[str]=None,
        before_round:Optional[Tuple[str,Any]]=None,
        noise_on_ancilla:Optional[bool]=True,
        before_measure_flip_probability:Optional[float]=0.0,
        after_reset_flip_probability:Optional[float]=0.0,
        after_CZ:Optional[Tuple[str,Any]]=None,
        measurement_order:str='z_first',
        loss_rate: Optional[float]=0.
        
    ):
        """
        Args:
            height (Integer): Height of the lattice
            
            width (Integer): Width of the lattice
            
            rounds (Integer): Rounds of stabilizer measurement
            
            initial_state (String): Defaults to None. The initial state in which the logical qubit is:
            '|0>', '|1>', '|+>' or '|->'.
            
            readout_direction (String): Defaults to None. The direction of the readout at the end:
            'X' or 'Z'.
            
            before_round (String, Any): Defaults to None. Indicate the error model (only 1 qubit)
            and the associated noise strength, applied to every data qubits
            (and possibly ancilla qubits (see following option)) at the start of
            a round of stabilizer measurements. 
            Example: (DEPOLARIZE1, p) Implement `DEPOLARIZE1(p)` operations.
            
           
            noise_on_ancilla (Boolean): Defaults to True. Indicate if the above error model
            also applies to ancilla qubits.
            
            before_measure_flip_probability (Float): Defaults to 0. The probability (p) of
            `X_ERROR(p)` operations applied to qubits before each measurement (X
            basis measurements use `Z_ERROR(p)` instead). The before-measurement
            flips are only included if this probability is not 0.
            
            after_reset_flip_probability (Float): Defaults to 0. The probability (p) of
            `X_ERROR(p)` operations applied to qubits after each reset (X basis
            resets use `Z_ERROR(p)` instead). The after-reset flips are only
            included if this probability is not 0.
  
            after_CZ(String, Any): Defaults to None. Indicate the error model 
            and the associated noise strength, applied to qubits
            after the CZ gate
            
            measurement_order(String): Defaults to 'z_first'. Order in which the measuremen is proceed . 
            For 'z_first', Z Stabilizer are measured prior to X stabilizer. For 'x_first', it is the other way around.
            Finally 'combined', measure both stabilizer types in parallel.
            
            loss_rate(Float):  Loss rate of atoms per CZ gate. Default value set to 0.
            
        

        """
        if height%2==1:
            self.height=height
        else:
            raise Expection("The height  must be odd")
        
        if width%2==1:
            self.width=width
        else:
            raise Expection("The width  must be odd")
        self.rounds=rounds
        self.n_data=height*width #number of data qubits
        self.n_ancilla_x=int((height+1)*(width-1)/2) #number of ancilla qubits in the sublattice x
        self.n_ancilla_z=int((height-1)*(width+1)/2) #number of ancilla qubits in the sublattice z
        self.initial_state=initial_state
        self.readout_direction=readout_direction
        self.before_round=before_round
        self.noise_on_ancilla=noise_on_ancilla
        
        self.before_measure_flip_probability=before_measure_flip_probability
        self.after_reset_flip_probability=after_reset_flip_probability
        self.after_CZ=after_CZ
        self.measurement_order=measurement_order
        self.loss_rate=loss_rate
        
        
        
    def coordinates_data_qubit(self, i: int) -> Tuple[float, float]:
        """
        Get the coordinates of a data qubit in the lattice.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[float, float]: Coordinates (x, y) of the qubit.

        Raises:
            Exception: If the index is out of bounds.
        """
        if i>=0 and i<self.n_data:
            dw=self.width
            y=i//dw+0.5
            x=i%dw+0.5
            return (x,y)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_data))
            
    def index_data_qubit(self, i: int) -> int:
        """
        Get the index of a data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            int: Index of the data qubit.
        """
        return i
    
    def index_ancilla_z_qubit(self, i: int) -> int:
        """
        Get the index of a Z-type ancilla qubit.

        Args:
            i (int): Index of the Z ancilla qubit.

        Returns:
            int: Adjusted index for the Z ancilla qubit.
        """
        return self.n_data+i
    
    def index_ancilla_x_qubit(self, i: int) -> int:
        """
        Get the index of an X-type ancilla qubit.

        Args:
            i (int): Index of the X ancilla qubit.

        Returns:
            int: Adjusted index for the X ancilla qubit.
        """
        return self.n_data+self.n_ancilla_z+i
    
          
    def coordinates_ancilla_z_qubit(self, i: int) -> Tuple[float, float]:
        """
        Returns the coordinates of a Z-stabilizer ancilla qubit.

        Args:
            i (int): Index of the ancilla qubit.

        Returns:
            Tuple[float, float]: Coordinates (x, y) of the ancilla qubit.

        Raises:
            Exception: If the index is out of bounds.
        """
        if i>=0 and i<self.n_ancilla_z:
            dw=self.width
            ancilla_z_per_row=(dw-1)/2
            y=i//ancilla_z_per_row
            col=i%ancilla_z_per_row
            if y%2==0:
                x=2*col+1
            else:
                x=2*(col+1)
            return (x,y)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_ancilla_z))
            
    def coordinates_ancilla_x_qubit(self, i: int) -> Tuple[float, float]:
        """
        Returns the coordinates of an X-stabilizer ancilla qubit.

        Args:
            i (int): Index of the ancilla qubit.

        Returns:
            Tuple[float, float]: Coordinates (x, y) of the ancilla qubit.

        Raises:
            Exception: If the index is out of bounds.
        """
        if i>=0 and i<self.n_ancilla_x:
            dw=self.width
            ancilla_x_per_row=(dw+1)/2
            row=i//ancilla_x_per_row
            col=i%ancilla_x_per_row
            y=row+1
            if row%2==0:
                x=2*col+1    
            else:
                x=2*col
            return (x,y)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_ancilla_x))
            
    
    def corner_indices_data_qubits(self)-> list:
        """
        Get the indices of the corner data qubits in the lattice.

        Returns:
            list: Indices of the corner qubits.
        """
        dw=self.width
        dh=self.height
        return [0,dw-1,dw*(dh-1),dw*dh-1]

    def horizontal_edge_indices_data_qubits(self)-> list:
        """
        Get the indices of the data qubits located on the horizontal edges of the lattice.

        Returns:
            list: Indices of horizontal edge qubits.
        """
        dw=self.width
        dh=self.height
        return list(range(1,dw-1))+list(range(dw*(dh-1)+1,dw*dh-1))
    
    def vertical_edge_indices_data_qubits(self)-> list:
        """
        Get the indices of the data qubits located on the vertical edges of the lattice.

        Returns:
            list: Indices of vertical edge qubits.
        """
        dw=self.width
        dh=self.height
        return [d for d in range(dw,dw*(dh-1))if d%dw==0 or d%dw==dw-1]
    
    def bulk_indices_data_qubits(self)-> list:
        """
        Get the indices of the data qubits located inside the lattice (not on edges or corners).

        Returns:
            list: Indices of bulk data qubits.
        """
        dw=self.width
        dh=self.height        
        return [d for d in range(dw+1,dw*(dh-1)-1) if d%dw>0 and d%dw<dw-1]
    
    def edge_indices_ancilla_z_qubits(self)-> list:
        """
        Get the indices of Z-stabilizer ancilla qubits located on the edges of the lattice.

        Returns:
            list: Indices of edge Z-stabilizer ancilla qubits.
        """
        dw=self.width
        dh=self.height
        return [self.index_ancilla_z_qubit(i) for i in range(dw//2)]+[self.index_ancilla_z_qubit(i) for i in range(dw*(dh-1)//2,dw*dh//2)]
    
    def bulk_indices_ancilla_z_qubits(self)-> list:
        """
        Get the indices of Z-stabilizer ancilla qubits located on the edges of the lattice.

        Returns:
            list: Indices of edge Z-stabilizer ancilla qubits.
        """
        dw=self.width
        dh=self.height
        return [self.index_ancilla_z_qubit(i) for i in range(dw//2,dw*(dh-1)//2)]
    
    def edge_indices_ancilla_x_qubits(self)-> list:
        """
        Get the indices of X-stabilizer ancilla qubits located on the edges of the lattice.

        Returns:
            list: Indices of edge Z-stabilizer ancilla qubits.
        """
        dw=self.width
        dh=self.height
        return [self.index_ancilla_x_qubit(i) for i in range(dw//2,dw*dh//2,dw+1)]+[self.index_ancilla_x_qubit(i) for i in range(dw//2+1,dw*dh//2,dw+1)]
    
    def bulk_indices_ancilla_x_qubits(self)-> list:
        """
        Get the indices of X-stabilizer ancilla qubits located inside the lattice (not on edges).

        Returns:
            list: Indices of bulk X-stabilizer ancilla qubits.
        """
        dw=self.width
        dh=self.height
        return [self.index_ancilla_x_qubit(i) for i in range(dw*dh//2) if i%(dw+1)!=(dw+1)/2 and i%(dw+1)!=(dw-1)/2]
    
    def logical_x(self, circuit: stim.Circuit):
        """
        Apply a logical X operation to the circuit (X string on the topmost data qubits).

        Args:
            circuit (stim.Circuit): Circuit object to modify.
        """
        circuit.append('X',[self.index_data_qubit(i)  for i in range(self.width)])
        
    def logical_z(self,circuit):
        """
        Apply a logical Z operation to the circuit (Z string on the leftmost data qubits).

        Args:
            circuit (stim.Circuit): Circuit object to modify.
        """
        circuit.append('Z',[self.index_data_qubit(i)  for i in range(0,self.n_data,self.width)])
                
        
    def initialization_circuit(self) -> stim.Circuit:
        """
        Generate the initialization circuit.

        Returns:
            stim.Circuit: Circuit for initializing the lattice.
        """
        circuit = stim.Circuit()
        initial_state=self.initial_state
        dw=self.width
        dh=self.height
        n_data=self.n_data
        n_ancilla_x=self.n_ancilla_x
        n_ancilla_z=self.n_ancilla_z
        indices_data=[self.index_data_qubit(i) for i in range(n_data)]
        indices_ancilla_z=[self.index_ancilla_z_qubit(i) for i in range(n_ancilla_z)]
        indices_ancilla_x=[self.index_ancilla_x_qubit(i) for i in range(n_ancilla_x)]
        for i in range(n_data):
            x,y=self.coordinates_data_qubit(i)
            circuit.append('QUBIT_COORDS',indices_data[i],(x,y))

        for i in range(n_ancilla_z):
            x,y=self.coordinates_ancilla_z_qubit(i)
            circuit.append('QUBIT_COORDS',indices_ancilla_z[i],(x,y))

        for i in range(n_ancilla_x):
            x,y=self.coordinates_ancilla_x_qubit(i)
            circuit.append('QUBIT_COORDS',indices_ancilla_x[i],(x,y))
        
        
        
        
        if initial_state!=None:
            
            if initial_state == '|0>' or initial_state == '|1>':
                circuit.append('R',indices_data)
                if self.after_reset_flip_probability!=0.0:
                    circuit.append('X_ERROR', indices_data, self.after_reset_flip_probability)
                circuit.append('TICK')
                
                if initial_state == '|1>':
                    self.logical_x(circuit)
                    circuit.append('TICK')

            elif initial_state =='|+>' or initial_state == '|->':
                circuit.append('RX',indices_data)
                if self.after_reset_flip_probability!=0.0:
                    circuit.append('Z_ERROR', indices_data, self.after_reset_flip_probability)
                    circuit.append('TICK')

                if initial_state == '|->':
                    self.logical_z(circuit)
                    circuit.append('TICK')
                    
            else:
                raise Exception('Initial state must be either |0>,|1>,|+> or |->')

        return circuit
    
    
    def top_left_data_of_ancilla_x(self, i: int) -> Optional[int]:
        """
        Get the data qubit index to the top-left of a given X-stabilizer ancilla.

        Args:
            i (int): Index of the X-stabilizer ancilla.

        Returns:
            Optional[int]: Index of the top-left data qubit, or None if out of bounds.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_x_qubit(i)
        if x!=0:
            index=int((x-1)+(y-1)*dw)
            return index
               
        else:
            return None
        
        
    def top_left_stabilizer_x(self) -> stim.Circuit:
        """
        Create the circuit for the operations between X-type ancilla qubits and their top-left data qubits.

        Returns:
            stim.Circuit: Circuit for top-left X stabilizers.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_x):
            ancilla=self.index_ancilla_x_qubit(i)
            data=self.top_left_data_of_ancilla_x(i) 
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                
                
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)              
        return circuit
    
    def top_right_data_of_ancilla_x(self, i: int) -> Optional[int]:
        """
        Get the data qubit index to the top-right of a given X-stabilizer ancilla.

        Args:
            i (int): Index of the X-stabilizer ancilla.

        Returns:
            Optional[int]: Index of the top-right data qubit, or None if out of bounds.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_x_qubit(i)
        if x!=dw:
            index=int(x+(y-1)*dw)
            return index
           
        else:
            return None
        
    
    def top_right_stabilizer_x(self) -> stim.Circuit:
        """
        Create the circuit for the operations between X-type ancilla qubits and their top-right data qubits.

        Returns:
            stim.Circuit: Circuit for top-right X stabilizers.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_x):
            data=self.top_right_data_of_ancilla_x(i)
            ancilla=self.index_ancilla_x_qubit(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                
                   
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    def down_left_data_of_ancilla_x(self, i: int) -> Optional[int]:
        """
        Get the data qubit index to the bottom-left of a given X-stabilizer ancilla.

        Args:
            i (int): Index of the X-stabilizer ancilla.

        Returns:
            Optional[int]: Index of the bottom-left data qubit, or None if out of bounds.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_x_qubit(i)
        if x!=0:
            index=int((x-1)+y*dw)
            return index
            
        else:
            return None
    
    def down_left_stabilizer_x(self) -> stim.Circuit:
        """
        Create the circuit for the operations between X-type ancilla qubits and their down_left data qubits.

        Returns:
            stim.Circuit: Circuit for bottom-left X stabilizers.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_x):
            data=self.down_left_data_of_ancilla_x(i)
            ancilla=self.index_ancilla_x_qubit(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                
                
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    
    def down_right_data_of_ancilla_x(self, i: int) -> Optional[int]:
        """
        Get the data qubit index to the bottom-right of a given X-stabilizer ancilla.

        Args:
            i (int): Index of the X-stabilizer ancilla.

        Returns:
            Optional[int]: Index of the bottom-right data qubit, or None if out of bounds.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_x_qubit(i)
        if x!=dw:
            index=int(x+y*dw)
            return index
            
        else:
            return None
    
    def down_right_stabilizer_x(self) -> stim.Circuit:
        """
        Create the circuit for the operations between X-type ancilla qubits and their down-right data qubits.

        Returns:
            stim.Circuit: Circuit for bottom-right X stabilizers.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_x):
            ancilla=self.index_ancilla_x_qubit(i)
            data=self.down_right_data_of_ancilla_x(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
            
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    def top_left_data_of_ancilla_z(self, i: int) -> Optional[int]:
        """
        Get the data qubit index to the top-left of a given Z-stabilizer ancilla.

        Args:
            i (int): Index of the Z-stabilizer ancilla.

        Returns:
            Optional[int]: Index of the top-left data qubit, or None if out of bounds.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_z_qubit(i)
        if y!=0:
            index=int(x-1+(y-1)*dw)
            return index
            
        else:
            return None
    
    def top_left_stabilizer_z(self) -> stim.Circuit:
        """
        Create the circuit for the operations between Z-type ancilla qubits and their top-left data qubits.

        Returns:
            stim.Circuit: Circuit for top-left stabilizer measurement.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_z):
            ancilla=self.index_ancilla_z_qubit(i)
            data=self.top_left_data_of_ancilla_z(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                
               
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    def top_right_data_of_ancilla_z(self, i: int) -> Optional[int]:
        """
        Get the top-right data qubit index associated with a Z-type ancilla qubit.

        Args:
            i (int): Index of the Z-type ancilla qubit.

        Returns:
            Optional[int]: Index of the top-right data qubit if it exists, otherwise None.
        """
        dw=self.width
        x,y=self.coordinates_ancilla_z_qubit(i)
        if y!=0:
            index=int(x+(y-1)*dw)
            return index
        else:
            return None
    
    
    def top_right_stabilizer_z(self) -> stim.Circuit:
        """
        Create the circuit for the operations between Z-type ancilla qubits and their top-right data qubits.

        Returns:
            stim.Circuit: Circuit for top-right stabilizer measurement.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_z):
            data=self.top_right_data_of_ancilla_z(i)
            ancilla=self.index_ancilla_z_qubit(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                

        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    def down_left_data_of_ancilla_z(self, i: int) -> Optional[int]:
        """
        Get the down-left data qubit index associated with a Z-type ancilla qubit.

        Args:
            i (int): Index of the Z-type ancilla qubit.

        Returns:
            Optional[int]: Index of the down-left data qubit if it exists, otherwise None.
        """
        dw=self.width
        dh=self.height
        x,y=self.coordinates_ancilla_z_qubit(i)
        if y!=dh:
            index=int(x-1+y*dw)
            return index
        else:
            return None
    

    def down_left_stabilizer_z(self) -> stim.Circuit:
        """
        Create the circuit for the operations between Z-type ancilla qubits and their down-left data qubits.

        Returns:
            stim.Circuit: Circuit for down-left stabilizer measurement.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_z):
            data=self.down_left_data_of_ancilla_z(i)
            ancilla=self.index_ancilla_z_qubit(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)
                
        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit

    def down_right_data_of_ancilla_z(self, i: int) -> Optional[int]:
        """
        Get the down-right data qubit index associated with a Z-type ancilla qubit.

        Args:
            i (int): Index of the Z-type ancilla qubit.

        Returns:
            Optional[int]: Index of the down-right data qubit if it exists, otherwise None.
        """
        dw=self.width
        dh=self.height
        x,y=self.coordinates_ancilla_z_qubit(i)
        if y!=dh:
            index=int(x+y*dw)
            return index
            
        else:
            return None
        
    
    def down_right_stabilizer_z(self) -> stim.Circuit:
        """
        Create the circuit for the operations between Z-type ancilla qubits and their down-right data qubits.


        Returns:
            stim.Circuit: Circuit for down-right stabilizer measurement.
        """
        circuit = stim.Circuit()
        qubits_indices=[]
        for i in range(self.n_ancilla_z):
            ancilla=self.index_ancilla_z_qubit(i)
            data=self.down_right_data_of_ancilla_z(i)
            if data!=None:
                qubits_indices.append(ancilla)
                qubits_indices.append(data)

        circuit.append('CZ',qubits_indices)
        if self.after_CZ!=None:
            error_model=self.after_CZ[0]
            proba=self.after_CZ[1]
            circuit.append(error_model,qubits_indices,proba)  
        return circuit
    
    def top_left_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the top-left ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if i>=0 and i<self.n_data:
            dw=self.width
            if i%2==1:
                if i%dw!=0:
                    return ('z_type',int((i-1)/2-(i+dw)//(2*dw)))
                else:
                    return (None,None)
            else:
                if i>dw:
                    return ('x_type',int((i-dw-1)/2+i//(2*dw)))
                else:
                    return (None,None)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_data))
            

    def top_right_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the top-right ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if i>=0 and i<self.n_data:
            dw=self.width
            if i%2==0:
                if i%dw!=dw-1:
                    return ('z_type',int(i/2-(i+dw)//(2*dw)))
                else:
                    return (None,None)
            else:
                if i>=dw:
                    return ('x_type',int((i-dw)/2+i//(2*dw)))
                else:
                    return (None,None)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_data))
            
    
    def down_left_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the down-left ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if i>=0 and i<self.n_data:
            dw=self.width
            if i%2==0:
                if i%dw!=0:
                    return ('z_type',int((dw-3)/2+i/2-i//(2*dw)))
                else:
                    return (None,None)
            else:
                dh=self.height
                if i<(dh-1)*dw:
                    return ('x_type',int((i-1)/2+(i+dw)//(2*dw)))
                else:
                    return (None,None)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_data))
            
    def down_right_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the down-right ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if i>=0 and i<self.n_data:
            dw=self.width
            if i%2==1:
                if i%dw!=dw-1:
                    return ('z_type',int((dw-1)/2+(i-1)/2-i//(2*dw)))
                else:
                    return (None,None)
            else:
                dh=self.height
                if i<(dh-1)*dw:
                    return ('x_type',int(i/2+(i+dw)//(2*dw)))
                else:
                    return (None,None)
        else:
            raise Exception('Qubit index must be between 0 and {}'.format(self.n_data))

    def trace_noise_on_1qubit(self, noise: Tuple[str, list[float]], qubit: int) -> Tuple[str, float]:
        """
        Trace noise from a 2-qubit error model to a single qubit.

        Args:
            noise (Tuple[str, list[float]]): Error model and associated probabilities.
            qubit (int): Target qubit index (0 or 1).

        Returns:
            Tuple[str, float]: Single-qubit error model and probability.

        Raises:
            Exception: If qubit index is not 0 or 1.
            NotImplementedError: If the error model is unsupported.
        """
        error_model=noise[0]
        proba=noise[1]
        noise_1_qubit=['DEPOLARIZE1','PAULI_CHANNEL_1','HERALDED_PAULI_CHANNEL_1','HERALDED_ERASE','X_ERROR','Y_ERROR','Z_ERROR']
        noise_2_qubit=['DEPOLARIZE2','PAULI_CHANNEL_2']
        if error_model in noise_1_qubit:
            return self.after_CZ

        elif error_model in noise_2_qubit:
            if error_model=='DEPOLARIZE2':
                if type(proba)==list:
                    proba=proba[0]
                return ('DEPOLARIZE1',12*proba/15)
            else:
                if qubit==0:
                    px=sum(proba[3:7])
                    py=sum(proba[7:11])
                    pz=sum(proba[11:15])
                    return ('PAULI_CHANNEL_1',(px,py,pz))
                elif qubit==1:
                    px=sum([proba[4*i] for i in range(4)])
                    py=sum([proba[4*i+1] for i in range(4)])
                    pz=sum([proba[4*i+2] for i in range(4)])
                    return ('PAULI_CHANNEL_1',(px,py,pz))
                else:
                    raise Exception('Qubit index must be 0 or 1.')

        else:
            raise NotImplementedError

            
                

    def effective_noise_detection_circuit_lost_qubit(self):
        """
        Computes the effective noise model resulting from multiple reapplication of the standard LDU due to the loss of the ancilla LDU
    
        Returns:
            tuple: A tuple representing the noise model and its parameters:
                   - ('DEPOLARIZE1', probability): For depolarizing noise, the effective depolarizing probability.
                   - ('PAULI_CHANNEL_1', probabilities): For Pauli channel noise, the effective probabilities.
                   - None: If `self.after_CZ` is not defined.
            Raises:
                NotImplementedError: If the noise type is not supported.
            """
        if self.after_CZ!=None:
            CZ_1qubit=self.trace_noise_on_1qubit(self.after_CZ,1)
        else:
            return None
        p=self.loss_rate
        if CZ_1qubit[0]=='DEPOLARIZE1':
            if type(CZ_1qubit[1])==list:
                proba=CZ_1qubit[1][0]
            else:
                proba=CZ_1qubit[1]
            fe=(1-p)**2/(1-(1-(1-p)**2)*(1-16*proba/15)**2)
            return ('DEPOLARIZE1',3*(1-fe)/4)

        elif CZ_1qubit[0]=='PAULI_CHANNEL_1':
            px,py,pz=CZ_1qubit[1]
            fi=1
            fx=(1-p)**2/(1-(1-(1-p)**2)*(1-2*py-2*pz)**2)
            fy=(1-p)**2/(1-(1-(1-p)**2)*(1-2*px-2*pz)*(1-2*px-2*py))
            fz=fy
            fidelities=[fi,fx,fy,fz]
            #Walsh-Hadamard transformation for 1 qubit
            WH=1/4*np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
            proba=np.dot(WH,fidelities)
            return ('PAULI_CHANNEL_1',proba[1:])

        else:
            raise NotImplementedError

    @staticmethod
    def Walsh_Hadamard() -> np.ndarray:
        """
        Compute the Walsh-Hadamard transformation matrix for 2 qubits.

        Returns:
            np.ndarray: Walsh-Hadamard transformation matrix.
        """
        A=1/16*np.ones((16,16))
        for i in range(16):
            iint=i//4
            irest=i%4
            for j in range(16):
                jint=j//4
                jrest=j%4
                if iint!=0 and jint!=0 and iint!=jint:
                    A[i,j]*=-1
                if irest!=0 and jrest!=0 and irest!=jrest:
                    A[i,j]*=-1
        return A

    def pauli_noise_to_fidelities(self, pauli_noise: np.ndarray) -> np.ndarray:
        """
        Convert Pauli noise probabilities to fidelities.

        Args:
            pauli_noise (np.ndarray): Pauli noise vector.

        Returns:
            np.ndarray: Fidelities corresponding to the input noise.
        """
        WH_inv=linalg.inv(self.Walsh_Hadamard())
        return np.dot(WH_inv,pauli_noise)

    def fidelities_to_pauli_noise(self, fidelities: np.ndarray) -> np.ndarray:
        """
        Convert fidelities to Pauli noise probabilities.

        Args:
            fidelities (np.ndarray): Fidelity vector.

        Returns:
            np.ndarray: Pauli noise vector corresponding to the input fidelities.
        """
        return np.dot(self.Walsh_Hadamard(),fidelities)

    @staticmethod
    def fidelities_conjugate_X2(fidelities: np.ndarray) -> np.ndarray:
        """
        Compute the resulting vector fidelities after conjugating by a X gate on the second qubit
        
        Args:
            fidelities (np.ndarray): Input fidelity vector.

        Returns:
            np.ndarray: Modified fidelities after conjugation.
        """
        fid_conj_X=fidelities.copy()
        for i,f in enumerate(fidelities):
            if i%4==2:
                fid_conj_X[i]=fidelities[i+1]
                fid_conj_X[i+1]=fidelities[i]
        return fid_conj_X

    @staticmethod
    def fidelities_conjugate_CZ(fidelities: np.ndarray) -> np.ndarray:
        """
        Compute the resulting vector fidelities after conjugating by a CZ gate
        
        Args:
            fidelities (np.ndarray): Input fidelity vector.

        Returns:
            np.ndarray: Modified fidelities after conjugation.
        """
        fid_conj_CZ=fidelities.copy()
        fid_conj_CZ[1]=fidelities[13]
        fid_conj_CZ[2]=fidelities[14]
        fid_conj_CZ[4]=fidelities[7]
        fid_conj_CZ[5]=fidelities[10]
        fid_conj_CZ[6]=fidelities[9]
        fid_conj_CZ[7]=fidelities[4]
        fid_conj_CZ[8]=fidelities[11]
        fid_conj_CZ[9]=fidelities[6]
        fid_conj_CZ[10]=fidelities[5]
        fid_conj_CZ[11]=fidelities[8]
        fid_conj_CZ[13]=fidelities[1]
        fid_conj_CZ[14]=fidelities[2]
        
        return fid_conj_CZ

   

    def effective_noise_detection_circuit_not_lost_qubit(self):
        """
        Computes the effective noise model resulting from the last application of the standard LDU (i.e. when the ancilla LDU is not lost)
    
        Returns:
            tuple: A tuple representing the noise model and its parameters:
                   - ('DEPOLARIZE1', probability): For depolarizing noise, the effective depolarizing probability.
                   - ('PAULI_CHANNEL_1', probabilities): For Pauli channel noise, the effective probabilities.
                   - None: If `self.after_CZ` is not defined.
            Raises:
                NotImplementedError: If the noise type is not supported.
            """
        if self.after_CZ!=None:
            error_model,proba=self.after_CZ
        
            if error_model=='DEPOLARIZE1':
                if type(proba)==list:
                    proba2=proba[0]
                else:
                    proba2=proba
                proba=[proba2/3]*3+[proba2/3]+[(proba2/3)**2]*3+[proba2/3]+[(proba2/3)**2]*3+[proba2/3]+[(proba2/3)**2]*3
                error_model='PAULI_CHANNEL_2'
            elif error_model=='PAULI_CHANNEL_1':
                proba2=proba.copy()
                proba=proba2+[proba2[0]]+[proba2[0]*p for p in proba2]+[proba2[1]]+[proba2[1]*p for p in proba2]+[proba2[2]]+[proba2[2]*p for p in proba2]
                error_model='PAULI_CHANNEL_2'
                
            if error_model=='DEPOLARIZE2':
                if type(proba)==list:
                    proba2=proba[0]
                else:
                    proba2=proba
                fe=(1-16*proba2/15)**2
                return ('DEPOLARIZE1',3/4*(1-fe))
    
            elif error_model=='PAULI_CHANNEL_2':
                pauli_noise=[1-sum(proba)]+proba
                fidelities=self.pauli_noise_to_fidelities(pauli_noise)
                fid_conj_X=self.fidelities_conjugate_X2(fidelities)
                fid_conj_CZ=self.fidelities_conjugate_CZ(fid_conj_X)
                fid_conj_X2=self.fidelities_conjugate_X2(fid_conj_CZ)
                final_fidelities=np.array(fid_conj_X)*np.array(fid_conj_X2)
                pauli_noise2=self.fidelities_to_pauli_noise(final_fidelities)
                
                
                return self.trace_noise_on_1qubit(('PAULI_CHANNEL_2',pauli_noise2[1:]),qubit=1)   
                
    
            else:
                raise NotImplementedError
                
        else:
            return None
        
        
    def effective_detection_loss_circuit(self) -> stim.Circuit:
        """
        Generate the effective noise associated to the standard LDU.

        Returns:
            stim.Circuit: Circuit for simulating the effective noise of the standard LDU.
        """
        n_data=self.n_data
        indices_data=[self.index_data_qubit(i) for i in range(n_data)]
        circuit = stim.Circuit()
        #effective noise accouting for all the trial where the detector qubit has been lost
        if self.after_CZ!=None:
            effective_noise,effective_proba=self.effective_noise_detection_circuit_lost_qubit()
            circuit.append(effective_noise,indices_data,effective_proba)

        if self.after_CZ!=None:
            effective_noise,effective_proba=self.effective_noise_detection_circuit_not_lost_qubit()
            circuit.append(effective_noise,indices_data,effective_proba)

        return circuit
    
    def logical_readout(self, no_observable: bool = False) -> stim.Circuit:
        """
        Generate the logical readout circuit.

        Args:
            no_observable (bool): If True, exclude observable measurements.

        Returns:
            stim.Circuit: Logical readout circuit.
        """
        circuit=stim.Circuit()
        readout_direction=self.readout_direction
        if readout_direction!=None:
            n_data=self.n_data
            n_ancilla_x=self.n_ancilla_x
            n_ancilla_z=self.n_ancilla_z
            dw=self.width
            dh=self.height
            indices_data=[self.index_data_qubit(i) for i in range(n_data)]
            indices_ancilla_z=[self.index_ancilla_z_qubit(i) for i in range(n_ancilla_z)]
            indices_ancilla_x=[self.index_ancilla_x_qubit(i) for i in range(n_ancilla_x)]
            
            if readout_direction=='X':
                if self.before_measure_flip_probability!=0.0:
                    circuit.append('Z_ERROR',indices_data)
                circuit.append('MX',indices_data )
                circuit.append('TICK')
                for i in range(n_ancilla_x):
                    x,y=self.coordinates_ancilla_x_qubit(i)
                    top_left=self.top_left_data_of_ancilla_x(i)
                    top_right=self.top_right_data_of_ancilla_x(i)
                    down_left=self.down_left_data_of_ancilla_x(i)
                    down_right=self.down_right_data_of_ancilla_x(i)
                    data_stab=[top_left,top_right,down_left,down_right]
                    record=[stim.target_rec(-n_ancilla_x-n_data+i)]
                    for data in data_stab:
                        if data !=None:
                            record.append(stim.target_rec(-n_data+data))
                    circuit.append('DETECTOR',record,(x,y,0))
                if not no_observable:
                    circuit.append('OBSERVABLE_INCLUDE',[stim.target_rec(-n_data+qubit)for qubit in range(dw)],0)
                

            elif readout_direction=='Z':
                if self.before_measure_flip_probability!=0.0:
                    circuit.append('X_ERROR',indices_data)
                circuit.append('M',indices_data)
                circuit.append('TICK')
                for i in range(n_ancilla_z):
                    x,y=self.coordinates_ancilla_z_qubit(i)
                    top_left=self.top_left_data_of_ancilla_z(i)
                    top_right=self.top_right_data_of_ancilla_z(i)
                    down_left=self.down_left_data_of_ancilla_z(i)
                    down_right=self.down_right_data_of_ancilla_z(i)
                    data_stab=[top_left,top_right,down_left,down_right]
                    record=[stim.target_rec(-n_ancilla_x-n_ancilla_z-n_data+i)]
                    for data in data_stab:
                        if data != None:
                            record.append(stim.target_rec(-n_data+data))
                    circuit.append('DETECTOR',record,(x,y,0))
                if not no_observable:
                    circuit.append('OBSERVABLE_INCLUDE',[stim.target_rec(-n_data+qubit) for qubit in range(0,n_data,dw)],0)

            else:
                raise Exception('Readout direction must be either X or Z')
        
        return circuit


    def rotated_surface_code_circuit(self, flatten_loops: bool = False, no_observable: bool = False) -> stim.Circuit:
        """
        Generate the full rotated surface code circuit, including initialization, stabilizer measurements, and logical readout.

        Args:
            flatten_loops (bool): If True, expands the loops for all stabilizer rounds.
            no_observable (bool): If True, exclude observable definitions.

        Returns:
            stim.Circuit: The complete rotated surface code circuit.
        """
        rounds=self.rounds
        n_data=self.n_data
        dh=self.height
        dw=self.width
        n_ancilla_x=self.n_ancilla_x
        n_ancilla_z=self.n_ancilla_z
        indices_data=[self.index_data_qubit(i) for i in range(n_data)]
        indices_ancilla_z=[self.index_ancilla_z_qubit(i) for i in range(n_ancilla_z)]
        indices_ancilla_x=[self.index_ancilla_x_qubit(i) for i in range(n_ancilla_x)]
            
        surface_code_circuit=self.initialization_circuit()
        if rounds!=0:
            stab_circuit=stim.Circuit()
            stab_circuit.append('RX',indices_ancilla_z+indices_ancilla_x)
            if self.after_reset_flip_probability!=0.0:
                stab_circuit.append('Z_ERROR', indices_ancilla_z+indices_ancilla_x, self.after_reset_flip_probability)
            stab_circuit.append('TICK')
            
            if self.before_round!=None:
                error_model=self.before_round[0]
                proba=self.before_round[1]
                if self.noise_on_ancilla:
                    stab_circuit.append(error_model,indices_data+indices_ancilla_z+indices_ancilla_x,proba)
                else:
                    stab_circuit.append(error_model,indices_data,proba)
                    stab_circuit.append('TICK')
            
            if self.measurement_order=='z_first':
                stab_circuit+=self.top_left_stabilizer_z()
                stab_circuit.append('TICK')        
                stab_circuit+=self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H',indices_data)
                stab_circuit.append('TICK')
                stab_circuit+=self.top_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.top_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit.append('H',indices_data)
                stab_circuit.append('TICK')
                
         
                
            elif self.measurement_order=='x_first':
                stab_circuit.append('H',indices_data)
                stab_circuit.append('TICK')
                stab_circuit+=self.top_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.top_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit.append('H',indices_data)
                stab_circuit.append('TICK')
                stab_circuit+=self.top_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.down_right_stabilizer_z()
                stab_circuit.append('TICK')
                
            elif self.measurement_order=='combined':
              
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data[:dw*(dh-1)]) if i%2==0])
                stab_circuit.append('TICK')
                stab_circuit+=self.top_left_stabilizer_x()
                stab_circuit+=self.top_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data[:dw*(dh-1)]) if i%2==0 ])
                stab_circuit.append('TICK')
                
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data) if i%2==1])
                stab_circuit.append('TICK')
                stab_circuit+=self.down_left_stabilizer_x()
                stab_circuit+=self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit+=self.top_right_stabilizer_x()
                stab_circuit+=self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data) if i%2==1])
                stab_circuit.append('TICK')
                
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data) if i%2==0 and i>=dw])
                stab_circuit.append('TICK')
                stab_circuit+=self.down_right_stabilizer_x()
                stab_circuit+=self.down_right_stabilizer_z() 
                stab_circuit.append('TICK')
                stab_circuit.append('H',[index_data for i,index_data in enumerate(indices_data) if i%2==0 and i>=dw])
                stab_circuit.append('TICK')
                
                
            if self.before_measure_flip_probability!=0.0:
                stab_circuit.append('Z_ERROR', indices_ancilla_z+indices_ancilla_x,self.before_measure_flip_probability)
              
                      
            stab_circuit.append('MX',indices_ancilla_z+indices_ancilla_x)

            
               
             
            stab_circuit.append('TICK')
            
            surface_code_circuit+=stab_circuit
            
            for a in range(n_ancilla_z):
                x,y=self.coordinates_ancilla_z_qubit(a)
                if self.initial_state=='|0>' or self.initial_state=='|1>':
                    surface_code_circuit.append('DETECTOR',stim.target_rec(-n_ancilla_x-n_ancilla_z+a),(x,y,0))
                    
                if rounds>1:
                    stab_circuit.append('DETECTOR',[stim.target_rec(-2*n_ancilla_x-2*n_ancilla_z+a),
                                                    stim.target_rec(-n_ancilla_x-n_ancilla_z+a)],(x,y,0))

            for a in range(n_ancilla_x):
                x,y=self.coordinates_ancilla_x_qubit(a)
                if self.initial_state=='|+>' or self.initial_state=='|->':
                    surface_code_circuit.append('DETECTOR',stim.target_rec(-n_ancilla_x+a),(x,y,0))

                if rounds>1:
                    stab_circuit.append('DETECTOR',[stim.target_rec(-2*n_ancilla_x-n_ancilla_z+a),
                                                    stim.target_rec(-n_ancilla_x+a)],(x,y,0))
                    
            surface_code_circuit.append('TICK')
            surface_code_circuit+=self.effective_detection_loss_circuit()
            surface_code_circuit.append('TICK')
            surface_code_circuit.append('SHIFT_COORDS',[],(0,0,1))

            stab_circuit.append('TICK')

                       
                
            
            if rounds>1:
                if flatten_loops:
                    for i in range(1,rounds-1):
                        surface_code_circuit+=stab_circuit
                        surface_code_circuit+=self.effective_detection_loss_circuit()
                        surface_code_circuit.append('TICK')
                        surface_code_circuit.append('SHIFT_COORDS',[],(0,0,1))
                    surface_code_circuit+=stab_circuit
                    surface_code_circuit.append('SHIFT_COORDS',[],(0,0,1))
                else:
                    if rounds>2:
                        stab_circuit2=stab_circuit.copy()
                        stab_circuit2+=self.effective_detection_loss_circuit()
                        stab_circuit2.append('TICK')
                        stab_circuit2.append('SHIFT_COORDS',[],(0,0,1))
                        stab_circuit2*=(rounds-2)
                        surface_code_circuit+=stab_circuit2
                    surface_code_circuit+=stab_circuit
                    surface_code_circuit.append('SHIFT_COORDS',[],(0,0,1))
        logical_readout= self.logical_readout(no_observable=no_observable)
        surface_code_circuit+=logical_readout
        return surface_code_circuit
    