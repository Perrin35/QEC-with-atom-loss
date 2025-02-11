from typing import Optional, Tuple, Any
import stim
import numpy as np
from scipy import linalg

class Rotated_Surface_Code_no_detection_loss:
    """
    Class to construct the circuit corresponding to the rotated surface code without LDU.
    This class defines the structure, initialization, and stabilizer circuits for the surface code.
    """

    def __init__(
        self,
        height: int,
        width: int,
        rounds: int,
        initial_state: Optional[str] = None,
        readout_direction: Optional[str] = None,
        before_round: Optional[Tuple[str, Any]] = None,
        noise_on_ancilla: Optional[bool] = True,
        before_measure_flip_probability: Optional[float] = 0.0,
        after_reset_flip_probability: Optional[float] = 0.0,
        after_CZ: Optional[Tuple[str, Any]] = None,
        measurement_order: str = 'z_first'
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
            
            measurement_order(String): Defaults to 'z_first'. Order in which the measurement is proceed . 
            For 'z_first', Z Stabilizer are measured prior to X stabilizer. For 'x_first', it is the other way around.
            Finally 'combined', measure both stabilizer types in parallel.
            
           

        """
        if height % 2 == 1:
            self.height = height
        else:
            raise Exception("The height must be odd.")

        if width % 2 == 1:
            self.width = width
        else:
            raise Exception("The width must be odd.")

        self.rounds = rounds
        self.n_data = height * width  # Total number of data qubits
        self.n_ancilla_x = int((height + 1) * (width - 1) / 2)  # Number of X ancilla qubits
        self.n_ancilla_z = int((height - 1) * (width + 1) / 2)  # Number of Z ancilla qubits
        self.initial_state = initial_state
        self.readout_direction = readout_direction
        self.before_round = before_round
        self.noise_on_ancilla = noise_on_ancilla
        self.before_measure_flip_probability = before_measure_flip_probability
        self.after_reset_flip_probability = after_reset_flip_probability
        self.after_CZ = after_CZ
        self.measurement_order = measurement_order

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
        if 0 <= i < self.n_data:
            dw = self.width
            y = i // dw + 0.5
            x = i % dw + 0.5
            return (x, y)
        else:
            raise Exception(f"Qubit index must be between 0 and {self.n_data - 1}")

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
        return self.n_data + i

    def index_ancilla_x_qubit(self, i: int) -> int:
        """
        Get the index of an X-type ancilla qubit.

        Args:
            i (int): Index of the X ancilla qubit.

        Returns:
            int: Adjusted index for the X ancilla qubit.
        """
        return self.n_data + self.n_ancilla_z + i

    def coordinates_ancilla_z_qubit(self, i: int) -> Tuple[int, int]:
        """
        Get the coordinates of a Z-type ancilla qubit.

        Args:
            i (int): Index of the Z ancilla qubit.

        Returns:
            Tuple[int, int]: Coordinates (x, y) of the Z ancilla qubit.

        Raises:
            Exception: If the index is out of bounds.
        """
        if 0 <= i < self.n_ancilla_z:
            dw = self.width
            ancilla_z_per_row = (dw - 1) / 2
            y = i // ancilla_z_per_row
            col = i % ancilla_z_per_row
            x = 2 * col + 1 if y % 2 == 0 else 2 * (col + 1)
            return (x, y)
        else:
            raise Exception(f"Qubit index must be between 0 and {self.n_ancilla_z - 1}")

    def coordinates_ancilla_x_qubit(self, i: int) -> Tuple[int, int]:
        """
        Get the coordinates of an X-type ancilla qubit.

        Args:
            i (int): Index of the X ancilla qubit.

        Returns:
            Tuple[int, int]: Coordinates (x, y) of the X ancilla qubit.

        Raises:
            Exception: If the index is out of bounds.
        """
        if 0 <= i < self.n_ancilla_x:
            dw = self.width
            ancilla_x_per_row = (dw + 1) / 2
            row = i // ancilla_x_per_row
            col = i % ancilla_x_per_row
            y = row + 1
            x = 2 * col + 1 if row % 2 == 0 else 2 * col
            return (x, y)
        else:
            raise Exception(f"Qubit index must be between 0 and {self.n_ancilla_x - 1}")

    def corner_indices_data_qubits(self) -> list:
        """
        Get the indices of data qubits at the corners of the lattice.

        Returns:
            list: Indices of corner data qubits.
        """
        dw = self.width
        dh = self.height
        return [0, dw - 1, dw * (dh - 1), dw * dh - 1]

    def horizontal_edge_indices_data_qubits(self) -> list:
        """
        Get the indices of data qubits along the horizontal edges of the lattice.

        Returns:
            list: Indices of horizontal edge data qubits.
        """
        dw = self.width
        dh = self.height
        return list(range(1, dw - 1)) + list(range(dw * (dh - 1) + 1, dw * dh - 1))

    def vertical_edge_indices_data_qubits(self) -> list:
        """
        Get the indices of data qubits along the vertical edges of the lattice.

        Returns:
            list: Indices of vertical edge data qubits.
        """
        dw = self.width
        dh = self.height
        return [d for d in range(dw, dw * (dh - 1)) if d % dw == 0 or d % dw == dw - 1]

    def bulk_indices_data_qubits(self) -> list:
        """
        Get the indices of data qubits in the bulk (non-edge) of the lattice.

        Returns:
            list: Indices of bulk data qubits.
        """
        dw = self.width
        dh = self.height
        return [d for d in range(dw + 1, dw * (dh - 1) - 1) if d % dw > 0 and d % dw < dw - 1]

    def logical_x(self, circuit: stim.Circuit):
        """
        Apply a logical X operation on the circuit.

        Args:
            circuit (stim.Circuit): The circuit to modify.
        """
        circuit.append('X', [self.index_data_qubit(i) for i in range(self.width)])

    def logical_z(self, circuit: stim.Circuit):
        """
        Apply a logical Z operation on the circuit.

        Args:
            circuit (stim.Circuit): The circuit to modify.
        """
        circuit.append('Z', [self.index_data_qubit(i) for i in range(0, self.n_data, self.width)])

    def initialization_circuit(self) -> stim.Circuit:
        """
        Construct the initialization circuit for the surface code.

        Returns:
            stim.Circuit: The initialization circuit.
        """
        circuit = stim.Circuit()
        indices_data = [self.index_data_qubit(i) for i in range(self.n_data)]
        indices_ancilla_z = [self.index_ancilla_z_qubit(i) for i in range(self.n_ancilla_z)]
        indices_ancilla_x = [self.index_ancilla_x_qubit(i) for i in range(self.n_ancilla_x)]

        for i in range(self.n_data):
            x, y = self.coordinates_data_qubit(i)
            circuit.append('QUBIT_COORDS', indices_data[i], (x, y))

        for i in range(self.n_ancilla_z):
            x, y = self.coordinates_ancilla_z_qubit(i)
            circuit.append('QUBIT_COORDS', indices_ancilla_z[i], (x, y))

        for i in range(self.n_ancilla_x):
            x, y = self.coordinates_ancilla_x_qubit(i)
            circuit.append('QUBIT_COORDS', indices_ancilla_x[i], (x, y))

        if self.initial_state:
            if self.initial_state in ['|0>', '|1>']:
                circuit.append('R', indices_data)
                if self.after_reset_flip_probability:
                    circuit.append('X_ERROR', indices_data, self.after_reset_flip_probability)
                circuit.append('TICK')

                if self.initial_state == '|1>':
                    self.logical_x(circuit)
                    circuit.append('TICK')

            elif self.initial_state in ['|+>', '|->']:
                circuit.append('RX', indices_data)
                if self.after_reset_flip_probability:
                    circuit.append('Z_ERROR', indices_data, self.after_reset_flip_probability)
                circuit.append('TICK')

                if self.initial_state == '|->':
                    self.logical_z(circuit)
                    circuit.append('TICK')

            else:
                raise Exception("Initial state must be either |0>, |1>, |+>, or |->")

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
        if 0 <= i < self.n_data:
            dw = self.width
            if i % 2 == 1:
                if i % dw != 0:
                    return ('z_type', int((i - 1) / 2 - (i + dw) // (2 * dw)))
                else:
                    return (None, None)
            else:
                if i > dw:
                    return ('x_type', int((i - dw - 1) / 2 + i // (2 * dw)))
                else:
                    return (None, None)
        else:
            raise Exception(f'Qubit index must be between 0 and {self.n_data - 1}')

    def top_right_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the top-right ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if 0 <= i < self.n_data:
            dw = self.width
            if i % 2 == 0:
                if i % dw != dw - 1:
                    return ('z_type', int(i / 2 - (i + dw) // (2 * dw)))
                else:
                    return (None, None)
            else:
                if i >= dw:
                    return ('x_type', int((i - dw) / 2 + i // (2 * dw)))
                else:
                    return (None, None)
        else:
            raise Exception(f'Qubit index must be between 0 and {self.n_data - 1}')

    def down_left_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the down-left ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if 0 <= i < self.n_data:
            dw = self.width
            if i % 2 == 0:
                if i % dw != 0:
                    return ('z_type', int((dw - 3) / 2 + i / 2 - i // (2 * dw)))
                else:
                    return (None, None)
            else:
                dh = self.height
                if i < (dh - 1) * dw:
                    return ('x_type', int((i - 1) / 2 + (i + dw) // (2 * dw)))
                else:
                    return (None, None)
        else:
            raise Exception(f'Qubit index must be between 0 and {self.n_data - 1}')

    def down_right_ancilla_of_data(self, i: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine the down-right ancilla type and index for a given data qubit.

        Args:
            i (int): Index of the data qubit.

        Returns:
            Tuple[Optional[str], Optional[int]]: A tuple containing the ancilla type ('z_type' or 'x_type')
            and the index, or (None, None) if not applicable.
        """
        if 0 <= i < self.n_data:
            dw = self.width
            if i % 2 == 1:
                if i % dw != dw - 1:
                    return ('z_type', int((dw - 1) / 2 + (i - 1) / 2 - i // (2 * dw)))
                else:
                    return (None, None)
            else:
                dh = self.height
                if i < (dh - 1) * dw:
                    return ('x_type', int(i / 2 + (i + dw) // (2 * dw)))
                else:
                    return (None, None)
        else:
            raise Exception(f'Qubit index must be between 0 and {self.n_data - 1}')

    def logical_readout(self, no_observable: bool = False) -> stim.Circuit:
        """
        Construct the logical readout circuit for the surface code.

        Args:
            no_observable (bool): If True, do not include observable definitions in the circuit.

        Returns:
            stim.Circuit: The logical readout circuit.
        """
        circuit = stim.Circuit()
        readout_direction = self.readout_direction
        if readout_direction is not None:
            n_data = self.n_data
            n_ancilla_x = self.n_ancilla_x
            n_ancilla_z = self.n_ancilla_z
            dw = self.width
            indices_data = [self.index_data_qubit(i) for i in range(n_data)]
            indices_ancilla_z = [self.index_ancilla_z_qubit(i) for i in range(n_ancilla_z)]
            indices_ancilla_x = [self.index_ancilla_x_qubit(i) for i in range(n_ancilla_x)]

            if readout_direction == 'X':
                if self.before_measure_flip_probability != 0.0:
                    circuit.append('Z_ERROR', indices_data)
                circuit.append('MX', indices_data)
                circuit.append('TICK')
                for i in range(n_ancilla_x):
                    x, y = self.coordinates_ancilla_x_qubit(i)
                    top_left = self.top_left_data_of_ancilla_x(i)
                    top_right = self.top_right_data_of_ancilla_x(i)
                    down_left = self.down_left_data_of_ancilla_x(i)
                    down_right = self.down_right_data_of_ancilla_x(i)
                    data_stab = [top_left, top_right, down_left, down_right]
                    record = [stim.target_rec(-n_ancilla_x - n_data + i)]
                    for data in data_stab:
                        if data is not None:
                            record.append(stim.target_rec(-n_data + data))
                    circuit.append('DETECTOR', record, (x, y, 0))
                if not no_observable:
                    circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-n_data + qubit) for qubit in range(dw)], 0)

            elif readout_direction == 'Z':
                if self.before_measure_flip_probability != 0.0:
                    circuit.append('X_ERROR', indices_data)
                circuit.append('M', indices_data)
                circuit.append('TICK')
                for i in range(n_ancilla_z):
                    x, y = self.coordinates_ancilla_z_qubit(i)
                    top_left = self.top_left_data_of_ancilla_z(i)
                    top_right = self.top_right_data_of_ancilla_z(i)
                    down_left = self.down_left_data_of_ancilla_z(i)
                    down_right = self.down_right_data_of_ancilla_z(i)
                    data_stab = [top_left, top_right, down_left, down_right]
                    record = [stim.target_rec(-n_ancilla_x - n_ancilla_z - n_data + i)]
                    for data in data_stab:
                        if data is not None:
                            record.append(stim.target_rec(-n_data + data))
                    circuit.append('DETECTOR', record, (x, y, 0))
                if not no_observable:
                    circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-n_data + qubit) for qubit in range(0, n_data, dw)], 0)

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
        rounds = self.rounds
        n_data = self.n_data
        dh = self.height
        dw = self.width
        n_ancilla_x = self.n_ancilla_x
        n_ancilla_z = self.n_ancilla_z
        indices_data = [self.index_data_qubit(i) for i in range(n_data)]
        indices_ancilla_z = [self.index_ancilla_z_qubit(i) for i in range(n_ancilla_z)]
        indices_ancilla_x = [self.index_ancilla_x_qubit(i) for i in range(n_ancilla_x)]

        surface_code_circuit = self.initialization_circuit()

        if rounds != 0:
            stab_circuit = stim.Circuit()
            stab_circuit.append('RX', indices_ancilla_z + indices_ancilla_x)

            if self.after_reset_flip_probability != 0.0:
                stab_circuit.append('Z_ERROR', indices_ancilla_z + indices_ancilla_x, self.after_reset_flip_probability)
            stab_circuit.append('TICK')

            if self.before_round is not None:
                error_model = self.before_round[0]
                proba = self.before_round[1]
                if self.noise_on_ancilla:
                    stab_circuit.append(error_model, indices_data + indices_ancilla_z + indices_ancilla_x, proba)
                else:
                    stab_circuit.append(error_model, indices_data, proba)
                stab_circuit.append('TICK')

            if self.measurement_order == 'z_first':
                stab_circuit += self.top_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.down_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H', indices_data)
                stab_circuit.append('TICK')
                stab_circuit += self.top_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.down_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.top_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.down_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit.append('H', indices_data)
                stab_circuit.append('TICK')

            elif self.measurement_order == 'x_first':
                stab_circuit.append('H', indices_data)
                stab_circuit.append('TICK')
                stab_circuit += self.top_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.top_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.down_left_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit += self.down_right_stabilizer_x()
                stab_circuit.append('TICK')
                stab_circuit.append('H', indices_data)
                stab_circuit.append('TICK')
                stab_circuit += self.top_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.down_right_stabilizer_z()
                stab_circuit.append('TICK')

            elif self.measurement_order == 'combined':
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data[:dw * (dh - 1)]) if i % 2 == 0])
                stab_circuit.append('TICK')
                stab_circuit += self.top_left_stabilizer_x()
                stab_circuit += self.top_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data[:dw * (dh - 1)]) if i % 2 == 0])
                stab_circuit.append('TICK')
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data) if i % 2 == 1])
                stab_circuit.append('TICK')
                stab_circuit += self.down_left_stabilizer_x()
                stab_circuit += self.top_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit += self.top_right_stabilizer_x()
                stab_circuit += self.down_left_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data) if i % 2 == 1])
                stab_circuit.append('TICK')
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data) if i % 2 == 0 and i >= dw])
                stab_circuit.append('TICK')
                stab_circuit += self.down_right_stabilizer_x()
                stab_circuit += self.down_right_stabilizer_z()
                stab_circuit.append('TICK')
                stab_circuit.append('H', [index_data for i, index_data in enumerate(indices_data) if i % 2 == 0 and i >= dw])
                stab_circuit.append('TICK')

            if self.before_measure_flip_probability != 0.0:
                stab_circuit.append('Z_ERROR', indices_ancilla_z + indices_ancilla_x, self.before_measure_flip_probability)

            stab_circuit.append('MX', indices_ancilla_z + indices_ancilla_x)
            stab_circuit.append('TICK')

            surface_code_circuit += stab_circuit

            for a in range(n_ancilla_z):
                x, y = self.coordinates_ancilla_z_qubit(a)
                if self.initial_state == '|0>' or self.initial_state == '|1>':
                    surface_code_circuit.append('DETECTOR', stim.target_rec(-n_ancilla_x - n_ancilla_z + a), (x, y, 0))

                if rounds > 1:
                    stab_circuit.append('DETECTOR', [stim.target_rec(-2 * n_ancilla_x - 2 * n_ancilla_z + a),
                                                     stim.target_rec(-n_ancilla_x - n_ancilla_z + a)], (x, y, 0))

            for a in range(n_ancilla_x):
                x, y = self.coordinates_ancilla_x_qubit(a)
                if self.initial_state == '|+>' or self.initial_state == '|->':
                    surface_code_circuit.append('DETECTOR', stim.target_rec(-n_ancilla_x + a), (x, y, 0))

                if rounds > 1:
                    stab_circuit.append('DETECTOR', [stim.target_rec(-2 * n_ancilla_x - n_ancilla_z + a),
                                                     stim.target_rec(-n_ancilla_x + a)], (x, y, 0))

            surface_code_circuit.append('TICK')
            surface_code_circuit.append('SHIFT_COORDS', [], (0, 0, 1))

            stab_circuit.append('TICK')

            if rounds > 1:
                if flatten_loops:
                    for i in range(1, rounds - 1):
                        surface_code_circuit += stab_circuit
                        surface_code_circuit.append('TICK')
                        surface_code_circuit.append('SHIFT_COORDS', [], (0, 0, 1))
                    surface_code_circuit += stab_circuit
                    surface_code_circuit.append('SHIFT_COORDS', [], (0, 0, 1))
                else:
                    if rounds > 2:
                        stab_circuit2 = stab_circuit.copy()
                        stab_circuit2.append('TICK')
                        stab_circuit2.append('SHIFT_COORDS', [], (0, 0, 1))
                        stab_circuit2 *= (rounds - 2)
                        surface_code_circuit += stab_circuit2
                    surface_code_circuit += stab_circuit
                    surface_code_circuit.append('SHIFT_COORDS', [], (0, 0, 1))

        logical_readout = self.logical_readout(no_observable=no_observable)
        surface_code_circuit += logical_readout
        return surface_code_circuit

