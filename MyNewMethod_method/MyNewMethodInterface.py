# This is the example interface file for connecting CHORAS to your simulation method

# Import the relevant functions from your package (/submodule)
from My_New_Method import simulation_method
from simulation_method_interface import SimulationMethod


# This function will be called from the main function below, while it runs
# in the container. The container is started in backend/app/simulation_service. 

class MyNewMethod(SimulationMethod):
    def __init__(self):
        super().__init__()

    def run_simulation(self, json_file_path: str):
        self._mynewmethod_method(json_file_path)

    def _mynewmethod_method(self, json_file_path=None):

        print("mynewmethod_method: starting simulation")

        # Call the appropriate function(s) in your package to simulate
        simulation_method(json_file_path)

        print("mynewmethod_method: simulation done!")


if __name__ == "__main__":
    import os

    from HelperFunctions import (
        find_input_file_in_subfolders,
        create_tmp_from_input,
        save_results,
    )

    # Load the input file
    file_name = find_input_file_in_subfolders(
        os.path.dirname(__file__), "exampleInput_MyNewMethod.json"
    )
    json_tmp_file = create_tmp_from_input(file_name)

    # Run the method
    my_new_method = MyNewMethod()
    my_new_method.run_simulation(json_tmp_file)

    # Save the results to a separate file
    save_results(json_tmp_file)
