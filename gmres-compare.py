import numpy as np
import scipy.io as sio
import time

# Function to read the sparse matrix from a Matrix Market file
def read_matrix(file_path):
    import scipy.io as sio
    return sio.mmread(file_path).tocsr()

# Importing the GMRES implementations
import igs_gmres
import mgs_gmres_1sync
import mgs_gmres_ss

# Function to run and compare the GMRES implementations
def compare_gmres(matrix_file, max_iter=100, tol=1e-6):
    # Read the sparse symmetric matrix from the file
    A = read_matrix(matrix_file)
    
    # Define the right-hand side vector and initial guess
    b = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    
    # List to store results for comparison
    results = []
    
    # Function to run a GMRES implementation and collect results
    def run_gmres(implementation, name):
        start_time = time.time()
        x, timings, mpi_allreduce_count, iterations = implementation(A, b, x0, max_iter, tol)
        total_execution_time = time.time() - start_time
        results.append({
            'name': name,
            'solution': x,
            'timings': timings,
            'mpi_allreduce_count': mpi_allreduce_count,
            'total_execution_time': total_execution_time,
            'iterations': iterations
        })
    
    # Run each GMRES implementation
    run_gmres(igs_gmres.IGS_GMRES, 'IGS-GMRES')
    run_gmres(mgs_gmres_1sync.MGS_GMRES_1SYNC, 'MGS-GMRES-1SYNC')
    run_gmres(mgs_gmres_ss.MGS_GMRES_SS, 'MGS-GMRES-SS')
    
    # Display results for comparison
    for result in results:
        print(f"--- {result['name']} Results ---")
        print(f"Solution x: {result['solution']}\n")
        print(f"Timings per iteration: {result['timings']}\n")
        print(f"Total iterations: {result['iterations']}")
        print(f"Total MPI_ALLreduce equivalent operations: {result['mpi_allreduce_count']}")
        print(f"Total execution time: {result['total_execution_time']:.6f} seconds\n")

if __name__ == "__main__":
    # Path to the matrix file
    matrix_file = './bcsstk26.mtx'
    
    # Run the comparison
    compare_gmres(matrix_file)