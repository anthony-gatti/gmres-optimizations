import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import time

# Function to read the sparse matrix from a Matrix Market file and convert it to CSR format
def read_matrix(file_path):
    matrix = sio.mmread(file_path).tocsr()
    return matrix

# IGS-GMRES implementation
def IGS_GMRES(A, b, x0, max_iter, tol):
    n = A.shape[0] # Size of the matrix
    V = np.zeros((n, max_iter + 1)) # Matrix to store Krylov vectors
    H = np.zeros((max_iter + 1, max_iter)) # Hessenberg matrix
    r0 = b - A @ x0 # Initial residual
    beta = np.linalg.norm(r0) # Norm of r0
    V[:, 0] = r0 / beta # Normalized r0
    
    timings = []
    iterations = 0
    mpi_allreduce_count = 0
    
    for k in range(max_iter):
        start_time = time.time()
        iterations += 1
        
        # Arnoldi process
        w = A @ V[:, k]
        # 2 Gauss-Seidel iterations
        for _ in range(2):
            for i in range(k + 1):
                H[i, k] = np.dot(V[:, i], w)
                mpi_allreduce_count += 1 # MPI_ALLreduce equivalent operation
                w -= H[i, k] * V[:, i]
        
        # Compute the new norm and normalize the orthogonal vector
        H[k + 1, k] = np.linalg.norm(w)
        if H[k + 1, k] != 0:
            V[:, k + 1] = w / H[k + 1, k]
        
        # Timing the iteration
        timings.append(time.time() - start_time)
        
        # Check for convergence
        y, residuals, rank, s = np.linalg.lstsq(H[:k + 2, :k + 1], beta * np.eye(k + 2, 1)[:, 0], rcond=None)
        x = x0 + V[:, :k + 1] @ y
        if np.linalg.norm(b - A @ x) / np.linalg.norm(b) < tol:
            break
    
    return x, timings, mpi_allreduce_count, iterations

# Main function
if __name__ == "__main__":
    # Read the sparse symmetric matrix from the file
    matrix_file = './bcsstk26.mtx'
    A = read_matrix(matrix_file)
    
    # Define the right-hand side vector and initial guess
    b = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    
    # Set maximum iterations and tolerance
    max_iter = 100
    tol = 1e-6
    
    # Measure the total execution time
    total_start_time = time.time()
    
    # Solve the linear system using IGS-GMRES
    x, timings, mpi_allreduce_count, iterations = IGS_GMRES(A, b, x0, max_iter, tol)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # Print the results
    print(f"Solution x: {x}\n")
    print(f"Timings per iteration: {timings}\n")
    print(f"Total iterations: {iterations}")
    print(f"Total MPI_ALLreduce equivalent operations: {mpi_allreduce_count}")
    print(f"Total execution time: {total_execution_time:.6f} seconds")