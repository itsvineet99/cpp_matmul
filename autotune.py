import subprocess
import re
import itertools

# --- Configuration ---
CPP_FILE = "adv_blocked_parallel_matmul.cpp"
EXECUTABLE = "./adv_blocked_parallel_matmul_tuned"
MATRIX_SIZE = 1024

# The block sizes you want to test
# 16, 32, 64, and 128 are standard hardware-friendly powers of 2
TILE_SIZES = [16, 32, 64, 128] 

any_opt = "third_party/anyoption/anyoption.cpp"
omp_flags = [
    "-Xpreprocessor",
    "-fopenmp",
    "-I/opt/homebrew/opt/libomp/include",
    "-L/opt/homebrew/opt/libomp/lib",
    "-lomp",
]

# The command used to compile the C++ code. 
# -O3 is MANDATORY for performance testing!
BASE_COMPILE_CMD = [
    "clang++",
    "-O3",
    "-std=c++17",
    *omp_flags,
    any_opt,
    CPP_FILE,
    "-o",
    EXECUTABLE,
]

def compile_and_run(tm, tn, tk):
    """Compiles the C++ code with specific macros and runs it."""
    
    # 1. Inject the macros into the compiler command
    # This acts exactly like the #define statements in C++
    macros = [f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}"]
    compile_cmd = BASE_COMPILE_CMD[:1] + macros + BASE_COMPILE_CMD[1:]
    
    try:
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {tm}x{tn}x{tk}:\n{e.stderr}")
        return -1.0

    # 2. Run the compiled executable
    run_cmd = [EXECUTABLE, "-m", str(MATRIX_SIZE), "-n", str(MATRIX_SIZE), "-k", str(MATRIX_SIZE)]
    
    try:
        result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Execution failed for {tm}x{tn}x{tk}:\n{e.stdout}{e.stderr}")
        return -1.0

    # 3. Parse the output for the Blocked GigaFLOPS
    # Looking for: "Matmul time for blocked implementation (ms): 123.4, GigaFLOPS: 45.67"
    match = re.search(r"Matmul time for blocked implementation.*?GigaFLOPS:\s*([0-9.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not parse GigaFLOPS for {tm}x{tn}x{tk}")
        return -1.0

def main():
    print(f"Starting Autotuner for {MATRIX_SIZE}x{MATRIX_SIZE} Matrix Multiplication...")
    print("Sweeping parameter space. This will take a few minutes.\n")
    print(f"{'TM':<5} | {'TN':<5} | {'TK':<5} | {'GigaFLOPS':<10}")
    print("-" * 35)

    best_gflops = 0.0
    best_config = None
    results = []

    # Generate all combinations: (16,16,16), (16,16,32) ... (128,128,128)
    combinations = list(itertools.product(TILE_SIZES, repeat=3))
    
    for tm, tn, tk in combinations:
        gflops = compile_and_run(tm, tn, tk)
        
        print(f"{tm:<5} | {tn:<5} | {tk:<5} | {gflops:<10.2f}")
        
        results.append(((tm, tn, tk), gflops))
        
        if gflops > best_gflops:
            best_gflops = gflops
            best_config = (tm, tn, tk)

    print("\n" + "=" * 35)
    print("TUNING COMPLETE")
    print("=" * 35)
    if best_config:
        print(f"Optimal Configuration found:")
        print(f"TM = {best_config[0]}, TN = {best_config[1]}, TK = {best_config[2]}")
        print(f"Peak Performance: {best_gflops:.2f} GigaFLOPS")
    else:
        print("Failed to find a valid configuration.")

if __name__ == "__main__":
    main()
