# Cache Line & False Sharing Lab - Student 230107140

**Instructor:** Sufyan bin Uzayr  
**Language:** Python 3  
**Submission:** GitHub repository (not Moodle)  
**GPU required?** No. The assignment requires a multicore CPU, preferably with four separate physical cores.

## Quick start in VS Code (Mac / Linux)

1. Extract `CACHE_LAB_230107140_START.zip`.
2. Open the extracted folder in Visual Studio Code: **File > Open Folder**.
3. Open **Terminal > New Terminal**.
4. Install requirements:

   ```bash
   python3 -m pip install numpy numba
   ```

5. Run the completed lab:

   ```bash
   python3 false_sharing_lab.py
   ```

6. Enter **your real GitHub username** when prompted. It is not necessarily the Student ID `230107140`.
7. Do not stop the 60,000,000-updates-per-thread benchmark early. It prints results for strides 1, 4, 8, 16 and the private control. On success look for:

   ```text
   EXACT STATE PARITY: True
   VERIFICATION PASSED: True
   ```

8. Three required files are created in this folder AUTOMATICALLY:

   - `false_sharing_lab.py`
   - `cache_lab_manifest.json`
   - `ANALYSIS.md`

9. Upload these **three separate files** to a GitHub repository named:

   `cache-line-false-sharing-YOUR-GITHUB-USERNAME`

   The script also generates `CACHE_LAB_230107140_GITHUB_FILES.zip` for convenience. **Extract the ZIP and upload the three files; do not submit the ZIP instead.**

## Quick start on Windows (PowerShell, VS Code)

```powershell
py -m pip install numpy numba
py false_sharing_lab.py
```

The code records that direct per-thread CPU affinity is not provided by standard Windows Python; the assignment asks for affinity **where supported**.

## Optional Colab runner

You can use `RUN_IN_COLAB_OPTIONAL.ipynb` if Colab actually assigns **at least four accessible CPU cores**. Many Colab instances expose only two and **cannot** perform this particular assignment correctly. Do not select GPU; it is irrelevant here. The runner checks CPU count before starting and prompts for your GitHub login.

## If the script fails

- **`Need >= 4 logical CPU cores`**: use a different 4+-core host; a GPU does not fix this.
- **`Need 4 CPU cores in the process affinity mask`**: this environment limits CPU access even if `os.cpu_count()` appears higher.
- **`No module named 'numba'`**: install dependencies in the *same* Python interpreter used by VS Code.
- **`verification_passed: false`**: do not submit; review the report and error. Reduced `--iterations` are only for tests and never count as a completed submission.
- **macOS on ARM (Apple Silicon)**: if `hw.cachelinesize` returns a value other than 64 B, the instructor's exact 64-B layout assumptions do not hold. Use a 64-B-line host for strict verification.

## Authenticity and measured data

The Python source is implemented and tested. The manifest and analysis are deliberately generated only at **runtime**, using the **actual** GitHub handle and the current **CPU model**. It is not valid to copy a manifest from somebody else's computer, to invent timings, or to rename a GitHub handle in a manifest after execution.

The LLVM audit also guards against a subtle optimization: a conventional Numba loop can be register promoted, invalidating a false-sharing benchmark. This implementation uses a *compiler-only* memory clobber to preserve an actual `load i64` and `store i64` inside the native loop, without inserting a hardware memory fence.

**Submit only these three files:** `false_sharing_lab.py`, `cache_lab_manifest.json`, `ANALYSIS.md`.
