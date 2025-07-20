# SearchEngine

A powerful search engine leveraging local embeddings and a web interface.

This guide will walk you through setting up and running the project.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.12 is highly recommended. If you use a different version, you may need to compile Cython extensions manually.
*   **Pip:** Python package installer.
*   **(Optional but Recommended for Cython) C++ Compiler:**
    *   On **Windows**: Microsoft Visual C++ (MSVC) Build Tools. You can get these by installing "Build Tools for Visual Studio" (select C++ build tools workload).


## Setup and Installation

1.  **Create and Activate a Virtual Environment:**
    It's strongly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Create a virtual environment (named 'venv' here) which runs on python 3.12 (preferable)
    py -3.12 -m venv venv

    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS and Linux:
    source venv/bin/activate
    ```
    You should see `(venv)` at the beginning of your command prompt.

2.  **Install Dependencies:**
    Install all required packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build Cython Extensions (if necessary):**
    *   If you are using **Python 3.12**, the pre-compiled Cython files should ideally work.
    *   If you are **not** using Python 3.12, or if you encounter issues related to Cython modules (e.g., `ImportError` for `.pyd` or `.so` files), you will need to build them manually.
    *   Ensure you have a C++ compiler installed (see Prerequisites).

    Run the following command from the project's root directory:
    ```bash
    python setup.py build_ext --inplace
    ```
    This will compile the `.pyx` files into C code and then into Python extension modules.

## Running the Application

You can run this project in two ways:

### 1. Web Application

To launch the web application:
```bash
python app.py
```
Once started, open your web browser and navigate to the address shown in the console (typically `http://127.0.0.1:5000` or `http://localhost:5000`).

### 2. Command-Line Interface (CLI)

To use the search engine directly from the command line:
```bash
python -m SearchEngine.main [arguments]
```
<!-- Optional: If your CLI takes arguments, briefly explain them here or link to more detailed documentation. -->
<!-- e.g., `python -m SearchEngine.main --query "your search query"` -->

## Important Considerations

*   **Embeddings Model:**
    For optimal performance and privacy, consider using a local model for generating embeddings. Configuration for this might be in a settings file or passed as an argument.

*   **CUDA and Performance:**
    *   Processing large documents or large datasets can be computationally intensive.
    *   If you have a **CUDA-enabled NVIDIA GPU**, the application may be able to leverage it for significantly faster embedding generation and other computations.
    *   You can usually check for CUDA availability in the console output/logs when the application starts.
    *   **If you do not have CUDA support**, be mindful when processing very large documents, as it may be slow and consume considerable CPU resources.

## Troubleshooting

*   **Cython Build Errors:** Ensure you have the correct C++ compiler and Python development headers installed for your operating system (see Prerequisites). Double-check that your virtual environment is active.
*   **`ModuleNotFoundError` for `.pyd` or `.so` files:** This usually means the Cython extensions were not built correctly or are not in the right place. Try re-running `python setup.py build_ext --inplace`.