# Usage Instructions

## Overview

This script automates the setup process for the project by installing dependencies, building the client, and preparing the necessary assets. However, the final execution step must be run manually by the user.

## Prerequisites

-   **Python** (either `python` or `python3` must be available in the system PATH)
-   **Node.js** (for `npm` commands)
-   **Pip** (for installing Python dependencies)

## Installation and Setup

**Run the setup script:**

```sh
python setup_script.py  # or use python3 if necessary
```

The script will:

1. Install Node.js dependencies (`npm install`)
2. Build the client (`npm run build`)
3. Install Python dependencies (`pip install .` or `pip3 install .`)
4. Copy `client/public/` to `src/semantra/client_public/` (this is so Semantra is able to see the build even if the system can not work with shortcuts or symlinks)
5. Prompt the user to run Semantra

## Running the Application

Once the setup script completes, manually run the following command to start the Semantra script:

```sh
python src/semantra/semantra.py  # Use python3 if needed
```

If you're unsure which Python version to use, refer to the final output of the script, which will print the exact command needed for your system.

You will not have to re-run `setup_script.py` every time you want to use Semantra, only when there are development changes, as any new changes will need to be installed on your system via this script.

Upon running this, you will be prompted with the PDF file to select for analysing via Semantra.

## Troubleshooting

-   If `python` is not recognized, try `python3` instead.
-   Ensure that all dependencies are installed correctly.
-   If the client build fails, make sure you have Node.js and npm installed.

This setup should work on Windows, it has not been tested for macOS or Linux systems.
