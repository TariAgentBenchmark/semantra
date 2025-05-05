# Usage Instructions

## Overview

This script automates the build process for the Semantra project by installing dependencies, building the Svelte frontend, copying the files to the Flask static folder, and running the Flask application.

## Prerequisites
Python (either python or python3 must be available in the system PATH)
-   **Python** (either `python` or `python3` must be available in the system PATH)
-   **Node.js** (for `npm` commands)
-   **Pip** (for installing Python dependencies)
-   **Visual Studio Build Tools with C++ development tools** - required for the Annoy library installation. You can download this from the Visual Studio Downloads page (Choose the "Build Tools for Visual Studio" option and select "Desktop development with C++" during installation)

## Installation and Setup

**Run the setup script:**

```sh
python setup_script.py  # or use python3 if necessary
```

The script will:

1. Install Node.js dependencies (`npm install`)
2. Build the client (`npm run build`)
3. Install Python dependencies (`pip install .` or `pip3 install .`)
4. Copy the build from  `client/public/` to `src/semantra/client_public/`
5. Start the Flask application automatically

Note that you should only need to run the setup script once during initial installation or when there are development changes. For regular usage, you can start Semantra directly using the command shown below.

## Running the Application

Once the setup script completes, manually run the following command to start the Semantra:

```sh
python src/semantra/semantra.py  # Use python3 if needed
```

## Controlling the Application

The Flask application will run until you stop it with Ctrl+C

The script will handle graceful shutdown of the server

## Troubleshooting

-   If python is not recognized, the script will automatically try python3 instead.
-   If the build fails, check for errors in the Node.js build process.
-   If you encounter errors during installation, ensure you have the Visual Studio Build Tools properly installed with  "Desktop development with C++" selected.
-   Make sure you have all required dependencies installed.
-   This setup has not been tested on macOS or Linux systems.

## Server Information

The Semantra server will run on the default port 5000.