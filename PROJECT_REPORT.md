# 3D GAN Project: Final Report & User Guide

## ⚡ Quick Start: How to Run (Terminal Commands)

To get the entire platform running, open **four** separate terminal windows and run these commands in order:

1.  **Terminal 1 (Redis)**:
    ```bash
    redis-server
    ```
2.  **Terminal 2 (AI Background Worker)**:
    ```bash
    celery -A celery_app worker --loglevel=info --pool=solo
    ```
3.  **Terminal 3 (Backend API)**:
    ```bash
    python main.py
    ```
4.  **Terminal 4 (Frontend UI)**:
    ```bash
    cd frontend
    npm run dev
    ```
    *Access the GUI at: `http://localhost:5173`*

---

## 1. Project Overview
The **3D GAN Project** is a sophisticated AI-driven platform designed to transform **2D images into high-fidelity 3D assets**. By leveraging Generative Adversarial Networks (GANs) and state-of-the-art transformer-based models (TripoSR), the system provides an end-to-end pipeline from image upload to interactive 3D visualization.

The project is built with a scalable, distributed architecture capable of handling heavy AI workloads asynchronously, ensuring a smooth user experience even during complex mesh generation.

---

## 2. Code Map: Where the Magic Happens

If you need to show exactly where the code for each feature is located, here is the directory of the core logic:

| Feature | Primary File(s) | Role in Project |
| :--- | :--- | :--- |
| **AI Model (GAN)** | `inference.py` / `models_extra.py` | Contains the `Generator`, `Discriminator`, and `Encoder` architectures. |
| **AI Training** | `train.py` | The main training loop where the model learns from the Pix3D dataset. |
| **Dataset Logic** | `dataset_pix3d.py` | Custom code to read 3D voxels (`.binvox`) and images from the dataset. |
| **Backend API** | `main.py` | The FastAPI server that handles all requests from the frontend. |
| **Task Queue** | `tasks.py` / `celery_app.py` | Logic for running the AI generation in the background so the UI doesn't freeze. |
| **Frontend UI** | `frontend/src/App.tsx` | The "brain" of the website that coordinates uploading and showing results. |
| **3D Rendering** | `frontend/src/components/Viewer3D.tsx` | The Three.js code that actually renders the 3D object on your screen. |
| **Tests & QA** | `tests/` | Comprehensive test suite for backend, worker, and integration. |
| **Notebooks** | `notebooks/` | Google Colab and local training walkthroughs. |

---

## 3. Project Layout
A clean, organized structure for scalable development:

```text
3D_GAN_PROJECT/
├── frontend/           # React + Three.js Application
├── notebooks/          # Colab & Jupyter training setups
├── outputs/            # Generated assets (.glb, .vox, .npy)
├── tests/              # Unit & Integration test suite
├── celery_app.py       # Worker configuration
├── main.py             # FastAPI API entry point
├── tasks.py            # AI Generation task logic
├── inference.py        # Core AI/GAN Inference pipeline
├── train.py            # Model training entry point
├── dataset_pix3d.py    # Pix3D Data pipeline
├── models_extra.py     # GAN Architectures
├── schemas.py          # API Pydantic models
└── requirements.txt    # Project dependencies
```

---

## 4. Key Achievements & Implementation Details

### 🧠 AI & Model Development
- **Custom GAN Architecture**: Implemented a Voxel-based GAN using PyTorch, featuring an `ImageEncoder`, `Generator`, and `Discriminator`.
- **Pix3D Dataset Integration**: Developed a robust data pipeline for the Pix3D furniture dataset, handling `.binvox` and `.mat` files for 3D supervision.
- **TripoSR Integration**: Integrated a high-performance transformer model for rapid 2D-to-3D conversion, supporting textured `.glb` exports.
- **Resilient Training**: Established a checkpoint-based training loop with automatic VRAM management to handle long-running sessions on consumer-grade GPUs (RTX 3050).

### ⚙️ Backend Engineering
- **Asynchronous Task Queue**: Built with **Celery** and **Redis** to dispatch 3D generation tasks to a background worker, preventing server timeouts.
- **FastAPI Endpoints**: Designed a clean REST API with endpoints for job submission, real-time status polling, and asset retrieval.
- **Multi-Format Export Engine**: Integrated **trimesh** to support diverse 3D industrial formats, including `.glb`, `.obj`, `.stl` (for 3D Printing), and a custom **Voxel Mesh (.vox)** representation.
- **Side-by-Side Comparison**: Automatic "Input in Output" visualization mode showing the original source image directly alongside the generated 3D volumes.
- **Static Asset Serving**: Automated the handling of generated files, serving them via a dedicated static mount with atomic write patterns.
- **Robust Validation**: Implemented Pydantic schemas for strict data validation and a comprehensive test suite (Unit, Integration, and E2E).

### 🎨 Frontend & UX (Premium Overhaul)
- **High-Contrast Design System**: A bespoke "Cyber-Dark" aesthetic featuring glassmorphism, depth-blurs, and vibrant accent glows.
- **Interactive Scanning Upload**: A dedicated reconstruction drop-zone with real-time "laser scanning" laser animations.
- **High-Tech Monitoring Panel**: Real-time progress tracking with hardware status indicators and precise progression metrics.
- **Museum-Grade 3D Viewer**: Integrated Three.js with city-preset environment mapping, contact shadows, and a holographic infinite grid.
- **Modern Typography**: Sophisticated use of 'Outfit' and 'JetBrains Mono' for a professional research-lab feel.

---

## 5. Explaining the GUI (User Flow)

The web interface is designed to be intuitive and visually striking using a **Glassmorphism** design style:

1.  **The Upload & Architecture Stage** (`frontend/src/components/UploadComponent.tsx`):
    -   **Visual**: A glowing glass-morphism container with an intuitive **Format Selector**.
    -   **Function**: Users can choose their target architecture (**GLB** for Web, **OBJ** for Blender, **STL** for 3D Printing, or **PLY**).
    -   **Submission**: It captures the image and selected format, sending them to the FastAPI backend.

2.  **The Generation Stage** (`frontend/src/components/ProgressPanel.tsx`):
    -   **Visual**: A dynamic progress bar with status text (e.g., "AI is dreaming up your mesh...").
    -   **Function**: The frontend "polls" the backend every 2 seconds. It asks: *"Is job #{ID} done yet?"*. The server responds with the current status from **Redis**.

3.  **The Visualization Stage** (`frontend/src/components/Viewer3D.tsx`):
    -   **Visual**: A high-performance 3D canvas using **Three.js**.
    -   **Function**: It downloads the completed `.glb` file and renders it. You can orbit, zoom, and pan to see the object from all angles.

---

## 6. How to Show it (Presentation Script)

If you are presenting this project, follow this flow to impress your audience:

1.  **Start with the Terminal**: Show the **four terminals** running. Explain that "Terminal 2 is the AI Brain (Worker) and Terminal 3 is the Gateway (API)."
2.  **Open the Browser**: Go to `http://localhost:5173`. Point out the clean, modern UI.
3.  **The "Wow" Moment**: Upload a clear image (like a chair). 
4.  **Explain the Wait**: While it processes, explain: "Right now, the image is being sent to a GPU cluster. A GAN is predicting thousands of 3D points (voxels) to recreate this object."
5.  **Multi-Format Power**: Demonstrate the **Format Selector**. Explain that "We don't just generate a preview; we generate ready-to-use industrial files like **STL for 3D printing** or **OBJ for professional animation**."
6.  **Interactive Result**: Once the model appears, **interact with it**. Spin it around. Show that it’s not just a flat image—it’s a real 3D asset.
6.  **Future Potential**: Scroll down and show the "Measurement" and "Physics" buttons as "Coming Soon" features to show the project is built for growth.

---

## 7. Installation & Detailed Setup
Before running the commands above, ensure you have set up the environment:
1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start Redis**:
    ```bash
    redis-server
    ```
3.  **Start Celery Worker** (in a new terminal):
    ```bash
    celery -A celery_app worker --loglevel=info --pool=solo
    ```
4.  **Start FastAPI Server** (in a new terminal):
    ```bash
    python main.py
    ```

### Frontend Setup
1.  **Navigate to Frontend**:
    ```bash
    cd frontend
    ```
2.  **Install Node Modules**:
    ```bash
    npm install
    ```
3.  **Run Development Server**:
    ```bash
    npm run dev
    ```
    The GUI will be available at `http://localhost:5173`.

### Training (Optional)
To train the custom GAN model on the Pix3D dataset:
```bash
python train.py
```

---

## 8. Technical Stack Summary
| Component | Technology |
| :--- | :--- |
| **Language** | Python, TypeScript |
| **AI Framework** | PyTorch, TripoSR, TorchVision |
| **3D Rendering** | Three.js, trimesh, skimage |
| **Backend** | FastAPI, Celery, Redis |
| **Frontend** | React, Vite, Three.js, Tailwind CSS |
| **Testing** | Playwright, Pytest |
| **Storage** | Local File System (outputs/), Redis (state) |
