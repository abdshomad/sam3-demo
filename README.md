# SAM3 Studio 🚀

SAM3 Studio is a premium web application for interactive segmenting and tracking of objects in images, videos, and animated GIFs powered by the Segment Anything Model 3 (SAM3).

---

## 🛠️ System Architecture

The application is built as a multi-service containerized stack managed via Docker Compose:

```
                      [ User Browser ]
                             │
                             ▼
                         Port 3058
                     [ Nginx Routing ]
                     ┌───────┴───────┐
                     ▼               ▼
                 / (Static)     /api (JSON/WS)
              [ Next.js ]      [ FastAPI ]
               (Port 3000)      (Port 8000)
```

- **Frontend**: Next.js 16 (App Router) + TypeScript styled using Vanilla CSS/Tailwind. Includes interactive canvas prompting (clicks and text) and an animated asset browser.
- **Backend**: FastAPI (Python) serving grounding queries, interactive click propagation, and frame decomposition routines using SAM3 on CUDA.
- **Routing**: Nginx reverse proxy routing static traffic to the frontend and API traffic to the backend, exposing only port `3058`.
- **Environment**: Managed using `uv` and containerized with BuildKit package caching for fast build stages.

---

## 🚀 Key Features

1. **Interactive Image Point Prompting**: 
   Uses SAM3's official `predict_inst` image interactivity API. Place positive (🟢 foreground) or negative (🔴 background) clicks directly on the original image to generate instant high-fidelity segment masks.
2. **Interactive Text Prompting**:
   Segment concepts in images or propagate object masks across video frames by typing search prompts (e.g., `"truck"`, `"wheel"`, `"person"`).
3. **Animated GIF & Video Propagation**:
   Decomposes uploaded or sample GIFs and videos into JPEG keyframes on-demand, running the SAM3 multiplex video tracker to trace target masks across all frames, exporting results as MP4 videos.
4. **Visual Asset Sidebar Grid**:
   A 3-column thumbnail-only gallery grid in the configuration sidebar with hover title tooltips for seamless sample selection.
5. **Interactive User Onboarding Tour**:
   A step-by-step onboarding walkthrough highlighting the sidebar gallery, prompt controller, and visualization output workspace with spotlight focus rings.
6. **Results Gallery Browser**:
   Inspect all generated segment overlays and tracked MP4 files side-by-side in a dedicated media gallery.

---

## 💻 Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU support)

### Installation & Launch

We provide utility scripts to control the stack in the background:

1. **Install Dependencies & Check Environment**:
   ```bash
   ./install.sh
   ```

2. **Start Services**:
   This runs the Docker Compose stack in the background. It reads port assignments (exposing port `3058` by default) from the `.env` file.
   ```bash
   ./start.sh
   ```

3. **Monitor Status**:
   View container status and output logs:
   ```bash
   ./monitor.sh
   ```

4. **Stop Services**:
   Shut down and clean up containers safely:
   ```bash
   ./stop.sh
   ```

---

## 📂 Project Structure

- `frontend/`: Next.js frontend application.
- `backend/`: FastAPI backend implementation.
- `nginx/`: Configuration files for the reverse proxy.
- `sam3/`: The SAM3 model submodule.
- `results/`: Directory storing static annotated test result assets.
- `plan/`: Architectural planning and feature logs.
- `start.sh`, `stop.sh`, `monitor.sh`, `install.sh`: Service orchestration shell scripts.
