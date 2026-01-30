"""
Web-based Human-in-the-Loop image selector for HITL evaluation.

This module provides a Flask web server that displays sample images for all environments
at once, allowing users to click to select the best sample for each environment.
"""

import os
import threading
import time
import cv2
from flask import Flask, render_template_string, jsonify, request, send_file
from typing import List, Optional, Set

# Global state for communication between main script and web server
_state = {
    'images': [],           # List of lists: images[env_idx][sample_idx] = image_path
    'exited_envs': set(),   # Set of environment indices that have been exited
    'waiting': False,       # Whether the main script is waiting for selection
    'selections': None,     # Selected indices after user submits
    'step_number': 0,       # Current step number for display
    'request_id': 0,        # Incremented each time new images are posted
}
_lock = threading.Lock()
_event = threading.Event()
_server_started = False
_app = Flask(__name__)

# Suppress Flask logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HITL Sample Selection</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: #16213e;
            border-radius: 8px;
        }
        .header h1 { margin: 0; font-size: 1.5em; }
        .status {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }
        .status.waiting { background: #4CAF50; }
        .status.idle { background: #666; }
        .submit-btn {
            padding: 12px 32px;
            font-size: 1.1em;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
        }
        .submit-btn:hover { background: #45a049; }
        .submit-btn:disabled { background: #666; cursor: not-allowed; }
        .env-row {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
        }
        .env-row.exited { opacity: 0.5; }
        .env-label {
            min-width: 80px;
            font-weight: bold;
            font-size: 1.1em;
        }
        .samples {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            flex: 1;
        }
        .sample {
            position: relative;
            cursor: pointer;
            border: 4px solid transparent;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s;
        }
        .sample:hover { transform: scale(1.02); }
        .sample.selected { border-color: #4CAF50; box-shadow: 0 0 15px #4CAF50; }
        .sample img {
            display: block;
            width: 200px;
            height: auto;
        }
        .sample-label {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .exit-btn {
            margin-left: 15px;
            padding: 8px 16px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .exit-btn:hover { background: #c0392b; }
        .exit-btn.exited { background: #666; }
        .message {
            text-align: center;
            padding: 60px;
            font-size: 1.3em;
            color: #888;
        }
        .incomplete-warning {
            color: #e74c3c;
            margin-left: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HITL Sample Selection - Step <span id="step">0</span></h1>
        <div>
            <span class="status idle" id="status">Waiting for images...</span>
            <span class="incomplete-warning" id="warning" style="display:none;">Select one sample per env!</span>
            <button class="submit-btn" id="submitBtn" onclick="submitSelections()" disabled>Submit Selections</button>
        </div>
    </div>
    <div id="content">
        <div class="message">Waiting for the script to generate images...</div>
    </div>

    <script>
        let currentRequestId = -1;
        let selections = {};  // env_idx -> sample_idx
        let exitedEnvs = new Set();
        let numEnvs = 0;

        function poll() {
            fetch('/api/images')
                .then(r => r.json())
                .then(data => {
                    if (data.request_id !== currentRequestId && data.waiting) {
                        currentRequestId = data.request_id;
                        selections = {};
                        exitedEnvs = new Set(data.exited_envs || []);
                        numEnvs = data.images.length;
                        renderImages(data);
                    }
                    updateStatus(data.waiting);
                })
                .catch(err => console.error('Poll error:', err))
                .finally(() => setTimeout(poll, 500));
        }

        function renderImages(data) {
            document.getElementById('step').textContent = data.step_number;
            const content = document.getElementById('content');

            if (!data.images || data.images.length === 0) {
                content.innerHTML = '<div class="message">Waiting for the script to generate images...</div>';
                return;
            }

            let html = '';
            data.images.forEach((samples, envIdx) => {
                const isExited = exitedEnvs.has(envIdx);
                html += `<div class="env-row ${isExited ? 'exited' : ''}" id="env-${envIdx}">`;
                html += `<div class="env-label">Env ${envIdx}</div>`;
                html += '<div class="samples">';
                samples.forEach((imgPath, sampleIdx) => {
                    const selected = selections[envIdx] === sampleIdx ? 'selected' : '';
                    html += `<div class="sample ${selected}" onclick="selectSample(${envIdx}, ${sampleIdx})">`;
                    html += `<span class="sample-label">${sampleIdx}</span>`;
                    html += `<img src="/image?path=${encodeURIComponent(imgPath)}" alt="Sample ${sampleIdx}">`;
                    html += '</div>';
                });
                html += '</div>';
                html += `<button class="exit-btn ${isExited ? 'exited' : ''}" onclick="toggleExit(${envIdx})">${isExited ? 'Exited' : 'Exit Env'}</button>`;
                html += '</div>';
            });
            content.innerHTML = html;
            updateSubmitButton();
        }

        function selectSample(envIdx, sampleIdx) {
            if (exitedEnvs.has(envIdx)) return;

            // Toggle selection
            if (selections[envIdx] === sampleIdx) {
                delete selections[envIdx];
            } else {
                selections[envIdx] = sampleIdx;
            }

            // Update UI
            document.querySelectorAll(`#env-${envIdx} .sample`).forEach((el, idx) => {
                el.classList.toggle('selected', idx === selections[envIdx]);
            });
            updateSubmitButton();
        }

        function toggleExit(envIdx) {
            if (exitedEnvs.has(envIdx)) {
                exitedEnvs.delete(envIdx);
            } else {
                exitedEnvs.add(envIdx);
                delete selections[envIdx];
            }

            const row = document.getElementById(`env-${envIdx}`);
            row.classList.toggle('exited', exitedEnvs.has(envIdx));
            row.querySelector('.exit-btn').textContent = exitedEnvs.has(envIdx) ? 'Exited' : 'Exit Env';
            row.querySelector('.exit-btn').classList.toggle('exited', exitedEnvs.has(envIdx));

            // Clear selection if exited
            if (exitedEnvs.has(envIdx)) {
                row.querySelectorAll('.sample').forEach(el => el.classList.remove('selected'));
            }
            updateSubmitButton();
        }

        function updateSubmitButton() {
            const btn = document.getElementById('submitBtn');
            const warning = document.getElementById('warning');

            // Check if all non-exited envs have a selection
            let allSelected = true;
            for (let i = 0; i < numEnvs; i++) {
                if (!exitedEnvs.has(i) && selections[i] === undefined) {
                    allSelected = false;
                    break;
                }
            }

            btn.disabled = !allSelected || numEnvs === 0;
            warning.style.display = (!allSelected && numEnvs > 0) ? 'inline' : 'none';
        }

        function updateStatus(waiting) {
            const status = document.getElementById('status');
            if (waiting) {
                status.textContent = 'Ready for selection';
                status.className = 'status waiting';
            } else {
                status.textContent = 'Processing...';
                status.className = 'status idle';
            }
        }

        function submitSelections() {
            const data = {
                selections: selections,
                exited_envs: Array.from(exitedEnvs)
            };

            fetch('/api/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('content').innerHTML = '<div class="message">Selections submitted! Waiting for next batch...</div>';
                    document.getElementById('submitBtn').disabled = true;
                    selections = {};
                    numEnvs = 0;
                }
            })
            .catch(err => console.error('Submit error:', err));
        }

        // Start polling
        poll();
    </script>
</body>
</html>
"""

@_app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@_app.route('/api/images')
def get_images():
    with _lock:
        return jsonify({
            'images': _state['images'],
            'exited_envs': list(_state['exited_envs']),
            'waiting': _state['waiting'],
            'step_number': _state['step_number'],
            'request_id': _state['request_id'],
        })

@_app.route('/api/submit', methods=['POST'])
def submit_selections():
    data = request.json
    with _lock:
        if not _state['waiting']:
            return jsonify({'success': False, 'error': 'Not waiting for selection'})

        _state['selections'] = data['selections']
        _state['exited_envs'].update(data['exited_envs'])
        _state['waiting'] = False

    _event.set()
    return jsonify({'success': True})

@_app.route('/image')
def serve_image():
    path = request.args.get('path', '')
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return 'Not found', 404


def _start_server(port: int = 5050):
    """Start the Flask server in a background thread."""
    global _server_started
    if _server_started:
        return
    _server_started = True

    def run():
        _app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"\n{'='*60}")
    print(f"HITL Web Selector running at: http://localhost:{port}")
    print(f"Open this URL in your browser to select samples")
    print(f"{'='*60}\n")
    time.sleep(1)  # Give server time to start


def get_user_input_web_selection(all_video_paths: List[List[str]],
                                  exited_envs: Optional[Set[int]] = None,
                                  step_number: int = 0,
                                  port: int = 5050) -> tuple:
    """
    Web-based replacement for get_user_input_direct_selection.

    Displays all environments at once in a web interface, allowing users to
    click to select the best sample for each environment.

    Args:
        all_video_paths: List of lists, where all_video_paths[env_idx][sample_idx] is a video path
        exited_envs: Optional set of environment indices that have already exited
        step_number: Current step number for display
        port: Port to run the web server on

    Returns:
        Tuple of (best_indices, raw_results) matching the original function signature
    """
    # Start server if not already running
    _start_server(port)

    if exited_envs is None:
        exited_envs = set()

    # Validate input
    if not all_video_paths or not isinstance(all_video_paths[0], (list, tuple)):
        raise ValueError("all_video_paths must be a list of lists of video paths per environment")

    n_envs = len(all_video_paths)
    n_samples = len(all_video_paths[0]) if n_envs > 0 else 0

    if n_samples < 2:
        # If only one sample, trivially choose index 0 for each env
        return [0 for _ in range(n_envs)], [[1.0] for _ in range(n_envs)]

    # Extract last frame from each video
    print(f"Extracting last frames from {n_envs * n_samples} videos...")
    all_image_paths = []
    for env_idx, env_videos in enumerate(all_video_paths):
        env_images = []
        for sample_idx, video_path in enumerate(env_videos):
            last_frame_path = _save_last_frame(video_path)
            env_images.append(last_frame_path)
        all_image_paths.append(env_images)

    # Update state and wait for selection
    with _lock:
        _state['images'] = all_image_paths
        _state['exited_envs'] = exited_envs.copy()
        _state['waiting'] = True
        _state['selections'] = None
        _state['step_number'] = step_number
        _state['request_id'] += 1

    _event.clear()
    print(f"Waiting for user selection at http://localhost:{port} ...")
    _event.wait()  # Block until user submits

    # Process results
    with _lock:
        selections = _state['selections']
        updated_exited = _state['exited_envs']

    # Update the caller's exited_envs set
    if exited_envs is not None:
        exited_envs.update(updated_exited)

    # Build return values matching original format
    best_indices = []
    raw_results = []

    for env_idx in range(n_envs):
        if env_idx in updated_exited:
            best_indices.append('exit')
            raw_results.append([0.0] * n_samples)
        else:
            selected = selections.get(str(env_idx), 0)  # JSON keys are strings
            best_indices.append(selected)
            scores = [1.0 if i == selected else 0.0 for i in range(n_samples)]
            raw_results.append(scores)

    if all(x == 'exit' for x in best_indices):
        return ['exit' for _ in range(len(best_indices))], ['exit' for _ in range(len(best_indices))]

    print(f"Selections received: {best_indices}")
    return best_indices, raw_results


def _save_last_frame(video_path: str) -> str:
    """Extract and save the last frame from a video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read the last frame.")

    base, _ = os.path.splitext(video_path)
    output_path = f"{base}_last_frame.png"
    cv2.imwrite(output_path, frame)
    return output_path


# For testing
if __name__ == '__main__':
    # Test with dummy data
    import tempfile
    import numpy as np

    # Create some dummy images
    tmpdir = tempfile.mkdtemp()
    test_videos = []
    for env in range(3):
        env_videos = []
        for sample in range(5):
            # Create a dummy "video" (just an image file for testing)
            img = np.random.randint(0, 255, (224, 672, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, f'env{env}_sample{sample}.mp4')
            # Save as image pretending to be video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 10, (672, 224))
            out.write(img)
            out.release()
            env_videos.append(path)
        test_videos.append(env_videos)

    print("Testing web selector...")
    results = get_user_input_web_selection(test_videos, step_number=1)
    print(f"Results: {results}")
