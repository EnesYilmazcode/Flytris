"""Modal GPU benchmark harness for Flytris.

Setup is intentionally separate from execution. Importing or deploying this app does
not start a GPU. A benchmark runs only through the local entrypoint, for example:

    modal run scripts/modal_flytris.py --gpu L4 --batch 384 --repeats 3

Results are committed to the ``flytris-runs`` Volume under ``benchmarks/``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "flytris"
DATA_MOUNT = "/root/data"
RUNS_MOUNT = "/root/runs"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("flytris-data")
runs_volume = modal.Volume.from_name("flytris-runs")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "torch==2.7.1",
    )
    .add_local_dir(ROOT / "flytris", remote_path="/root/flytris", copy=True)
)

common = {
    "image": image,
    "volumes": {DATA_MOUNT: data_volume, RUNS_MOUNT: runs_volume},
    "cpu": 4,
    "memory": 16384,
    "timeout": 3600,
    "retries": 1,
}
representation_common = {**common, "retries": 0}


def _benchmark(gpu_label: str, batch: int, repeats: int) -> dict:
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from flytris.fly_player import AfterstateFlyPlayer

    if batch < 1 or repeats < 1:
        raise ValueError("batch and repeats must be positive")

    with np.load(Path(RUNS_MOUNT) / "phase1b_boards.npz") as source:
        source_boards = np.concatenate([source["sel_b"], source["prac_b"]])
        source_pieces = np.concatenate([source["sel_p"], source["prac_p"]])
    take = np.arange(batch) % len(source_boards)
    boards = source_boards[take]
    pieces = source_pieces[take]

    torch.cuda.reset_peak_memory_stats()
    player = AfterstateFlyPlayer(
        groups_file=Path(RUNS_MOUNT) / "readout_groups_after.npz",
        max_batch=batch,
    )

    # First pass pays CUDA initialization and allocator costs. Timed passes measure the
    # exact eyes -> recurrent sparse brain -> pooled readout feature path used by Flytris.
    reference = player.features(boards, pieces)
    torch.cuda.synchronize()
    times = []
    max_error = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        observed = player.features(boards, pieces)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        max_error = max(max_error, float(np.max(np.abs(observed - reference))))

    median_seconds = float(np.median(times))
    props = torch.cuda.get_device_properties(0)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_gpu": gpu_label,
        "actual_gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch": batch,
        "brain_steps": int(player.steps),
        "repeats": repeats,
        "seconds": times,
        "median_seconds": median_seconds,
        "fly_seconds_per_wall_second": batch * player.steps * 1e-3 / median_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "total_vram_gib": props.total_memory / 2**30,
        "repeatability_max_abs_error": max_error,
    }
    if max_error != 0.0:
        raise RuntimeError(f"determinism check failed: max abs error {max_error}")

    out = Path(RUNS_MOUNT) / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{gpu_label.lower()}-b{batch}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf8")
    runs_volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(gpu="L4", **common)
def benchmark_l4(batch: int = 384, repeats: int = 3) -> dict:
    return _benchmark("L4", batch, repeats)


@app.function(gpu="A10", **common)
def benchmark_a10(batch: int = 384, repeats: int = 3) -> dict:
    return _benchmark("A10", batch, repeats)


@app.function(gpu="L40S", **common)
def benchmark_l40s(batch: int = 384, repeats: int = 3) -> dict:
    return _benchmark("L40S", batch, repeats)


@app.function(gpu="A100", **common)
def benchmark_a100(batch: int = 384, repeats: int = 3) -> dict:
    return _benchmark("A100", batch, repeats)


@app.function(gpu="H100!", **common)
def benchmark_h100(batch: int = 384, repeats: int = 3) -> dict:
    return _benchmark("H100", batch, repeats)


@app.function(gpu="L4", **representation_common)
def build_v3_representation(batch: int = 384, steps: int = 150,
                            bin_steps: int = 25, keep: int = 512,
                            eyes_mode: str = "pixel", readout_types: str = "",
                            stimulus_mode: str = "after", run_tag: str = "") -> dict:
    """Select spatially distinct, time-binned fly neurons and test teacher agreement."""
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from flytris.brain import Brain, load_neurons
    from flytris.eyes import Eyes, PixelEyes
    from flytris.v3_policy import LinearAfterstatePolicy, enumerate_transitions

    if steps % bin_steps:
        raise ValueError("steps must be divisible by bin_steps")
    with np.load(Path(RUNS_MOUNT) / "phase1b_boards.npz") as source:
        parents = np.concatenate([source["sel_b"], source["prac_b"]])
        pieces = np.concatenate([source["sel_p"], source["prac_p"]])
    boards, landed_boards, lines, groups = [], [], [], []
    for group, (board, piece) in enumerate(zip(parents, pieces)):
        landed, after, cleared, dead = enumerate_transitions(board, int(piece))
        for move in np.flatnonzero(~dead):
            landed_boards.append(landed[move])
            boards.append(after[move])
            lines.append(cleared[move])
            groups.append(group)
    boards = np.asarray(boards, np.uint8)
    landed_boards = np.asarray(landed_boards, np.uint8)
    lines = np.asarray(lines)
    groups = np.asarray(groups, np.int32)

    teacher = LinearAfterstatePolicy.load(Path(RUNS_MOUNT) / "v3" / "model.npz")
    target = teacher.scores(boards, lines)
    target_sum = np.bincount(groups, weights=target, minlength=len(parents))
    group_count = np.bincount(groups, minlength=len(parents))
    target = target - target_sum[groups] / group_count[groups]
    parent_ids = np.arange(len(parents))
    np.random.default_rng(20260915).shuffle(parent_ids)
    train_ids, test_ids = parent_ids[:640], parent_ids[640:]
    train = np.isin(groups, train_ids)

    neurons = load_neurons(["idx", "type", "superclass"])
    if readout_types:
        requested_types = [value.strip() for value in readout_types.split(",") if value.strip()]
        rec_np = neurons.loc[neurons.type.isin(requested_types), "idx"].sort_values().to_numpy(np.int64)
        readout_label = "types_" + "_".join(requested_types)
    else:
        requested_types = []
        rec_np = neurons.loc[neurons.superclass == "visual_projection", "idx"].sort_values().to_numpy(np.int64)
        readout_label = "visual_projection"
    if not len(rec_np):
        raise ValueError("readout selection contains no neurons")
    rec_idx = torch.as_tensor(rec_np, device="cuda")
    if eyes_mode == "pixel":
        eyes = PixelEyes(board_hz=500)
    elif eyes_mode == "features":
        eyes = Eyes(board_hz=500, piece_hz=0.0)
    else:
        raise ValueError("eyes_mode must be pixel or features")
    if stimulus_mode not in ("after", "transition"):
        raise ValueError("stimulus_mode must be after or transition")
    brain = Brain(threshold=5, w_scale=1.0)
    n_bins = steps // bin_steps
    n_features = len(rec_np) * n_bins
    sx = np.zeros(n_features, np.float64)
    sxx = np.zeros(n_features, np.float64)
    sxy = np.zeros(n_features, np.float64)
    sy = syy = 0.0
    n = 0
    started = time.time()

    def simulate(start, stop):
        zeros = np.zeros(stop - start, np.int64)
        if stimulus_mode == "transition":
            p = eyes.probs(eyes.channels(landed_boards[start:stop], zeros))
            p_after = eyes.probs(eyes.channels(boards[start:stop], zeros))
            return brain.run(eyes.in_idx, p, steps, rec_idx, eyes.phase,
                             bin_steps=bin_steps, p_in_after=p_after,
                             switch_step=steps // 2)[0]
        p = eyes.probs(eyes.channels(boards[start:stop], zeros))
        return brain.run(eyes.in_idx, p, steps, rec_idx, eyes.phase,
                         bin_steps=bin_steps)[0]

    # Pass one ranks every individual neuron in every time bin without storing the
    # full 15k x 55k matrix.
    for start in range(0, len(boards), batch):
        stop = min(start + batch, len(boards))
        counts = simulate(start, stop)
        use = train[start:stop]
        x = counts[:, use].T.cpu().numpy().astype(np.float64)
        y = target[start:stop][use]
        sx += x.sum(0)
        sxx += np.square(x).sum(0)
        sxy += x.T @ y
        sy += y.sum()
        syy += y @ y
        n += len(y)
        print(f"screen pass: {stop}/{len(boards)}", flush=True)
    cov = sxy - sx * sy / n
    varx = np.maximum(sxx - sx * sx / n, 0)
    vary = max(syy - sy * sy / n, 1e-12)
    corr = cov / np.sqrt(varx * vary + 1e-12)
    selected = np.argsort(-np.abs(corr), kind="stable")[:keep]

    # Pass two materializes only the selected neural signals for ridge fitting.
    features = np.empty((len(boards), keep), np.float32)
    for start in range(0, len(boards), batch):
        stop = min(start + batch, len(boards))
        counts = simulate(start, stop)
        features[start:stop] = counts[selected].T.cpu().numpy()
        print(f"materialize pass: {stop}/{len(boards)}", flush=True)

    # Candidate scores matter only within a parent board, so remove each parent's
    # shared activity before fitting the ranking readout.
    sums = np.zeros((len(parents), keep), np.float64)
    np.add.at(sums, groups, features)
    x = features - sums[groups] / group_count[groups, None]
    mean, scale = x[train].mean(0), x[train].std(0)
    scale[scale < 1e-6] = 1
    z = (x - mean) / scale

    def top_accuracy(score):
        correct = total = 0
        for group in test_ids:
            idx = np.flatnonzero(groups == group)
            if len(idx):
                correct += idx[np.argmax(score[idx])] == idx[np.argmax(target[idx])]
                total += 1
        return float(correct / total)

    trials = []
    best = None
    for lam in (0.1, 1, 10, 100, 1000, 10000):
        xt = z[train]
        w = np.linalg.solve(xt.T @ xt + lam * np.eye(keep), xt.T @ target[train])
        acc = top_accuracy(z @ w)
        trials.append({"lambda": lam, "heldout_top_move_agreement": acc})
        if best is None or acc > best[0]:
            best = acc, lam, w
    passed = bool(float(best[0]) >= 0.50)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actual_gpu": torch.cuda.get_device_name(0),
        "parents": len(parents), "afterstates": len(boards),
        "recorded_neurons": len(rec_np), "steps": steps, "bin_steps": bin_steps,
        "eyes_mode": eyes_mode, "stimulus_mode": stimulus_mode, "readout": readout_label,
        "screened_features": n_features, "selected_features": keep,
        "heldout_top_move_agreement": float(best[0]), "gate": 0.50, "passed": passed,
        "best_lambda": float(best[1]), "trials": trials,
        "seconds": float(time.time() - started),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    out = Path(RUNS_MOUNT) / "v3"
    if run_tag and any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_tag):
        raise ValueError("run_tag must contain only letters, digits, underscores, and hyphens")
    stem = run_tag or f"{eyes_mode}_{stimulus_mode}_{readout_label}_temporal"
    np.savez(out / f"{stem}_readout.npz",
             neuron_idx=rec_np[selected % len(rec_np)], time_bin=selected // len(rec_np),
             mean=mean, scale=scale, weights=best[2], steps=steps, bin_steps=bin_steps,
             heldout_top_move_agreement=best[0], passed=passed,
             eyes_mode=eyes_mode, stimulus_mode=stimulus_mode, readout=readout_label)
    (out / f"{stem}_gate.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    runs_volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(gpu="L4", **representation_common)
def evaluate_v3_fly(games: int = 8, cap: int = 300, seed0: int = 1_200_000,
                    readout_name: str = "features_transition_types_L1_L2_temporal_readout.npz",
                    run_tag: str = "", control: str = "none") -> dict:
    """Play held-out games whose placement scores come only from connectome spikes."""
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from flytris.tetris import Batch
    from flytris.v3_brain_player import TemporalNeuronPlayer, play_transition_games

    readout = Path(RUNS_MOUNT) / "v3" / readout_name
    player = TemporalNeuronPlayer(readout, max_batch=384)
    if control == "silenced":
        player.weights[:] = 0
    elif control == "shuffled_readout":
        player.weights[:] = player.weights[np.random.default_rng(99173).permutation(len(player.weights))]
    elif control == "shuffled_input":
        permutation = np.random.default_rng(99173).permutation(len(player.eyes.in_idx))
        player.eyes.in_idx = player.eyes.in_idx[torch.as_tensor(permutation, device="cuda")]
    elif control != "none":
        raise ValueError("control must be none, silenced, shuffled_readout, or shuffled_input")
    batches = [Batch(1, seed0 + i) for i in range(games)]
    started = time.time()
    played = play_transition_games(batches, player, cap, record_states=True)
    torch.cuda.synchronize()
    lines = [int(batch.lines[0]) for batch in batches]
    pieces = [int(batch.placed[0]) for batch in batches]
    alive = [bool(batch.alive[0]) for batch in batches]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "controller": f"fly_connectome_{readout.stem}",
        "control": control,
        "seed_range": [seed0, seed0 + games - 1], "cap": cap,
        "games": games, "lines": lines, "pieces": pieces, "alive": alive,
        "mean_lines": float(np.mean(lines)), "best_lines": max(lines),
        "survived": int(sum(alive)), "seconds": float(time.time() - started),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    suffix = run_tag or readout.stem.removesuffix("_readout")
    out = Path(RUNS_MOUNT) / "v3" / f"fly_eval_{suffix}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf8")
    state_boards, state_pieces, state_games = [], [], []
    for game, game_states in enumerate(played["states"]):
        for board, piece in game_states:
            state_boards.append(board)
            state_pieces.append(piece)
            state_games.append(game)
    max_moves = max((len(game_moves) for game_moves in played["moves"]), default=0)
    replay_moves = np.full((max_moves, games), 255, dtype=np.uint8)
    for game, game_moves in enumerate(played["moves"]):
        replay_moves[:len(game_moves), game] = np.asarray(game_moves, dtype=np.uint8)
    np.savez(Path(RUNS_MOUNT) / "v3" / f"fly_states_{suffix}.npz",
             boards=np.asarray(state_boards, np.uint8), pieces=np.asarray(state_pieces),
             games=np.asarray(state_games), seeds=np.arange(seed0, seed0 + games),
             moves=replay_moves, lines=np.asarray(lines), placed=np.asarray(pieces))
    runs_volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(gpu="L4", **representation_common)
def dagger_v3_iteration(
    readout_name: str = "features_transition_types_L1_L2_temporal_readout.npz",
    states_name: str = "fly_states_features_transition_types_L1_L2_temporal.npz",
    iteration: int = 1,
    teacher_parents: int = 300,
    batch: int = 384,
    head: str = "linear",
) -> dict:
    """Retrain the neural ranking head on teacher and fly-visited transition states."""
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from flytris.v3_brain_player import TemporalNeuronPlayer
    from flytris.v3_policy import LinearAfterstatePolicy, enumerate_transitions, play_policy

    root = Path(RUNS_MOUNT) / "v3"
    source_readout = root / readout_name
    fly_board_parts, fly_piece_parts, fly_game_parts = [], [], []
    game_offset = 0
    for name in [value.strip() for value in states_name.split(",") if value.strip()]:
        with np.load(root / name) as fly:
            fly_board_parts.append(fly["boards"].copy())
            fly_piece_parts.append(fly["pieces"].copy())
            games_part = fly["games"].copy() + game_offset
            fly_game_parts.append(games_part)
            game_offset = int(games_part.max()) + 1
    fly_boards = np.concatenate(fly_board_parts)
    fly_pieces = np.concatenate(fly_piece_parts)
    fly_games = np.concatenate(fly_game_parts)

    teacher = LinearAfterstatePolicy.load(root / "model.npz")
    teacher_states, teacher_games = [], []
    teacher_seed = 1_300_000
    game = 0
    while len(teacher_states) < teacher_parents:
        # Short independent trajectories ensure both train and held-out teacher games.
        result = play_policy(teacher, teacher_seed + game, cap=75, record_states=True)
        take = min(len(result["states"]), teacher_parents - len(teacher_states))
        teacher_states.extend(result["states"][:take])
        teacher_games.extend([game] * take)
        game += 1
    teacher_boards = np.asarray([state[0] for state in teacher_states], np.uint8)
    teacher_pieces = np.asarray([state[1] for state in teacher_states])

    parents = np.concatenate([fly_boards, teacher_boards])
    pieces = np.concatenate([fly_pieces, teacher_pieces])
    trajectory = np.concatenate([fly_games, np.asarray(teacher_games) + fly_games.max() + 1])
    landed_all, after_all, lines_all, groups_all = [], [], [], []
    for group, (board, piece) in enumerate(zip(parents, pieces)):
        landed, after, lines, dead = enumerate_transitions(board, int(piece))
        valid = np.flatnonzero(~dead)
        landed_all.append(landed[valid])
        after_all.append(after[valid])
        lines_all.append(lines[valid])
        groups_all.append(np.full(len(valid), group, np.int32))
    landed = np.concatenate(landed_all)
    after = np.concatenate(after_all)
    lines = np.concatenate(lines_all)
    groups = np.concatenate(groups_all)
    target = teacher.scores(after, lines)
    group_count = np.bincount(groups, minlength=len(parents))
    target_sum = np.bincount(groups, weights=target, minlength=len(parents))
    target = target - target_sum[groups] / group_count[groups]

    unique_trajectories = np.unique(trajectory)
    # Hold out the final fly game and final teacher trajectory as complete sequences.
    test_trajectories = np.array([fly_games.max(), unique_trajectories.max()])
    parent_train = ~np.isin(trajectory, test_trajectories)
    train = parent_train[groups]
    test_parent_ids = np.flatnonzero(~parent_train)

    player = TemporalNeuronPlayer(source_readout, max_batch=batch)
    started = time.time()
    features = player.transition_features(landed, after).astype(np.float64)
    sums = np.zeros((len(parents), features.shape[1]), np.float64)
    np.add.at(sums, groups, features)
    x = features - sums[groups] / group_count[groups, None]

    def accuracy(score):
        correct = total = 0
        for group in test_parent_ids:
            idx = np.flatnonzero(groups == group)
            if len(idx):
                correct += idx[np.argmax(score[idx])] == idx[np.argmax(target[idx])]
                total += 1
        return float(correct / total)

    trials, best = [], None
    xt = x[train]
    prior = player.weights.astype(np.float64)
    mlp_arrays = {}
    if head == "linear":
        for lam in (0.1, 1, 10, 100, 1000, 10000):
            w = np.linalg.solve(xt.T @ xt + lam * np.eye(xt.shape[1]),
                                xt.T @ target[train] + lam * prior)
            acc = accuracy(x @ w)
            trials.append({"lambda": lam, "heldout_top_move_agreement": acc})
            if best is None or acc > best[0]:
                best = acc, lam, w
    elif head in ("mlp", "mlp_rank"):
        torch.manual_seed(20260915 + iteration)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.Tanh(),
                                    torch.nn.Linear(64, 1)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        tx = torch.as_tensor(x[train], dtype=torch.float32, device="cuda")
        target_scale = float(np.std(target[train]) + 1e-6)
        ty = torch.as_tensor(target[train] / target_scale, dtype=torch.float32, device="cuda")
        if head == "mlp_rank":
            train_groups = [group for group in np.flatnonzero(parent_train) if group_count[group] > 0]
            width = max(group_count[group] for group in train_groups)
            group_matrix = np.zeros((len(train_groups), width), np.int64)
            group_valid = np.zeros_like(group_matrix, bool)
            group_label = np.zeros(len(train_groups), np.int64)
            for row, group in enumerate(train_groups):
                idx = np.flatnonzero(groups == group)
                group_matrix[row, :len(idx)] = idx
                group_valid[row, :len(idx)] = True
                group_label[row] = int(np.argmax(target[idx]))
            group_matrix = torch.as_tensor(group_matrix, device="cuda")
            group_valid = torch.as_tensor(group_valid, device="cuda")
            group_label = torch.as_tensor(group_label, device="cuda")
            all_x = torch.as_tensor(x, dtype=torch.float32, device="cuda")
        rng = np.random.default_rng(20260915 + iteration)
        best_state = None
        epochs = 100 if head == "mlp_rank" else 60
        for epoch in range(epochs):
            if head == "mlp_rank":
                score_all = model(all_x)[:, 0]
                grouped = score_all[group_matrix].masked_fill(~group_valid, -1e9)
                loss = torch.nn.functional.cross_entropy(grouped, group_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                order = rng.permutation(len(tx))
                for start in range(0, len(order), 2048):
                    idx = torch.as_tensor(order[start:start + 2048], device="cuda")
                    loss = torch.nn.functional.mse_loss(model(tx[idx])[:, 0], ty[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                prediction = model(torch.as_tensor(x, dtype=torch.float32, device="cuda"))[:, 0]
                score = prediction.cpu().numpy() * target_scale
            acc = accuracy(score)
            trials.append({"epoch": epoch + 1, "heldout_top_move_agreement": acc})
            if best is None or acc > best[0]:
                best = acc, epoch + 1, None
                best_state = {key: value.detach().cpu().numpy().copy()
                              for key, value in model.state_dict().items()}
        mlp_arrays = {
            "mlp_w1": best_state["0.weight"].T,
            "mlp_b1": best_state["0.bias"],
            "mlp_w2": best_state["2.weight"][0] * target_scale,
            "mlp_b2": np.asarray(best_state["2.bias"][0] * target_scale),
        }
    else:
        raise ValueError("head must be linear, mlp, or mlp_rank")

    out_name = f"dagger_{iteration:02d}_{head}_{Path(readout_name).name}"
    with np.load(source_readout) as source:
        arrays = {key: source[key] for key in source.files}
    if head == "linear":
        arrays["weights"] = best[2].astype(np.float32)
    arrays.update(mlp_arrays)
    arrays["head"] = np.asarray(head)
    arrays["dagger_iteration"] = np.asarray(iteration)
    arrays["heldout_top_move_agreement"] = np.asarray(best[0])
    np.savez(root / out_name, **arrays)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "iteration": iteration, "head": head,
        "source_readout": readout_name, "output_readout": out_name,
        "fly_parents": len(fly_boards), "teacher_parents": len(teacher_boards),
        "candidate_transitions": len(landed),
        "heldout_trajectories": test_trajectories.tolist(),
        "heldout_top_move_agreement": best[0],
        ("best_lambda" if head == "linear" else "best_epoch"): best[1],
        "trials": trials, "seconds": float(time.time() - started),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
    }
    (root / f"dagger_{iteration:02d}_{head}_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf8")
    runs_volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(gpu="L4", **representation_common)
def evolve_v3_fly(
    generations: int = 1,
    pop: int = 16,
    shortlist: int = 4,
    elites: int = 2,
    race_seeds: int = 3,
    cap: int = 150,
    init_readout: str = "features_transition_types_L1_L2_temporal_readout.npz",
    init_std: float = 0.05,
    run_name: str = "evolution",
) -> dict:
    """Checkpointed CEM using only genuine connectome-controlled gameplay fitness."""
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/root")
    from flytris.tetris import Batch
    from flytris.v3_brain_player import TemporalNeuronPlayer, play_transition_population

    if not 0 < elites <= shortlist <= pop:
        raise ValueError("require elites <= shortlist <= pop")
    if not run_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_name):
        raise ValueError("run_name must contain only letters, digits, underscores, and hyphens")
    root = Path(RUNS_MOUNT) / "v3" / run_name
    root.mkdir(parents=True, exist_ok=True)
    source_path = Path(RUNS_MOUNT) / "v3" / init_readout
    player = TemporalNeuronPlayer(source_path, max_batch=384)
    state_path = root / "state.npz"
    if state_path.exists():
        with np.load(state_path) as state:
            generation = int(state["generation"])
            mean, std = state["mean"], state["std"]
            champion = state["champion"]
            champion_fitness = float(state["champion_fitness"])
        expected = len(player.weights) + 1
        if len(mean) != expected or len(champion) != expected:
            raise ValueError(
                f"checkpoint has {len(mean)} parameters but readout requires {expected}; "
                "use a new run_name"
            )
    else:
        generation = 0
        mean = np.r_[player.weights, np.float32(0)].astype(np.float32)
        std = np.full_like(mean, init_std)
        champion, champion_fitness = mean.copy(), -np.inf
    started = time.time()
    summaries = []
    for _ in range(generations):
        rng = np.random.default_rng([73, generation])
        candidates = (mean + rng.standard_normal((pop, len(mean))) * std).astype(np.float32)
        candidates[0] = champion if np.isfinite(champion_fitness) else mean
        seed0 = 1_400_000 + generation * race_seeds
        primary = play_transition_population(Batch(pop, seed0), candidates, player, cap)
        primary_fitness = primary.lines.astype(np.float64) * 10_000 + primary.placed
        finalist = np.argsort(-primary_fitness, kind="stable")[:shortlist]
        all_fitness = [primary_fitness[finalist]]
        race_lines = [primary.lines[finalist].copy()]
        for offset in range(1, race_seeds):
            race = play_transition_population(Batch(shortlist, seed0 + offset),
                                              candidates[finalist], player, cap)
            all_fitness.append(race.lines.astype(np.float64) * 10_000 + race.placed)
            race_lines.append(race.lines.copy())
        aggregate = np.stack(all_fitness).mean(0)
        order = np.argsort(-aggregate, kind="stable")
        elite = finalist[order[:elites]]
        best_row = int(order[0])
        best_idx = int(finalist[best_row])
        if aggregate[best_row] > champion_fitness:
            champion = candidates[best_idx].copy()
            champion_fitness = float(aggregate[best_row])
        mean = candidates[elite].mean(0)
        std = np.maximum(candidates[elite].std(0), 0.01)
        summary = {
            "generation": generation, "seed0": seed0,
            "primary_best_lines": int(primary.lines.max()),
            "race_best_mean_lines": float(np.stack(race_lines)[:, best_row].mean()),
            "champion_fitness": champion_fitness,
        }
        summaries.append(summary)
        np.savez(root / f"gen_{generation:04d}.npz", candidates=candidates,
                 primary_lines=primary.lines, finalist=finalist,
                 race_lines=np.stack(race_lines), aggregate_fitness=aggregate, elites=elite)
        generation += 1
        np.savez(state_path, generation=generation, mean=mean, std=std,
                 champion=champion, champion_fitness=champion_fitness)
        with np.load(source_path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["weights"] = champion[:-1]
        arrays["head"] = np.asarray("linear")
        arrays["evolution_generation"] = np.asarray(generation - 1)
        np.savez(root / "champion_readout.npz", **arrays)
        runs_volume.commit()
        print(json.dumps(summary, indent=2), flush=True)
    result = {"completed_generations": generations, "next_generation": generation,
              "pop": pop, "shortlist": shortlist, "race_seeds": race_seeds,
              "cap": cap, "summaries": summaries,
              "seconds": float(time.time() - started)}
    (root / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    runs_volume.commit()
    return result


@app.local_entrypoint()
def main(gpu: str = "L4", batch: int = 384, repeats: int = 3,
         job: str = "benchmark", steps: int = 150, bin_steps: int = 25,
         keep: int = 512, eyes_mode: str = "pixel", readout_types: str = "",
         games: int = 8, cap: int = 300, seed0: int = 1_200_000,
         stimulus_mode: str = "after",
         readout_name: str = "features_transition_types_L1_L2_temporal_readout.npz",
         run_tag: str = "",
         control: str = "none",
         states_name: str = "fly_states_features_transition_types_L1_L2_temporal.npz",
         iteration: int = 1, teacher_parents: int = 300, head: str = "linear",
         generations: int = 1, pop: int = 16, shortlist: int = 4,
         elites: int = 2, race_seeds: int = 3, init_std: float = 0.05,
         run_name: str = "evolution") -> None:
    if job.lower() == "representation":
        result = build_v3_representation.remote(batch=batch, steps=steps,
                                                bin_steps=bin_steps, keep=keep,
                                                eyes_mode=eyes_mode,
                                                readout_types=readout_types,
                                                stimulus_mode=stimulus_mode,
                                                run_tag=run_tag)
        print(json.dumps(result, indent=2))
        return
    if job.lower() == "fly-eval":
        result = evaluate_v3_fly.remote(games=games, cap=cap, seed0=seed0,
                                        readout_name=readout_name, run_tag=run_tag,
                                        control=control)
        print(json.dumps(result, indent=2))
        return
    if job.lower() == "dagger":
        result = dagger_v3_iteration.remote(readout_name=readout_name,
                                            states_name=states_name,
                                            iteration=iteration,
                                            teacher_parents=teacher_parents,
                                            batch=batch, head=head)
        print(json.dumps(result, indent=2))
        return
    if job.lower() == "evolve":
        result = evolve_v3_fly.remote(generations=generations, pop=pop,
                                      shortlist=shortlist, elites=elites,
                                      race_seeds=race_seeds, cap=cap,
                                      init_readout=readout_name, init_std=init_std,
                                      run_name=run_name)
        print(json.dumps(result, indent=2))
        return
    if job.lower() != "benchmark":
        raise ValueError("job must be benchmark, representation, fly-eval, dagger, or evolve")
    choices = {
        "L4": benchmark_l4,
        "A10": benchmark_a10,
        "L40S": benchmark_l40s,
        "A100": benchmark_a100,
        "H100": benchmark_h100,
    }
    key = gpu.upper()
    if key not in choices:
        raise ValueError(f"gpu must be one of {', '.join(choices)}")
    result = choices[key].remote(batch=batch, repeats=repeats)
    print(json.dumps(result, indent=2))
