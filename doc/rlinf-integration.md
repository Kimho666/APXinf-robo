# RLinf integration

`apxinf_robo.integrations.rlinf` presents the engine as an RLinf `BasePolicy`.
The quick start in the [README](../README.md) is the whole procedure; this note
is why it has that shape.

The seam may still move. It has been run end to end from RLinf's own evaluation
entrypoint at one episode per task, not at the published 500, so treat the
shape as settled and the number as not yet reproduced through this path.

**Why those files.** The model registry is module-global state, and RLinf's
rollout and env workers are separate Ray processes that import `rlinf.models`
fresh. Registering from `main()` would populate the driver — enough for
`validate_cfg` to accept the model type — and leave the worker that actually
calls `get_model` holding the built-in registry only.

`rlinf/config.py` is the second file for a different reason: RLinf keeps the
legal `model_type` *names* in `SupportedModel`, a registry separate from the
builder one, and the env worker reads it through `rlinf/envs/action_utils.py`
without ever importing `rlinf.models`. One hook therefore reaches one process.
Register only the builder and the run loads the checkpoint, infers a chunk, and
dies on the first env step — after the model has already produced actions,
which is a confusing place to learn the name was missing.

**Which axis.** RLinf has two extension points: `rollout.rollout_backend`
selects the *worker* — the rollout loop and its transport — and
`rollout.model.model_type` selects the *action model*. Only the second has a
registry; the first is a hardcoded `if`/`elif` in
`evaluations/eval_embodied_agent.py`. Registering on the model axis is what
keeps this to six lines across two files rather than a fork, and it means the
ApxInf number and the openpi number come out of one rollout loop, one env group,
and one set of metrics.

**Why the names are paired.** ApxInf is an engine that hosts several models, so
the registered names are per pair — `pi05_apxinf`, `walloss_apxinf` — rather
than a bare `apxinf`: the slot names a model, so a model goes in it. Inside the
config the two registries sit one level apart and mean different things:

```yaml
model:
  model_type: pi05_apxinf   # RLinf's axis: which builder
  apxinf:
    model_type: pi05        # ApxInf's axis: which policy class
```

Only `pi05_apxinf` has been exercised on LIBERO.

**Why the wrapper script.** [scripts/rlinf_eval.sh](../scripts/rlinf_eval.sh)
stands in for RLinf's `evaluations/run_eval.sh`, which resolves `--config-path`
from the config name and so only finds files inside RLinf's own tree. The
wrapper exports what `run_eval.sh` exports for LIBERO and calls the same
`evaluations/eval_embodied_agent.py`. Copying the config into
`$RLINF/evaluations/libero/` and using `run_eval.sh` works too — both routes
reach the same entrypoint.

**Evaluation only.** `task_type: embodied_eval` launches a rollout group and an
env group, no actor and no optimizer, which is exactly what the engine can do: a
Rust inference runtime with no autograd, so `mode="train"` and `default_forward`
raise rather than returning zero-signal data an RL loop would consume. Train
with RLinf's openpi backend and evaluate the export here.

**The observation bridge.** RLinf hands over positional view stacks
(`main_images` / `wrist_images` / `extra_view_images`); an ApxInf policy wants
named wire keys from a robot preset. Slot *i* of the first goes to key *i* of
the second, which is the slot order the checkpoint's weights were trained with.
Note that RLinf's LIBERO env already rotates frames 180 degrees to match
training, so the bridge deliberately does *not* rotate again — unlike
`apxinf_robo.envs.libero`, which drives the simulator itself and does.

## Shortening a run

The published config is the full 500-episode protocol. Two knobs cut it down
without touching RLinf:

```bash
# one episode per task: the eval reset pool is interleaved by task, so the
# first 10 entries are trial 0 of each of the 10 LIBERO-10 tasks
RLINF=/path/to/RLinf scripts/rlinf_eval.sh libero_10_apxinf_robo_pi05_eval \
    rollout.model.model_path=/ckpt/pi05_libero \
    env.eval.total_num_envs=10

# wiring smoke test: one truncated episode on one task
RLINF=/path/to/RLinf scripts/rlinf_eval.sh libero_10_apxinf_robo_pi05_eval \
    rollout.model.model_path=/ckpt/pi05_libero \
    +env.eval.task_id_filter=[0] \
    env.eval.total_num_envs=1 \
    env.eval.max_episode_steps=80 \
    env.eval.max_steps_per_rollout_epoch=80 \
    env.eval.video_cfg.save_video=False
```

Two constraints RLinf asserts in `validate_cfg`: `total_num_envs` must divide
evenly across the env processes (one per GPU under `env,rollout: all`, so pin
`component_placement` to a single GPU for a small run), and
`max_steps_per_rollout_epoch` must be a multiple of `num_action_chunks`.

Wall clock is not proportional to episode count. The eval loop runs
`max_steps_per_rollout_epoch // num_action_chunks` sequential steps regardless
of how many environments are open; fewer environments make each step cheaper,
not fewer. Shortening `max_steps_per_rollout_epoch` truncates episodes, so a
success rate measured that way is not comparable to the published one.

## Reading the numbers

A run reports **two** success rates, and they are not interchangeable:

```
eval/success_once = 1.0    eval/success_at_end = 0.9    eval/episode_len = 520
```

Both come from the same per-step goal predicate LIBERO returns — *is the bowl in
the drawer?* — and differ only in when it is read. `success_once` is the running
OR over the episode (`rlinf/envs/libero/libero_env.py:738`): did the goal ever
hold. `success_at_end` is that predicate at the final step (`:911`): does it
still hold when the episode stops.

**Why they can differ.** The config sets `ignore_terminations: True`, so success
does not end the episode:

```python
# rlinf/envs/libero/libero_env.py:910-912
if self.ignore_terminations:
    infos["episode"]["success_at_end"] = to_tensor(terminations)
    terminations[:] = False
```

Every episode runs the full `max_episode_steps` — hence `episode_len = 520`
above, for all ten. A policy has no "stop" action and no notion that it is done,
so it keeps predicting chunks after the goal is reached, and continued motion
can undo it: drag the object back out, knock it over. The numbers above are ten
LIBERO-10 tasks at one episode each, and they say every task was solved and one
was then unsolved by its own follow-through.

**Which one to compare.** Against another number from this same protocol,
either, as long as it is the same one on both sides. The number to compare
against is RLinf PR #1537's ApxInf backend run on LIBERO-10 —
`success_once=0.94` / `success_at_end=0.85`, measured on 100 episodes (10 envs x
10 rollout epochs) — so the comparison for the run above is `1.0` vs `0.94`.
Reading our `0.9` against that `0.94` compares `success_at_end` to
`success_once` and inverts the result — this is the easiest mistake to make
here, because both numbers are printed side by side and neither is labelled as
the headline. RLinf publishes no openpi LIBERO-10 number of its own, so there is
no vendor baseline to read these against; #1537's run is ApxInf through a
different seam, not a different engine.

Those 100 episodes are reproducible from this config, and are a *prefix* of it.
The eval reset pool is walked by a cursor that rewinds only at `is_start`
(`libero_env.py:831`) and otherwise advances monotonically (`:567-582`), and
`use_fixed_reset_state_ids: True` comes from the shared `env/libero_10` defaults
on both sides. So `total_num_envs=10 rollout_epoch=50` runs #1537's exact 100
episodes first, in the same order, and then 400 more.

**Against `apxinf-robo eval-libero`, use `success_once`.** That harness stops
the instant success is detected, so its episodes end well short of 520 steps and
it can only ever measure `success_once`; `success_at_end` is not a number it
reports differently, it is a number it structurally cannot observe. The same
holds for ApxInf's own `scripts/eval_libero.py`. Measured on LIBERO-10 at one
episode per task, all three routes agree at 10/10 on that metric.

**`ignore_terminations: True` is not incidental.** It is in the config because
the config is shaped line for line after RLinf's
`libero_10_openpi_pi05_eval.yaml`, so an openpi run of that config and an ApxInf
run of this one are comparable by construction. Turning it off would make
`success_at_end` equal `success_once` and would also break that comparability.

## The engine is not run-to-run reproducible

Measured on Thor, 2026-09-10. **The same command, same seed, same container,
run twice back to back, does not produce the same episodes.** Ten episodes of
LIBERO-10 task 2, `--seed 7 --precision bf16`, in-process:

```
                              first-action ck   replans   success
repeat A vs repeat B                    3/10      3/10      9/10
```

`first_warm_noise_abs_checksum` was identical 10/10 in every comparison, so the
sampling noise is correctly seeded and is *not* the source. The divergence is in
the model forward pass. `eval-libero` exposes no `--autotune` flag and defaults
it off, so it is not kernel autoselection either.

**It is rare, not pervasive.** The forward pass is bitwise reproducible for a
long stretch and then, once, is not. Two runs that start from the same state
stay identical for anywhere between ~90 and ~5700 consecutive inferences before
the first difference appears — two orders of magnitude apart, on the same box,
with nothing else running. That rules out a systematically nondeterministic
reduction and points at a path taken only under particular timing or occupancy.
It also means a short smoke test can easily show perfect agreement and prove
nothing about the next thousand inferences.

**What this costs.** A LIBERO-10 success rate from this stack has run-to-run
variance on top of its binomial error bar, and the two are not separable from a
single run. Three 500-episode runs of the same checkpoint on the same box the
same night:

| route | success_once | success_at_end |
| --- | --- | --- |
| RLinf seam (`scripts/rlinf_eval.sh`) | 0.922 | 0.794 |
| `apxinf-robo eval-libero` | 0.912 | — |
| ApxInf `scripts/eval_libero.py` | 0.940 | — |

Those three numbers span 2.8 points and **that spread is not evidence that the
three routes differ**, because a route compared against *itself* moves too.
Treat any difference below ~3 points on 500 episodes as unresolved.

**What it does not cost.** Equivalence between routes is still observable, in
the prefix. `eval-libero` and `scripts/eval_libero.py` agree bitwise —
first-action checksum, replan count, step count — for the first **103
consecutive episodes** (task 0 x50, task 1 x50, task 2 trials 0-2) and then
separate. 103 identical episodes is not coincidence; the two harnesses are the
same computation. Nondeterminism accumulates until it swamps the signal, so a
short run is the *stronger* equivalence test here, not the weaker one.

**Why it cascades.** Engine state carries across episodes. Task 2 trial 0 is
bitwise identical between two runs that reach it the same way, and different
between a run that reaches it after 100 other episodes and one that starts
there. Once an episode diverges mid-way, every later episode starts from a
different state, so a single mid-episode difference is permanent.

