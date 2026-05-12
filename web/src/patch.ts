export type TrainOptions = { numSteps: number; seed: number };
export type InferOptions = { temperature: number; numSamples: number };

export const DEFAULT_TRAIN: TrainOptions = { numSteps: 200, seed: 42 };
export const DEFAULT_INFER: InferOptions = { temperature: 0.5, numSamples: 20 };

// The one literal that must be rewritten before execution. `random.seed`
// runs at module-load time and seeds the `random.gauss(...)` calls that
// initialize `state_dict`, so it can't be conveyed through a kwarg.
function replaceSeedLiteral(code: string, seed: number): string {
  return code.replace(/^(random\.seed\()\s*\d+\s*(\).*)$/m, `$1${seed}$2`);
}

function replaceLoadDatasetUrl(code: string, inputUrl: string): string {
  return code.replace(
    /^(def load_dataset\(input_url=)(['"]).+?\2(\).*)$/m,
    `$1$2${inputUrl}$2$3`,
  );
}

/**
 * Prepare `microgpt.py` for a Train run.
 *
 * Three changes:
 *   1. Patch the seed literal so `state_dict` is reproducibly initialized
 *      with the user's chosen seed.
 *   2. Patch the `load_dataset` default URL so the module-level
 *      `docs = load_dataset()` reads the file the worker pre-wrote into
 *      Pyodide's FS (the Python derives the filename from the URL).
 *   3. Strip the two demo call sites at the bottom (`train()` and
 *      `infer()`). Running the script then only defines functions and
 *      initializes state; the worker drives `train(num_steps=…)` and
 *      `infer(temperature=…, num_samples=…)` explicitly so it can time
 *      them and surface progress separately.
 *
 * Every other UI-controlled value (num_steps, temperature, num_samples)
 * flows through Python kwargs at call time — no source rewriting needed.
 */
export function prepareTrainingScript(code: string, seed: number, inputUrl: string): string {
  return replaceLoadDatasetUrl(replaceSeedLiteral(code, seed), inputUrl)
    .replace(/^train\(\)\s*$/m, "")
    .replace(/^infer\(\)\s*$/m, "");
}

/**
 * Project the UI's current slider values onto the displayed source so
 * users see "the script that would run if they `python microgpt.py`'d
 * this file." Pure cosmetic — the worker never runs this output. The
 * runtime equivalents are passed as kwargs from the worker, but they
 * derive from the same React state as the values patched here, so the
 * displayed defaults and the live run cannot disagree.
 *
 * Each substitution targets a single column-zero anchor: a top-level
 * statement or a `def` line. We patch only the function signature
 * defaults (not the trailing `train()` / `infer()` call sites), so the
 * source stays idiomatic — defaults declared once, called by name.
 */
export function patchDisplayCode(
  code: string,
  train: TrainOptions,
  infer: InferOptions,
  inputUrl: string,
): string {
  if (!code) return code;
  return replaceLoadDatasetUrl(replaceSeedLiteral(code, train.seed), inputUrl)
    .replace(/^(def train\(num_steps=)\d+(\):.*)$/m, `$1${train.numSteps}$2`)
    .replace(
      /^(def infer\(temperature=)[\d.]+(,\s*num_samples=)\d+(\):.*)$/m,
      `$1${infer.temperature}$2${infer.numSamples}$3`,
    );
}
