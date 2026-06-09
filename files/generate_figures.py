
Claude finished the response
1780840705055_image.png
msc_thesis_biadrg.pdf
Act as an expert quantum machine learning engineer.
Context: I am writing a master's thesis on "Federated Quantum Transformer Architectures for Time-Series Forecasting." I need to adapt three architectures to my federated time-series data:

QuLTSF (from the chariharasuthan/qultsf repository)
QSANN (Quantum Self-Attention Neural Networks)
QuantumTSTransformer (from the Transconnectome/TSQuantumTransformer repository) Read the attached thesis draft and analyse the attached directory structure image. Provide a highly detailed, step-by-step implementation guide for each architecture. Required Steps:
Setup and Installation

Provide the exact git clone commands for the repositories.
List the environment setup commands (e.g., Conda, pip) and identify any specific quantum libraries (like PennyLane or Qiskit) I need.
Data Preprocessing

Write Python code to load my data from the provided directory structure and format it for each model.
Detail the steps to normalise the data and split it across simulated decentralised nodes for federated learning.
Explain how to adapt QSANN from text classification to continuous time-series data.
Explain how to convert QuantumTSTransformer from fMRI spatial-temporal data to standard time-series data.
Implementation and Federated Integration

Provide the exact terminal commands to run the algorithms.
Write the specific Python code to wrap these architectures in a federated learning loop (e.g., FedAvg protocol) to match my thesis framework.
If a model needs classical layers replaced by quantum circuits, provide the exact code to do so.
Analysis

State exactly what metrics (MSE, MAE, etc.) to monitor.
Explain how to track the quantum attention mechanisms during training to evaluate any structural sources of quantum advantage.
Provide guidance on logging and visualising the results to include in my thesis.
Show more
Jun 7

Verified thesis legitimacy, strategized repository inspection to avoid hallucination, planned comprehensive implementation guide
Verified thesis legitimacy, strategized repository inspection to avoid hallucination, planned comprehensive implementation guide
I'll approach this properly — rather than write a guide from memory (which risks giving you fictional file paths and entry points for a thesis), I'm going to actually inspect the three repositories and the dataset formats, then build the guide against what's really there. Let me start.


Locating official QSANN repository and cloning quantum architecture code
Locating official QSANN repository and cloning quantum architecture code
Both cloned. Now let me find the QSANN reference implementation and inspect everything.


Identified codebase structure and prioritized key files for analysis
Identified codebase structure and prioritized key files for analysis
Good — real structure recovered. Let me read the key files in each repo.


Deciphered QuLTSF data architecture, then pivoted toward quantum transformer examination
Deciphered QuLTSF data architecture, then pivoted toward quantum transformer examination
Now the TSQuantumTransformer (Quixer) repo.


Verifying torchquantum compatibility and reconciling repository dependencies
Verifying torchquantum compatibility and reconciling repository dependencies
I have full, accurate pictures of QuLTSF and TSQuantumTransformer. Two quick verifications before I write — the QSANN reference situation and the torchquantum dependency.


Identified framework conflicts, flagged architectural mismatches, architected comprehensive implementation strategy
Identified framework conflicts, flagged architectural mismatches, architected comprehensive implementation strategy
I now have an accurate, ground-truth picture of all three codebases. Before writing the guide, let me smoke-test the most novel pieces (the from-scratch QSANN and the framework-agnostic FedAvg harness) so I hand you code that actually runs, not plausible-looking fiction.


Validated quantum circuits with PennyLane, devised smoke tests for core components
Validated quantum circuits with PennyLane, devised smoke tests for core components
PennyLane 0.45 works (torch won't fit on disk, but the error-prone parts are the quantum circuit math and the FedAvg averaging, which I can validate without torch). Let me smoke-test those.


Validated quantum circuits, architected modular implementation strategy across federated components
Validated quantum circuits, architected modular implementation strategy across federated components
Everything checks out — the quantum Q/K/V circuit, the classical softmax assembly, and the FedAvg weighted averaging all produce correct results. I now have ground truth on all three repos and validated the novel code. Writing the deliverables now.


Devised syntax validation strategy circumventing missing dependencies
Devised syntax validation strategy circumventing missing dependencies
Let me syntax-check all four modules (py_compile doesn't need the heavy deps installed).


Orchestrated federated training pipeline with driver and analysis modules
Orchestrated federated training pipeline with driver and analysis modules
Now the federated driver and the analysis/visualization module, then the master guide.


Architected comprehensive master guide structuring quantum federated learning implementation
Architected comprehensive master guide structuring quantum federated learning implementation
Now the master guide that ties it together. This is the primary deliverable.


Orchestrated comprehensive guide compilation across quantum architectures
Orchestrated comprehensive guide compilation across quantum architectures
I built this against the actual repositories rather than from memory — I cloned both chariharasuthan/qultsf and Transconnectome/TSQuantumTransformer, read their source, and validated the novel code (the QSANN Q/K/V circuit, the softmax assembly, and the FedAvg averaging all ran correctly). That surfaced three things that change how you should frame the thesis, so I'll lead with those rather than the install commands.

The reality check (Section 0 of the guide):

QuLTSF is not a transformer and has no attention. The real model is Linear → AmplitudeEmbedding → StronglyEntanglingLayers → ⟨PauliZ⟩ → Linear, applied channel-independently. Your Step 4 ("track the quantum attention") is literally inapplicable to it. The right move is to demote QuLTSF to the quantum baseline / no-attention ablation — which is what makes the other two models' attention curves interpretable. Trying to report "QuLTSF attention" would be a reviewer red flag.
The three live on two frameworks. QuLTSF + QSANN are PennyLane; Quixer is torchquantum. Their pinned dependencies genuinely conflict, so the guide uses two conda environments and a framework-agnostic FedAvg loop that only ever touches state_dict() — one harness wraps all three unchanged.
Quixer hard-requires power-of-two seq_len (n_ctrl_qubits = log2(n_timesteps)). And its model is already generic time-series — the fMRI specificity is only in the driver, so "converting from fMRI" is mostly discarding the phenotype loader and feeding forecasting windows, plus using its ignored second return value (LCU success probability) as a loss regulariser.
The QSANN adaptation (text→TS) and Quixer adaptation (fMRI→TS) are both fully implemented, with the exact thesis equations (16–19, plus the positional-embedding choice from your Sec. V-E). For the "structural advantage" analysis, only QSANN gives a true L×L attention map — I added an attention-entropy tracker so you can show the quantum attention specialising across federated rounds, which is your cleanest advantage figure.

Two honest caveats: the data loaders are written against the documented column layouts of these public datasets because the actual CSV contents weren't in what you uploaded (only the directory screenshot and the PDF) — verify the *_COLS constants against your files. And the torch-dependent modules are syntax-checked but couldn't be fully executed here (no GPU/torch in this sandbox); the quantum-circuit math and FedAvg core were executed and pass.

Want me to also wire up the QuLTSF-into-FedAvg adapter as a runnable file (Section 3.2 sketches it but I left it as a stub), or draft the experiment-runner that sweeps the {encoding} × {dataset} × {model} grid from Section 4.3?


Thesis implementation guide
Document · MD 

Federated
PY 

Qsann ts
PY 

Quixer forecast
PY 

Partition data
PY 

Train federated
PY 

Analysis
PY 
i have downloaded the files. please provide steps for running the implementation.

Jun 8
Before I give you run commands, let me check one thing that will determine the exact steps — whether you're on Windows (WSL Ubuntu, from your screenshot) or running natively on Linux.

Looking at your directory screenshot: BIADRG [WSL: UBUNTU] — so you're on Windows with WSL2. That changes a few things (path style, conda setup, torchquantum build). Let me verify the actual state of your environment before writing steps that don't work.


Architected comprehensive WSL-based execution roadmap for quantum federated learning experiments
Architected comprehensive WSL-based execution roadmap for quantum federated learning experiments
Good — I have everything I need. The use_gpu defaults to True but auto-falls back to CPU if CUDA isn't available (torch.cuda.is_available()), so WSL users are safe. Let me now write the exact run guide.


Created a file, read a file
Created a file, read a file
Eight parts, fully grounded in what I actually read from your repos. A few things worth calling out explicitly:

--num_workers 0 is non-negotiable on WSL. The default is 10, and WSL's process forking breaks DataLoader silently — it either hangs forever or kills workers with no error message. Every run_longExp.py call needs that flag.

The Quixer PYTHONPATH export (export PYTHONPATH=$PYTHONPATH:.../TSQuantumTransformer) needs to happen before any Quixer run, because quixer_forecast.py does from QuixerTSModel import QuixerTimeSeries. Put the export in ~/.bashrc so it survives new terminals.

The recommended development iteration order: run Part 4 (QuLTSF, --train_epochs 3) first. That single command validates your entire data pipeline — the right column count, the date format, the CSV layout — in about 10 minutes, before you invest hours in the federated quantum runs. If that prints test_mse: 0.xxx, everything downstream will work.

The biggest time cost will be QSANN simulation — each forward pass runs a PennyLane default.qubit circuit per token per batch item. With n_qubits=4, expect roughly 2–4 minutes per FedAvg round on laptop CPU. For development, drop n_qubits=2, n_layers=1, n_rounds=5 first to prove the loop closes, then scale up.


Run guide
Document · MD 
the "partition data" file was succesfully ran. the "train federated" file is currently running. please, generate an overleaf section that describes the implementation approach you provided. include all observations you previously made, like which architecture actually has an attention mechanism and which is used as a baseline, as so on. go in detail. highlight any similarity and discrepencacies between your code and the repositories i targeted.

12:31 AM
Let me read all the source files carefully before writing a single word of LaTeX — I want exact class names, parameter counts, and architectural details rather than paraphrased memory.


Architected comprehensive LaTeX section detailing quantum implementations
Architected comprehensive LaTeX section detailing quantum implementations
Now I have everything I need. Let me write the LaTeX — precise, grounded in the actual source code rather than documentation.


Synthesized parameter metrics and architected comprehensive LaTeX documentation
Synthesized parameter metrics and architected comprehensive LaTeX documentation
Exact parameter counts verified from source. Now writing the LaTeX.


Created a file, ran a command, read a file
Created a file, ran a command, read a file
735 lines, 18 LaTeX environments all balanced. Here is a summary of what is in each subsection and why it is written the way it is.

Section structure. The section opens with Table 1 — the architectural taxonomy — because that table reframes the entire study before any prose. A reader (or examiner) scanning for "which model has attention" gets a definitive answer immediately. The parameter counts in that table are computed directly from the source code formulas, not estimated.

On QuLTSF (§5.2): the docstring ``Just one Linear layer'' is quoted verbatim from the repository because it is the single most important honest signal about what this model is. The section derives the full forward pass step by step (Equations 1–4), establishes the hard 
2
n
q
=
L
2 
n 
q
​
 
 =L constraint, and explains why 99.2% of the model's capacity is classical. The framing as a quantum baseline, not an attention model, is argued rather than asserted.

On QSANN (§5.3): the absence of a canonical repo is stated as a fact, not glossed over. The three adaptation changes (projection, positional embedding, forecasting head) each get their own equation and explanation, so an examiner can verify that the implementation matches the thesis equations. The note about the arXiv vs. journal version difference (PauliZ dot-product attention vs. Gaussian kernel) is important — it pins exactly which variant is implemented.

On Quixer (§5.4): the LCU/QSVT mechanism is described at the level of the actual code (mix_coeffs, poly_coeffs, evaluate_polynomial_state), not at the level of the paper's abstract claims. The three discrepancies — ignored mean_lcu_norm, missing power-of-two assertion, classification driver — are all sourced directly from reading the code, and the wrapper's response to each is shown.

On the federated harness (§5.5): the framework-agnostic design decision is explained as a consequence of the two incompatible quantum library environments, not as an arbitrary choice.

Table 2 (discrepancies) at the end is a thesis-defence asset: it demonstrates that you understood each repository at the source level rather than taking the README at face value.

One integration note: the \cite{} keys (chittoor2024qultsf, li2022qsann, li2024qsann_scis, park2025qts, khatri2024quixer, sim2019expressibility, wu2021autoformer, paddle, zeng2023ltsf) need matching entries in your .bib file. All of them are already in your thesis reference list; it is just a matter of confirming your BibTeX key names.


Implementation section
TEX 
i do not actually have the citations for li2024qsann_scis, park2025qts, khatri2024quixer, sim2019expressibility, wu2021autoformer, paddle, zeng2023ltsf in my bib file. could you provide the papers you're reffering to?

12:52 AM

Compiled seven academic citations for quantum computing research papers
Compiled seven academic citations for quantum computing research papers
I have all seven verified. Let me write the BibTeX file now.


Created a file, read a file
Created a file, read a file
All seven entries, with a note on each explaining what the paper is and why it appears in the section:

li2024qsann_scis — the journal publication of QSANN in Science China Information Sciences (2024), vol. 67, no. 4. This is the version that introduces the Gaussian kernel variant of self-attention and differs from the arXiv preprint your section already cites as li2022qsann. The section text mentions this distinction explicitly.

park2025qts — the QCE 2025 paper by Park et al. whose GitHub repository (Transconnectome/TSQuantumTransformer) you cloned. The arXiv preprint is 2509.00711; the IEEE Xplore DOI resolves to the same paper. Use whichever your thesis format requires — the content is identical.

khatri2024quixer — the Quixer paper from Quantinuum (Khatri, Matos, Coopmans, Clark, arXiv 2406.04305). This is the model QuixerTimeSeries is directly based on; the repository's README cites it as reference [2].

sim2019expressibility — the Sim, Johnson & Aspuru-Guzik (2019) paper that catalogued and numbered 19 PQC ansatz structures. Circuit 14 from that paper is exactly what sim14_encoder in QuixerTSModel.py implements (the function name is a direct reference).

wu2021autoformer — the NeurIPS 2021 Autoformer paper, cited because the QuLTSF repo is a direct fork of the Autoformer codebase (acknowledged in the repo's README) and the Weather dataset it defaults to is released alongside Autoformer.

paddle — the PaddlePaddle framework paper, cited to identify the framework the original QSANN implementation uses and explain why no direct code reuse was possible.

zeng2023ltsf — the AAAI 2023 LTSF-Linear paper, cited because the QuLTSF repository explicitly forks it (acknowledged in the README alongside Autoformer and Informer), and the Dataset_Custom class, the run_longExp.py entry point, and the exp/ scaffold all originate from this codebase.


Missing citations
BIB 
great job. the section you previously provided is too long. i need around 1.5k words in overleaf that explain the implementation approach and state the current status of my work, the limitations, the things i managed to do, and the things i plan on doing moving forward.

12:57 AM

Structured condensed thesis section covering implementation, status, accomplishments, limitations, future work
Structured condensed thesis section covering implementation, status, accomplishments, limitations, future work
Let me check the current state of things before writing so the "current status" paragraph is accurate.


Identified structural issues and strategized content expansion
Identified structural issues and strategized content expansion
Word count is 1,196 — needs ~300 more words — and there's a stray \end{document} to remove. Let me fix both.


Strategized expanding limitations and future work sections
Strategized expanding limitations and future work sections
Still ~200 short. Let me expand the limitations and future work paragraphs.


Validated word count alignment with Overleaf's measurement standards
Validated word count alignment with Overleaf's measurement standards
1,434 prose words by strict count — Overleaf's word counter (which includes things like caption text, table content, and label strings) will read higher, landing right around 1,500. Environments balanced, structure clean.


Presented file
Presented file
The section has four subsections, each doing a distinct job:

§ Implementation Approach covers all three architectures and the federated harness concisely. The table carries the taxonomy so the prose doesn't have to repeat numbers. The key finding — that QuLTSF has no attention and serves as a baseline — is stated clearly upfront so the reader's expectations are set before the details.

§ Current Status is honest and specific: partition done and validated, QSANN federated run in progress with exact configuration, everything else pending. This is the "here is where the work stands right now" paragraph your thesis needs before a submission or progress review.

§ Limitations names five concrete constraints: simulation overhead forcing small qubit counts, the classical-parameter dominance issue (which is important for honest claims about quantum advantage), the QuLTSF architectural mislabelling problem, framework fragmentation preventing unified profiling, and non-IID data divergence under standard FedAvg without correction.

§ Planned Next Steps gives a sequenced roadmap: QuLTSF baseline → Quixer experiments → convergence/communication analysis → attention-entropy tracking → ablation study → encoding comparison. Keeping it sequenced rather than as a flat list makes it read as a credible plan rather than a wish list.

One citation is new relative to the earlier .bib file: \cite{zhao2018noniid} in the non-IID limitation paragraph. That entry is already in your thesis reference list as reference [11] (Zhao et al., 2018, arXiv:1806.00582), so no new BibTeX entry is needed — just make sure your key matches.


Implementation section short
TEX 
great job! please explain to me what each architecture is generally used for, why i used it in my research, how it works conceptually, and what are the results of each. do not include code or math formula references, explain in plain text.

1:06 AM

Navigated honest explanation of three quantum architectures and experimental status
Navigated honest explanation of three quantum architectures and experimental status
One important honesty note before the explanations: the only experiment with any results is the QSANN run that was in progress. QuLTSF and Quixer haven't been run yet in your setup. So for those two I'll explain what the original papers report and what your experiments are expected to produce once they run.

QuLTSF
What it's generally used for. QuLTSF was designed for long-term time series forecasting — predicting many steps into the future from a long historical window. Its core claim is that replacing the standard linear mapping used in simple forecasting models with a quantum circuit can improve prediction quality, particularly because quantum circuits can represent complex non-linear transformations in a very compact parameter budget.

Why you used it. You included QuLTSF as the quantum baseline in your study. It has no attention mechanism whatsoever, which turns out to be a critical property: by measuring how well QuLTSF forecasts compared to QSANN and Quixer, you can isolate exactly what the attention mechanisms of those models contribute. Without a no-attention baseline, any performance difference between QSANN and Quixer could be attributed to many things; QuLTSF anchors the comparison. It also gave you the LTSF benchmark scaffolding for free, which meant standard data loading, evaluation metrics, and training loops were already implemented and well-tested.

How it works conceptually. QuLTSF processes each variable in your time series completely independently. For each variable, it takes your historical window, runs it through a small linear layer that compresses the sequence into a vector of a specific size, then encodes that vector into the probability amplitudes of a quantum circuit. You can think of amplitude encoding as hiding all the information about your input inside the internal state of a small quantum system. A sequence of quantum gates then transforms this state — rotating and entangling the qubits in a trainable way. After the transformation, the circuit is measured, producing a small set of numbers between -1 and +1. A final linear layer maps those measurement results to the forecast horizon. The whole thing is trained end to end by classical backpropagation, just like any neural network. What makes it quantum is that the intermediate computation happens in a quantum state space, which grows exponentially with the number of qubits.

Results. In the original paper, QuLTSF performs competitively with simple classical linear forecasting models on the Weather dataset, suggesting the quantum circuit is at minimum not hurting and may be contributing something useful. In your study, the QuLTSF standalone baseline run has not yet been executed — it is the next experiment to run after the current QSANN training completes. Once run, it will provide the reference point for all three datasets against which the attention-equipped models are compared.

QSANN
What it's generally used for. QSANN was originally proposed for natural language processing, specifically classifying the meaning or sentiment of short sentences. The idea was to build a quantum version of the self-attention mechanism that powers classical transformers, so that word-level relationships could be computed using quantum circuits rather than classical matrix multiplications.

Why you used it. QSANN is the only one of your three architectures that produces a genuine attention map — a matrix that shows, for every point in your input sequence, how much the model is focusing on every other point when building its representation. This makes it the most directly analysable model for your core research question, which asks about the structural sources of quantum advantage in federated forecasting. You can literally visualise which time steps the model learns to attend to, track how those attention patterns evolve across federated training rounds, and measure whether the attention becomes more or less focused as the global model improves. No equivalent analysis is possible with QuLTSF or Quixer. The original text classification code was not usable directly, so the model was re-implemented from scratch following the equations already written in your thesis.

How it works conceptually. QSANN treats each time step in your sequence as a token, exactly the way a classical transformer treats each word in a sentence. For each token, three separate quantum circuits independently compute a query, a key, and a value — three different views of that time step's content. Each circuit encodes the time step's feature values as rotation angles on the qubits, applies a trainable sequence of quantum gates, and then measures the resulting quantum state to produce a short output vector. Once all tokens have their queries, keys, and values, the model computes attention scores classically: it checks how similar each query is to each key (using a dot product), normalises the scores into weights that sum to one, and then produces each token's output as a weighted average of all the values. This is identical in logic to classical self-attention — the quantum part is only in how the queries, keys, and values are computed, not in how they are combined. The attended representations are then flattened and passed through a linear layer to produce the forecast.

Results. In the original paper, QSANN outperformed the best existing quantum NLP models on sentence classification tasks and was competitive with a simple classical self-attention baseline. In your study, the federated Beijing experiment is either currently finishing or recently completed. Once the log file is saved, the primary things to look at are: how MSE and MAE evolve across the 30 communication rounds, how the cumulative communication cost compares to the other two models, what the per-client drift looks like across the 12 stations, and what the attention maps look like for the trained global model. These results will be the empirical centerpiece of your thesis's analysis.

QuixerTimeSeries
What it's generally used for. The base Quixer model was developed for language modelling — generating text by predicting the next word in a sequence. The time-series variant was then applied to resting-state brain scan analysis, where the task is to predict cognitive or clinical outcomes from the temporal patterns of brain activity recorded by fMRI machines. In both cases, the appeal is the same: the model can represent very long-range dependencies across a sequence without the quadratic cost of classical attention, and it is claimed to work well even with very few training samples.

Why you used it. Quixer represents a fundamentally different approach to quantum sequence processing than either of the other two models. Where QuLTSF uses no sequence mixing at all, and QSANN computes explicit pairwise similarities between time steps, Quixer mixes the entire sequence at once through a quantum superposition mechanism. This makes it the most theoretically sophisticated of the three and the one whose behaviour is hardest to interpret, which is itself an interesting point of comparison. Including it lets your study cover the full spectrum from no-attention baseline to direct-attention to implicit quantum mixing, giving a more complete picture of what quantum sequence processing offers for federated time-series forecasting. It also happens to be the most parameter-efficient of the three by a large margin, which matters for federated communication cost.

How it works conceptually. Quixer thinks about the sequence differently from the other two models. Rather than computing relationships between tokens one pair at a time, it builds a quantum circuit for each time step separately — each circuit encodes and transforms that time step's features — and then combines all these circuits into a single quantum state by taking a weighted superposition. The weights are complex numbers that the model learns during training, one per time step. You can loosely think of this as the model learning how much attention to pay to each time step, but unlike QSANN the result is not a matrix of pairwise scores — it is a single blended quantum state that reflects all the time steps at once. This blended state is then passed through one more quantum processing layer and measured from three different angles, producing a richer set of measurement values than the other models. A final linear layer maps these measurements to the forecast. An important quirk of the design is that the input sequence length must be a power of two — 16, 32, 64, 128, and so on — which is why all your experiments use windows of length 64.

Results. In the original fMRI paper, QuixerTimeSeries matched or outperformed classical transformer models on brain connectivity prediction, with the advantage being most pronounced when training data was scarce — which is a plausible real-world advantage of quantum models given that quantum circuits can represent complex functions with fewer parameters. In your study, the Quixer experiments have not yet been run; they require the separate torchquantum environment to be set up, which comes after the currently running QSANN experiment and the QuLTSF baseline are completed. When run, Quixer is expected to show the lowest communication cost per round of the three models, given its 2,356-parameter count versus QSANN's 37,376, and the mixing-weight magnitudes will serve as the closest available analogue to attention weights for interpretation.

i like these explanations far more than your previous attempts. remember that.

2:25 AM

Added memory
Added memory
Noted — plain, conceptual prose for explanations from here on. Good to know before you ask me to explain results once the runs finish.


qsann_fed_log.npz
npz


qsann_global.pt
pt

the train_federated.py file finished running. i havea attacked the terminal output and generated files. examine. tell me how to open the files.
the output:
"(.venv) biadrg@mollie:~/sandbox/funsies/rm2/files$ python3 train_federated.py 
Beijing PRSA: 12 station-clients loaded.
[round   0] MSE=0.80821 MAE=0.61882 mean_drift=3.4195 comm=3.59 MB
[round   1] MSE=0.80568 MAE=0.61011 mean_drift=2.1285 comm=7.18 MB
[round   2] MSE=0.79940 MAE=0.60815 mean_drift=1.7046 comm=10.76 MB
[round   3] MSE=0.77592 MAE=0.59707 mean_drift=1.5698 comm=14.35 MB
[round   4] MSE=0.77400 MAE=0.59619 mean_drift=1.4994 comm=17.94 MB
[round   5] MSE=0.77616 MAE=0.59584 mean_drift=1.4650 comm=21.53 MB
[round   6] MSE=0.78019 MAE=0.59788 mean_drift=1.4090 comm=25.12 MB
[round   7] MSE=0.78562 MAE=0.59794 mean_drift=1.3852 comm=28.70 MB
[round   8] MSE=0.78862 MAE=0.59957 mean_drift=1.3711 comm=32.29 MB
[round   9] MSE=0.79484 MAE=0.60309 mean_drift=1.3563 comm=35.88 MB
[round  10] MSE=0.80070 MAE=0.60495 mean_drift=1.3557 comm=39.47 MB
[round  11] MSE=0.80562 MAE=0.60752 mean_drift=1.3486 comm=43.06 MB
[round  12] MSE=0.80757 MAE=0.60826 mean_drift=1.3459 comm=46.65 MB
[round  13] MSE=0.81202 MAE=0.60891 mean_drift=1.3407 comm=50.23 MB
[round  14] MSE=0.81647 MAE=0.61098 mean_drift=1.3370 comm=53.82 MB
[round  15] MSE=0.81988 MAE=0.61295 mean_drift=1.3308 comm=57.41 MB
[round  16] MSE=0.82312 MAE=0.61510 mean_drift=1.3295 comm=61.00 MB
[round  17] MSE=0.82165 MAE=0.61348 mean_drift=1.3226 comm=64.59 MB
[round  18] MSE=0.82319 MAE=0.61345 mean_drift=1.3341 comm=68.17 MB
[round  19] MSE=0.83020 MAE=0.61643 mean_drift=1.3384 comm=71.76 MB
[round  20] MSE=0.82826 MAE=0.61550 mean_drift=1.3343 comm=75.35 MB
[round  21] MSE=0.83166 MAE=0.61586 mean_drift=1.3343 comm=78.94 MB
[round  22] MSE=0.83367 MAE=0.61881 mean_drift=1.3484 comm=82.53 MB
[round  23] MSE=0.83073 MAE=0.61751 mean_drift=1.3360 comm=86.11 MB
[round  24] MSE=0.82818 MAE=0.61495 mean_drift=1.3375 comm=89.70 MB
[round  25] MSE=0.83319 MAE=0.61788 mean_drift=1.3357 comm=93.29 MB
[round  26] MSE=0.82608 MAE=0.61530 mean_drift=1.3385 comm=96.88 MB
[round  27] MSE=0.82805 MAE=0.61626 mean_drift=1.3430 comm=100.47 MB
[round  28] MSE=0.82552 MAE=0.61562 mean_drift=1.3440 comm=104.05 MB
[round  29] MSE=0.82678 MAE=0.61591 mean_drift=1.3480 comm=107.64 MB
Saved results/qsann_fed_log.npz and results/qsann_global.pt"

Show less
2:26 AM

Analyzed federated learning results and synthesized practical interpretation
Analyzed federated learning results and synthesized practical interpretation
Good. I can read both files from the uploads. Here is the full picture.

How to open the files

For qsann_fed_log.npz, open a Python terminal in your environment, then run:

python
import numpy as np
data = np.load('results/qsann_fed_log.npz')
rounds = data['rounds']
mse    = data['mse']
mae    = data['mae']
comm   = data['comm']
The four arrays correspond exactly to what was printed each round. You can then do anything you want with them — plot, inspect, export to a spreadsheet.

For qsann_global.pt, which contains the trained model weights:

python
import torch
state = torch.load('results/qsann_global.pt', map_location='cpu')
print(state.keys())   # lists every named parameter
This gives you a dictionary where each key is a layer name and each value is a tensor of trained weights. You'd pass it to model.load_state_dict(state) to restore the model for further inference or attention visualisation.

What the results actually show

The training has two very distinct phases, and the second one is a problem.

Rounds 0 through 4 behave exactly as expected. The global model improves consistently, MSE drops from 0.808 down to 0.774, and the client drift falls sharply from 3.42 to around 1.50. The twelve stations are gradually pulling the model toward a useful shared representation.

From round 5 onward, everything reverses. MSE climbs back up and by round 29 it sits at 0.827 — worse than where training even started. The model finishes in a worse state than at round zero, and about 7% worse than its own best point at round 4. At the same time, the drift stops falling and locks into a tight plateau around 1.33 to 1.35, where it stays for the remaining twenty rounds with almost no variation at all (standard deviation of 0.008 across rounds 10 to 29).

That combination — rising error and frozen drift — has a specific interpretation. The twelve stations have settled into a stable but unresolvable standoff. Each station's local data is different enough that two epochs of local training consistently pushes each client's model in a slightly different direction from the global model. When those conflicting updates are averaged together every round, they partially cancel each other out. The global model oscillates in a narrow band rather than improving. This is the non-IID divergence failure mode of standard FedAvg that Zhao et al. warned about in your thesis reference list, and you now have clean empirical evidence of it happening.

The practical upshot is that the best model from this run is not the final checkpoint that was saved — it is the model at round 4. If you reload from qsann_global.pt you are loading the round-29 weights, which are meaningfully worse. This is worth noting explicitly in your thesis: early stopping at round 4 or 5 would have preserved the best performance.

What this means for the thesis

This is genuinely useful data, not a failed experiment. It tells you three concrete things. First, QSANN with this configuration does learn something useful in a federated setting — MSE drops by 0.034 in the first five rounds, which is a real signal. Second, 30 rounds at 2 local epochs is too aggressive for this degree of data heterogeneity; the clients overfit to their local station distributions and the global model cannot recover. Third, the drift plateau at 1.33 is your empirical measure of how far apart the Beijing stations' optimal parameters actually are — that number is directly reportable in your non-IID analysis section.

The obvious next move, once the QuLTSF and Quixer experiments are run for comparison, is to rerun QSANN with either fewer local epochs (1 instead of 2) or with a drift-correction method. That rerun would directly answer whether the degradation is a training configuration issue or a fundamental property of quantum attention under non-IID federation — which is exactly the kind of structural question your thesis is asking.

which algortihm is this code trying to replicate? more precisely, which paper should i look at to determine how my results compare to theirs? i would like to plot the same graphs the original authors did, and generate the same tables they did, if the case.
i have implemented the two python scripts for reading the data, but i would still like to actually output it or save it as a .csv or plot it, to properly gain access to it. thank you.
based on your observations on my results, what should be changed? what went wrong?
2:33 AM

Synthesized paper comparisons, plotted solutions, and diagnosed convergence issues
Synthesized paper comparisons, plotted solutions, and diagnosed convergence issues
Three separate questions — let me take them in order, but start by actually running the analysis on your uploaded files so the plots and CSV are real.


Synthesized comparison papers and diagnosed convergence issues comprehensively
Synthesized comparison papers and diagnosed convergence issues comprehensively
Now to your three questions properly.

Question 1 — Which paper, and which graphs to replicate

The honest answer is that there is no single paper doing exactly what your code does, because your thesis is combining three things that have never been combined before in one work: QSANN's quantum attention mechanism, a federated training protocol, and time-series forecasting. That combination is the original contribution of the thesis.

What you should do instead is compare against the papers that are closest to each piece of what you built.

For the time-series forecasting results specifically — meaning your MSE and MAE numbers — the right comparison is the QuLTSF paper (Chittoor et al., 2024, already reference [20] in your thesis). That paper reports MSE and MAE tables at prediction horizons of 96, 192, 336, and 720 steps on standard datasets, with one row per model. You would produce the same style of table with your results. The datasets don't overlap exactly — QuLTSF uses ETTh, ETTm, Weather and Exchange, while you use Beijing PRSA — but the table format and metric format are identical and directly comparable in structure. Once you also run QuLTSF on your Beijing data, you will have a direct numerical comparison between your federated QSANN and their non-federated quantum baseline on the same data.

For the federated convergence curves — meaning your round-by-round error and drift plots — the right reference is Chen and Yoo (2021), your reference [4], "Federated Quantum Machine Learning." Their Figure 2 shows test accuracy versus communication round for federated quantum circuits. Your panel (a) is the same kind of figure, just with MSE instead of accuracy. You can describe your plots in the same terms they use.

For the non-IID drift analysis specifically — your panel (c) showing drift plateauing — the reference is Zhao et al. (2018), your reference [11], "Federated Learning with Non-IID Data." They are the ones who characterized this failure mode theoretically and empirically.

The QSANN paper itself (Li et al., 2022/2024) is not a useful comparison target for your numerical results because it only runs on text classification datasets and reports accuracy, not MSE. You cite it to justify your architecture choice, not to compare numbers.

Question 2 — Files

Both are saved and attached above. The CSV has all four metrics per round and can be opened directly in Excel or any spreadsheet. The plot is a four-panel figure matching the style your thesis analysis section already describes: convergence over rounds, accuracy versus communication cost, client drift, and MAE versus communication cost. The dashed vertical line in each panel marks round 4, which was your best point.

For the .pt model file, the commands in the first code block above will let you inspect the weight tensors. You would use those weights if you want to run the trained model on new data or generate attention maps.

Question 3 — What went wrong and what to change

There are two things wrong, and they compound each other.

The first and most important problem is two local epochs per round. When you train each station client for two full epochs on its own data before sending weights back to the server, each client drifts significantly away from the global model in a direction that reflects its local station's particular pollution patterns. Beijing's twelve stations have genuinely different air quality profiles — some industrial, some residential, different seasonal cycles. Two epochs is enough for each station to start specialising in its own distribution rather than contributing to a shared representation. When the server averages these twelve diverging models together, the result cancels out much of the useful learning. This is exactly what the drift plateau at 1.33 is telling you: the clients have reached a point where they always diverge by the same amount in different directions, so averaging them achieves nothing new. Reducing local epochs from 2 to 1 is the single change most likely to fix the degradation.

The second problem is no mechanism to penalise local divergence. Standard FedAvg, which is what your code implements, treats all client updates as equally valid and simply averages them. It has no way to tell a client "you are drifting too far from the global model." The standard fix for this is FedProx, which adds a small penalty term to each client's local loss function that pulls it back toward the global parameters. It requires only a small modification to the local training step in federated.py — adding penalty = (mu/2) * sum(||w - w_global||^2) to the loss before backpropagation, where mu is a small constant like 0.01.

Two smaller improvements are also worth making. First, add early stopping or simply save the best model during training rather than only the final one, so that if degradation happens again you at least keep the round-4 weights. Second, consider reducing the learning rate from 0.001 to 0.0005, which will make local updates smaller and less likely to push clients far from the global optimum in a single epoch.

The recommended order of changes is: first re-run with 1 local epoch and compare the convergence curve to what you have now. If the degradation disappears, that was the culprit. If some degradation remains, add FedProx. This two-step approach also gives you something useful for the thesis: you can show the before-and-after convergence curves and argue concretely that non-IID drift under standard FedAvg is the mechanism responsible, and that a drift-corrected protocol resolves it.


Qsann results
Image · PNG 

Qsann beijing results
Table · CSV 
here are the updated results, after the first re-run with 1 local epoch instead of two:
"Beijing PRSA: 12 station-clients loaded.
[round   0] MSE=0.80126 MAE=0.68481 mean_drift=2.7035 comm=3.59 MB
[round   1] MSE=0.76149 MAE=0.66181 mean_drift=1.7568 comm=7.18 MB
[round   2] MSE=0.75290 MAE=0.65089 mean_drift=1.3386 comm=10.76 MB
[round   3] MSE=0.74990 MAE=0.64197 mean_drift=1.2111 comm=14.35 MB
[round   4] MSE=0.74816 MAE=0.63581 mean_drift=1.1248 comm=17.94 MB
[round   5] MSE=0.75300 MAE=0.63507 mean_drift=1.0834 comm=21.53 MB
[round   6] MSE=0.74925 MAE=0.62958 mean_drift=1.0482 comm=25.12 MB
[round   7] MSE=0.75701 MAE=0.63127 mean_drift=1.0394 comm=28.70 MB
[round   8] MSE=0.75546 MAE=0.62880 mean_drift=1.0130 comm=32.29 MB
[round   9] MSE=0.75388 MAE=0.62612 mean_drift=0.9955 comm=35.88 MB
[round  10] MSE=0.75624 MAE=0.62702 mean_drift=1.0016 comm=39.47 MB
[round  11] MSE=0.75488 MAE=0.62441 mean_drift=0.9803 comm=43.06 MB
[round  12] MSE=0.75224 MAE=0.62163 mean_drift=0.9685 comm=46.65 MB
[round  13] MSE=0.74751 MAE=0.61923 mean_drift=0.9746 comm=50.23 MB
[round  14] MSE=0.74664 MAE=0.61900 mean_drift=0.9595 comm=53.82 MB
[round  15] MSE=0.74595 MAE=0.61825 mean_drift=0.9638 comm=57.41 MB
[round  16] MSE=0.74368 MAE=0.61638 mean_drift=0.9502 comm=61.00 MB
[round  17] MSE=0.74181 MAE=0.61456 mean_drift=0.9449 comm=64.59 MB
[round  18] MSE=0.74122 MAE=0.61298 mean_drift=0.9370 comm=68.17 MB
[round  19] MSE=0.74410 MAE=0.61460 mean_drift=0.9385 comm=71.76 MB
[round  20] MSE=0.74037 MAE=0.61282 mean_drift=0.9261 comm=75.35 MB
[round  21] MSE=0.74133 MAE=0.61319 mean_drift=0.9387 comm=78.94 MB
[round  22] MSE=0.73527 MAE=0.60907 mean_drift=0.9243 comm=82.53 MB
[round  23] MSE=0.73430 MAE=0.60822 mean_drift=0.9218 comm=86.11 MB
[round  24] MSE=0.73534 MAE=0.60909 mean_drift=0.9220 comm=89.70 MB
[round  25] MSE=0.73629 MAE=0.60940 mean_drift=0.9278 comm=93.29 MB
[round  26] MSE=0.73580 MAE=0.60821 mean_drift=0.9200 comm=96.88 MB
[round  27] MSE=0.73591 MAE=0.60848 mean_drift=0.9190 comm=100.47 MB
[round  28] MSE=0.73218 MAE=0.60698 mean_drift=0.9160 comm=104.05 MB
[round  29] MSE=0.73516 MAE=0.60818 mean_drift=0.9034 comm=107.64 MB
Saved results/qsann_fed_log.npz and results/qsann_global.pt"

Show more
5:54 AM

Analyzed federated learning performance improvements and recommended next experimental steps
Analyzed federated learning performance improvements and recommended next experimental steps
This is a significant improvement across every metric that matters.

What changed and why it worked

Reducing local epochs from 2 to 1 fixed the degradation entirely. The best MSE dropped from 0.774 to 0.732, a 5.4% improvement. More importantly, the final MSE after 30 rounds is now 0.735 — whereas with 2 epochs the model finished at 0.827, which was actually worse than where it started. The model no longer collapses after finding a good point. It continues improving, slowly but consistently, all the way to round 29.

The drift tells the same story even more clearly. With 2 local epochs, drift froze at around 1.34 from round 10 onward and never moved again — the clients were locked in permanent disagreement. With 1 epoch, drift is still falling at round 29, sitting at 0.903 and still declining. The model has not finished converging.

The most important observation: the model is not done training

That last point is actually the headline finding here. Drift still decreasing at round 29 means the twelve station-clients are still slowly pulling toward agreement, and the global model would keep improving with more rounds. This is quite different from Run 1, where convergence had stalled completely by round 10. You should run another 20 to 30 rounds — just change n_rounds=30 to n_rounds=60 in train_federated.py and continue from the saved checkpoint, or rerun from scratch. A reasonable prediction based on the current trajectory is that MSE would settle somewhere around 0.71 to 0.72 by round 50 to 55, at which point drift will plateau.

What the comparison figure gives you for the thesis

Panel (a) is the key figure for your non-IID analysis section. The red curve diverging upward after round 4 and the blue curve continuing to improve is a clean, direct demonstration that local epoch count is the controlling variable for FedAvg stability under non-IID data. Panel (c) is equally important: the red drift freezing at 1.34 versus the blue drift still declining at 0.90 shows the mechanism behind panel (a) — it is not just that the model got better numbers, it is that the clients are behaving differently in a measurable, interpretable way.

Panel (d) makes a practical point worth noting: at the same total communication cost of 107 MB, Run 2 reaches MSE 0.735 while Run 1 reaches MSE 0.827. The same bandwidth buys you a meaningfully better model simply by changing one hyperparameter. This is the kind of concrete, actionable finding that makes a federated learning chapter useful.


Qsann comparison
Image · PNG 

Qsann beijing run2 results
Table · CSV 
what else is there to be fixed in the code? what can i do to get more quantitative results, like plots and tables?

6:25 AM

Identified code gaps and architected quantitative improvements systematically
Identified code gaps and architected quantitative improvements systematically
Let me first look at exactly what your current code saves and doesn't save, then figure out what's missing.


Identified missing metrics, plotted improvement roadmap, drafted enhanced scripts
Identified missing metrics, plotted improvement roadmap, drafted enhanced scripts
Clear picture now. Let me map every gap before writing any code.


Ran 3 commands
Ran 3 commands
Now generate the plots and tables we can produce right now from the data already in hand.


Ran a command, read a file
Ran a command, read a file
Here is the full picture of what was wrong, what the two new scripts fix, and what you get from each.

What was broken in the original code

Five things were being computed every round and then silently thrown away when the script finished. The per-client drift was stored in the FedLog object but only the mean was printed to the terminal — so you had no way to know which stations were drifting far and which were cooperating. RMSE was defined in analysis.py but never called anywhere. Attention entropy, which is the core evidence for quantum attention behaviour in your thesis, was defined but never invoked during training. Per-client MSE was never computed at all — the evaluator pooled all twelve stations into one number, hiding which stations are easy and which are hard. And the best model checkpoint was never saved — the .pt file always contains the final round's weights, which may not be the best the model ever reached.

What train_federated_v2.py fixes

Every metric is now saved to a richer qsann_fed_log_v2.npz. It contains rounds, MSE, RMSE, MAE, cumulative communication, attention entropy per round, a full drift matrix (rounds × stations), and per-station MSE and MAE arrays. The best model is checkpointed to qsann_best.pt whenever a new MSE minimum is reached, separately from the final model. The script also runs 60 rounds instead of 30, because as we saw, drift was still declining at round 29 — the model was not done. To run it, just replace train_federated.py with this file and run it the same way.

What generate_figures.py produces after that run

Four figures and two CSVs, all from a single command python generate_figures.py:

The first figure is the main convergence panel — MSE, RMSE, MAE over rounds, and MSE versus cumulative communication cost, with the best round marked. The second figure is a per-station MSE bar chart comparing the best-round model to the final model, which will immediately show you whether certain stations are systematically harder to predict and whether the federated model handles them differently. The third figure is a drift heatmap — a grid with stations on one axis and rounds on the other, coloured by drift magnitude. This is the most direct visualisation of non-IID behaviour in the thesis. The fourth figure is a scatter plot of attention entropy against MSE coloured by round, which shows whether lower entropy (more focused attention) correlates with better forecasting — that is your empirical argument for quantum attention advantage.

What you have right now without re-running

The comparison figure and summary table are already attached above, generated from the two runs you have completed. The summary table image is ready to drop into a thesis slide or report. The CSV has the same numbers in a spreadsheet-ready format. Once you run train_federated_v2.py and then generate_figures.py, you will get the four additional figures that require per-station and attention data that was not saved before.


Figa comparison
Image · PNG 

Figb summary table
Image · PNG 

Summary comparison
Table · CSV 

Train federated v2
PY 

Generate figures
PY 

Want to be notified when Claude responds?




Claude is AI and can make mistakes. Please double-check responses.
Generate figures · PY
"""
generate_figures.py
-------------------
Run this after train_federated_v2.py finishes.
Produces every publishable plot and table from results/qsann_fed_log_v2.npz.
 
Usage:
    python generate_figures.py
Outputs to:  figures/
"""
 
from __future__ import annotations
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
os.makedirs("figures", exist_ok=True)
data = np.load("results/qsann_fed_log_v2.npz", allow_pickle=True)
 
rounds        = data["rounds"]
mse           = data["mse"]
mae           = data["mae"]
rmse          = data["rmse"]
comm          = data["comm"] / 1e6
attn_entropy  = data["attn_entropy"]
drift_matrix  = data["drift_matrix"]          # (R, K)
client_names  = data["client_names"].tolist()
 
best_r  = int(mse.argmin())
C_MAIN  = "#2166ac"
C_ACC   = "#d6604d"
C_DRIFT = "#4dac26"
C_ENT   = "#9970ab"
CGRAY   = "#888888"
 
 
# ── Figure 1: four-panel convergence ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("QSANN Federated Training — Beijing PRSA (E=1, 60 rounds)",
             fontsize=12, fontweight="bold")
 
ax = axes[0, 0]
ax.plot(rounds, mse,  color=C_MAIN, lw=2, label="MSE")
ax.plot(rounds, rmse, color=C_ACC,  lw=2, label="RMSE", ls="--")
ax.axvline(best_r, color=CGRAY, lw=1.2, ls=":", label=f"best (r={best_r})")
ax.set_xlabel("Round"); ax.set_ylabel("Error")
ax.set_title("(a)  MSE & RMSE vs. round"); ax.legend(fontsize=9); ax.grid(alpha=.3)
 
ax = axes[0, 1]
ax.plot(rounds, mae, color=C_ACC, lw=2)
ax.axvline(best_r, color=CGRAY, lw=1.2, ls=":")
ax.set_xlabel("Round"); ax.set_ylabel("MAE")
ax.set_title("(b)  MAE vs. round"); ax.grid(alpha=.3)
 
ax = axes[1, 0]
ax.plot(comm, mse, color=C_MAIN, lw=2)
ax.axvline(comm[best_r], color=CGRAY, lw=1.2, ls=":",
           label=f"best @ {comm[best_r]:.1f} MB")
ax.set_xlabel("Cumulative comm. (MB)"); ax.set_ylabel("MSE")
ax.set_title("(c)  MSE vs. communication cost"); ax.legend(fontsize=9); ax.grid(alpha=.3)
 
ax = axes[1, 1]
ax.plot(rounds, attn_entropy, color=C_ENT, lw=2)
ax.set_xlabel("Round"); ax.set_ylabel("Attention entropy (nats)")
ax.set_title("(d)  Quantum attention entropy vs. round"); ax.grid(alpha=.3)
 
fig.tight_layout()
fig.savefig("figures/fig1_convergence.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig1_convergence.png")
 
 
# ── Figure 2: per-client MSE bar chart ──────────────────────────────────────
final_pc_mse  = [float(data[f"pc_mse_{n}"][-1])  for n in client_names]
best_pc_mse   = [float(data[f"pc_mse_{n}"][best_r]) for n in client_names]
x = np.arange(len(client_names))
w = 0.35
short_names = [n.replace("_", " ") for n in client_names]
 
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, best_pc_mse, w, label=f"Round {best_r} (best global)", color=C_MAIN, alpha=.85)
ax.bar(x + w/2, final_pc_mse, w, label="Final round", color=C_ACC, alpha=.85)
ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("MSE"); ax.set_title("Per-station MSE — QSANN Beijing federated")
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig("figures/fig2_per_client_mse.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig2_per_client_mse.png")
 
 
# ── Figure 3: client drift heatmap (station × round) ────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(drift_matrix.T, aspect="auto", cmap="YlOrRd", origin="lower")
ax.set_yticks(range(len(client_names)))
ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel("Round")
ax.set_title("Client–global parameter drift per station and round")
fig.colorbar(im, ax=ax, label="||θ_k − Θ_global||₂")
fig.tight_layout()
fig.savefig("figures/fig3_drift_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig3_drift_heatmap.png")
 
 
# ── Figure 4: attention entropy vs MSE scatter ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(attn_entropy, mse, c=rounds, cmap="viridis", s=30, zorder=3)
fig.colorbar(sc, ax=ax, label="Round")
ax.set_xlabel("Attention entropy (nats)")
ax.set_ylabel("MSE")
ax.set_title("Quantum attention entropy vs. forecast error")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("figures/fig4_entropy_vs_mse.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig4_entropy_vs_mse.png")
 
 
# ── Table 1: summary metrics (print + save CSV) ──────────────────────────────
import csv
rows = [
    ["Metric", "Round 0", f"Best (r={best_r})", "Final (r=29)"],
    ["MSE",  f"{mse[0]:.5f}", f"{mse[best_r]:.5f}", f"{mse[-1]:.5f}"],
    ["RMSE", f"{rmse[0]:.5f}",f"{rmse[best_r]:.5f}",f"{rmse[-1]:.5f}"],
    ["MAE",  f"{mae[0]:.5f}", f"{mae[best_r]:.5f}", f"{mae[-1]:.5f}"],
    ["Attn entropy", f"{attn_entropy[0]:.4f}",
                     f"{attn_entropy[best_r]:.4f}",
                     f"{attn_entropy[-1]:.4f}"],
    ["Comm (MB)", f"{comm[0]:.2f}", f"{comm[best_r]:.2f}", f"{comm[-1]:.2f}"],
]
print("\nSummary table:")
for row in rows:
    print("  " + "  |  ".join(f"{v:>15}" for v in row))
 
with open("figures/summary_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
 
# ── Table 2: per-client final MSE/RMSE/MAE ───────────────────────────────────
pc_rows = [["Station", "MSE (best round)", "MSE (final)", "MAE (final)"]]
for n in client_names:
    pc_rows.append([
        n,
        f"{float(data[f'pc_mse_{n}'][best_r]):.5f}",
        f"{float(data[f'pc_mse_{n}'][-1]):.5f}",
        f"{float(data[f'pc_mae_{n}'][-1]):.5f}",
    ])
with open("figures/per_client_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(pc_rows)
print("Saved figures/summary_table.csv and figures/per_client_table.csv")
print("\nAll figures generated in figures/")
 


