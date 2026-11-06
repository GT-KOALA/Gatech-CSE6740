'use strict';
// Each step adds a distinct teaching point.
const LESSON=[
  {
    "id": "context",
    "title": "Why does a word need a memory?",
    "scene": "context",
    "steps": [
      [
        "One word is ambiguous",
        "A message ends with <b>“free”</b>. Is it spam? That word appears in both promotions and ordinary conversations. A classifier that sees only the current word receives the same input in both cases.",
        "current word: “free”",
        "A word alone can leave the decision ambiguous."
      ],
      [
        "Earlier words change the meaning",
        "Watch two messages arrive. One promises a prize; the other describes a friend giving away a spare ticket. Both end in “free”. These are invented examples for reasoning about context, not predictions from our trained models.",
        "",
        "Earlier words provide evidence that the final word cannot supply."
      ],
      [
        "Carry context forward",
        "At each word, update a summary of the words already read and pass it onward. The two messages reach “free” with different histories. A hidden state is a learned vector, not a stored sentence. To handle different message lengths, an RNN applies the same learned update at every token. Next, we will open that update.",
        "h₀ = 0\nhₜ = update(hₜ₋₁, eₜ)",
        "The state changes with the message; the update rule is shared."
      ]
    ]
  },
  {
    "id": "rnn",
    "title": "Build a recurrent neural network",
    "scene": "rnn",
    "steps": [
      [
        "Convert a token into numbers",
        "Split a message into tokens: words, numbers, and punctuation. A lookup table maps each token to an embedding eₜ (a learned numerical vector). Our experiment learns 32 numbers per token. Embedding values and colors in this diagram are schematic.",
        "token xₜ → embedding eₜ ∈ ℝ³²",
        "Embeddings give the network a numerical input."
      ],
      [
        "Combine input with the previous state",
        "The blue path brings the previous hidden state hₜ₋₁; the green path brings the current embedding eₜ. Separate matrices transform them, and a bias shifts their sum. Our hidden state has 48 components.",
        "aₜ = Wₓeₜ + Wₕhₜ₋₁ + b",
        "Both the current word and its history enter the update."
      ],
      [
        "Create the new hidden state",
        "Apply tanh component by component. This nonlinear function maps each value into (−1, 1). The resulting hₜ becomes the next step’s memory. The recurrent state compresses history into a fixed-size vector.",
        "hₜ = tanh(aₜ)",
        "Each token transforms the same 48-dimensional state."
      ],
      [
        "Unroll the same cell",
        "The single cell becomes a chain of time steps. These boxes are repeated applications of one cell, not independently learned networks. Wₓ, Wₕ, and b are shared. h₀ is zero for each new message.",
        "h₁ → h₂ → … → hₜ\nshared: Wₓ, Wₕ, b",
        "Sequence length changes computation, not the number of recurrent parameters."
      ],
      [
        "Read out a spam probability",
        "After the last real token, a linear layer maps h_T to two logits. Softmax turns them into probabilities for ham (legitimate SMS) and spam. During training, cross entropy compares this final prediction to the true label.",
        "z = Wᵧh_T + bᵧ\np = softmax(z)\nL = −log p(true label)",
        "The final state must retain the evidence needed for classification."
      ]
    ]
  },
  {
    "id": "reading",
    "title": "Watch a trained model read",
    "scene": "reading",
    "steps": [
      [
        "Watch, pause, and inspect",
        "Watch recorded predictions from the trained RNN and LSTM on held-out messages. Choose a message or scrub to a token to inspect how its spam score changes. Gray tokens are visible to you but have not been processed by the models.<br><br>Each score uses the whole prefix. These intermediate readouts are diagnostics: training supervised only the final real token. Early scores need not be reliable.",
        "training loss is attached to h_T",
        "How does a late classification error teach the network to retain early evidence? Next, follow the gradient backward."
      ]
    ]
  },
  {
    "id": "gradient",
    "title": "Learning across a long chain",
    "scene": "gradient",
    "steps": [
      [
        "Reverse the direction",
        "Forward computation carries states to the right. Backpropagation carries loss derivatives to the left through the same updates. Each step multiplies by a local derivative. Shared weights receive contributions from all their uses.",
        "∂L/∂hₜ₋₁ = (∂hₜ/∂hₜ₋₁)ᵀ ∂L/∂hₜ",
        "Training has to connect a late error to earlier computations."
      ],
      [
        "Length amplifies derivative products",
        "Use a scalar linear recurrence to isolate the effect: hₜ = w hₜ₋₁ + xₜ. Its derivative over T steps is wᵀ. Try w = 0.8, 1.0, and 1.1, then increase T to compare vanishing, preserved, and growing signals.<br><br>This is an illustration, not a measured spam-model gradient. A tanh RNN also includes activation derivatives; vector RNNs multiply Jacobians. Gradient clipping limits large updates but does not restore vanished gradients.",
        "∂h_T/∂h₀ = wᵀ\nJₜ = diag(1 − hₜ²) Wₕ",
        "Many moderately small factors can produce a tiny derivative; factors above one can amplify it."
      ],
      [
        "Make a path for retained memory",
        "An LSTM adds a cell state with an additive update. Gates regulate what is retained, written, and exposed. This creates a direct cell-state path whose local derivative is the forget gate when other paths are held fixed. It helps; it does not guarantee perfect long-range learning.",
        "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ",
        "Next, construct this memory path one operation at a time."
      ]
    ]
  },
  {
    "id": "lstm",
    "title": "Build an LSTM cell",
    "scene": "lstm",
    "steps": [
      [
        "Separate memory from visible output",
        "An LSTM carries two vectors: cell state cₜ (amber) and hidden state hₜ (blue). The cell state provides an additive memory path. The hidden state is exposed to the classifier and used with the next input to compute gates.",
        "state = (cₜ, hₜ)",
        "Cell state and hidden state have different roles."
      ],
      [
        "Choose what to retain",
        "The forget gate fₜ scales the previous cell state component by component. Values near 1 retain a component; values near 0 erase it. A sigmoid σ produces values between 0 and 1 from the input and previous hidden state. The notation [eₜ; hₜ₋₁] means concatenating these two vectors.",
        "fₜ = σ(W_f[eₜ; hₜ₋₁] + b_f)\nretained = fₜ ⊙ cₜ₋₁",
        "Forgetting is learned and depends on context."
      ],
      [
        "Choose what to write",
        "A tanh layer proposes candidate content gₜ. The input gate iₜ controls how much of it is written. Both depend on the same concatenated input and previous hidden state, but have their own learned parameters.",
        "iₜ = σ(W_i[eₜ; hₜ₋₁] + b_i)\ngₜ = tanh(W_g[eₜ; hₜ₋₁] + b_g)",
        "Candidate proposes content; input gate controls its contribution."
      ],
      [
        "Add retained and new content",
        "The two branches meet at addition. A cell component can preserve its old value, write new content, or combine both. Here ⊙ means elementwise multiplication. Cell values are not restricted to (−1, 1).",
        "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ",
        "An additive path lets the model preserve memory without rewriting all of it."
      ],
      [
        "Choose what to expose",
        "The output gate oₜ scales tanh(cₜ) to form hₜ. Keeping something in the cell and exposing it in the hidden state are distinct operations. All four affine transformations have separate learned parameters.",
        "oₜ = σ(W_o[eₜ; hₜ₋₁] + b_o)\nhₜ = oₜ ⊙ tanh(cₜ)",
        "The classifier reads h_T; the next step receives both cₜ and hₜ."
      ],
      [
        "Carry both states through time",
        "Unroll the cell again. Both amber c and blue h move forward. Every time step reuses the same gate parameters. In our classifier, the last real hidden state feeds the same two-class readout used by the RNN.",
        "(cₜ₋₁, hₜ₋₁) → (cₜ, hₜ)",
        "An LSTM is a recurrent cell with controlled memory updates."
      ]
    ]
  },
  {
    "id": "gates",
    "title": "Experiment with memory",
    "scene": "gates",
    "steps": [
      [
        "Preserve, overwrite, or combine",
        "Start with the illustrative scalar cell cₜ₋₁ = 1. Set f = 1 and i = 0 to preserve it; set f = 0 and i = 1 to replace it with g. Intermediate settings combine retained and new content.<br><br>Change o to see that visible hidden state differs from stored memory. These sliders are interventions, not trained gates. Exact gate endpoints are idealizations: sigmoid gates approach them for finite inputs.",
        "cₜ = f × 1 + i × g\nhₜ = o × tanh(cₜ)",
        "Memory storage and visible output can change independently."
      ],
      [
        "Inspect the direct path",
        "The plot shows fᵀ for a constant forget gate with all other paths held fixed. Compare f = 0.5, 0.95, and 1. Even 0.95 decays over many steps. The full LSTM gradient includes additional dependencies through gates and hidden states.",
        "direct cell path: ∏ fₜ",
        "LSTMs can improve retention; they do not eliminate every gradient problem."
      ]
    ]
  },
  {
    "id": "memory",
    "title": "Open the trained LSTM",
    "scene": "memory",
    "steps": [
      [
        "Inspect one learned memory update",
        "Choose a held-out message, token, and one of 48 LSTM units. The gates and states come from the selected checkpoint. Follow retained content fₜcₜ₋₁ and written content iₜgₜ into their sum; rounding may affect the last displayed decimal.<br><br>The plot follows this unit across the message. A unit can mix features, so a large activation does not identify a “spam neuron”. Explicit unrolling was verified against PyTorch. Next, assess complete predictions with test metrics.",
        "cₜ = fₜcₜ₋₁ + iₜgₜ",
        "The same update connects the architecture, manual illustration, and trained model."
      ]
    ]
  },
  {
    "id": "evaluation",
    "title": "What did the models learn?",
    "scene": "evaluation",
    "steps": [
      [
        "Learn on training data; select on validation",
        "The experiment removes exact token-sequence duplicates, then uses a seeded stratified 80/10/10 split. Vocabulary is built from training data only. Each model learns 32-dimensional embeddings and a 48-unit recurrent layer. Watch recorded training and validation spam F1 over 18 epochs.",
        "F1 = 2 × precision × recall / (precision + recall)",
        "Validation loss selects the checkpoint; test data do not select it."
      ],
      [
        "Accuracy can hide missed spam",
        "Most messages are ham. Always predicting ham achieves high accuracy while finding no spam. Precision asks how many flagged messages are spam; recall asks how much actual spam is found. Use the confusion counts to see both mistakes.",
        "precision = TP / (TP + FP)\nrecall = TP / (TP + FN)",
        "A useful metric must reflect the task and class imbalance."
      ],
      [
        "Read the actual comparison",
        "The RNN performs better in this seeded experiment. This does not contradict the LSTM mechanism: short SMS messages may not require long memory, and optimization and parameter counts differ. Near-duplicate templates may remain. These results are a classroom experiment, not a universal benchmark.",
        "",
        "Architectural capability does not guarantee a win on every dataset."
      ],
      [
        "Check your understanding",
        "A new message arrives. We reset the state, embed its tokens, update recurrent memory, and classify the last real hidden state. Inputs are capped at 64 tokens; unseen tokens map to <UNK>. Use the question below to check the central idea.",
        "",
        "RNN: shared state update. LSTM: shared update with gated cell memory."
      ]
    ]
  }
];
