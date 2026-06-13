# AI-Optimized Ecosystem Simulation
**An Ecological Decision Support System for Habitat Management and Species Reintroduction**

**Course:** CSE-307 Introduction to Artificial Intelligence  
**Instructor:** Dr. Syed Ali Raza  
**Institution:** IBA Karachi (Spring 2026)  

**Team Members:**
* Huzaifa Iqbal (31563)
* Muhammad Usman Hussain (28992)
* Ali Hamza (30618)
* Daniyaal Bokhari (31571)

---

## 1. The Yellowstone Problem

Wildlife conservationists and park rangers frequently face the important and very fragile task of reintroducing endangered species into an ecosystem or managing wildlife reserves (e.g., the famous reintroduction of wolves to Yellowstone National Park). This process is incredibly unstable.

* **The Problem:** Ecosystems rely on a fragile domino effect. If conservationists introduce too many predators, the prey population is hunted and forced to become extinct, which then follows through to the predators dying of starvation. On the other hand, if they introduce too few predators, the herbivores overpopulate, overgraze the plant life, and cause total ecological collapse. Currently, the process of determining the perfect starting populations and managing the biological parameters of a habitat relies mainly on trial, error, and human guesswork, which is very costly and dangerous.
* **The AI Solution:** This project acts as an Ecological Decision Support System. By simulating specific climates on a digital grid, our software uses four artificial intelligence algorithms to calculate the optimal populations, predict long-term survival, and extract actionable conservation policies to guarantee stability in fragile habitats.

---

## 2. Project Architecture

The whole code is neatly organized into three separate parts: the physics rules, the AI models, and the visual interface. Most importantly, we built the whole AI completely from scratch without using any existing libraries or tools.

* **simulation.py:** The main physics and biology engine.
* **genetic_algorithm.py:** Evolutionary optimizer (GA).
* **particle_swarm.py:** Swarm intelligence optimizer (PSO).
* **ml_models.py:** Custom Neural Network (Predictor) and Decision Tree (Analyzer).
* **analysis.py:** Computes data for all 6 required experiments.
* **gui.py:** The Tkinter visual dashboard.

---

## 3. The GUI Dashboard

To make our AI easy to use for conservationists, we built a professional, multi-layered Tkinter dashboard. Our interface is split into 8 dedicated tabs, allowing researchers to run simulations, optimize parameters, and analyze data in real time.

### 3.1 Tab 1: Live Simulation
This is the visualized grid. It allows users to watch the ecosystem interact live on a monthly basis.
* **The Grid:** A dynamic cellular automata map rendering using emojis. It tracks plant spread, herbivore grazing, and carnivore hunting visually.
* **Seasonal Rendering:** The grid background changes color based on the season (Spring/Summer/Autumn/Winter).
* **Live Controls:** Users can set the grid size, hemisphere, and simulation speed.
* **Habitat Presets:** Allows users to select predefined environments like Tropical Forest, Desert/Arid, or Arctic Region.
* **'Use AI-Optimized Params' button:** Allows the user to instantly apply the best parameters discovered by the GA or PSO, visually proving that the AI's algorithm creates a sustainable ecosystem compared to default human guesswork or trial and error.
* **Populations Graph:** Plots the real-time counts of all three species as the months tick by.
* **Monthly Details:** Displays enhanced monthly statistics including birth/death rates, food index, and gender ratios.

### 3.2 Tab 2: Genetic Algorithm (Optimizer)
This tab is dedicated to running the genetic evolutionary algorithm.
* **Controls:** Users can set the Population Size, Generations, and Mutation Rate.
* **Live Charts:** While the GA runs, the GUI plots three real-time charts: Fitness over Generations, Population Diversity, and a live-updating normalized bar chart of the exact 24 parameters the GA is currently recommending to save the ecosystem. Applying these parameters automatically syncs them to the manual sliders.

### 3.3 Tab 3: Particle Swarm Optimization (Optimizer)
This tab is dedicated to running the swarm intelligence optimization algorithm.
* **Controls:** Users can set the Swarm Size, Iterations, Inertia (w), Cognitive (c1), and Social (c2) coefficients.
* **Live Charts:** Plots fitness convergence, swarm diversity, and a bar chart showing the improved change of fitness per iteration.

### 3.4 Tab 4: Neural Network (Predictor)
This tab predicts ecosystem survival without needing to run slow simulations.
* **Data Collection:** A button that runs hundreds of simulations in the background to generate a fresh X and y dataset.
* **Training View:** After being trained, the GUI plots the MSE Loss curve over epochs.
* **Evaluation View:** The GUI displays an "Actual vs. Predicted" scatter plot which calculates the R² score and an absolute error distribution histogram showing the Mean Absolute Error (MAE).
* **Test and Compare:** Allows testing current parameters and visually comparing the NN's prediction directly against an actual simulation run.

### 3.5 Tab 5: Decision Tree (Policy Analyzer)
This tab translates complex data into human-readable policies.
* **Rule Extraction:** A button pops up a window displaying the IF/THEN rules generated by the tree.
* **Visuals:** It renders a horizontal bar chart of the top feature importances and a heatmap confusion matrix of the tree's accuracy.

### 3.6 Tab 6: Research Analysis
The main hub for our research experiments. Clicking any experiment button triggers `analysis.py` to compile the data and render Matplotlib figures directly in the GUI.

### 3.7 Tab 7: Parameters
This tab provides manual control over the ecosystem.
* **Parameter Sliders:** Grants access to manually tune all 24 ecosystem parameters with sliders and numeric inputs.
* **Presets and Exporting:** Users can apply habitat presets, reset to defaults, or export the currently selected parameters directly to the simulation.

### 3.8 Tab 8: Population Analysis
This tab offers deep ecosystem insights and visual analytics after a simulation run.
* **Predator-Prey Dynamics:** A graph detailing the population cycles between herbivores and carnivores.
* **Population Stability:** A 12-month rolling stability score graph identifying stable and warning thresholds.
* **Gender Ratios & Biodiversity:** Pie charts for species gender ratios and a Shannon Biodiversity Index graph to track ecological balance.

---

## 4. The Core Simulation Engine (simulation.py)

The environment is working based on 24 parameters (`EcosystemParams`), which control everything from initial populations to metabolic rates.

* **Metabolism & Starvation:** Animals have finite energy. Moving and aging deducts energy. Eating restores energy. If energy hits 0, they die.
* **Reproduction:** Herbivores and Carnivores have genders (M/F). Females require a mate, a pregnancy timer, and a nursing timer to produce offspring, introducing realistic population bottlenecks.
* **Seasons:** Plant growth is modified based on the month and chosen Hemisphere (North vs. South).
* **Fitness Metric:** Success is calculated by combining survival time (months), Shannon diversity (species balance), and population stability (variance).

---

## 5. The 4 AI Algorithms

1. **Genetic Algorithm:** This replicates natural selection. We use tournament selection (k = 3), uniform crossover, and Gaussian mutation to evolve the 24 parameters toward maximum fitness.
2. **Particle Swarm Optimization:** This replicates flocking behavior. Particles update their velocity vectors based on their personal best and the swarm's global best parameters, trying to find the best and highest fitness.
3. **Neural Network:** A 24-64-32-1 feedforward architecture using ReLU activations and linear output. Trained using custom mini-batch SGD and backpropagation to minimize Mean Squared Error (MSE).
4. **Decision Tree:** Uses information gain and Shannon entropy to recursively split the parameter space. It classifies ecosystems into binary outcomes ("Survived" vs. "Collapsed") to extract ecological thresholds.

---

## 6. The 6 Research Experiments

Located in `analysis.py`, this is the rigorous research component required in the project:

1. **GA vs PSO Comparison:** Compares and analyses which algorithm converges faster and gets a higher fitness peak.
2. **Parameter Sensitivity:** Uses One-at-a-Time (OAT) variance testing to separate which biological traits cause the ecosystem to collapse the fastest.
3. **NN Accuracy:** Validates the predictive power of the neural network using R² and MAE metrics.
4. **Hemisphere Comparison:** Statistically shows distinction of survival outcomes between Northern and Southern seasonal phases.
5. **Evolutionary Dynamics:** Analyzes the trade-off between genetic exploration (diversity) and exploitation (convergence) during a GA run.
6. **DT Rule Extraction:** Maps the feature importance of biological constraints.
7. **Grid Scalability:** Tests if optimized parameters hold up when scaling the physical grid from 10x10 up to 40x40.

---

## 7. Installation & Usage

The system requires standard scientific libraries but strictly avoids ML frameworks.

```bash
# 1. Install required dependencies
pip install numpy matplotlib scipy

# 2. Launch the Interactive GUI
python gui.py
