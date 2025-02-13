# CNG Usage Optimization for Sinotruk HOWO MT13

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Methodology](#methodology)  
4. [Installation](#installation)  
5. [Usage Instructions](#usage-instructions)  
6. [Project Structure](#project-structure)  
7. [Interpreting Results](#interpreting-results)  
8. [Future Work](#future-work)  
9. [License](#license)

---

## Project Overview
This repository contains a **linear optimization** (Simplex) demonstration for **Compressed Natural Gas (CNG) usage** in a **Sinotruk HOWO MT13.43-50** 6x4 tractor truck:

- **Engine:** 430 HP, Euro 5 Emission Standard (MT13.43-50)
- **Objective:** To optimize engine performance and CNG efficiency, subject to emission and power constraints.
- **Approach:** Proof-of-concept **linear** model using [PuLP](https://github.com/coin-or/pulp) for optimization and [Streamlit](https://streamlit.io/) for the user interface.

> **Disclaimer:**  
> Real engine calibration is typically **nonlinear** and governed by advanced thermodynamics. The linear model here is strictly a **demonstration** of an optimization approach.

---

## Key Features
1. **Four Main Decision Variables**  
   - **CNG Flow Rate** (kg/h)  
   - **Valve Opening Time** (ms)  
   - **Air-Fuel Ratio (AFR)**  
   - **Injection Timing** (°BTDC)

2. **Linear Objective Function**  
   - **Maximize**:  
     \[
       \text{Performance} = a \times \text{FlowRate}
                          + b \times \text{OpeningTime}
                          + c \times \text{AFR}
                          + d \times \text{InjectionTiming}
     \]

3. **Constraints**  
   - **Emission Constraint**:  
     \[
       e_1 \times \text{FlowRate}
       + e_2 \times \text{OpeningTime}
       + e_3 \times \text{AFR}
       + e_4 \times \text{InjectionTiming} \;\le\; \text{EmissionsLimit}
     \]  
   - **Power Constraint**:  
     \[
       p_1 \times \text{FlowRate}
       + p_2 \times \text{OpeningTime}
       + p_3 \times \text{AFR}
       + p_4 \times \text{InjectionTiming} \;\ge\; \text{MinPowerRequired}
     \]  

4. **Interactive Frontend**  
   - **Streamlit** UI to adjust variable ranges, coefficients, and constraints in real-time.
   - **Altair Charts** to visualize objective function sensitivity around the optimal point.

---

## Methodology
1. **Linear Formulation**: A simplified linear equation approximates how each decision variable influences overall engine performance.  
2. **Decision Variables**: Bounded by realistic min/max feasible values (e.g., Flow Rate from 10 kg/h to 50 kg/h).  
3. **Constraints**: Illustrative constraints on emissions (must not exceed a certain threshold) and power (must exceed a minimum).  
4. **Simplex Solver**: Utilizes PuLP’s default CBC solver to find an **optimal** (maximum) solution.  

Because the real engine behavior is highly nonlinear (knock limits, turbocharger maps, stoichiometry, etc.), this linear approach is meant only as a conceptual framework for how you might **structure** an optimization problem.

---

## Installation

1. **Clone or Download this Repository**:
   ```bash
   git clone https://github.com/pzalms/cng-optimization-demo.git
   cd cng-optimization-demo
   ```

2. **Install Required Packages** (e.g., in a virtual environment):
   ```bash
   pip install streamlit pulp altair
   ```
   - [**Streamlit**](https://streamlit.io/) for creating the web-based dashboard.  
   - [**PuLP**](https://pypi.org/project/PuLP/) for linear programming and the Simplex algorithm.  
   - [**Altair**](https://altair-viz.github.io/) for plotting interactive charts.

---

## Usage Instructions
1. **Run the Streamlit App**:
   ```bash
   streamlit run cng_optimization_app.py
   ```
2. **Open Your Web Browser**:  
   - By default, Streamlit will launch at [http://localhost:8501](http://localhost:8501).  

3. **Adjust Parameters in the Sidebar**:
   - **Decision Variable Bounds** (Flow Rate, Opening Time, AFR, Injection Timing)  
   - **Objective Function Coefficients** (weights for each variable in the performance equation)  
   - **Emission & Power Constraints** (linear approximations)

4. **Click "Optimize"**:
   - The solver (CBC) will find the feasible combination of variables that **maximizes** the performance metric.  
   - If a solution is found, you’ll see an **optimal** set of decision variables.

5. **View Results & Charts**:
   - The app displays the **Optimal Variables**, **Objective Function** value, and **Constraint Values** at the optimum.  
   - **Altair charts** show how performance changes near the optimal region when you vary each variable by ±20%.

---

## Project Structure

```
cng-optimization-demo/
│
├── cng_optimization_app.py     # Main Streamlit application
├── README.md                   # This README file
└── requirements.txt            # (Optional) list of dependencies
```

- **`cng_optimization_app.py`**: 
  - Defines the decision variables (Flow Rate, Valve Opening Time, AFR, Injection Timing).  
  - Sets up the objective function (Max Performance).  
  - Imposes linear constraints (Emissions ≤ Limit, Power ≥ Minimum).  
  - Uses PuLP for solving.  
  - Generates interactive Altair charts for a parametric sensitivity analysis.

---

## Interpreting Results
Once you press **"Optimize"** and receive a solution:
1. **Optimal Decision Variables**: The best set of Flow Rate, Opening Time, AFR, and Injection Timing that yields the highest objective function value—subject to constraints.  
2. **Objective Function Value**: A single metric representing the simplified "Engine Performance."  
3. **Emission & Power Values**: Check that the final solution does not exceed the **EmissionsLimit** and meets/exceeds the **MinPower** requirement.  
4. **Parametric Analysis Charts**:  
   - Let you see how **sensitive** the performance is to small changes in each variable around its **optimal** point.  
   - If performance changes significantly with small variable deviations, the solution might require fine-tuned controls in real-world applications.

---

## Future Work
1. **Nonlinear Modeling**  
   - Real engine calibration typically uses more complex models (e.g., polynomial, multi-dimensional maps, or thermodynamic simulation).  
   - Incorporate knock limits, turbo maps, real stoichiometric constraints, etc.

2. **Multi-Objective Optimization**  
   - Simultaneous minimization of emissions and maximization of efficiency.  
   - Could use weighted-sum approaches or Pareto-based methods.

3. **Real-World Data Integration**  
   - Use actual test data or advanced simulation tools (e.g., GT-Power) to refine the objective and constraints.

4. **Robustness & Uncertainty**  
   - Account for uncertainties (fuel quality variability, ambient conditions, sensor drift).

---

## License
This project is intended as an **educational/academic** resource. You may adapt or extend the code for personal or research use. For commercial or production use, consult appropriate legal terms and conditions.

---

**© 2025 Sinotruk HOWO MT13 CNG Optimization Demonstration**  
Developed as a proof-of-concept for project defense and academic demonstration.
