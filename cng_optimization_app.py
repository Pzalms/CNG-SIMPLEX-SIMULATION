import streamlit as st
import pulp
import pandas as pd
import numpy as np
import altair as alt

def main():
    # ----------------------------------------------------------------------------
    # 1. Title and High-Level Description
    # ----------------------------------------------------------------------------
    st.title("OPTIMISATION OF CNG USAGE IN SINOTRUK HOWO MT13 ENGINE USING SIMPLEX ALGORITHM.")

    st.markdown(
        """
        ## Project Background

        This application demonstrates a **linear** (proof-of-concept) optimization model to 
        **minimize** emissions and **maximize** engine performance of a **Sinotruk HOWO MT13** 
        6x4 tractor truck that runs on **Compressed Natural Gas (CNG)**. The engine has:
        
        - **430 HP**
        - Proven reliability, strong power for logistics and construction
        - Suitable for towing various semi-trailers
        
        The goal is to **optimize** key parameters such as **CNG flow rate**, **valve opening time**, 
        **air-fuel ratio (AFR)**, and **injection timing** to simultaneously:
        
        1. Keep emissions below regulatory thresholds (Euro 5).  
        2. Ensure the engine produces enough power/torque for heavy haulage.  
        3. Achieve optimal (efficient) usage of CNG.  
        
        ---
        """
    )

    # ----------------------------------------------------------------------------
    # 2. Sidebar Inputs
    # ----------------------------------------------------------------------------
    st.sidebar.header("1) Decision Variables: Bounds")

    # Bounds for the 4 decision variables:
    flow_rate_min = st.sidebar.number_input("Min CNG Flow Rate [kg/h]", value=10.0)
    flow_rate_max = st.sidebar.number_input("Max CNG Flow Rate [kg/h]", value=50.0)

    opening_time_min = st.sidebar.number_input("Min Valve Opening Time [ms]", value=1.0)
    opening_time_max = st.sidebar.number_input("Max Valve Opening Time [ms]", value=5.0)

    afr_min = st.sidebar.number_input("Min AFR", value=14.0)
    afr_max = st.sidebar.number_input("Max AFR", value=18.0)

    inj_timing_min = st.sidebar.number_input("Min Injection Timing [deg BTDC]", value=5.0)
    inj_timing_max = st.sidebar.number_input("Max Injection Timing [deg BTDC]", value=25.0)

    st.sidebar.markdown("---")

    st.sidebar.header("2) Objective Function Coefficients (Performance)")
    st.sidebar.markdown("**Maximize** `Perf = a*FlowRate + b*OpeningTime + c*AFR + d*InjTiming`")

    coef_flow = st.sidebar.number_input("a: Coeff for Flow Rate", value=2.0)
    coef_opening = st.sidebar.number_input("b: Coeff for Opening Time", value=3.0)
    coef_afr = st.sidebar.number_input("c: Coeff for AFR", value=1.0)
    coef_inj = st.sidebar.number_input("d: Coeff for Inj Timing", value=2.0)

    st.sidebar.markdown("---")

    st.sidebar.header("3) Constraints: Emissions & Power")
    st.sidebar.markdown("We add two *illustrative* linear constraints:")

    # Example linear constraints for Emissions & Required Power:
    # We'll treat "Emissions" as: E = e1*FlowRate + e2*OpeningTime + e3*AFR + e4*InjTiming
    # We want E <= EmissionsLimit
    # We'll treat "Power" as: P = p1*FlowRate + p2*OpeningTime + p3*AFR + p4*InjTiming
    # We want P >= MinPower
    #
    # The coefficients are purely for demonstration!
    e1 = st.sidebar.number_input("Emission Coeff (FlowRate)", value=0.5)
    e2 = st.sidebar.number_input("Emission Coeff (OpeningTime)", value=0.7)
    e3 = st.sidebar.number_input("Emission Coeff (AFR)", value=0.2)
    e4 = st.sidebar.number_input("Emission Coeff (InjTiming)", value=0.3)
    emissions_limit = st.sidebar.number_input("Max Emissions Allowed", value=30.0)

    p1 = st.sidebar.number_input("Power Coeff (FlowRate)", value=4.0)
    p2 = st.sidebar.number_input("Power Coeff (OpeningTime)", value=2.0)
    p3 = st.sidebar.number_input("Power Coeff (AFR)", value=1.0)
    p4 = st.sidebar.number_input("Power Coeff (InjTiming)", value=1.0)
    min_power = st.sidebar.number_input("Min Power Required", value=80.0)

    st.sidebar.markdown("---")

    st.write("**Click 'Optimize' when you're ready to run the Simplex solver.**")

    # ----------------------------------------------------------------------------
    # 3. Optimization Button
    # ----------------------------------------------------------------------------
    if st.button("Optimize"):
        # 3a. Set up the linear optimization problem
        model = pulp.LpProblem("CNG_Engine_Optimization", pulp.LpMaximize)

        # 3b. Decision Variables
        flow_rate = pulp.LpVariable('FlowRate',
                                    lowBound=flow_rate_min,
                                    upBound=flow_rate_max,
                                    cat='Continuous')
        opening_time = pulp.LpVariable('OpeningTime',
                                       lowBound=opening_time_min,
                                       upBound=opening_time_max,
                                       cat='Continuous')
        afr = pulp.LpVariable('AFR',
                              lowBound=afr_min,
                              upBound=afr_max,
                              cat='Continuous')
        inj_timing = pulp.LpVariable('InjectionTiming',
                                     lowBound=inj_timing_min,
                                     upBound=inj_timing_max,
                                     cat='Continuous')

        # 3c. Objective Function
        # Maximize: Perf = a*FlowRate + b*OpeningTime + c*AFR + d*InjTiming
        model += (
            coef_flow * flow_rate
            + coef_opening * opening_time
            + coef_afr * afr
            + coef_inj * inj_timing
        ), "EnginePerformance"

        # 3d. Constraints
        # Emissions constraint: e1*FlowRate + e2*OpenTime + e3*AFR + e4*InjTiming <= emissions_limit
        model += (
            e1 * flow_rate
            + e2 * opening_time
            + e3 * afr
            + e4 * inj_timing
        ) <= emissions_limit, "EmissionConstraint"

        # Power constraint: p1*FlowRate + p2*OpenTime + p3*AFR + p4*InjTiming >= min_power
        model += (
            p1 * flow_rate
            + p2 * opening_time
            + p3 * afr
            + p4 * inj_timing
        ) >= min_power, "PowerConstraint"

        # Solve
        solution_status = model.solve(pulp.PULP_CBC_CMD(msg=False))

        # ----------------------------------------------------------------------------
        # 4. Results
        # ----------------------------------------------------------------------------
        if solution_status == 1:  # 1 => Optimal
            st.success("Status: Optimal solution found!")
            opt_flow_rate = pulp.value(flow_rate)
            opt_opening_time = pulp.value(opening_time)
            opt_afr = pulp.value(afr)
            opt_inj_timing = pulp.value(inj_timing)
            objective_val = pulp.value(model.objective)

            st.markdown(
                f"""
                **Optimal Decision Variables:**
                - **Flow Rate:** {opt_flow_rate:.3f} kg/h
                - **Valve Opening Time:** {opt_opening_time:.3f} ms
                - **AFR:** {opt_afr:.3f}
                - **Injection Timing:** {opt_inj_timing:.3f} °BTDC

                **Maximum Performance Value:** {objective_val:.3f}
                """
            )

            # Check Emissions & Power at Optimal
            emissions_at_opt = (e1 * opt_flow_rate
                                + e2 * opt_opening_time
                                + e3 * opt_afr
                                + e4 * opt_inj_timing)
            power_at_opt = (p1 * opt_flow_rate
                            + p2 * opt_opening_time
                            + p3 * opt_afr
                            + p4 * opt_inj_timing)
            st.markdown(
                f"""
                **Constraint Values at Optimum:**
                - Emissions = {emissions_at_opt:.3f} (Limit = {emissions_limit})
                - Power = {power_at_opt:.3f} (Minimum Required = {min_power})
                """
            )

            # ----------------------------------------------------------------------------
            # 5. Parametric Analysis & Interactive Charts
            # ----------------------------------------------------------------------------
            st.markdown("### Parametric Analysis Around the Optimum")
            st.write(
                """
                Below, we visualize how the **objective function** (Performance) changes 
                if we vary each variable **±20%** around the optimal value, while keeping 
                the others at their optimal levels.
                """
            )

            # Create a function to compute objective given the 4 variables
            def compute_objective(fr, ot, a, it):
                return (coef_flow * fr
                        + coef_opening * ot
                        + coef_afr * a
                        + coef_inj * it)

            # We will loop over each variable independently
            # We'll create separate line charts for each variable
            # Variation of ±20%
            var_data = []
            n_points = 21  # how many points to plot for each variable
            frac_range = np.linspace(0.8, 1.2, n_points)

            # Helper to clamp values within min & max range
            def clamp(val, minv, maxv):
                return max(min(val, maxv), minv)

            # 5a. Flow Rate Variation
            fr_list = []
            perf_list = []
            for frac in frac_range:
                test_fr = clamp(opt_flow_rate * frac, flow_rate_min, flow_rate_max)
                perf = compute_objective(
                    test_fr, opt_opening_time, opt_afr, opt_inj_timing
                )
                fr_list.append(test_fr)
                perf_list.append(perf)
            df_flow = pd.DataFrame({"FlowRate": fr_list, "Performance": perf_list})

            # 5b. Opening Time Variation
            ot_list = []
            perf_list2 = []
            for frac in frac_range:
                test_ot = clamp(opt_opening_time * frac, opening_time_min, opening_time_max)
                perf = compute_objective(
                    opt_flow_rate, test_ot, opt_afr, opt_inj_timing
                )
                ot_list.append(test_ot)
                perf_list2.append(perf)
            df_open = pd.DataFrame({"OpeningTime": ot_list, "Performance": perf_list2})

            # 5c. AFR Variation
            afr_list = []
            perf_list3 = []
            for frac in frac_range:
                test_afr = clamp(opt_afr * frac, afr_min, afr_max)
                perf = compute_objective(
                    opt_flow_rate, opt_opening_time, test_afr, opt_inj_timing
                )
                afr_list.append(test_afr)
                perf_list3.append(perf)
            df_afr = pd.DataFrame({"AFR": afr_list, "Performance": perf_list3})

            # 5d. Injection Timing Variation
            it_list = []
            perf_list4 = []
            for frac in frac_range:
                test_it = clamp(opt_inj_timing * frac, inj_timing_min, inj_timing_max)
                perf = compute_objective(
                    opt_flow_rate, opt_opening_time, opt_afr, test_it
                )
                it_list.append(test_it)
                perf_list4.append(perf)
            df_inj = pd.DataFrame({"InjectionTiming": it_list, "Performance": perf_list4})

            # ----------------------------------------------------------------------------
            # 6. Plot Each Variation Using Altair
            # ----------------------------------------------------------------------------
            # 6a. Flow Rate Chart
            flow_chart = alt.Chart(df_flow).mark_line(point=True).encode(
                x=alt.X('FlowRate', title='CNG Flow Rate [kg/h]'),
                y=alt.Y('Performance', title='Objective Function'),
                tooltip=['FlowRate', 'Performance']
            ).properties(
                width=350, height=300, title='Performance vs. Flow Rate'
            )

            # 6b. Opening Time Chart
            open_chart = alt.Chart(df_open).mark_line(point=True).encode(
                x=alt.X('OpeningTime', title='Valve Opening Time [ms]'),
                y=alt.Y('Performance', title='Objective Function'),
                tooltip=['OpeningTime', 'Performance']
            ).properties(
                width=350, height=300, title='Performance vs. Opening Time'
            )

            # 6c. AFR Chart
            afr_chart = alt.Chart(df_afr).mark_line(point=True).encode(
                x=alt.X('AFR', title='Air-Fuel Ratio'),
                y=alt.Y('Performance', title='Objective Function'),
                tooltip=['AFR', 'Performance']
            ).properties(
                width=350, height=300, title='Performance vs. AFR'
            )

            # 6d. Injection Timing Chart
            inj_chart = alt.Chart(df_inj).mark_line(point=True).encode(
                x=alt.X('InjectionTiming', title='Injection Timing [°BTDC]'),
                y=alt.Y('Performance', title='Objective Function'),
                tooltip=['InjectionTiming', 'Performance']
            ).properties(
                width=350, height=300, title='Performance vs. Injection Timing'
            )

            # Arrange charts in a grid
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(flow_chart, use_container_width=True)
            with col2:
                st.altair_chart(open_chart, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.altair_chart(afr_chart, use_container_width=True)
            with col4:
                st.altair_chart(inj_chart, use_container_width=True)

        else:
            st.error(
                f"Could not find an optimal solution. Solver Status = {pulp.LpStatus[solution_status]}"
            )

    # ----------------------------------------------------------------------------
    # 7. Usage Instructions
    # ----------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## How to Use / Defend This Model

        1. **Adjust Decision Variable Ranges** in the sidebar (e.g., Flow Rate, Opening Time, etc.).  
        2. **Set Coefficients** in the objective function to reflect how each parameter influences 
           overall performance in your simplified model.  
        3. **Set Emission & Power Constraints** to reflect Euro 5 compliance and the minimum 
           power requirement for heavy haulage.  
        4. **Click 'Optimize'** to run the Simplex solver. The app will either show 
           an optimal solution or indicate infeasibility.  
        5. **Analyze the Result**:  
           - **Decision Variable** outputs (Flow Rate, Opening Time, etc.)  
           - **Objective Value** (a numeric performance metric)  
           - **Emissions & Power** at the optimum  
           - **Parametric Charts** to see how performance changes near the optimum  

        ### Important Notes

        - In reality, engine performance is **nonlinear** and must be modeled via advanced 
          thermodynamic / empirical methods (e.g., GT-Power, experimental data).  
        - Additional constraints (e.g., cylinder pressure limits, knock thresholds, turbocharger 
          maps) are usually required for a realistic calibration.  
        - This **linear** model is purely a demonstration of how you might structure an 
          optimization approach for a project defense or academic prototype.
        """
    )

    st.markdown(
        """
        ---
        **© 2025 Sinotruk HOWO MT13 Optimization Demo** 
        """
    )

if __name__ == "__main__":
    main()
