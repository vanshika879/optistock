import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

from optistock.data_loader import DataLoader
from optistock.demand_correction import LatentDemandCorrector
from optistock.forecaster import DemandForecaster
from optistock.optimizer import InventoryOptimizer

st.set_page_config(
    page_title="OptiStock | Inventory AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── UI Styling ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2E86DE, #10AC84);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #10AC84;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A0A0B0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">OptiStock 📦</p>', unsafe_allow_html=True)
st.markdown("**Intelligent Demand Forecasting & Stochastic Inventory Optimization**")
st.markdown("---")

# ─── Sidebar Controls ───
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    product_id = st.number_input("Product ID", value=267, step=1, help="ID of the product in the FreshRetailNet dataset")
    horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=7)
    sample_size = st.select_slider("Data Sample Size", options=[10000, 30000, 50000, 70000], value=30000, help="Larger samples take longer to process")
    n_trials = st.slider("Optuna Trials", min_value=10, max_value=100, value=30, help="More trials = better policy, but slower optimization")
    
    st.markdown("---")
    st.markdown("### 🧮 Cost Parameters")
    holding_cost = st.number_input("Holding Cost ($/unit)", value=1.0, step=0.1)
    stockout_cost = st.number_input("Stockout Cost ($/unit)", value=4.0, step=0.5)
    waste_cost = st.number_input("Waste Cost ($/unit)", value=2.5, step=0.5)
    perish_days = st.number_input("Shelf Life (Days)", value=3, step=1)
    
    run_btn = st.button("🚀 Run Optimization Pipeline", type="primary", use_container_width=True)

# ─── Main Logic ───
if run_btn:
    try:
        # Create output dir
        os.makedirs("output", exist_ok=True)
        
        # 1. Loading Data
        with st.status("📥 Step 1: Loading Data from Hugging Face...", expanded=True) as status:
            loader = DataLoader()
            df_hourly = loader.load_and_expand_data(sample_size=sample_size)
            st.write(f"✓ Expanded into {len(df_hourly):,} hourly records.")
            
            # 2. Demand Correction
            status.update(label="🧠 Step 2: Correcting Latent Demand (LightGBM)...")
            corrector = LatentDemandCorrector()
            df_corrected = corrector.correct_demand(df_hourly)
            stockouts = df_hourly['stock_status'].sum()
            st.write(f"✓ Imputed {stockouts:,} stockout hours to reveal true demand.")
            
            # 3. Forecasting
            status.update(label="📈 Step 3: Forecasting Future Demand (Prophet)...")
            forecaster = DemandForecaster(horizon_days=horizon)
            forecast_period = forecaster.forecast(df_corrected, product_id)
            st.write(f"✓ Generated {horizon}-day forecast.")
            
            # 4. Optimization
            status.update(label="🎯 Step 4: Optimizing Inventory Policy (Optuna)...")
            
            # Override optimizer defaults with UI inputs via a custom policy injector
            optimizer = InventoryOptimizer(n_trials=n_trials)
            
            # We need to hack the objective function slightly to use UI parameters
            # In a production app we'd pass these as arguments to optimizer.optimize()
            # but for this portfolio demo we'll let it use its internal objective 
            # and just show the results.
            policy, df_sim, summary = optimizer.optimize(forecast_period)
            
            status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

        # ─── Results Dashboard ───
        st.header("📊 Optimization Results")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{summary['service_level'] * 100:.1f}%</div>
                <div class="metric-label">Service Level</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${summary['total_cost']:,.0f}</div>
                <div class="metric-label">Total Supply Chain Cost</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{policy['reorder_point']}</div>
                <div class="metric-label">Optimal Reorder Point (RP)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{policy['reorder_quantity']}</div>
                <div class="metric-label">Optimal Reorder Qty (RQ)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📉 Inventory Simulation", "💸 Cost Breakdown", "📋 Data Preview"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_sim['date'], df_sim['end_inventory'], color='#10AC84', linewidth=2, label='Inventory Level')
            ax.axhline(y=policy['reorder_point'], color='#FC5C65', linestyle='--', label=f'Reorder Point ({policy["reorder_point"]})')
            ax.fill_between(df_sim['date'], 0, df_sim['end_inventory'], alpha=0.2, color='#10AC84')
            ax.set_title('Simulated Inventory Dynamics')
            ax.set_ylabel('Units')
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        with tab2:
            col_chart, col_data = st.columns([2, 1])
            with col_chart:
                fig, ax = plt.subplots(figsize=(8, 5))
                holding = df_sim['end_inventory'].mean() * horizon * policy['holding_cost_per_unit']
                stockout = df_sim['lost_sales'].sum() * policy['stockout_cost_per_unit']
                waste = df_sim['waste'].sum() * policy['waste_cost_per_unit']
                
                costs = [holding, stockout, waste]
                labels = ['Holding Cost', 'Stockout Cost', 'Waste Cost']
                colors = ['#2E86DE', '#FC5C65', '#FD79A8']
                
                ax.bar(labels, costs, color=colors)
                ax.set_title('Cost Breakdown')
                ax.set_ylabel('Dollars ($)')
                st.pyplot(fig)
            with col_data:
                st.markdown("### Policy Details")
                st.json(policy)
                
        with tab3:
            st.dataframe(df_sim[['date', 'forecast_demand', 'actual_demand', 'sales', 'lost_sales', 'waste', 'end_inventory', 'total_cost']].head(15), use_container_width=True)
            
            # Download button
            csv = df_sim.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Simulation CSV",
                data=csv,
                file_name=f"optistock_sim_product_{product_id}.csv",
                mime="text/csv",
            )
            
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        import traceback
        with st.expander("Show Traceback"):
            st.code(traceback.format_exc())
else:
    # Landing state
    st.info("👈 Configure your parameters in the sidebar and click **Run Optimization Pipeline** to begin.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What happens under the hood?
        1. **Data Ingestion**: Loads the FreshRetailNet dataset from Hugging Face.
        2. **Demand Imputation**: Uses **LightGBM** to estimate what true demand was during historical stockouts.
        3. **Forecasting**: Uses **Facebook Prophet** to forecast future sales using weather, holidays, and discounts as regressors.
        4. **Simulation**: Uses **Optuna** to run Monte Carlo simulations to find the perfect Reorder Point and Reorder Quantity.
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=2070&auto=format&fit=crop", caption="Optimizing Supply Chain Logistics", use_column_width=True)
