import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="System Metrics Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    /* Center the tabs container */
    .css-1d391kg {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Style the tabs like buttons */
    .stTabs button {
        background-color: #6f2c91;  /* Purple color */
        color: white;
        border: 2px solid #6f2c91;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin: 5px;
        transition: background-color 0.3s, transform 0.3s;
    }

    .stTabs button:hover {
        background-color: #5a2383;  /* Darker purple on hover */
        transform: scale(1.05);  /* Slight scale effect on hover */
    }

    /* Active tab styling */
    .stTabs button:active {
        background-color: #9c4da2;  /* Lighter purple on click */
    }

    /* Optional: Remove the underline and default border */
    .css-1r6slb0 {
        border: none;
        text-decoration: none;
    }
    /* Theme-compatible styles */
    .block-container {
        padding: 0.47rem;
        margin-top: 0rem;
        max-width: 100%;
        margin-bottom: 0rem;
    }

    /* Remove space above the tabs */
    .stTabs {
        margin-top: 0 !important; 
        padding-top: 0 !important;
    }

    /* Dashboard title - increase font size */
    .dashboard-title {
        font-size: 2.3rem !important;
        margin-bottom: 0rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text-color);
    }

    /* Align the logo with the title at the top */
    .header-container {
        display: flex;
        align-items: center;
        position: sticky;
        top: 0;  /* Ensures it stays at the top */
        padding: 0px 0px;
        background-color: var(--background-color); /* Set a background for better visibility */
        z-index: 0;
    }

    .logo-container {
        position: absolute;
        top: 5px;
        right: 0px;
        width: 133px; /* Adjust size if needed */
    }

    /* Add spacing for header elements */
    .stWrite {
        margin-top: 0rem !important;
    }

    /* Floating footer - theme compatible */
    .floating-footer {
        position: fixed;
        right: 16px;
        bottom: 16px;
        z-index: 999;
        background-color: #6f2c91;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Hero Section - Upper Part */
    .hero-section {
        background-image: url('data:image/jpeg;base64,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'); /* Replace with your background image */
        background-size: cover;
        background-position: center;
        color: #fff;
        text-align: center;
        padding: 0;  /* Remove padding */
        margin-top: 0;  /* Remove margin */
        margin-bottom: 0; /* Remove margin */
        height: auto; /* Let content define the height */
    }

    .hero-section h1 {
        font-size: 3rem;
        margin-bottom: 0px;
        color: white;
    }

    .hero-section p {
        font-size: 1.5rem;
        margin-bottom: 30px;
    }

    .hero-section .button {
        padding: 15px 30px;
        background-color: #5a2383;
        color: #fff;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .hero-section .button:hover {
        background-color: #5a2383;
    }

    /* Overview Section */
    .overview {
        background-color: #fff;
        padding: 50px 0;
        text-align: center;
    }

    .overview h2 {
        font-size: 2.5rem;
        margin-bottom: 30px;
        color: #333;
    }

    /* Cards for Features in Overview */
    .feature-cards {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
    }

    .card {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        width: 30%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card img {
        width: 100%;
        height: auto;
        border-radius: 8px;
        transition: transform 0.3s ease;
    }

    .card h3 {
        margin-top: 15px;
        font-size: 1.5rem;
    }

    .card p {
        color: #555;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Hover Effects for Cards */
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .card:hover img {
        transform: scale(1.05);
    }

    .hero-section .card {
    height: 345px; /* You can adjust this to your desired height */
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between; /* Ensures content is evenly spaced */
    align-items: center; /* Centers the content horizontally */
    text-align: center; /* Centers the text inside the card */
}

.hero-section .card .card-body {
    flex-grow: 1; /* Ensures content stretches within the card to fill height */
}

    /* Responsive Design */
    @media screen and (max-width: 768px) {
        .feature-cards {
            flex-direction: column;
            align-items: center;
        }

        .card {
            width: 100%;
            margin-bottom: 0px;
        }

        .hero-section {
            padding: 80px 20px;
        }

        .hero-section h1 {
            font-size: 2.5rem;
        }

        .hero-section p {
            font-size: 1.2rem;
        }
    }
    .logo-container-dxc {
                position: absolute;
                top: 5px;
                right: 0px;
                width: 133px; /* Adjust size if needed */
            }
            /* Additional style for the landing page button */
            .landing-button-container {
                margin-top: 30px;  /* Position the button below the text */
            }
    </style>
""", unsafe_allow_html=True)

def landing_page():
    st.markdown("""
        <div class="hero-section">
            <h1>Welcome to DXC System Metrics Dashboard</h1>
            <div class="logo-container-dxc">
                <img src="https://dxc.com/content/dam/dxc/projects/dxc-com/us/images/about-us/newsroom/logos-for-media/vertical/DXC%20Logo_Purple+Black%20RGB.png" 
                     alt="DXC Logo">
            </div>
            <p>Track your CPU and Memory and Disk usage in real-time</p>
            <div class="feature-cards">
                <div class="card">
                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTXAkKt0-B6aiQwXW7haMZ0dvHPQtp9UMDMsw&s" alt="CPU Usage">
                    <h3>CPU Usage</h3>
                    <p>Track your system's CPU performance in real-time.</p>
                </div>
                <div class="card">
                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROVFHKRuLPhdC25KAvqFuUH4ROueTmuZa2cA&s" alt="Memory Usage">
                    <h3>Memory Usage</h3>
                    <p>Monitor memory utilization to ensure optimal performance.</p>
                </div>
                <div class="card">
                    <img src="https://dxc.scene7.com/is/image/dxc/AdobeStock_298559228?$landscape_desktop$" alt="Disk Usage">
                    <h3>Disk Usage</h3>
                    <p>Get insights on disk space usage and manage your storage better.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    # Specify the engine explicitly
    cpu_data = pd.read_excel(file_path, sheet_name='CPU', engine='openpyxl')  # Use openpyxl for .xlsx files
    memory_data = pd.read_excel(file_path, sheet_name='Memory', engine='openpyxl')
    disk_data = pd.read_excel(file_path, sheet_name='Disk', engine='openpyxl')

    cpu_columns = [
        "FQDN", "Datetime",
        "GBL_CPU_TOTAL_UTIL", "forecastvalues"
    ]
    cpu_data.columns = cpu_columns
    cpu_data['Datetime'] = pd.to_datetime(cpu_data['Datetime'], errors='coerce')
    cpu_data['Hour'] = cpu_data['Datetime'].dt.hour
    cpu_data['Minute'] = cpu_data['Datetime'].dt.minute
    cpu_data['DayOfWeek'] = cpu_data['Datetime'].dt.dayofweek
    cpu_data['IsWeekend'] = cpu_data['DayOfWeek'].isin([5, 6]).astype(int)

    memory_columns = [
        "FQDN", "Datetime",
        "GBL_MEM_UTIL", "forecastvalues"
    ]
    memory_data.columns = memory_columns
    memory_data['Datetime'] = pd.to_datetime(memory_data['Datetime'], errors='coerce')
    memory_data['Hour'] = memory_data['Datetime'].dt.hour
    memory_data['Minute'] = memory_data['Datetime'].dt.minute
    memory_data['DayOfWeek'] = memory_data['Datetime'].dt.dayofweek
    memory_data['IsWeekend'] = memory_data['DayOfWeek'].isin([5, 6]).astype(int)
    
    disk_columns = [
        "FQDN", "FS_DIRNAME", "Datetime", "FS_SPACE_UTIL", "forecastvalues"
    ]
    disk_data.columns = disk_columns
    disk_data['Datetime'] = pd.to_datetime(disk_data['Datetime'], errors='coerce')
    disk_data['Hour'] = disk_data['Datetime'].dt.hour
    disk_data['Minute'] = disk_data['Datetime'].dt.minute
    disk_data['DayOfWeek'] = disk_data['Datetime'].dt.dayofweek
    disk_data['IsWeekend'] = disk_data['DayOfWeek'].isin([5, 6]).astype(int)

    return cpu_data, memory_data, disk_data



file_path = "CPU_MEMORY_DISK_ACTUAL_PREDICTIONS.xlsx" 

def display_overview_tab(cpu_data, memory_data, disk_data):
    st.subheader("📊 **Overview** 👀")

    # Clean up column names (remove spaces and hidden characters)
    cpu_data.columns = cpu_data.columns.str.replace(r'\s+', '', regex=True)
    memory_data.columns = memory_data.columns.str.replace(r'\s+', '', regex=True)
    disk_data.columns = disk_data.columns.str.replace(r'\s+', '', regex=True)

    cpu_data.columns = cpu_data.columns.str.strip()
    memory_data.columns = memory_data.columns.str.strip()
    disk_data.columns = disk_data.columns.str.strip()

    # Ensure 'FS_SPACE_UTIL' is numeric and clean data
    disk_data['FS_SPACE_UTIL'] = pd.to_numeric(disk_data['FS_SPACE_UTIL'], errors='coerce')
    disk_data = disk_data.dropna(subset=['FS_SPACE_UTIL'])

    # Convert columns to numeric (if they aren't already)
    cpu_data['GBL_CPU_TOTAL_UTIL'] = pd.to_numeric(cpu_data['GBL_CPU_TOTAL_UTIL'], errors='coerce')
    memory_data['GBL_MEM_UTIL'] = pd.to_numeric(memory_data['GBL_MEM_UTIL'], errors='coerce')

    # Drop rows where any of the required columns have NaN values
    cpu_data = cpu_data.dropna(subset=['GBL_CPU_TOTAL_UTIL'])
    memory_data = memory_data.dropna(subset=['GBL_MEM_UTIL'])
    disk_data = disk_data.dropna(subset=['FS_SPACE_UTIL'])

    # Calculate average CPU and memory utilization per server
    server_cpu_mean = cpu_data.groupby('FQDN')['GBL_CPU_TOTAL_UTIL'].mean().reset_index()
    server_memory_mean = memory_data.groupby('FQDN')['GBL_MEM_UTIL'].mean().reset_index()
    server_disk_mean = disk_data.groupby('FQDN')['FS_SPACE_UTIL'].mean().reset_index()  # For disk

    # Sort the server data by utilization
    server_cpu_mean = server_cpu_mean.sort_values(by='GBL_CPU_TOTAL_UTIL', ascending=False)
    server_memory_mean = server_memory_mean.sort_values(by='GBL_MEM_UTIL', ascending=False)
    server_disk_mean = server_disk_mean.sort_values(by='FS_SPACE_UTIL', ascending=False)  # For disk

    # Round the values for better readability
    server_cpu_mean['GBL_CPU_TOTAL_UTIL'] = server_cpu_mean['GBL_CPU_TOTAL_UTIL'].round(2)
    server_memory_mean['GBL_MEM_UTIL'] = server_memory_mean['GBL_MEM_UTIL'].round(2)
    server_disk_mean['FS_SPACE_UTIL'] = server_disk_mean['FS_SPACE_UTIL'].round(2)  # For disk

    # Create columns to display data
    col1, col2, col3 = st.columns(3)  # Adding one more column for disk utilization

    with col1:
        st.write("#### 💻 **Top 5 Servers with Highest Average CPU Utilization:** 🔥")
        server_cpu_mean = server_cpu_mean[['FQDN', 'GBL_CPU_TOTAL_UTIL']]
        server_cpu_mean.columns = ['Server Name', 'Average CPU Utilization (%)']
        st.write(server_cpu_mean.head(5).to_html(index=False), unsafe_allow_html=True)

    with col2:
        st.write("#### 🧠 **Top 5 Servers with Highest Average Memory Utilization:** ⚡️")
        server_memory_mean = server_memory_mean[['FQDN', 'GBL_MEM_UTIL']]
        server_memory_mean.columns = ['Server Name', 'Average Memory Utilization (%)']
        st.write(server_memory_mean.head(5).to_html(index=False), unsafe_allow_html=True)

    with col3:
        st.write("#### 💾 **Top 5 Servers with Highest Average Disk Utilization:** 🚀")
        server_disk_mean = server_disk_mean[['FQDN', 'FS_SPACE_UTIL']]
        server_disk_mean.columns = ['Server Name', 'Average Disk Utilization (%)']
        st.write(server_disk_mean.head(5).to_html(index=False), unsafe_allow_html=True)

    # Plot CPU utilization:
    col1, col2, col3 = st.columns(3)

    with col1:
        plt.figure(figsize=(9, 7)) 
        bars = plt.bar(server_cpu_mean['Server Name'].head(5), server_cpu_mean['Average CPU Utilization (%)'].head(5), color='purple')
        plt.xlabel('Servers')
        plt.ylabel('Average CPU Utilization (%)')
        plt.title('Top 5 Servers with Highest Average CPU Utilization')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

        plt.ylim(bottom=1)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    with col2:
        plt.figure(figsize=(9, 7)) 
        bars = plt.bar(server_memory_mean['Server Name'].head(5), server_memory_mean['Average Memory Utilization (%)'].head(5), color='blue')
        plt.xlabel('Servers')
        plt.ylabel('Average Memory Utilization (%)')
        plt.title('Top 5 Servers with Highest Average Memory Utilization')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

        plt.ylim(bottom=1)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    with col3:
        plt.figure(figsize=(9, 7)) 
        bars = plt.bar(server_disk_mean['Server Name'].head(5), server_disk_mean['Average Disk Utilization (%)'].head(5), color='green')
        plt.xlabel('Servers')
        plt.ylabel('Average Disk Utilization (%)')
        plt.title('Top 5 Servers with Highest Average Disk Utilization')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

        plt.ylim(bottom=1)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(plt)



def display_trends_tab(cpu_data, memory_data, disk_data):
    col1, col2, col3 = st.columns(3)  # Add the first column for the header and second column for the dropdown

    with col1:
        st.subheader("📊 Key Metrics and Trends")
    
    with col2:
        # Add the dropdown to select prediction period (In Hours)
        prediction_period = st.selectbox(
            "⏳ Select prediction period (In Hours)",
            [1, 2, 6, 9],  # Prediction period options in hours
            key="prediction_period"
        )

    
    with col3:
        server_names = cpu_data['FQDN'].unique()
        selected_server = st.selectbox("🔧 Select a server", server_names, key="server_dropdown")

    if selected_server:
        server_cpu_data = cpu_data[cpu_data['FQDN'] == selected_server]
        server_memory_data = memory_data[memory_data['FQDN'] == selected_server]
        server_disk_data = disk_data[disk_data['FQDN'] == selected_server]  # Fetch disk data for selected server
    else:
        st.warning("Please select a server.")
        return

    cpu_utilization_column = "GBL_CPU_TOTAL_UTIL"
    memory_utilization_column = "GBL_MEM_UTIL"
    disk_utilization_column = "FS_SPACE_UTIL"  # Disk space utilization column

    if cpu_utilization_column not in cpu_data.columns or memory_utilization_column not in memory_data.columns or disk_utilization_column not in disk_data.columns:
        st.error(f"Columns {cpu_utilization_column}, {memory_utilization_column}, or {disk_utilization_column} not found in the dataset.")
        return
    
    # Convert the relevant columns to numeric to avoid errors
    cpu_data[cpu_utilization_column] = pd.to_numeric(cpu_data[cpu_utilization_column], errors='coerce')
    memory_data[memory_utilization_column] = pd.to_numeric(memory_data[memory_utilization_column], errors='coerce')
    disk_data[disk_utilization_column] = pd.to_numeric(disk_data[disk_utilization_column], errors='coerce')

    # Filter the data to remove missing values
    server_cpu_data = server_cpu_data[server_cpu_data[cpu_utilization_column].notna()]
    server_memory_data = server_memory_data[server_memory_data[memory_utilization_column].notna()]
    server_disk_data = server_disk_data[server_disk_data[disk_utilization_column].notna()]  # Filter out missing disk data

    time_features = ['Hour', 'Minute', 'DayOfWeek', 'IsWeekend']
    target_column = cpu_utilization_column

    numeric_columns = cpu_data.select_dtypes(include=np.number).drop(
        columns=['Datetime', 'FQDN', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend'], 
        errors='ignore'
    )

    correlations = numeric_columns.corr()[target_column].sort_values(ascending=False)
    top_features = correlations.index[1:4].tolist() 
    final_features = top_features + time_features

    all_predictions = []
    prediction_times = []

    # Machine learning models for prediction
    model_cpu = XGBRegressor(objective='reg:squarederror', random_state=42)
    model_memory = XGBRegressor(objective='reg:squarederror', random_state=42)
    model_disk = XGBRegressor(objective='reg:squarederror', random_state=42)  # New model for disk utilization

    # Prepare the data and make predictions
    X_cpu = server_cpu_data[final_features]
    y_cpu = server_cpu_data[cpu_utilization_column]
    X_memory = server_memory_data[final_features]
    y_memory = server_memory_data[memory_utilization_column]
    X_disk = server_disk_data[final_features]  # Use the same features for disk
    y_disk = server_disk_data[disk_utilization_column]

    # Train the models
    model_cpu.fit(X_cpu, y_cpu)
    model_memory.fit(X_memory, y_memory)
    model_disk.fit(X_disk, y_disk)  # Train the disk model

    # Modify the prediction to the selected period in hours
    prediction_hours = prediction_period  # Use the selected period in hours for prediction

    # Predict for the selected period (e.g., 1, 2, 6, or 9 hours)
    predictions_cpu = []
    predictions_memory = []
    predictions_disk = []

    last_hour_data_cpu = X_cpu.iloc[-12:]
    last_hour_data_memory = X_memory.iloc[-12:]
    last_hour_data_disk = X_disk.iloc[-12:]  # For disk utilization

    # Loop to predict for the full period (e.g., 1, 2, 6, or 9 hours)
    for i in range(prediction_hours):
        # For each time point, predict the value
        pred_cpu = model_cpu.predict(last_hour_data_cpu)
        pred_memory = model_memory.predict(last_hour_data_memory)
        pred_disk = model_disk.predict(last_hour_data_disk)

        predictions_cpu.append(pred_cpu[-1])  # Take the last prediction from the output
        predictions_memory.append(pred_memory[-1])
        predictions_disk.append(pred_disk[-1])

        # Shift the data for the next prediction (e.g., use the last predicted value as the input for the next prediction)
        last_hour_data_cpu = last_hour_data_cpu.shift(-1, fill_value=pred_cpu[-1])
        last_hour_data_memory = last_hour_data_memory.shift(-1, fill_value=pred_memory[-1])
        last_hour_data_disk = last_hour_data_disk.shift(-1, fill_value=pred_disk[-1])

    # Create prediction times for the full period (e.g., 1 hour, 2 hours, 6 hours, etc.)
    last_datetime = server_cpu_data["Datetime"].iloc[-1]
    prediction_times = pd.date_range(last_datetime, periods=prediction_hours + 1, freq="H")[1:]

    # Store predictions in the all_predictions list
    for time, pred in zip(prediction_times, predictions_cpu):
        all_predictions.append({
            "FQDN": selected_server,
            "Datetime": time,
            "GBL_CPU_TOTAL_UTIL": pred
        })
    for time, pred in zip(prediction_times, predictions_memory):
        all_predictions.append({
            "FQDN": selected_server,
            "Datetime": time,
            "GBL_MEM_UTIL": pred
        })
    for time, pred in zip(prediction_times, predictions_disk):
        all_predictions.append({
            "FQDN": selected_server,
            "Datetime": time,
            "FS_SPACE_UTIL": pred
        })

    # Plotting the predictions
    col1, col2, col3 = st.columns(3)  # Create 3 columns for CPU, Memory, and Disk plots

    with col1:
        st.write(f"#### Predicted Average CPU Utilization for Next {prediction_period} Hours: {np.mean(predictions_cpu):.2f}%")
        plt.figure(figsize=(9, 5))  
        plt.plot(server_cpu_data["Datetime"], server_cpu_data[cpu_utilization_column], label='Actual CPU Utilization', color='blue', linestyle='-', marker='o')
        plt.plot(prediction_times, predictions_cpu, label='Predicted CPU Utilization', color='green', linestyle='-', marker='s')
        plt.title(f"Actual vs Predicted CPU Utilization for {selected_server}")
        plt.xlabel("Datetime")
        plt.ylabel("CPU Utilization (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    with col2:
        st.write(f"#### Predicted Average Memory Utilization for Next {prediction_period} Hours: {np.mean(predictions_memory):.2f}%")
        plt.figure(figsize=(9, 5)) 
        plt.plot(server_memory_data["Datetime"], server_memory_data[memory_utilization_column], label='Actual Memory Utilization', color='blue', linestyle='-', marker='o')
        plt.plot(prediction_times, predictions_memory, label='Predicted Memory Utilization', color='green', linestyle='-', marker='s')
        plt.title(f"Actual vs Predicted Memory Utilization for {selected_server}")
        plt.xlabel("Datetime")
        plt.ylabel("Memory Utilization (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    with col3:
        st.write(f"#### Predicted Average Disk Utilization for Next {prediction_period} Hours: {np.mean(predictions_disk):.2f}%")
        plt.figure(figsize=(9, 5))  
        
        if not server_disk_data.empty:
            # Plot actual disk utilization if data exists
            plt.plot(server_disk_data["Datetime"], server_disk_data[disk_utilization_column], 
                    label='Actual Disk Utilization', color='blue', linestyle='-', marker='o')
        else:
            st.warning("No actual disk utilization data available.")
        
        # Plot predicted disk utilization
        plt.plot(prediction_times, predictions_disk, label='Predicted Disk Utilization', color='green', linestyle='-', marker='s')
        
        plt.title(f"Actual vs Predicted Disk Utilization for {selected_server}")
        plt.xlabel("Datetime")
        plt.ylabel("Disk Utilization (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)


    # Save the predictions to an Excel file
    predictions_df = pd.DataFrame(all_predictions)
    output_file = "predictions_output.xlsx"
    predictions_df.to_excel(output_file, index=False)






def display_host_performance_tab(cpu_data, memory_data, disk_data):

    col1, col2 = st.columns([8, 2]) 
    with col1:
        st.subheader("🖥️ Host Performance Metrics 📈")
    
    with col2:
        server_names = cpu_data['FQDN'].unique()
        selected_server = st.selectbox("🔧 Select a server", server_names, key="server_dropdown_host_performance")

    if selected_server:
        server_cpu_data = cpu_data[cpu_data['FQDN'] == selected_server]
        server_memory_data = memory_data[memory_data['FQDN'] == selected_server]
        server_disk_data = disk_data[disk_data['FQDN'] == selected_server]  # Fetch disk data for selected server
    else:
        st.warning("⚠️ Please select a server.")
        return

    # Clean CPU data
    server_cpu_data['GBL_CPU_TOTAL_UTIL'] = pd.to_numeric(server_cpu_data['GBL_CPU_TOTAL_UTIL'], errors='coerce')
    server_cpu_data = server_cpu_data[server_cpu_data['GBL_CPU_TOTAL_UTIL'].notna()]
    server_cpu_data = server_cpu_data[server_cpu_data['GBL_CPU_TOTAL_UTIL'] <= 100.0] 

    overall_cpu_utilization = server_cpu_data['GBL_CPU_TOTAL_UTIL'].mean()

    # Clean Memory data
    server_memory_data['GBL_MEM_UTIL'] = pd.to_numeric(server_memory_data['GBL_MEM_UTIL'], errors='coerce')
    server_memory_data = server_memory_data[server_memory_data['GBL_MEM_UTIL'].notna()]
    server_memory_data = server_memory_data[server_memory_data['GBL_MEM_UTIL'] <= 100.0]

    overall_memory_utilization = server_memory_data['GBL_MEM_UTIL'].mean()

    # Clean Disk data
    server_disk_data['FS_SPACE_UTIL'] = pd.to_numeric(server_disk_data['FS_SPACE_UTIL'], errors='coerce')
    server_disk_data = server_disk_data[server_disk_data['FS_SPACE_UTIL'].notna()]
    server_disk_data = server_disk_data[server_disk_data['FS_SPACE_UTIL'] <= 100.0] 

    overall_disk_utilization = server_disk_data['FS_SPACE_UTIL'].mean()  # Average disk utilization

    # Define features for prediction
    time_features = ['Hour', 'Minute', 'DayOfWeek', 'IsWeekend']
    final_features = time_features  

    # CPU prediction model
    X_cpu = server_cpu_data[final_features]
    y_cpu = server_cpu_data['GBL_CPU_TOTAL_UTIL']
    X_cpu = X_cpu[~y_cpu.isna()]
    y_cpu = y_cpu[~y_cpu.isna()]
    model_cpu = XGBRegressor(objective='reg:squarederror', random_state=42)
    model_cpu.fit(X_cpu, y_cpu)

    last_hour_data_cpu = X_cpu.iloc[-12:]
    predictions_cpu = model_cpu.predict(last_hour_data_cpu)
    predicted_avg_cpu = predictions_cpu.mean()

    # Memory prediction model
    X_memory = server_memory_data[final_features]
    y_memory = server_memory_data['GBL_MEM_UTIL']
    X_memory = X_memory[~y_memory.isna()]
    y_memory = y_memory[~y_memory.isna()]
    model_memory = XGBRegressor(objective='reg:squarederror', random_state=42)
    model_memory.fit(X_memory, y_memory)

    last_hour_data_memory = X_memory.iloc[-12:]
    predictions_memory = model_memory.predict(last_hour_data_memory)
    predicted_avg_memory = predictions_memory.mean()

    # Disk prediction model (same as memory and CPU)
    X_disk = server_disk_data[final_features]
    y_disk = server_disk_data['FS_SPACE_UTIL']
    X_disk = X_disk[~y_disk.isna()]
    y_disk = y_disk[~y_disk.isna()]
    model_disk = XGBRegressor(objective='reg:squarederror', random_state=42)
    model_disk.fit(X_disk, y_disk)

    last_hour_data_disk = X_disk.iloc[-12:]
    predictions_disk = model_disk.predict(last_hour_data_disk)
    predicted_avg_disk = predictions_disk.mean()

    # Risk Level for CPU
    if overall_cpu_utilization < 65 and predicted_avg_cpu < 65:
        cpu_color = 'green'
        cpu_status = "🟢 Normal (<65%)"
        cpu_response_action = "Low Risk: Continue monitoring and maintenance."
    elif overall_cpu_utilization < 65:
        cpu_color = 'green'
        cpu_status = "🟢 Normal (<65%)"
        cpu_response_action = "Low Risk: Continue monitoring and maintenance."
    elif 65 <= overall_cpu_utilization <= 85 or 65 <= predicted_avg_cpu <= 85:
        cpu_color = 'yellow'
        cpu_status = "🟡 At Risk (65-85%)"
        cpu_response_action = "Medium Risk: Develop improvement plan within 24 hours."
    else:
        cpu_color = 'red'
        cpu_status = "🔴 Reached Threshold (>85%)"
        cpu_response_action = "High Risk: Immediate investigation and corrective action required."

    # Risk Level for Memory
    if overall_memory_utilization < 65 and predicted_avg_memory < 65:
        memory_color = 'green'
        memory_status = "🟢 Normal (<65%)"
        memory_response_action = "Low Risk: Continue monitoring and maintenance."
    elif overall_memory_utilization < 65:
        memory_color = 'green'
        memory_status = "🟢 Normal (<65%)"
        memory_response_action = "Low Risk: Continue monitoring and maintenance."
    elif 65 <= overall_memory_utilization <= 85 or 65 <= predicted_avg_memory <= 85:
        memory_color = 'yellow'
        memory_status = "🟡 At Risk (65-85%)"
        memory_response_action = "Medium Risk: Develop improvement plan within 24 hours."
    else:
        memory_color = 'red'
        memory_status = "🔴 Reached Threshold (>85%)"
        memory_response_action = "High Risk: Immediate investigation and corrective action required."

    # Risk Level for Disk
    if overall_disk_utilization < 65 and predicted_avg_disk < 65:
        disk_color = 'green'
        disk_status = "🟢 Normal (<65%)"
        disk_response_action = "Low Risk: Continue monitoring and maintenance."
    elif overall_disk_utilization < 65:
        disk_color = 'green'
        disk_status = "🟢 Normal (<65%)"
        disk_response_action = "Low Risk: Continue monitoring and maintenance."
    elif 65 <= overall_disk_utilization <= 85 or 65 <= predicted_avg_disk <= 85:
        disk_color = 'yellow'
        disk_status = "🟡 At Risk (65-85%)"
        disk_response_action = "Medium Risk: Develop improvement plan within 24 hours."
    else:
        disk_color = 'red'
        disk_status = "🔴 Reached Threshold (>85%)"
        disk_response_action = "High Risk: Immediate investigation and corrective action required."

    # Display CPU, Memory, and Disk Status
    col1, col2 = st.columns([1, 1])  

    with col1:
        st.markdown(f"#### Overall CPU Utilization: {overall_cpu_utilization:.2f}%")
        if predicted_avg_cpu > overall_cpu_utilization:
            cpu_arrow = "↑"  
            cpu_arrow_color = 'green'
        elif predicted_avg_cpu < overall_cpu_utilization:
            cpu_arrow = "↓" 
            cpu_arrow_color = 'red'
        else:
            cpu_arrow = "-"  
            cpu_arrow_color = 'gray'

        st.markdown(f"#### Predicted Average CPU Utilization for Next Hour: {predicted_avg_cpu:.2f}% {cpu_arrow}", unsafe_allow_html=True)
        st.markdown(f'<div style="color:{cpu_color}; font-size:24px;">{cpu_status}</div>', unsafe_allow_html=True)
        st.write(cpu_response_action)


    with col2:
        st.markdown(f"#### Overall Memory Utilization: {overall_memory_utilization:.2f}%")
        if predicted_avg_memory > overall_memory_utilization:
            memory_arrow = "↑"  
            memory_arrow_color = 'green'
        elif predicted_avg_memory < overall_memory_utilization:
            memory_arrow = "↓" 
            memory_arrow_color = 'red'
        else:
            memory_arrow = "-"  
            memory_arrow_color = 'gray'

        st.markdown(f"#### Predicted Average Memory Utilization for Next Hour: {predicted_avg_memory:.2f}% {memory_arrow}", unsafe_allow_html=True)
        st.markdown(f'<div style="color:{memory_color}; font-size:24px;">{memory_status}</div>', unsafe_allow_html=True)
        st.write(memory_response_action)

    with col1:
        st.markdown(f"#### Overall Disk Utilization: {overall_disk_utilization:.2f}%")
        if predicted_avg_disk > overall_disk_utilization:
            disk_arrow = "↑"  
            disk_arrow_color = 'green'
        elif predicted_avg_disk < overall_disk_utilization:
            disk_arrow = "↓" 
            disk_arrow_color = 'red'
        else:
            disk_arrow = "-"  
            disk_arrow_color = 'gray'

        st.markdown(f"#### Predicted Average Disk Utilization for Next Hour: {predicted_avg_disk:.2f}% {disk_arrow}", unsafe_allow_html=True)
        st.markdown(f'<div style="color:{disk_color}; font-size:24px;">{disk_status}</div>', unsafe_allow_html=True)
        st.write(disk_response_action)

    with col2:
        st.write(""" #### Utilization Status Indicators:
    - 🟢 Normal (<65%)
    - 🟡 At Risk (65-85%)
    - 🔴 Reached Threshold (>85%)
                    """)

        st.write(""" ### Response Actions:
    - High Risk (Red): Immediate investigation and corrective action required
    - Medium Risk (Yellow): Develop improvement plan within 24 hours
    - Low Risk (Green): Continue monitoring and maintenance
                    """)




def display_info_tab():
    st.subheader("ℹ️ About System Metrics Dashboard")
    st.write("""
    Welcome to the **System Metrics Prediction Dashboard**! This dashboard provides insights into server performance metrics, helping IT administrators monitor and optimize the health of servers in real-time. The dashboard leverages historical data to predict future server states, ensuring efficient resource management and proactive troubleshooting.
    """)
    st.write("""
    This dashboard analyzes key server performance metrics, including:
    - **CPU Utilization**: Track how much of the CPU's total processing power is in use.
    - **Memory Utilization**: Monitor the percentage of RAM in use by processes.
    - **Disk I/O Rates**: Analyze read/write operations on the server’s disk drives.
    - **Network Activity**: Evaluate the incoming and outgoing packet rates across the server network.

    The **XGBoost Machine Learning Model** is used to predict future CPU utilization based on past data trends. This enables system administrators to forecast server performance, avoid system overloads, and take necessary actions before performance degradation occurs.
    """)

    st.write(""" ##### Key Features:
    - **Real-time Monitoring**: View real-time CPU, memory, disk, and network metrics.
    - **Historical Data Insights**: Analyze historical performance trends and identify anomalies.
    - **Predictions**: Get predicted server performance for the next hour based on past data, using an advanced machine learning model.
    """)

    st.write("""
    The dashboard uses a correlation-based approach to identify which metrics are most strongly associated with CPU utilization. This helps in selecting the most relevant features for machine learning predictions, improving the accuracy of future forecasts.
    """)

    st.write("""
    In addition to real-time insights, the dashboard generates predictions for the next hour based on current trends in CPU utilization. This proactive approach can assist in:
    - **Capacity Planning**: Predicting when additional resources might be needed.
    - **Anomaly Detection**: Identifying sudden spikes or drops in system performance that could indicate potential issues.
    """)

    st.write("""
    Use the tabs above to explore different analyses, such as key metrics and trends, host performance, and more detailed predictions. This dashboard is designed to be an all-in-one solution for server health monitoring, enabling efficient and effective system management.
    """)

def display_insights_tab(cpu_data, memory_data, disk_data):
    st.subheader("🔍 **Insights** 📊")

    # Clean CPU data
    cpu_data['GBL_CPU_TOTAL_UTIL'] = pd.to_numeric(cpu_data['GBL_CPU_TOTAL_UTIL'], errors='coerce')
    cpu_data = cpu_data.dropna(subset=['GBL_CPU_TOTAL_UTIL'])

    # Clean Memory data
    memory_data['GBL_MEM_UTIL'] = pd.to_numeric(memory_data['GBL_MEM_UTIL'], errors='coerce')
    memory_data = memory_data.dropna(subset=['GBL_MEM_UTIL'])

    # Clean Disk data
    disk_data['FS_SPACE_UTIL'] = pd.to_numeric(disk_data['FS_SPACE_UTIL'], errors='coerce')
    disk_data = disk_data.dropna(subset=['FS_SPACE_UTIL'])

    # CPU Insights
    highest_cpu_server = cpu_data.groupby('FQDN')['GBL_CPU_TOTAL_UTIL'].mean().idxmax()
    lowest_cpu_server = cpu_data.groupby('FQDN')['GBL_CPU_TOTAL_UTIL'].mean().idxmin()
    avg_cpu_utilization = cpu_data.groupby('FQDN')['GBL_CPU_TOTAL_UTIL'].mean()
    attention_cpu_servers = avg_cpu_utilization[avg_cpu_utilization > 85].index.tolist()

    # Memory Insights
    highest_memory_server = memory_data.groupby('FQDN')['GBL_MEM_UTIL'].mean().idxmax()
    lowest_memory_server = memory_data.groupby('FQDN')['GBL_MEM_UTIL'].mean().idxmin()
    avg_memory_utilization = memory_data.groupby('FQDN')['GBL_MEM_UTIL'].mean()
    attention_memory_servers = avg_memory_utilization[avg_memory_utilization > 85].index.tolist()

    # Disk Insights
    highest_disk_server = disk_data.groupby('FQDN')['FS_SPACE_UTIL'].mean().idxmax()
    lowest_disk_server = disk_data.groupby('FQDN')['FS_SPACE_UTIL'].mean().idxmin()
    avg_disk_utilization = disk_data.groupby('FQDN')['FS_SPACE_UTIL'].mean()
    attention_disk_servers = avg_disk_utilization[avg_disk_utilization > 85].index.tolist()

    # Display CPU Insights
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 🏆 Highest CPU Utilization Server:")
        st.write(f" {highest_cpu_server}")
    with col2:
        st.write("#### ❄️ Lowest CPU Utilization Server:")
        st.write(f" {lowest_cpu_server}")

    # Display Memory Insights
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 🧠 Highest Memory Utilization Server:")
        st.write(f" {highest_memory_server}")
    with col2:
        st.write("#### 🧊 Lowest Memory Utilization Server:")
        st.write(f" {lowest_memory_server}")

    # Display Disk Insights
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 💾 Highest Disk Utilization Server:")
        st.write(f" {highest_disk_server}")
    with col2:
        st.write("#### 💿 Lowest Disk Utilization Server:")
        st.write(f" {lowest_disk_server}")

    # Display Attention Insights for CPU, Memory, and Disk
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### ⚠️ Servers Needing Attention (CPU > 85%):")
        if len(attention_cpu_servers) > 0:
            st.write("\n".join(attention_cpu_servers))
        else:
            st.write("No servers need attention based on CPU utilization. ✅")
    with col1:
        st.write("#### ⚠️ Servers Needing Attention (Disk > 85%):")
        if len(attention_disk_servers) > 0:
            st.write("\n".join(attention_disk_servers))
        else:
            st.write("No servers need attention based on Disk utilization. ✅") 
    with col2:
        st.write("#### ⚠️ Servers Needing Attention (Memory > 85%):")
        if len(attention_memory_servers) > 0:
            st.write("\n".join(attention_memory_servers))
        else:
            st.write("No servers need attention based on Memory utilization. ✅")

    


def main():
    
    st.markdown("""
    <style>
        body {
            background-color: #D3D3D3;  /* Light grey background */
            font-family: 'Arial', sans-serif; /* Change font style */
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            position: relative;
            margin-top: 0px;
            width: 100%;
        }
        .dashboard-title {
            font-size: 40px;
            font-weight: bold;
            white-space: nowrap;  /* Prevent text from wrapping */
            animation: scrollTitle 10s linear infinite; /* Animation for scrolling text */
            margin-right: 240px;  /* Add enough space for the logo */
            color: #4B0082;  /* Purple color for text */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); /* Shadow effect for text */
        }
        .logo-container {
            position: absolute;
            right: 20px;  /* Adjust this value if needed */
            top: 0;
        }
        .graphic-container {
            position: absolute;
            left: 0;
            bottom: 10px;
            display: flex;
            gap: 10px;
        }
        .graphic-container img {
            width: 50px; /* Size of the icons */
            opacity: 0.8;
        }
        @keyframes scrollTitle {
            0% {
                transform: translateX(50%); /* Start off-screen to the right */
            }
            100% {
                transform: translateX(-1%); /* Move to off-screen to the left */
            }
        }
    </style>

    <div class="header-container">
        <h1 class="dashboard-title">🎯 System Metrics Prediction Dashboard</h1>
        <div class="logo-container">
            <img src="https://dxc.com/content/dam/dxc/projects/dxc-com/us/images/about-us/newsroom/logos-for-media/vertical/DXC%20Logo_Purple+Black%20RGB.png" 
                 alt="DXC Logo" width="120"> <!-- Adjust the logo size if needed -->
        </div>
    </div>
""", unsafe_allow_html=True)

    with st.spinner("Processing data..."):
        try:
            cpu_data, memory_data, disk_data = load_data(file_path)


            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([ 
                "📈 Overview", 
                "🔍 Insights",
                "📊 Key Metrics and Trends", 
                "🖥️ Host Performance", 
                "ℹ️ About System Metrics Dashboard"
            ])
            
            with tab1:
                display_overview_tab(cpu_data, memory_data, disk_data) 
            with tab3:
                display_trends_tab(cpu_data, memory_data, disk_data) 
            with tab4:
                display_host_performance_tab(cpu_data, memory_data, disk_data) 
            with tab2:
                display_insights_tab(cpu_data, memory_data, disk_data)
            with tab5:
                display_info_tab()

        except Exception as e:
            st.error(f"An error occurred: {e}")

        st.markdown(
            """
            <div class="floating-footer">
                © 2024 DXC Technology
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    # Initialize session state if it doesn't exist
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = True

    # Add space before the button
    st.markdown("<div style='margin-bottom: 37px'></div>", unsafe_allow_html=True)

    # Check if the button has already been clicked
    if not st.session_state.button_clicked:
        landing_page()  # Show the landing page initially
        if st.button("System Metrics Dashboard"):
            st.session_state.button_clicked = True  # Mark that the button has been clicked

    # Show the main content if the button has been clicked
    if st.session_state.button_clicked:
        main()
