import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

acc_r = df["Acc_x"] ** 2 + df["Acc_y"] ** 2 + df["Acc_z"] ** 2
gyr_r = df["Gyr_x"] ** 2 + df["Gyr_y"] ** 2 + df["Gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = squat_df
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Acc_x"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Acc_y"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Acc_z"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Gyr_x"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Gyr_y"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["Gyr_z"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000/200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["Set"] == bench_df["Set"].unique()[0]]
squat_set = squat_df[squat_df["Set"] == squat_df["Set"].unique()[0]]
row_set = row_df[row_df["Set"] == row_df["Set"].unique()[0]]
ohp_set = ohp_df[ohp_df["Set"] == ohp_df["Set"].unique()[0]]
dead_set = dead_df[dead_df["Set"] == dead_df["Set"].unique()[0]]

bench_set["acc_r"].plot()

column = "Acc_y"
LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color = "red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["Category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

    return len(peaks)

count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="Gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["Reps"] = df["Category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "Category", "Set"])["Reps"].max().reset_index()
rep_df["Reps_pred"] = 0

for s in df["Set"].unique():
    subset = df[df["Set"] == s]

    column = "acc_r"
    cutoff = 0.4

    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    
    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
        col = "Gyr_x"

    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_reps(subset, cutoff=cutoff, column=column)

    rep_df.loc[rep_df["Set"] == s, "Reps_pred"] = reps

rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["Reps"], rep_df["Reps_pred"]).round(2)
rep_df.groupby(["label", "Category"])["Reps", "Reps_pred"].mean().plot.bar()
