import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df=pd.read_pickle("../../data/interim/01_data_processed.pkl")
df
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["Set"] == 1]
plt.plot(set_df["Acc_y"])

plt.plot(set_df["Acc_y"].reset_index(drop=True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["Acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["Acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("Participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["Category"])["Acc_y"].plot()
ax.set_ylabel("Acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("Participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["Participant"])["Acc_y"].plot()
ax.set_ylabel("Acc_y")
ax.set_xlabel("Samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
Participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(f"Participant == '{Participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["Acc_x", "Acc_y", "Acc_z"]].plot(ax=ax)
ax.set_ylabel("Acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
Participants = df["Participant"].unique()

for label in labels:
    for Participant in Participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"Participant == '{Participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:

            fig, ax = plt.subplots()
            all_axis_df[["Acc_x", "Acc_y", "Acc_z"]].plot(ax=ax)
            ax.set_ylabel("Acc_y")
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({Participant})".title())
            plt.legend()

for label in labels:
    for Participant in Participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"Participant == '{Participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:

            fig, ax = plt.subplots()
            all_axis_df[["Gyr_x", "Gyr_y", "Gyr_z"]].plot(ax=ax)
            ax.set_ylabel("Gyr_y")
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({Participant})".title())
            plt.legend()
# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
Participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"Participant == '{Participant}'")
    .reset_index(drop=True))

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
all_axis_df[["Acc_x", "Acc_y", "Acc_z"]].plot(ax=ax[0])
all_axis_df[["Gyr_x", "Gyr_y", "Gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("Samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
Participants = df["Participant"].unique()

for label in labels:
    for Participant in Participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"Participant == '{Participant}'")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            combined_plot_df[["Acc_x", "Acc_y", "Acc_z"]].plot(ax=ax[0])
            combined_plot_df[["Gyr_x", "Gyr_y", "Gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("Samples")


            plt.savefig(f"../../reports/figures/{label.title()} ({Participant}).png")

            plt.show()